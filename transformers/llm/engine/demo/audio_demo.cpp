//
//  audio_demo.cpp
//
//  示例：使用 MNN LLM Omni 模型做语音输入 + 语音输出
//
//  依赖：
//  1. 编译时打开：
//     -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON
//  2. 模型目录为通过 llmexport.py --omni 导出的 Qwen2.5-Omni，
//     例如：D:/Project/models/qwen2.5 内含 config.json / llm.mnn / talker.mnn 等文件。
//

#include "llm/llm.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/AutoTime.hpp>

#include "audio/audio.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Transformer;

static void saveWaveToFile(const std::vector<float>& waveform,
                           const std::string& outPath,
                           int sampleRate = 24000) {
    if (waveform.empty()) {
        MNN_PRINT("No waveform data, skip save.\n");
        return;
    }
    auto var = _Const(waveform.data(), {(int)waveform.size()}, NCHW, halide_type_of<float>());
    bool ok = AUDIO::save(outPath.c_str(), var, sampleRate);
    if (!ok) {
        MNN_ERROR("Save wav to %s failed.\n", outPath.c_str());
    } else {
        MNN_PRINT("Waveform saved to: %s\n", outPath.c_str());
    }
}

/**
 * 构造一个带音频输入标签的 prompt：
 *   <audio>audio_path</audio> + 用户自定义问题
 */
static std::string buildAudioPrompt(const std::string& audioPath,
                                    const std::string& userQuestion) {
    std::ostringstream os;
    os << "<audio>" << audioPath << "</audio>";
    if (!userQuestion.empty()) {
        os << userQuestion;
    } else {
        os << "请你用中文总结一下这段音频的内容。";
    }
    return os.str();
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0]
                  << " <config.json> <audio.wav> [output.wav] [question]\n\n";
        std::cout << "示例：\n";
        std::cout << "  " << argv[0]
                  << " D:/Project/models/qwen2.5/config.json "
                     "D:/audio/input.wav output.wav \"请把内容翻译成中文并简要概括\"\n";
        return 0;
    }

    std::string configPath = argv[1];
    std::string audioPath  = argv[2];
    std::string outWave    = "output.wav";
    if (argc >= 4) {
        outWave = argv[3];
    }

    std::string question;
    if (argc >= 5) {
        // 剩余参数合并成一句话
        std::ostringstream q;
        for (int i = 4; i < argc; ++i) {
            if (i > 4) q << " ";
            q << argv[i];
        }
        question = q.str();
    }

    std::cout << "Config : " << configPath << std::endl;
    std::cout << "Audio  : " << audioPath << std::endl;
    std::cout << "OutWav : " << outWave << std::endl;

    // 为了和官方 demo 行为一致，这里仍然创建 ExecutorScope
    BackendConfig backendConfig;
    auto executor = Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    Express::ExecutorScope scope(executor);

    std::unique_ptr<Llm> llm(Llm::createLLM(configPath));
    if (!llm) {
        MNN_ERROR("Create LLM failed, check config.json path.\n");
        return 1;
    }

    // 可选配置：设置临时目录（mmap / kv mmap 会用到）
    llm->set_config(R"({"tmp_path":"tmp"})");
    // 关闭异步生成，保证 response 返回前已经完成文本解码
    // 否则 context->gen_seq_len 会一直是 0，generateWavform 也拿不到对应语音 token
    llm->set_config(R"({"async":false})");

    {
        AUTOTIME;
        bool ok = llm->load();
        if (!ok) {
            MNN_ERROR("LLM load failed, please check your Omni model files.\n");
            return 1;
        }
    }

    // 如果需要，可以在这里覆盖部分 Omni 语音相关配置（也可以直接写在模型目录下的 config.json 里）
    // 例如：限定语音长度 / 选择说话人：
    //
    // llm->set_config(R"({
    //     "talker_max_new_tokens": 1200,
    //     "talker_speaker": "Chelsie"
    // })");

    // 收集并保存语音输出
    std::vector<float> waveform;
    llm->setWavformCallback([&](const float* ptr, size_t size, bool last_chunk) {
        waveform.insert(waveform.end(), ptr, ptr + size);
        if (last_chunk) {
            saveWaveToFile(waveform, outWave);
            waveform.clear();
        }
        // 返回 true 表示继续接收后续音频块
        return true;
    });

    // 构造带音频标签的 prompt
    auto prompt = buildAudioPrompt(audioPath, question);
    std::cout << "\n==== Prompt ====\n" << prompt << "\n================\n";

    auto context = llm->getContext();

    // 文本 + 语音生成（文本走 stdout，语音通过回调单独保存）
    //
    // 注意：这里必须让 LLM 实际「解码生成」文本 token，
    // 才能驱动 Omni 的 Talker 分支生成对应语音。
    // 不能把 max_new_tokens 设为 0，否则 gen_seq_len 始终为 0，语音会很短且无意义。
    //
    // 这里直接使用默认生成逻辑，相当于：直到 EOS 或达到 config.json 里配置的 max_new_tokens。
    llm->response(prompt, &std::cout);

    // 触发 Omni Talker 部分，把已经生成好的语音 token 转成波形
    llm->generateWavform();

    std::cout << "\n\n===== Stats =====\n";
    std::cout << "Prompt tokens : " << context->prompt_len << "\n";
    std::cout << "Decode tokens : " << context->gen_seq_len << "\n";
    std::cout << "Audio input s : " << context->audio_input_s << "\n";
    std::cout << "Audio proc  s : " << (context->audio_us / 1e6) << "\n";
    if (context->audio_input_s > 0.0f) {
        std::cout << "Audio RTF     : "
                  << (context->audio_us / 1e6 / context->audio_input_s) << "\n";
    }
    std::cout << "=================\n";

    return 0;
}

