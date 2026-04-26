#!/bin/bash
# bench_omni.sh - Build, run 10x, and parse MNN Omni audio pipeline benchmarks
# Usage: bash bench_omni.sh <branch> [config_name]
# Example: bash bench_omni.sh raw
#          bash bench_omni.sh v2 async_config
set -euo pipefail

BRANCH="${1:?Usage: bench_omni.sh <branch> [config_name]}"
CONFIG_NAME="${2:-config}"
# strip .json suffix if user included it
CONFIG_NAME="${CONFIG_NAME%.json}"
RESULTS_DIR="$HOME/Project/results/$BRANCH"
MODEL_DIR="$HOME/Project/models/qwen2.5"
BUILD_DIR="$HOME/Project/github/MNN/build"
REPO_DIR="$HOME/Project/github/MNN"
PARSER_SRC="$REPO_DIR/transformers/llm/engine/src/parse_perf_log.py"
GOOGLETEST_CACHE="$HOME/.cache/mnn/googletest"

echo "=== bench_omni: branch=$BRANCH config=${CONFIG_NAME}.json ==="

# --- 1. checkout ---
cd "$REPO_DIR"
git checkout "$BRANCH"

# --- 2. build ---
echo "--- Building ---"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
mkdir -p tmp
echo "Who are you?" > prompt.txt

# ensure googletest is cached locally (avoid network download every time)
if [ ! -d "$GOOGLETEST_CACHE" ]; then
    echo "  Downloading googletest (one-time) ..."
    mkdir -p "$(dirname "$GOOGLETEST_CACHE")"
    git clone --depth 1 https://github.com/google/googletest.git "$GOOGLETEST_CACHE" 2>/dev/null || {
        echo "  WARNING: googletest download failed, trying to build without tests"
        GOOGLETEST_CACHE=""
    }
fi

cmake .. -G Ninja \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_SHARED_LIBS=OFF \
    -DMNN_BUILD_LLM=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_BUILD_AUDIO=ON \
    -DMNN_OPENCL=ON \
    -DMNN_USE_SSE=ON \
    -DCMAKE_CXX_FLAGS="-DDUMP_TALKER_PERFORMANCE -DENABLE_PERF_LOGGING" \
    ${GOOGLETEST_CACHE:+-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$GOOGLETEST_CACHE}

ninja llm_demo
echo "--- Build OK ---"

# --- 3. run 10x ---
echo "--- Running 10 benchmarks ---"
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/perf_log_run_*.txt

for i in $(seq 1 10); do
    printf "  Run %2d/10 ... " "$i"
    if ./llm_demo "$MODEL_DIR/${CONFIG_NAME}.json" prompt.txt > /dev/null 2>&1; then
        mv perf_log.txt "$RESULTS_DIR/perf_log_run_${i}.txt"
        echo "done"
    else
        rc=$?
        echo "FAILED (exit=$rc)"
        if [ -f perf_log.txt ]; then
            mv perf_log.txt "$RESULTS_DIR/perf_log_run_${i}.txt"
            echo "         (partial log saved)"
        fi
    fi
done

# --- 4. parse ---
echo "--- Parsing results ---"
PARSER="$RESULTS_DIR/parse_perf_log.py"
if [ -f "$PARSER_SRC" ]; then
    cp "$PARSER_SRC" "$PARSER"
fi

for i in $(seq 1 10); do
    LOG="$RESULTS_DIR/perf_log_run_${i}.txt"
    if [ -f "$PARSER" ]; then
        python3 "$PARSER" "$LOG" --json    > "$RESULTS_DIR/perf_run_${i}.json"    2>/dev/null || true
        python3 "$PARSER" "$LOG" --csv     > "$RESULTS_DIR/perf_run_${i}.csv"     2>/dev/null || true
        python3 "$PARSER" "$LOG" --summary > "$RESULTS_DIR/perf_run_${i}_summary.txt" 2>/dev/null || true
    fi
done

# --- 5. compute averages ---
echo "--- Computing averages ---"
python3 -c "
import json, glob, statistics
d = '$RESULTS_DIR'
files = sorted(glob.glob(f'{d}/perf_run_*.json'))
if not files:
    print('No JSON files found, skipping average.')
    exit(0)
summaries = []
for f in files:
    with open(f) as fp:
        data = json.load(fp)
        for rec in data.get('raw_records', []):
            if rec.get('stage') == 'summary':
                summaries.append(rec); break
if not summaries:
    print('No summary records in JSON, computing from phase data...')
    all_dit, all_voc, all_total = [], [], []
    for f in files:
        with open(f) as fp:
            for rec in json.load(fp).get('raw_records', []):
                p = rec.get('phase','')
                if p == 'dit_end': all_dit.append(rec.get('dit_time_ms',0))
                elif p == 'vocoder_end': all_voc.append(rec.get('vocoder_time_ms',0))
                elif p == 'generate_end': all_total.append(rec.get('total_time_ms',0))
    avg = {
        'runs': len(files),
        'dit_samples': len(all_dit), 'voc_samples': len(all_voc),
        'avg_dit_ms': {'mean': statistics.mean(all_dit) if all_dit else 0, 'stddev': statistics.stdev(all_dit) if len(all_dit)>1 else 0},
        'avg_vocoder_ms': {'mean': statistics.mean(all_voc) if all_voc else 0, 'stddev': statistics.stdev(all_voc) if len(all_voc)>1 else 0},
        'total_time_ms': {'mean': statistics.mean(all_total) if all_total else 0},
    }
else:
    dit  = [r.get('avg_dit_ms',0) for r in summaries]
    voc  = [r.get('avg_vocoder_ms',0) for r in summaries]
    chk  = [r.get('total_chunks',0) for r in summaries]
    avg = {
        'runs': len(summaries),
        'total_chunks': {'mean': statistics.mean(chk)},
        'avg_dit_ms':     {'mean': statistics.mean(dit), 'stddev': statistics.stdev(dit) if len(dit)>1 else 0, 'min': min(dit), 'max': max(dit)},
        'avg_vocoder_ms': {'mean': statistics.mean(voc), 'stddev': statistics.stdev(voc) if len(voc)>1 else 0, 'min': min(voc), 'max': max(voc)},
    }
with open(f'{d}/perf_avg.json','w') as fp: json.dump(avg, fp, indent=2)
with open(f'{d}/perf_avg_summary.txt','w') as fp:
    fp.write(f'$BRANCH Branch - {len(files)}-Run Average\\n')
    fp.write('=' * 50 + '\\n')
    for k,v in avg.items():
        if isinstance(v, dict) and 'mean' in v:
            fp.write(f'{k}: mean={v[\"mean\"]:.3f} stddev={v.get(\"stddev\",0):.3f}\\n')
print(f'  avg_dit={avg.get(\"avg_dit_ms\",{}).get(\"mean\",\"N/A\")}  avg_voc={avg.get(\"avg_vocoder_ms\",{}).get(\"mean\",\"N/A\")}')
"

echo "=== Done: results in $RESULTS_DIR ==="
ls -la "$RESULTS_DIR"/perf_log_run_*.txt | wc -l | xargs echo "  Log files:"
ls -la "$RESULTS_DIR"/perf_avg.json 2>/dev/null && echo "  Averages: OK" || echo "  Averages: MISSING"
