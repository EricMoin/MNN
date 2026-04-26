@echo off

set ANDROID_NDK="C:/Users/wu_mi/AppData/Local/Android/Sdk/ndk/27.0.12077973"

rem Parallel build jobs for ninja/cmake
set "JOBS=8"
rem ==============================================================

rem Resolve repo root (this script lives in MNN\.build\)
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "MNN_ROOT=%%~fI"

echo [INFO] MNN_ROOT: %MNN_ROOT%
echo [INFO] ANDROID_NDK: %ANDROID_NDK%

if not exist "%MNN_ROOT%\project\android" (
  echo [ERROR] Expected "%MNN_ROOT%\project\android" not found.
  exit /b 1
)

if not exist "%ANDROID_NDK%\build\cmake\android.toolchain.cmake" (
  echo [ERROR] android.toolchain.cmake not found under ANDROID_NDK.
  echo         Got: "%ANDROID_NDK%\build\cmake\android.toolchain.cmake"
  exit /b 1
)

call :require_tool cmake || exit /b 1
call :require_tool ninja || exit /b 1

call :build_mnn || exit /b 1

echo [OK] Done.
exit /b 0

rem --------------------------------------------------------------
:require_tool
where %~1 >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Required tool not found on PATH: %~1
  exit /b 1
)
exit /b 0

rem --------------------------------------------------------------
:build_mnn
set "BUILD_DIR=%MNN_ROOT%\project\android\build_64"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%" >nul 2>nul

pushd "%BUILD_DIR%" || (
  echo [ERROR] Failed to enter build dir: "%BUILD_DIR%"
  exit /b 1
)

echo [INFO] Configuring MNN (Android arm64-v8a)...
cmake "%MNN_ROOT%" -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK%\build\cmake\android.toolchain.cmake" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DANDROID_ABI=arm64-v8a ^
  -DANDROID_STL=c++_static ^
  -DANDROID_NATIVE_API_LEVEL=android-21 ^
  -DMNN_BUILD_FOR_ANDROID_COMMAND=true ^
  -DNATIVE_LIBRARY_OUTPUT=. ^
  -DNATIVE_INCLUDE_OUTPUT=. ^
  -DCMAKE_INSTALL_PREFIX=. ^
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384" ^
  -DCMAKE_MODULE_LINKER_FLAGS="-Wl,-z,max-page-size=16384" ^
  -DMNN_BUILD_SHARED_LIBS=ON ^
  -DMNN_BUILD_TEST=ON ^
  -DMNN_BUILD_BENCHMARK=ON ^
  -DMNN_USE_SSE=OFF ^
  -DMNN_LOW_MEMORY=true ^
  -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true ^
  -DMNN_BUILD_LLM=true ^
  -DMNN_SUPPORT_TRANSFORMER_FUSE=true ^
  -DMNN_ARM82=true ^
  -DMNN_USE_LOGCAT=true ^
  -DMNN_OPENCL=true ^
  -DLLM_SUPPORT_VISION=true ^
  -DMNN_BUILD_OPENCV=true ^
  -DMNN_IMGCODECS=true ^
  -DLLM_SUPPORT_AUDIO=true ^
  -DMNN_BUILD_AUDIO=true ^
  -DMNN_BUILD_DIFFUSION=ON ^
  -DMNN_SEP_BUILD=OFF

if errorlevel 1 (
  echo [ERROR] CMake configure failed.
  popd
  exit /b 1
)

echo [INFO] Building + installing into project/android/build_64 ...
cmake --build . --target install -- -j %JOBS%
if errorlevel 1 (
  echo [ERROR] Build/install failed.
  popd
  exit /b 1
)

if not exist "%BUILD_DIR%\lib\libMNN.so" (
  echo [WARN] Expected "%BUILD_DIR%\lib\libMNN.so" not found after install.
  echo        The app CMake expects MNN_INSTALL_ROOT=project/android/build_64.
)

popd
exit /b 0
