# PTX-EMU Copilot Instructions

## Big picture (architecture)
- 这是一个 C++/CUDA 的 PTX 模拟器：核心库在 src/ 下，包含 `ptx_ir`（IR 类型/语义上下文）、`ptxsim`（执行内核）和 `cudart`（CUDA runtime 替代实现，拦截并模拟 CUDA API）。见 [src/CMakeLists.txt](src/CMakeLists.txt)。
- PTX 解析使用 ANTLR4：语法文件在 src/grammar/，生成源码输出到 build/antlr4_generated_src，并编进 `cudart`。ANTLR 运行时来自 antlr4/antlr4-cpp-runtime-4.13.1-source。见 [CMakeLists.txt](CMakeLists.txt)。
- 执行层次：GPUContext → SMContext → CTAContext → WarpContext → ThreadContext；实现位于 src/ptxsim/core/，指令实现位于 src/ptxsim/instructions/。参见 [docs/gpgpu_arch.md](docs/gpgpu_arch.md)。

## 关键目录
- src/ptxsim/：核心执行与指令处理。
- src/memory/、src/register/：内存与寄存器抽象。
- src/cudart/：CUDA runtime API 替代实现。
- src/grammar/：ptxLexer.g4 / ptxParser.g4。
- include/：对外头文件。
- configs/：GPU 架构 JSON 与调试/日志 INI（如 configs/debug_config.ini）。
- tests/：Catch2 + CUDA PTX 单元测试集合。
- bench/：样例与基准程序（生成 bin/<name>）。

## 构建与运行
- 先执行环境脚本：`. env.sh`，它会设置 CUDA_PATH、CLASSPATH，并把项目 lib/ 加入 LD_LIBRARY_PATH（fake libcudart.so 依赖）。
- 构建（推荐 CMake）：
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build build`
  - 或使用 build.sh（封装了 env.sh 与 Release 构建）。
- 运行：构建产物在 build/bin；lib/libcudart.so 会被软链接到项目根目录 lib/。
- 测试：在 build 目录运行 `ctest` 或 `make test`；tests/CMakeLists.txt 强制 PTX 编译模式并保留中间 PTX。

## 项目特定约定
- `cudart` 是关键入口：新增/修改 CUDA API 行为时优先在 src/cudart/ 查找实现。
- PTX 语法/语义变更需同步更新 src/grammar/ 与对应的解析/执行逻辑，并确保 ANTLR 生成目标可重新生成。
- GPU 架构参数由 configs/*.json 驱动（如 ampere_a100.json、hopper_h100.json）。

## 调试与日志
- 调试/日志由 INI 控制（见 configs/config.ini 与 [docs/debugging_guide.md](docs/debugging_guide.md)）。支持组件级日志（emu/exec/mem/reg/thread/func）与断点配置。
