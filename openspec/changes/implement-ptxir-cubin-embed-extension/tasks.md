# Tasks: implement-ptxir-cubin-embed-extension

## 1. Phase 12.2 Commit 1 — PTXIRLoader 类 + Unit 测试

> **策略**：TDD 5 步结构（Write failing test → Verify fail → Implement → Verify pass → Commit）。Phase 内每个 task 验证后再 commit。
> **依赖**：无（独立 commit）
> **风险**：PTXIR_EMBED_MAGIC 字面值选择需在 Commit 1 完成前通过 ADR-0024 governance check

### 1.1 PTXIR_EMBED_MAGIC 字面值验证与注册

- [ ] 1.1.1 用 `cuobjdump --dump-elf` + `strings` 扫描 NVIDIA 已有 cubin magic 列表，确认 `PTXIR_EMBED_MAGIC` 不冲突
- [ ] 1.1.2 在 `include/cudart/ptxir_loader.h` 定义 `static constexpr uint64_t PTXIR_EMBED_MAGIC = 0x<8byte>` 常量
- [ ] 1.1.3 **MUST**：任何 magic 字面值变更必须触发 ADR-0024 §合规检查第 6 项重审（governance check 守门）

### 1.2 PTXIRLoader 类骨架 + 单元测试失败

- [ ] 1.2.1 编写 `tests/unit/test_ptxir_loader.cpp` 骨架，含 4 个 public static 方法的 fixture
- [ ] 1.2.2 编写失败测试 `hasEmbeddedPTXIR_legitimateEmbedded_returnsTrue`
- [ ] 1.2.3 编写失败测试 `hasEmbeddedPTXIR_plainCubin_returnsFalse`
- [ ] 1.2.4 编写失败测试 `hasEmbeddedPTXIR_truncatedInput_returnsFalse`
- [ ] 1.2.5 编写失败测试 `hasEmbeddedPTXIR_fakeMagic_returnsFalse`（首 4 字节匹配但后 4 字节不匹配）
- [ ] 1.2.6 编写失败测试 `extractPTXIR_legitimateEmbedded_returnsSection`
- [ ] 1.2.7 编写失败测试 `extractPTXIR_plainCubin_returnsNullopt`
- [ ] 1.2.8 编写失败测试 `extractPTXIR_zeroSizeInput_returnsNullopt`
- [ ] 1.2.9 编写失败测试 `extractPureCubin_legitimateEmbedded_returnsBytes`
- [ ] 1.2.10 编写失败测试 `extractPureCubin_plainCubin_passthrough`
- [ ] 1.2.11 编写失败测试 `extractPureCubin_hashMismatch_returnsNullopt`
- [ ] 1.2.12 编写失败测试 `deserializeForCubin_legitimateSection_returnsContexts`
- [ ] 1.2.13 编写失败测试 `deserializeForCubin_corruptedHeader_returnsEmpty`
- [ ] 1.2.14 编写失败测试 `deserializeForCubin_hashCheckFails_returnsEmpty`
- [ ] 1.2.15 **NOTE**：以上 13 个测试预期全部失败（PTXIRLoader 类尚未实现）
- [ ] 1.2.16 验证测试失败：`cmake --build build && ctest --output-on-failure -R test_ptxir_loader`（expect 13 failures）

### 1.3 PTXIRLoader 实现 — Red → Green

- [ ] 1.3.1 创建 `include/cudart/ptxir_loader.h`，声明 4 个 `public static` 方法 + magic 常量
- [ ] 1.3.2 创建 `src/cudart/ptxir_loader.cpp`，实现 `hasEmbeddedPTXIR()`：尾部 8 字节 O(1) 比对
- [ ] 1.3.3 实现 `extractPTXIR()`：基于 magic 边界定位 + Section TOC 解析
- [ ] 1.3.4 实现 `extractPureCubin()`：提取 cubin 前缀 + SHA-256 hash 校验
- [ ] 1.3.5 实现 `deserializeForCubin()`：复用 ADR-0023 PTXIRHeader 反序列化路径 + cubin_hash 校验
- [ ] 1.3.6 验证测试通过：`cmake --build build && ctest --output-on-failure -R test_ptxir_loader`（expect 13 passes）
- [ ] 1.3.7 测试覆盖率检查：`cmake --build build --target coverage && gcov` → PTXIRLoader 覆盖率 ≥ 90%
- [ ] 1.3.8 **MUST**：实现中所有失败路径返回 `nullopt` / 空 vector，不抛异常（与 design §决策 2 一致）

### 1.4 Commit 1

- [ ] 1.4.1 `git add include/cudart/ptxir_loader.h src/cudart/ptxir_loader.cpp tests/unit/test_ptxir_loader.cpp`
- [ ] 1.4.2 `git commit -m "feat(cudart): PTXIRLoader class with 4 public static methods + unit tests"`（独立可合并，无运行时影响）

## 2. Phase 12.2 Commit 2 — cudart dispatch 集成 + Integration 测试

> **策略**：TDD 5 步结构。Commit 2 依赖 Commit 1。
> **风险**：修改 `__cudaRegisterFatBinary`（关键 ABI 入口），必须确保 `PTXIR_MODE=off` 完全 bypass

### 2.1 config::isPTXIRModeEnabled() 实现与测试

- [ ] 2.1.1 编写 `tests/unit/test_config_ptxir_mode.cpp`：测试 `PTXIR_MODE` env var + INI 双源
- [ ] 2.1.2 失败测试：`isPTXIRModeEnabled_PTXIR_MODE_off_returnsFalse`
- [ ] 2.1.3 失败测试：`isPTXIRModeEnabled_PTXIR_MODE_auto_returnsTrue`
- [ ] 2.1.4 失败测试：`isPTXIRModeEnabled_unset_returnsFalse`（默认 OFF）
- [ ] 2.1.5 失败测试：`isPTXIRModeEnabled_iniOverridesEnv`（INI 优先 vs env var 优先，需 design 中确认）
- [ ] 2.1.6 实现 `config::isPTXIRModeEnabled()` 函数（`src/cudart/config.cpp` 或现有 config 模块）
- [ ] 2.1.7 添加 `PTXIR_MODE` 字段到 `configs/debug_config.ini`（默认 `off`）
- [ ] 2.1.8 验证 4 个测试通过

### 2.2 __cudaRegisterFatBinary dispatch — Integration 测试

- [ ] 2.2.1 创建 `tests/integration/test_ptxir_cubin_loader.cpp` 骨架
- [ ] 2.2.2 失败测试 `dispatch_embeddedCubin_PTXIR_MODE_auto_loadsViaPTXIR`
- [ ] 2.2.3 失败测试 `dispatch_embeddedCubin_PTXIR_MODE_off_loadsViaStandardPath`
- [ ] 2.2.4 失败测试 `dispatch_plainCubin_PTXIR_MODE_auto_loadsViaStandardPath`
- [ ] 2.2.5 失败测试 `dispatch_corruptedPTXIR_PTXIR_MODE_auto_gracefulDegradation`
- [ ] 2.2.6 失败测试 `dispatch_cubinSizeExceedsMagicWindow_logsAndFallbacksToStandardPath`
- [ ] 2.2.7 **NOTE**：以上 5 个测试预期失败（dispatch 分支尚未实现）
- [ ] 2.2.8 验证失败：`ctest -R test_ptxir_cubin_loader`（expect 5 failures）

### 2.3 dispatch 分支实现

- [ ] 2.3.1 修改 `src/cudart/cudart_sim.cpp`：`__cudaRegisterFatBinary` 入口增加 dispatch
- [ ] 2.3.2 在 dispatch 中检查 `config::isPTXIRModeEnabled()`，OFF 时直接走现有路径
- [ ] 2.3.3 调用 `PTXIRLoader::hasEmbeddedPTXIR()` 检测嵌入段
- [ ] 2.3.4 true → 调用 `extractPTXIR()` + `deserializeForCubin()` + 走现有 `gpu.registerFatBinary()` 主路径
- [ ] 2.3.5 false / 任意失败 → 走现有标准 cubin 路径（优雅降级，无 log spam）
- [ ] 2.3.6 **MUST**：dispatch 不修改 `__cudaRegisterFatBinary` 签名，不修改 fatbin 句柄本身
- [ ] 2.3.7 **MUST**：`PTXIR_MODE=off` 时 dispatch 调用成本 O(1)（env var 静态缓存）
- [ ] 2.3.8 验证 5 个 integration 测试通过
- [ ] 2.3.9 验证 `PTXIR_MODE=off` 行为字节级等价：跑 `tests/integration/` 全套测试，无回归

### 2.4 Commit 2

- [ ] 2.4.1 `git add src/cudart/cudart_sim.cpp src/cudart/config.cpp configs/debug_config.ini tests/integration/test_ptxir_cubin_loader.cpp tests/unit/test_config_ptxir_mode.cpp`
- [ ] 2.4.2 `git commit -m "feat(cudart): wire PTXIRLoader dispatch into __cudaRegisterFatBinary + config gate"`（默认 OFF，运行时行为不变）

## 3. Phase 12.2 Commit 3 — CLI 工具 + E2E 测试

> **策略**：TDD 5 步结构。Commit 3 依赖 Commit 1（PTXIRLoader），独立于 Commit 2。
> **风险**：e2e 测试需真实 nvcc + cuobjdump 环境

### 3.1 ptxir_embed CLI 工具

- [ ] 3.1.1 创建 `tools/ptxir_embed.cpp` 骨架，CLI 参数解析（`--in-cubin`/`--in-ptxir`/`--out`/`--help`/`--version`）
- [ ] 3.1.2 失败测试 `embed_legitimateInputs_producesEmbeddedCubin`（`tests/e2e/test_ptxir_cubin_embed.cu`）
- [ ] 3.1.3 失败测试 `embed_missingInputFile_exitsWithError`（exit code 2）
- [ ] 3.1.4 失败测试 `embed_help_printsUsage`（exit code 0）
- [ ] 3.1.5 失败测试 `embed_version_printsVersion`（exit code 0）
- [ ] 3.1.6 实现 `ptxir_embed.cpp`：复用 `PTXIRLoader` static 方法 + `PTXIR_EMBED_MAGIC` 常量
- [ ] 3.1.7 验证 4 个测试通过

### 3.2 ptxir_extract CLI 工具

- [ ] 3.2.1 创建 `tools/ptxir_extract.cpp` 骨架，CLI 参数解析（`--in`/`--out-cubin`/`--out-ptxir`/`--help`/`--version`）
- [ ] 3.2.2 失败测试 `extract_legitimateEmbedded_producesPureCubinAndPTXIR`
- [ ] 3.2.3 失败测试 `extract_plainCubin_passthrough`（`--out-cubin` 字节级相同）
- [ ] 3.2.4 失败测试 `extract_hashMismatch_exitsWithError`（exit code 3）
- [ ] 3.2.5 失败测试 `extract_help_printsUsage`
- [ ] 3.2.6 实现 `ptxir_extract.cpp`：复用 `PTXIRLoader` + SHA-256 校验 + hash mismatch 错误码
- [ ] 3.2.7 验证 5 个测试通过

### 3.3 CMake 注册 + Tools README

- [ ] 3.3.1 修改 `src/cudart/CMakeLists.txt`：注册 `ptxir_loader.cpp` 子目标（如尚未注册）
- [ ] 3.3.2 修改 `tools/CMakeLists.txt`：注册 `ptxir_embed` / `ptxir_extract` 工具目标
- [ ] 3.3.3 创建 `tools/README.md`：说明 ptxir_embed/ptxir_extract 用法、限制、PTXIR_MODE 配置

### 3.4 E2E 测试 — 真实 nvcc + cuobjdump 双向验证

- [ ] 3.4.1 创建 `tests/e2e/test_ptxir_cubin_embed.cu` 骨架（含 ≥3 个不同复杂度 CUDA kernel fixture）
- [ ] 3.4.2 失败测试 `e2e_nvccCompile_embed_ptxemu_executesCorrectly`（≥3 个 kernel）
- [ ] 3.4.3 失败测试 `e2e_embed_extract_cuobjdump_byteIdenticalToOriginalCubin`
- [ ] 3.4.4 失败测试 `e2e_embed_cuobjdump_dumpSASS_direct`（Oracle review blocking fix — 验证 cuModuleLoadData 容忍尾部 magic）
- [ ] 3.4.5 失败测试 `e2e_embed_cuobjdump_dumpPTX_normal`
- [ ] 3.4.6 失败测试 `e2e_cuModuleLoadData_noDriver_explicitSkip`（Oracle review blocking fix — 输出 `[SKIP] cuModuleLoadData test — no driver`）
- [ ] 3.4.7 实现 e2e 测试驱动（`build_e2e.sh` 或 ctest 集成）
- [ ] 3.4.8 验证 ≥5 个 e2e 测试场景全部通过（或按设计输出显式 SKIP）
- [ ] 3.4.9 **MUST**：e2e 测试用真实 nvcc 编译 ≥3 个不同复杂度 kernel

### 3.5 全套验证

- [ ] 3.5.1 `cmake --build build && ctest --output-on-failure` 全部通过
- [ ] 3.5.2 `./scripts/sanity.sh` 全部通过
- [ ] 3.5.3 `ptxir_extract --help` 与 `ptxir_embed --help` 输出正常
- [ ] 3.5.4 静态分析（`clang-tidy` / `cppcheck`）无新增 warning

### 3.6 Commit 3

- [ ] 3.6.1 `git add tools/ptxir_embed.cpp tools/ptxir_extract.cpp tools/README.md tools/CMakeLists.txt src/cudart/CMakeLists.txt tests/e2e/test_ptxir_cubin_embed.cu`
- [ ] 3.6.2 `git commit -m "feat(tools): ptxir_embed/extract CLI + e2e tests with nvcc/cuobjdump verification"`

## 4. ADR-0024 合规检查（最终）

- [ ] 4.1 验证 `PTXIR_MODE=off` 完全 bypass 检测分支（CI 守门）
- [ ] 4.2 验证 `ptxir_extract` 保留原 cubin 字节内容（hash 相等）
- [ ] 4.3 验证嵌入段 `.ptxir.section` 使用 ADR-0023 Section TOC
- [ ] 4.4 验证 PTXIRLoader 所有 4 个函数有 unit 测试（覆盖率 ≥ 90%）
- [ ] 4.5 验证 e2e 测试用 nvcc + cuobjdump 验证 NVIDIA 兼容性（含 Oracle review 新增的 2 个直接对 embedded cubin 解析场景）
- [ ] 4.6 验证 `PTXIR_EMBED_MAGIC` 字面值未在 proposal 层面单方面修改（governance check）
- [ ] 4.7 在 ADR-0024.md 添加变更记录（§合规检查 6 项全部 ✅）

## 5. Archival 准备

- [ ] 5.1 `openspec validate implement-ptxir-cubin-embed-extension --strict` 全部通过
- [ ] 5.2 3 个 commits 全部合并到 main 分支
- [ ] 5.3 等待 guide-ship 启动执行（创建 worktree、生成 Prometheus 计划、逐 commit 实施 + 验证）