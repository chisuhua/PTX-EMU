# Tasks: implement-ptxir-cubin-embed-extension

## 0. Commit 0 — ADR-0024 v1.1 Amendment（已完成 2026-08-07）

> **策略**：governance check 必须先于任何代码 commit
> **状态**：✅ 已 2026-08-07 提交（commit `pending`）
> **风险**：magic literal 变更触发 §合规检查 #6 — 必须先 amend ADR-0024

- [x] 0.1 实施前用 `cuobjdump --dump-elf` + `strings` 扫描 NVIDIA 已有 magic 列表，确认 `{'P','T','X','E','M','B','\x01','\x00'}` 不冲突
- [x] 0.2 更新 ADR-0024 §决策内容 §1 layout 为 footer pattern（`prefix[N] || section[M] || uint32_le size || magic[8]`）
- [x] 0.3 更新 ADR-0024 §决策内容 §1 magic literal 为 `{'P','T','X','E','M','B','\x01','\x00'}`
- [x] 0.4 更新 ADR-0024 §影响范围表（新增 PtxContextAdapter / ptxir_config / tools 目录等条目）
- [x] 0.5 更新 ADR-0024 §前置依赖（澄清 `config::` 命名空间实际不存在）
- [x] 0.6 修复 ADR-0024 §风险与缓解 表 line 343 typo（`.ptxir\x00\x01\x00` → `PTXEMB\x01\x00`）
- [x] 0.7 添加 ADR-0024 §更新记录 entry（2026-08-07 footer layout + magic literal change + Oracle findings）

## 1. Phase 12.2 Commit 1 — PTXIRLoader + PtxContextAdapter + config + Unit 测试

> **策略**：TDD 5 步结构（Write failing test → Verify fail → Implement → Verify pass → Commit）
> **依赖**：Commit 0 (ADR amendment)
> **风险**：PTXIR_EMBED_MAGIC 字面值已 2026-08-07 governance check 通过；PtxContextAdapter 字段填充正确性

### 1.1 PTXIR_EMBED_MAGIC + EmbeddedKernelManifest 注册

- [ ] 1.1.1 在 `include/cudart/ptxir_loader.h` 定义 `static constexpr uint8_t PTXIR_EMBED_MAGIC[8] = {'P','T','X','E','M','B','\x01','\x00'}`（与 ADR-0024 v1.1 同步）
- [ ] 1.1.2 在 `include/cudart/ptx_context_adapter.h` 定义 `struct EmbeddedKernelManifest { std::string kernelName; std::vector<ParamContext> params; int ptxAddressSize = 64; }`
- [ ] 1.1.3 **MUST**：magic 字面值修改必须先 amend ADR-0024 v1.1+（governance check 守门）

### 1.2 PTXIRLoader 类骨架 + footer-layout 单元测试失败

- [ ] 1.2.1 编写 `tests/unit/cudart/test_ptxir_loader.cpp` 骨架，含 4 个 public static 方法的 fixture
- [ ] 1.2.2 失败测试 `hasEmbeddedPTXIR_legitimateEmbedded_returnsTrue`（footer-layout: prefix + section + size_le + magic）
- [ ] 1.2.3 失败测试 `hasEmbeddedPTXIR_plainCubin_returnsFalse`
- [ ] 1.2.4 失败测试 `hasEmbeddedPTXIR_truncatedInput_returnsFalse`（size < 12）
- [ ] 1.2.5 失败测试 `hasEmbeddedPTXIR_fakeMagic_returnsFalse`（首字节不匹配）
- [ ] 1.2.6 失败测试 `hasEmbeddedPTXIR_sizeFieldMismatch_returnsFalse`（magic 正确但 size_le 指向 cubin 外部）
- [ ] 1.2.7 失败测试 `extractPTXIR_legitimateEmbedded_returnsSection`
- [ ] 1.2.8 失败测试 `extractPTXIR_plainCubin_returnsNullopt`
- [ ] 1.2.9 失败测试 `extractPTXIR_zeroSizeInput_returnsNullopt`
- [ ] 1.2.10 失败测试 `extractPureCubin_legitimateEmbedded_returnsBytes`
- [ ] 1.2.11 失败测试 `extractPureCubin_plainCubin_passthrough`
- [ ] 1.2.12 失败测试 `extractPureCubin_hashMismatch_returnsNullopt`
- [ ] 1.2.13 失败测试 `deserializeForCubin_legitimateSection_returnsContexts`
- [ ] 1.2.14 失败测试 `deserializeForCubin_corruptedHeader_returnsEmpty`（try/catch 包裹 deserializeFromString）
- [ ] 1.2.15 失败测试 `deserializeForCubin_hashCheckFails_returnsEmpty`
- [ ] 1.2.16 **NOTE**：以上 14 个测试预期全部失败（PTXIRLoader 类尚未实现）
- [ ] 1.2.17 验证测试失败：`cmake --build build && ctest --output-on-failure -R test_ptxir_loader`（expect 14 failures）

### 1.3 PTXIRLoader 实现 — Red → Green

- [ ] 1.3.1 创建 `include/cudart/ptxir_loader.h`，声明 4 个 `public static` 方法 + magic 常量
- [ ] 1.3.2 创建 `src/cudart/ptxir_loader.cpp`
- [ ] 1.3.3 实现 `hasEmbeddedPTXIR(const uint8_t* data, size_t size)`：尾部 8 字节比对 magic + 检查 size ≥ 12 + size_le 边界检查
- [ ] 1.3.4 实现 `extractPTXIR(const uint8_t* data, size_t size, size_t* out_size)`：基于 magic 边界 + size_le 定位 section 起始
- [ ] 1.3.5 实现 `extractPureCubin(const uint8_t* data, size_t size)`：返回 `data[0 .. size-12-size_le)` + SHA-256 hash 校验
- [ ] 1.3.6 实现 `deserializeForCubin(const uint8_t* ptxir_data, size_t ptxir_size)`：try/catch 包裹 `deserialize_from_string()` + hash 校验，返回 `std::vector<StatementContext>`
- [ ] 1.3.7 验证 14 个测试通过：`ctest -R test_ptxir_loader`（expect 14 passes）
- [ ] 1.3.8 测试覆盖率检查：PTXIRLoader 覆盖率 ≥ 90%
- [ ] 1.3.9 **MUST**：所有失败路径返回 `nullopt` / 空 vector，不抛异常（与 design §决策 2 一致）

### 1.4 PtxContextAdapter 类骨架 + 单元测试失败

- [ ] 1.4.1 编写 `tests/unit/cudart/test_ptx_context_adapter.cpp` 骨架
- [ ] 1.4.2 失败测试 `fromEmbedded_emptyManifest_populatesDefaults`（kernelName=""，ptxAddressSize=64）
- [ ] 1.4.3 失败测试 `fromEmbedded_withKernelName_setsKernelName`（manifest.kernelName="myKernel" → PtxContext.ptxKernels[0].kernelName == "myKernel"）
- [ ] 1.4.4 失败测试 `fromEmbedded_withParams_populatesKernelParams`（manifest 含 2 个 ParamContext → PtxContext.ptxKernels[0].kernelParams.size() == 2）
- [ ] 1.4.5 失败测试 `fromEmbedded_withAddressSize_setsPtxAddressSize`（manifest.ptxAddressSize=32 → PtxContext.ptxAddressSize == 32）
- [ ] 1.4.6 失败测试 `fromEmbedded_stmtsBecomeKernelStatements`（StatementContext[] 长度 N → kernelStatements 长度 N）
- [ ] 1.4.7 **NOTE**：以上 5 个测试预期全部失败
- [ ] 1.4.8 验证测试失败：`ctest -R test_ptx_context_adapter`（expect 5 failures）

### 1.5 PtxContextAdapter 实现 — Red → Green

- [ ] 1.5.1 创建 `include/cudart/ptx_context_adapter.h`，声明 `EmbeddedKernelManifest` + `PtxContextAdapter::fromEmbedded()`
- [ ] 1.5.2 创建 `src/cudart/ptx_context_adapter.cpp`
- [ ] 1.5.3 实现 `fromEmbedded(stmts, manifest)`：
  - `KernelContext kc; kc.kernelName = manifest.kernelName; kc.kernelParams = manifest.params; kc.kernelStatements = stmts; kc.ifEntryKernel = true;`
  - `PtxContext ctx; ctx.ptxAddressSize = manifest.ptxAddressSize; ctx.ptxKernels.push_back(kc); return ctx;`
- [ ] 1.5.4 验证 5 个测试通过
- [ ] 1.5.5 测试覆盖率检查：PtxContextAdapter 覆盖率 ≥ 90%

### 1.6 config::isPTXIRModeEnabled() 实现 + 单元测试

- [ ] 1.6.1 编写 `tests/unit/cudart/test_ptxir_config.cpp`：测试 env var + INI 双源
- [ ] 1.6.2 失败测试 `isPTXIRModeEnabled_PTXIR_MODE_off_returnsFalse`
- [ ] 1.6.3 失败测试 `isPTXIRModeEnabled_PTXIR_MODE_auto_returnsTrue`
- [ ] 1.6.4 失败测试 `isPTXIRModeEnabled_unset_returnsFalse`（默认 OFF）
- [ ] 1.6.5 失败测试 `isPTXIRModeEnabled_envOverridesIni_returnsTrue`（env `auto` wins over INI `off` — 遵循 cudart_sim.cpp:277-281 precedent）
- [ ] 1.6.6 创建 `include/cudart/ptxir_config.h`，声明 `namespace config { bool isPTXIRModeEnabled(); }`
- [ ] 1.6.7 创建 `src/cudart/ptxir_config.cpp`：
  - Meyers singleton 缓存（首次 call 读 env var，后续 O(1)）
  - INI 加载由 `initialize_environment()` 调用 `setPTXIRModeFromIni(bool)`
  - env var wins over INI
- [ ] 1.6.8 验证 4 个测试通过

### 1.7 INI 配置 + integration 到 cudart_sim.cpp

- [ ] 1.7.1 添加 `[ptxir]` 段到 `configs/config.ini`、`configs/debug_config.ini`、`configs/release_config.ini` 等（默认 `mode = off`）
- [ ] 1.7.2 在 `src/cudart/cudart_sim.cpp::initialize_environment()` 中加载 INI `[ptxir]` 段 → `config::setPTXIRModeFromIni(bool)`
- [ ] 1.7.3 验证 `PTXIR_MODE=off` 行为字节级等价：跑 `tests/integration/` 全套测试，无回归

### 1.8 Commit 1

- [ ] 1.8.1 `git add openspec/changes/implement-ptxir-cubin-embed-extension/ include/cudart/ptxir_loader.h include/cudart/ptx_context_adapter.h include/cudart/ptxir_config.h src/cudart/ptxir_loader.cpp src/cudart/ptx_context_adapter.cpp src/cudart/ptxir_config.cpp src/cudart/cudart_sim.cpp configs/*.ini tests/unit/cudart/test_ptxir_loader.cpp tests/unit/cudart/test_ptx_context_adapter.cpp tests/unit/cudart/test_ptxir_config.cpp`（NEW-5: per ptx-lessons-learned §6 — 显式 stage openspec artifacts 以保证后续 spec/design/tasks 修订被跟踪）
- [ ] 1.8.2 `git commit -m "feat(cudart): PTXIRLoader + PtxContextAdapter + config + unit tests (default PTXIR_MODE=off)"`（独立可合并，无运行时影响）

## 2. Phase 12.2 Commit 2 — cudart dispatch 集成 + Integration 测试

> **策略**：TDD 5 步结构。Commit 2 依赖 Commit 1。
> **风险**：修改 `__cudaRegisterFatBinary`（关键 ABI 入口），必须确保 `PTXIR_MODE=off` 完全 bypass + 不解引用 `fat_bin`

### 2.1 __cudaRegisterFatBinary dispatch — Integration 测试

- [ ] 2.1.0 **NEW-3 fix**：在根 `tests/CMakeLists.txt` 添加 `add_subdirectory(integration/cudart)`（与现有 `add_subdirectory(integration/memory)` 等并列）— **必须先于此小节任何 task 完成**
- [ ] 2.1.1 创建 `tests/integration/cudart/` 目录 + `CMakeLists.txt`（新集成测试子目录）
- [ ] 2.1.2 创建 `tests/integration/cudart/test_ptxir_cubin_loader.cpp` 骨架
- [ ] 2.1.3 失败测试 `dispatch_embeddedExe_PTXIR_MODE_auto_loadsViaPTXIR`（需用 self_exe tail overlay fixture）
- [ ] 2.1.4 失败测试 `dispatch_embeddedExe_PTXIR_MODE_off_loadsViaStandardPath`
- [ ] 2.1.5 失败测试 `dispatch_plainExe_PTXIR_MODE_auto_loadsViaStandardPath`
- [ ] 2.1.6 失败测试 `dispatch_corruptedPTXIR_PTXIR_MODE_auto_gracefulDegradation`
- [ ] 2.1.7 失败测试 `dispatch_exeSizeLessThan12_logsAndFallbacksToStandardPath`
- [ ] 2.1.8 失败测试 `dispatch_fatBinNullPtr_doesNotCrash`（Oracle R10：fat_bin 不解引用）
- [ ] 2.1.9 **NOTE**：以上 6 个测试预期失败（dispatch 分支尚未实现）
- [ ] 2.1.10 验证失败：`ctest -R test_ptxir_cubin_loader`（expect 6 failures）

### 2.2 dispatch 分支实现

- [ ] 2.2.1 修改 `src/cudart/cudart_sim.cpp`：`__cudaRegisterFatBinary` 在 `readlink("/proc/self/exe")` (line 377) 之后立即增加 dispatch
- [ ] 2.2.2 在 dispatch 中检查 `config::isPTXIRModeEnabled()`，OFF 时直接走现有路径
- [ ] 2.2.3 调用 `PTXIRLoader::hasEmbeddedPTXIR()` 检测嵌入段（读取 `/proc/self/exe` 末尾 12 字节）
- [ ] 2.2.4 true → 调用 `extractPTXIR()` + `deserializeForCubin()` + 构造 `EmbeddedKernelManifest` (kernelName 来自 PTXIR section 中的 MANIFEST 段，由 CLI `--kernel-name` flag 在 ptxir_embed 阶段写入 — 详见 §3.1.1) + `PtxContextAdapter::fromEmbedded()` → `g_ptx_interpreter->set_ptx_context(*ctx)`
- [ ] 2.2.5 false / 任意失败 → 走现有标准 cubin 路径（优雅降级，无 log spam）
- [ ] 2.2.6 **MUST**：dispatch 不修改 `__cudaRegisterFatBinary` 4 参签名（cudart_sim.cpp:354），不修改 fatbin 句柄，不解引用 `fat_bin`
- [ ] 2.2.7 **MUST**：`PTXIR_MODE=off` 时 dispatch 调用成本 O(1)（Meyers singleton 静态缓存，遵循 §1.6.7）
- [ ] 2.2.8 **MUST**：byte source = `/proc/self/exe` 末尾（非 `fat_bin`）
- [ ] 2.2.9 验证 6 个 integration 测试通过
- [ ] 2.2.10 ABI stability test (NEW-4 修正)：
  - `nm -D lib/libcudart.so | grep cudaRegisterFatBinary` — 前后 unmangled symbol 名 `cudaRegisterFatBinary` 必须 unchanged（extern "C" linkage preserved）
  - 跑一个 nvcc 编译的 CUDA program 调用 `__cudaRegisterFatBinary(...)` — 必须 link 成功无 undefined reference
  - (rationale: `nm -D` 的 "size" 字段对 extern "C" 函数不直接有意义；真正 ABI 契约是 mangled/unmangled name 一致 + 调用约定匹配)

### 2.3 Commit 2

- [ ] 2.3.1 `git add openspec/changes/implement-ptxir-cubin-embed-extension/ src/cudart/cudart_sim.cpp tests/integration/cudart/CMakeLists.txt tests/integration/cudart/test_ptxir_cubin_loader.cpp tests/CMakeLists.txt`
- [ ] 2.3.2 `git commit -m "feat(cudart): wire PTXIRLoader dispatch into __cudaRegisterFatBinary (PTXIR_MODE=auto only, byte source = /proc/self/exe)"`（默认 OFF，运行时行为不变）

## 3. Phase 12.2 Commit 3 — tools/ CLI + E2E 测试

> **策略**：TDD 5 步结构。Commit 3 依赖 Commit 1（PTXIRLoader + PtxContextAdapter），**部分独立于 Commit 2**（e2e 中 `cuobjdump --dump-sass` 不需要 dispatch；`PTX-EMU executes embedded exe` 需要 Commit 2 — 见 §3.4 拆分）
> **风险**：e2e 测试需真实 nvcc + cuobjdump 环境；`tools/` 目录当前不存在需新建

### 3.0 新增 tools/ 目录 + CMakeLists.txt

- [ ] 3.0.1 创建 `tools/` 目录（当前不存在）
- [ ] 3.0.2 创建 `tools/CMakeLists.txt`，注册 `ptxir_embed` / `ptxir_extract` 工具目标
- [ ] 3.0.3 在根 `CMakeLists.txt` 添加 `add_subdirectory(tools)`
- [ ] 3.0.4 创建 `tools/README.md`：说明 ptxir_embed/ptxir_extract 用法、限制、PTXIR_MODE 配置

### 3.1 ptxir_embed CLI 工具

- [ ] 3.1.1 创建 `tools/ptxir_embed.cpp` 骨架，CLI 参数解析（`--in-exe`/`--in-cubin` 二选一 + `--in-ptxir` + `--out` + `--kernel-name` 必填 + `--help`/`--version`）
- [ ] 3.1.2 失败测试 `embed_legitimateExe_producesEmbeddedExe`（`tests/e2e/test_ptxir_cubin_embed.cu`）
- [ ] 3.1.3 失败测试 `embed_legitimateCubin_producesEmbeddedCubin`（NVIDIA-compat 场景）
- [ ] 3.1.4 失败测试 `embed_missingKernelName_exitsWithError`（exit code 4）
- [ ] 3.1.5 失败测试 `embed_missingInputFile_exitsWithError`（exit code 2）
- [ ] 3.1.6 失败测试 `embed_help_printsUsage`（exit code 0）
- [ ] 3.1.7 失败测试 `embed_version_printsVersion`（exit code 0）
- [ ] 3.1.8 实现 `ptxir_embed.cpp`：复用 `PTXIR_EMBED_MAGIC` + footer layout 拼装
- [ ] 3.1.9 验证 6 个测试通过

### 3.2 ptxir_extract CLI 工具

- [ ] 3.2.1 创建 `tools/ptxir_extract.cpp` 骨架，CLI 参数解析（`--in` + `--out-cubin` + `--out-ptxir` + `--help`/`--version`）
- [ ] 3.2.2 失败测试 `extract_legitimateEmbedded_producesPurePrefixAndPTXIR`
- [ ] 3.2.3 失败测试 `extract_plainCubin_passthrough`（`--out-cubin` 字节级相同）
- [ ] 3.2.4 失败测试 `extract_hashMismatch_exitsWithError`（exit code 3）
- [ ] 3.2.5 失败测试 `extract_help_printsUsage`
- [ ] 3.2.6 实现 `ptxir_extract.cpp`：复用 `PTXIRLoader` + SHA-256 校验 + hash mismatch 错误码
- [ ] 3.2.7 验证 4 个测试通过

### 3.3 E2E 测试 — 真实 nvcc + cuobjdump 双向验证

> **拆分**：3.3.1-3.3.5 独立于 Commit 2；3.3.6 需要 Commit 2

#### 3.3.x 独立于 Commit 2 的 e2e 测试（验证 CLI + cuobjdump 兼容性）

- [ ] 3.3.1 创建 `tests/e2e/test_ptxir_cubin_embed.cu` 骨架（含 ≥3 个不同复杂度 CUDA kernel fixture）
- [ ] 3.3.2 失败测试 `e2e_nvccCompile_embed_cuobjdumpDumpSassMatchesOriginal`（`--in-cubin` 路径，验证 embed 后 cuobjdump 仍能解析）
- [ ] 3.3.3 失败测试 `e2e_embed_extract_cuobjdump_byteIdenticalToOriginalCubin`（验证 extract 后字节级一致）
- [ ] 3.3.4 失败测试 `e2e_embed_cuobjdump_dumpPTX_normal`
- [ ] 3.3.5 失败测试 `e2e_cuModuleLoadData_noDriver_explicitSkip`（Oracle review blocking fix — 输出 `[SKIP] cuModuleLoadData test — no driver`）

#### 3.3.x 依赖 Commit 2 的 e2e 测试（验证 PTX-EMU 通过 dispatch 加载 embedded exe）

- [ ] 3.3.6 失败测试 `e2e_nvccCompile_embedExe_ptxemu_executesCorrectly`（≥3 个 kernel，需 Commit 2 的 dispatch 已合并）

### 3.4 全套验证

- [ ] 3.4.1 `cmake --build build && ctest --output-on-failure` 全部通过
- [ ] 3.4.2 `./scripts/sanity.sh` 全部通过
- [ ] 3.4.3 `ptxir_extract --help` 与 `ptxir_embed --help` 输出正常
- [ ] 3.4.4 静态分析（`clang-tidy` / `cppcheck`）无新增 warning

### 3.5 Commit 3

- [ ] 3.5.1 `git add openspec/changes/implement-ptxir-cubin-embed-extension/ tools/ptxir_embed.cpp tools/ptxir_extract.cpp tools/CMakeLists.txt tools/README.md CMakeLists.txt tests/e2e/test_ptxir_cubin_embed.cu`
- [ ] 3.5.2 `git commit -m "feat(tools): ptxir_embed/extract CLI + e2e tests with nvcc/cuobjdump verification"`

## 4. Phase 12.2 Commit 4 — 文档同步

- [ ] 4.1 修改 `roadmap.md`：新增 Phase 12.2 (PTXIR Cubin 集成) 条目
- [ ] 4.2 修改根 `README.md` §已实现功能：添加 PTXIR-Embedded CUBIN 支持
- [ ] 4.3 修改根 `README.md` §已知限制：移除"PTXIR 仅在内部 pipeline 中可用"项
- [ ] 4.4 更新 `docs/adr/README.md` 索引：ADR-0024 v1.1 更新条目
- [ ] 4.5 `git add openspec/changes/implement-ptxir-cubin-embed-extension/ roadmap.md README.md docs/adr/README.md` && `git commit -m "docs: Phase 12.2 (PTXIR Cubin 集成) roadmap + README sync"`

## 5. ADR-0024 合规检查（最终）

- [ ] 5.1 验证 `PTXIR_MODE=off` 完全 bypass 检测分支（CI 守门）
- [ ] 5.2 验证 `ptxir_extract` 保留原 cubin/prefix 字节内容（hash 相等）
- [ ] 5.3 验证嵌入段 `.ptxir.section` 使用 ADR-0023 Section TOC
- [ ] 5.4 验证 PTXIRLoader 所有 4 个函数有 unit 测试（覆盖率 ≥ 90%）
- [ ] 5.5 验证 e2e 测试用 nvcc + cuobjdump 验证 NVIDIA 兼容性（含 Oracle review 新增的 2 个直接对 embedded 解析场景）
- [ ] 5.6 验证 ADR-0024 §更新记录已包含 2026-08-07 magic + layout 变更条目（governance check 通过）
- [ ] 5.7 ABI stability test: `nm -D lib/libcudart.so | grep cudaRegisterFatBinary` 前后 symbol size delta == 0
- [ ] 5.8 验证 `fat_bin=nullptr` 不导致 crash（Oracle R10）

## 6. Archival 准备

- [ ] 6.1 `openspec validate implement-ptxir-cubin-embed-extension --strict` 全部通过
- [ ] 6.2 5 个 commits (0-4) 全部合并到 main 分支
- [ ] 6.3 等待 guide-ship 启动执行（创建 worktree、生成 Prometheus 计划、逐 commit 实施 + 验证）