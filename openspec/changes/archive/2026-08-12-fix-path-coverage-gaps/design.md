## Context

PTX-EMU 当前 cudart 加载路径覆盖矩阵（实测 grep + Oracle 验证）：

| 路径 | unit | integration | e2e 真实执行 | 状态 |
|------|------|-------------|--------------|------|
| 1A Legacy PTX | ⚠️ 故意 skip | ⚠️ SingletonGuard only | ✅ test_blackwell_gemm.cu + test_tcgen05_*.cu + tests/e2e/divergence/*.cu | 现有 e2e 间接覆盖 |
| 1B PTXIR fat-binary | ✅ test_ptxir_config | ✅ test_ptxir_cubin_loader（调 dispatch 函数） | ❌ 缺失 | proposal 写"PTX-EMU 加载"但实际只验证格式 |
| 1C Driver API | ✅ test_cuda_driver_api + helpers | ✅ test_cuda_driver_api + mutation | ❌ 缺失 | test_cuda_driver_api 测 load/get_function/unload；cuLaunchKernel 仅在 test_error_mapping.cpp 覆盖错误路径 |
| 2D Image Executor | ✅ test_cpptlm_module (rc==0) | ✅ test_libptxemu_device + 5 个 | ⚠️ 部分 | 缺输出正确性验证 |

**架构假设尚未被任何 e2e 验证**：假设 `try_ptxir_dispatch_from_memory` 在生产场景中真的能从 `/proc/self/exe` 正确反序列化 PTXIR 并 dispatch 到 `g_ptx_interpreter`。这是一个关键的架构盲点。

**关联 ADR / 已存档 OpenSpec**：
- ADR-0024（PTXIR-Embedded CUBIN, 2026-08-06 Accepted）— Risk 1: NVIDIA cuobjdump 必须容忍尾部 PTXIR。test_ptxir_cubin_embed.cpp 验证 Risk 1 成立，但**未验证 PTX-EMU 真的能加载并执行**该 embedded binary
- ADR-0029（PTX-EMU Image Executor, 2026-08-10 ship）— D6: SINGLE-GPU-INSTANCE 假设。test_cpptlm_module.cpp 仅验证 API 调用成功，未验证 RMSNorm 输出正确
- 已存档 OpenSpec `implement-ptxir-cubin-embed-extension`（2026-08-07 ship）— proposal §Capabilities 声称 e2e 测试"PTX-EMU 加载 + ptxir_extract"但**实际只验证了格式 round-trip**。交付文件 `.cu` → `.cpp` 后缀变化是 silent descoping 的证据

**结构性约束（Oracle 补充）**：`cuModuleLoadData` 显式拒绝 `kExecutableTailPtxir`（`cudart_sim.cpp:532-537`）— driver API 路径拒绝恰好是 Path 1B 接受的 fat-binary 形式。意味着 Path 1C coverage 无法用作 Path 1B 的覆盖替身，3 个缺口是结构性独立问题，必须各自补齐。

**新增债务编号**：ADR-0021 已定义 D-PTX-1 至 D-PTX-6，本改进新增 D-PTX-7（PTXIR fat-binary 端到端未验证）+ D-PTX-8（Driver API 真实成功 kernel 执行未验证），须登记到 ADR-0021 附录避免与 D-PTX-1 ~ D-PTX-6 编号冲突：
- D-PTX-7（proposed）：PTXIR fat-binary 端到端未验证
- D-PTX-8（proposed）：Driver API 真实成功 kernel 执行未验证

## Goals / Non-Goals

**Goals:**
- Path 1B/1C/2D 各自补齐 e2e 真实执行覆盖（不是仅验证格式/rc）
- `tests/e2e/` 重组织为路径化目录结构，每个 path 子目录独立 CMakeLists
- ctest label 支持 `-L path_1B` 单路径回归过滤
- 修复归档 change `implement-ptxir-cubin-embed-extension` 文档不一致（silent descoping）
- 维护现有 4-path cudart 测试覆盖率从 3/4 (75%) → 4/4 (100%)
- 维护现有 e2e output-correctness 覆盖率从 1/4 → 4/4 (100%)

**Non-Goals:**
- 不修改 Path 1A/1B/1C/2D 的实现代码（仅补测试，不动 cudart_sim.cpp / cpptlm_module.cpp / ptxir_loader.cpp）
- 不修复 `multi-entry-handle-api` 任务未勾选状态（archive gate 的 process gap，需另立 improvement）
- 不引入新测试框架（沿用 Catch2 + add_catch_test）
- 不创建新的 PTXIR fixture 生成工具
- 不修改 openspec CLI / openspec validate 规则
- 不修改 ctest 标签体系（仅添加新 label）
- 不动 PTX-EMU 整体测试目录结构（仅修改 tests/e2e/ 子树）

## Decisions

### Decision 1: 5 Phase 串行 + 各自独立可回退

**选择**：将本改进拆为 5 个 Phase，每个 Phase 独立可 archive/revert。

**理由**：
- Phase 1/2/3 各自测试不同的路径（Path 1B/1C/2D），Phase 4 重组织 + Phase 5 文档修正是横切关注
- 5 Phase 共享技术约束（CTEST_LABEL 段、anti-fallback 等）但产出独立 e2e 文件

**放弃方案**：
- 单一 Phase 一次性写完所有 4 个 path 子目录 + 重组织 + 文档修正 → 失败时定位困难、archive 粒度过粗

### Decision 2: Anti-fallback guard 用 PATH="" 而非 dispatch marker

**选择**：Phase 1 用 `PATH=""`（unset cuobjdump location）阻止 `extract_ptx_with_cuobjdump` 子进程调用。

**理由**：
- 简单（环境变量设置即可）
- 不依赖 PTX-EMU 内部计数器
- 测试可观测：binary 输出非预期内容即证明 fallback 生效

**替代**：
- Dispatch marker ABI（`extern "C" uint32_t ptxemu_ptxir_dispatch_hits()`）—— 需要改 cpptlm_module.cpp 暴露新 ABI，超出 Non-Goals
- 文件系统检查（mock cuobjdump 二进制）—— 复杂度高，且 PATH="" 已足够

### Decision 3: cute_rmsnorm baseline 格式自定

**选择**：Phase 3 定义 baseline 文件格式 = 8-byte magic `PTXR_OUT\0\0` + 4-byte LE u32 size + bytes。

**理由**：
- magic header 便于将来版本迁移（PTXR_OUT v1 → v2 增字段）
- size 前缀便于 memcmp 前预校验
- 当前 repo 无此 magic（验证：未污染）

**替代**：
- 裸二进制（无 magic）—— 难以 debug、未来加字段无迁移路径
- JSON/MessagePack —— 增加外部依赖

### Decision 4: 4 个 path_X/CMakeLists.txt 新建（非继承父目录）

**选择**：每个 path 子目录新建独立 CMakeLists.txt（`add_subdirectory(path_X/)` 调用），不复用 `tests/e2e/CMakeLists.txt` 的现有 `add_catch_test` 包装。

**理由**：
- 子目录独立构建（可在 path_1B/ 内单独 cmake build）
- 标签集中（每个子目录 CMakeLists 显式标注 `LABELS "e2e;path_1X"`）
- 便于将来 path_X 之间添加 path-specific 编译选项

**替代**：
- 父目录单一 CMakeLists + 条件分支 —— 标签分散、可读性差

### Decision 5: ctest label 段含 `e2e`

**选择**：新测试 LABELS 必含 `e2e` 段（`e2e;path_1B`），保证 `regression.sh -L e2e` 覆盖。

**理由**：
- regression.sh 现有 `-L e2e` 过滤是主要回归脚本
- 缺失 `e2e` 段会导致新测试被静默 skip（commit ab55e06 已有先例教训）

**替代**：
- 单一 path label（`path_1B`）—— regression.sh 范围不一致

### Decision 6: Phase 5 仅修改归档 change 的 proposal.md §Capabilities 文案

**选择**：Phase 5 用 inline 修正标记 `[修正: 2026-08-12, see fix-path-coverage-gaps]` 修改归档 proposal.md。

**理由**：
- ERRATA inline-merge 惯例已有先例（`docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md`）
- 不修改 tasks.md（任何 checkbox 状态变化都是 archive gate violation）
- 不重命名归档目录（避免 archive history 篡改）

**替代**：
- 建独立 ERRATA 文档 + 在原 proposal 加交叉引用 —— 增加文档数量，不符合 inline-merge 惯例
- 不修改归档 change（保持 silent descoping）—— 文档不一致问题持续存在

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Phase 1 standalone binary 构建时间拖慢 ctest | `set_tests_properties(<test> PROPERTIES TIMEOUT 60)` + ctest label 隔离（`-L path_1B` 单独跑） |
| cute_rmsnorm output baseline 可能因 simulator 微小变化失效（per ADR-0029 D6 SINGLE-GPU-INSTANCE 假设）| baseline 更新流程：simulator 改动必须同步更新 baseline + AC-3.5 commit 验证 |
| Phase 4 重组织涉及 ~10 文件 git mv，若有未提交修改会冲突 | 重组织前 `git status` 干净检查 + 选用 commit 间隔期 |
| `cuModuleUnload` 修改 func2name 的实现位置在 `cudart_sim.cpp:573-592`，Phase 2 测试依赖此行为不要变 | Phase 2 测试仅做 `Scenario 2.5` 验证（cuModuleUnload func2name 失效），不改实现 |
| Anti-fallback PATH="" 在某些 CI 环境可能不生效（cuobjdump 不在 PATH） | PATH="" + 显式 unset `CUDA_BIN_PATH`（若存在），双重保障 |
| Dispatch marker ABI 暴露需 Phase 5 后续改进（cpptlm_module.cpp:227-262 现有 8 extern "C" 符号）| Phase 1 仅用 PATH="" 方案，Phase 5 不引入新 ABI；后续若需可另立 improvement |

## Migration Plan

**部署步骤**：
1. Phase 1 提交：`tests/e2e/path_1B_ptxir_fatbinary/` 子目录新建（含 CMakeLists.txt + `test_ptxir_fatbinary_exec.cpp` + `path_1B_kernels.cu`）
2. Phase 2 提交：`tests/e2e/path_1C_driver_api/` 子目录新建（含 CMakeLists.txt + `test_cuda_driver_exec.cpp` + fixture）
3. Phase 3 提交：
   - `tests/e2e/path_2D_image_executor/` 子目录新建（含 CMakeLists.txt + 增强 `test_libptxemu_device.cpp`）
   - `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` baseline 文件 commit
   - `tests/ptxir/baselines/baseline_format.md` 文档定义格式
4. Phase 4 提交：4 个 path_X/ 子目录 `git mv` 现存测试 + tests/e2e/CMakeLists.txt 更新
5. Phase 5 提交：归档 proposal.md §Capabilities 文案修改（git log 显示追加 commit）

**回滚策略**：
- 单 Phase 回滚：`git revert <phase-commit>` —— 因 Phase 独立可 archive，revert 不影响其他 Phase
- 全量回滚：跳过 ship 即可（无需 archive 因本 change 还没 ship）

## Open Questions

- Phase 1 是否需要考虑 `try_ptxir_dispatch_from_memory` 在生产场景下可能走 `__cudaRegisterFatBinary` 二次调用导致 SingletonGuard FATAL？答：Phase 1 用 fork+exec 启动（cudart_sim.cpp:329-335 约束）规避，不动生产代码
- Phase 3 baseline 是否需考虑 cute_rmsnorm kernel 自身版本变更？答：当前 fixture 5294 B 固定，simulator 微小变化由 baseline 更新流程处理；cute_rmsnorm kernel 升级需另立 improvement
- Phase 4 重组织是否需要保留 `tests/e2e/kernel/` 目录作为兼容层？答：不保留（Oracle 建议），所有 path-related 测试都迁移，遗留测试（test_test3_cfg_full 等）保留在原位