## 1. Phase 1 — Path 1B PTXIR fat-binary 真实 e2e

- [ ] 1.1 创建 `tests/e2e/path_1B_ptxir_fatbinary/` 子目录
- [ ] 1.2 编写 `tests/e2e/path_1B_ptxir_fatbinary/CMakeLists.txt`（新模式，含 `add_catch_test(e2e_ptxir_fatbinary_exec ...)` + `LABELS "e2e;path_1B"` + `TIMEOUT 60`）
- [ ] 1.3 编写 `tests/e2e/path_1B_ptxir_fatbinary/path_1B_kernels.cu`（≥3 kernels: vector_add, matmul, reduction）
- [ ] 1.4 编写 `tests/e2e/path_1B_ptxir_fatbinary/build_standalone.sh`（nvcc 编译 .cu → cubin + `ptxir_embed` 多次追加 + link PTX-EMU `lib/libcudart.so`）
- [ ] 1.5 编写 `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`（fork+exec standalone binary + 验证 stdout + Scenario 1.1/1.2/1.3/1.5 全部覆盖）
- [ ] 1.6 实现 Anti-fallback guard（PATH="" + unset CUDA_BIN_PATH 在 test fixture）
- [ ] 1.7 验证 Scenario 1.4 字节级一致性（与 Path 1A 编译对比 binary stdout）
- [ ] 1.8 添加 `.gitignore` 白名单 `!tests/e2e/path_1B_ptxir_fatbinary/**/*.ptx`
- [ ] 1.9 验证 AC-1.1 ~ AC-1.8 全部满足（`ctest -L path_1B` + `ldd` 验证 + `xxd` 验证 magic）

## 2. Phase 2 — Path 1C Driver API 真实 e2e

- [ ] 2.1 创建 `tests/e2e/path_1C_driver_api/` 子目录
- [ ] 2.2 编写 `tests/e2e/path_1C_driver_api/CMakeLists.txt`（含 `add_catch_test(e2e_cuda_driver_exec ...)` + `LABELS "e2e;path_1C"`）
- [ ] 2.3 准备 PTXIR image fixture（v2 manifest `kernels[]` 非空，满足 NOT_FOUND 测试要求）
- [ ] 2.4 编写 `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp`（cuModuleLoadData → cuModuleGetFunction → cuLaunchKernel 全链路 + Scenario 2.1/2.2/2.3/2.4 全部覆盖）
- [ ] 2.5 验证 Scenario 2.1 output buffer 与 Path 1B 字节级一致
- [ ] 2.6 验证 Scenario 2.5 cuModuleUnload func2name 失效（per `cudart_sim.cpp:573-592`）
- [ ] 2.7 添加 `.gitignore` 白名单 `!tests/e2e/path_1C_driver_api/**`
- [ ] 2.8 验证 AC-2.1 ~ AC-2.6 全部满足（`ctest -L path_1C`）

## 3. Phase 3 — Path 2D Image Executor 输出正确性

- [ ] 3.1 创建 `tests/e2e/path_2D_image_executor/` 子目录
- [ ] 3.2 编写 `tests/e2e/path_2D_image_executor/CMakeLists.txt`（含 `add_catch_test(e2e_image_executor_output ...)` + `LABELS "e2e;path_2D"`）
- [ ] 3.3 编写 `tests/ptxir/baselines/baseline_format.md`（8-byte `PTXR_OUT\0\0` magic + 4-byte LE size + bytes 格式规范）
- [ ] 3.4 生成 golden output baseline：执行 `cute_rmsnorm.ptxir` fixture → 输出 buffer → 写入 `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin`（含 magic header）
- [ ] 3.5 验证 simulator 输出与 baseline 字节级一致（手动验证一次以确认 baseline 正确）
- [ ] 3.6 commit baseline 文件到 git（`git add tests/ptxir/baselines/`）
- [ ] 3.7 编写 `tests/integration/cudart/test_libptxemu_device.cpp` 增强：新增 cute_rmsnorm output correctness 测试 + D3 mutation 回归（RED PHASE header comment）
- [ ] 3.8 添加 ≥4 个 new error path tests（load garbage, execute invalid handle, unload invalid handle, kernel_name 不存在）
- [ ] 3.9 添加 `.gitignore` 白名单 `!tests/e2e/path_2D_image_executor/**` + `!tests/ptxir/baselines/*.bin`（baseline 必须 commit 不被 ignore）
- [ ] 3.10 验证 AC-3.1 ~ AC-3.7 + AC-3.3-RED 全部满足（`ctest -L path_2D` + baseline file git ls-files 验证）

## 4. Phase 4 — `tests/e2e/` 重组织（路径化目录）

- [ ] 4.1 `git mv tests/e2e/kernel/test_blackwell_gemm.cu tests/e2e/path_1A_legacy_ptx/`
- [ ] 4.2 `git mv tests/e2e/kernel/test_tcgen05_*.cu tests/e2e/path_1A_legacy_ptx/`（保留 Path 1A 守护）
- [ ] 4.3 `git mv tests/e2e/divergence/*.cu tests/e2e/path_1A_legacy_ptx/`（整目录内容）
- [ ] 4.4 `git mv tests/e2e/kernel/test_ptxir_cubin_embed.cpp tests/e2e/path_1B_ptxir_fatbinary/`（format-level 共存）
- [ ] 4.5 创建 `tests/e2e/path_1A_legacy_ptx/CMakeLists.txt`（复用父目录 CUDA flags）
- [ ] 4.6 修改 `tests/e2e/CMakeLists.txt`：删除被移走的 `add_catch_test` 调用 + 新增 4 个 `add_subdirectory(path_X/)`
- [ ] 4.7 给 path_1A 现有测试加 `LABELS "e2e;path_1A"`（Oracle 修订：必须含 `e2e` 段）
- [ ] 4.8 验证 `ctest -L path_1A/1B/1C/2D` 各自仅运行对应子目录测试（AC-4.3 ~ AC-4.5）
- [ ] 4.9 验证 `ctest --output-on-failure` 全量通过（AC-4.6）
- [ ] 4.10 验证 `git log --follow <file>` file history 保留（AC-4.8 用 `git mv` 验证）
- [ ] 4.11 验证现有 kernel/cosim 测试不变（AC-4.7）：test_test3_cfg_full, test_barrier_warp_sync, test_ldglobal_simple, 3 个 shared_memory 测试, test_flashattention_mini, test_printf + 整个 tests/e2e/cosim/ 保留在原位

## 5. Phase 5 — Proposal 描述修正（一致性）

- [ ] 5.1 修改 `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` §Capabilities 中 `tests/e2e/test_ptxir_cubin_embed.cu` 描述
- [ ] 5.2 添加 disclaimer：**Note [修正: 2026-08-12, see fix-path-coverage-gaps]** — 此 e2e 验证 PTXIR-Embedded CUBIN 格式兼容性（Phase 12.2 R5 / ADR-0024 Risk 1），**不验证 PTX-EMU 真实加载执行**
- [ ] 5.3 disclaimer 交叉引用 Phase 1 新测试位置（`tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`）
- [ ] 5.4 验证 tasks.md 不变（AC-5.6）
- [ ] 5.5 验证 archive 目录名 `2026-08-07-implement-ptxir-cubin-embed-extension` 不变（AC-5.5）
- [ ] 5.6 验证 `git log -- openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` 显示本次修改为后续追加 commit（AC-5.4）

## 6. 验收 & 提交

- [ ] 6.1 验证 AC-G1 `./scripts/sanity.sh` 通过
- [ ] 6.2 验证 AC-G2 `ctest --output-on-failure -L "e2e;integration;unit"` 100% pass
- [ ] 6.3 验证 AC-G3 `./scripts/regression.sh` 通过
- [ ] 6.4 验证 AC-G4 `clang-format --dry-run --Werror <changed-files>` 返回 0
- [ ] 6.5 验证 AC-G5 5 个 Phase 全部 ship（`openspec status` 显示 archived，iteration.json 同步）
- [ ] 6.6 验证 AC-N1/N2 新测试保留 `e2e_` 前缀 + LABELS 含 `e2e` 段
- [ ] 6.7 验证 AC-M1/M2 cudart 路径覆盖率 3/4 → 4/4 + output-correctness 1/4 → 4/4
- [ ] 6.8 验证 AC-M3 openspec 文档一致性修复 1 处（Phase 5）
- [ ] 6.9 验证 AC-M4 `ctest -L path_1X` 可作为单路径回归命令

## 7. 归档

- [ ] 7.1 执行 `openspec archive fix-path-coverage-gaps --yes`
- [ ] 7.2 验证 iteration.json 更新 `fix-path-coverage-gaps` status=archived
- [ ] 7.3 验证归档目录 `openspec/changes/archive/2026-08-12-fix-path-coverage-gaps/` 创建
- [ ] 7.4 清理工作 worktree（`git worktree remove`）