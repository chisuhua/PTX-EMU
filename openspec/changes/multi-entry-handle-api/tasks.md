# Tasks: multi-entry-handle-api

> **Per [ptx-lessons-learned](.opencode/skills/ptx-lessons-learned/SKILL.md) §3**: 复杂迁移分 Phase commit, 每个 Phase 独立可回退。
> 6 个 commit (C1-C6), 顺序依赖: writer → fixture → cudart → cpptlm → test → api+abi。

## 1. Phase C1: v2 PTXIR writer (P0)

- [x] 1.1 在 `src/ptx_ir/ptxir_writer.cpp` 添加 `writeMultiKernels(const ManifestSection&)` 函数
- [x] 1.2 修改 `writeManifestSection()` 同时写 `kernels` vector + 保留 `kernel_name` 字段
- [x] 1.3 添加 `ManifestSection` validation: `kernels.empty() && kernel_name.empty()` 抛 `std::invalid_argument`
- [x] 1.4 **测试先行 (TDD Red)**: 新建 `tests/unit/ptxir/test_multi_entry_roundtrip.cpp`，含 6 测试用例 (单 entry / 多 entry / 空 vector / 大端 / 异常 / fixture)
- [x] 1.5 **验证失败 (Red)**: `./build.sh && ctest -R multi_entry_roundtrip` 预期失败 (handler 未实现)
- [x] 1.6 **实现 + 验证 (Green)**: 运行 ctest, 6 测试通过
- [x] 1.7 **Commit C1**: `feat(ptxir): v2 writer multi-entry 完整实现 (commit C1, ref: openspec multi-entry-handle-api)`
- [x] 1.8 **回退验证**: `git revert HEAD` 后 reader backward-compat synthesis 仍可用 (跑 `test_multi_kernel_selection.cpp` 验证)

## 2. Phase C2: Multi-entry fixture (P0)

- [x] 2.1 创建 `tests/fixtures/ptx/multi_kernel_basic.ptx`: ≥3 kernel (vec_add + mat_mul + reduce_sum)
- [x] 2.2 创建 `tests/scripts/gen_multi_kernel_ptxir.py`: 从 PTX 生成 multi-entry PTXIR (复用 `ptxir_loader.cpp` 路径)
- [x] 2.3 在 `tests/CMakeLists.txt` 添加 fixture 注册 (确保 fixture 在 ctest 中可访问)
- [x] 2.4 **测试先行 (TDD Red)**: `tests/unit/ptxir/test_fixture_load.cpp` 验证 fixture 加载 ≥3 kernel
- [x] 2.5 **验证失败 (Red)**: fixture 不存在导致测试失败
- [x] 2.6 **实现 + 验证 (Green)**: fixture 创建 + test 通过
- [x] 2.7 **Commit C2**: `test(fixture): multi_kernel_basic.ptx + generator script (commit C2)`
- [x] 2.8 **回退验证**: `git revert HEAD` 后 fixture 测试 skip (CMake fixture 引用更新)

## 3. Phase C3: cuModuleGetFunction handle 映射 (P0)

- [ ] 3.1 **测试先行 (TDD Red)**: 在 `tests/integration/cudart/test_cuda_driver_api.cpp` 添加 3 测试场景 (name 查找 / 重名 / 不存在)
- [ ] 3.2 **验证失败 (Red)**: stub 返回 invalid handle, 测试失败
- [ ] 3.3 在 `include/cudart/module_registry.h` 添加 per-module `std::unordered_map<std::string, CUfunction>` 字段
- [ ] 3.4 在 `src/cudart/cuda_driver.cpp` 实现 `insert_function()` 真实逻辑 (替换 stub): name lookup → insert → 返回 handle
- [ ] 3.5 修改 `src/cudart/cudart_sim.cpp:556-570` `cuModuleGetFunction`: 替换 stub → 调用真实 `ModuleRegistry::insert_function`
- [ ] 3.6 线程安全: `std::lock_guard<std::mutex>` 保护 per-module registry (复用既有 mutex)
- [ ] 3.7 **实现 + 验证 (Green)**: ctest 3 场景通过
- [ ] 3.8 **Commit C3**: `feat(cudart): cuModuleGetFunction multi-kernel name→handle 映射 (commit C3)`
- [ ] 3.9 **回退验证**: `git revert HEAD` 后 stub 行为恢复, cudart_sim.cpp 编译通过

## 4. Phase C4: cpptlm_module multi-entry handle (P0)

- [ ] 4.1 **测试先行 (TDD Red)**: 在 `tests/integration/cudart/test_in_memory_mutation.cpp` 添加 4 测试场景 (load + get_handle + execute + unload per kernel)
- [ ] 4.2 **验证失败 (Red)**: 3 新 API 未定义, 编译失败
- [ ] 4.3 修改 `include/cudart/cpptlm_module.h`: 添加 3 `extern "C"` 函数 (`ptxemu_image_kernel_count` / `_kernel_name_at` / `_execute_named`) + `CPPTLM_MODULE_VERSION 1→2` bump
- [ ] 4.4 在 `src/cudart/cpptlm_module.cpp` 实现 3 函数 + 替换 `kernels[0]` fallback (`src/cudart/cpptlm_module.cpp:120-127`)
- [ ] 4.5 锁顺序契约: `execute_named` 保持 `exec_mu_` → `mu_` 顺序 (per `ptx-lessons-learned` §3)
- [ ] 4.6 截断契约: `_kernel_name_at` buf_size=0 返回 -1, buf_size 不足截断但不溢出
- [ ] 4.7 **实现 + 验证 (Green)**: ctest 4 场景通过 + SC-5 stale handle 测试
- [ ] 4.8 **Commit C4**: `feat(cpptlm): multi-entry handle API + VERSION 1→2 (commit C4)`
- [ ] 4.9 **回退验证**: `git revert HEAD` 后 `CPPTLM_MODULE_VERSION` 同步回退到 1, 旧 binary 仍兼容

## 5. Phase C5: test_multi_kernel_selection 升级 (P1)

- [ ] 5.1 **测试先行 (TDD Red)**: 在 `tests/unit/cudart/test_multi_kernel_selection.cpp` 添加 ≥3 真实测试 (替换 `SUCCEED("placeholder")`)
- [ ] 5.2 **验证失败 (Red)**: placeholder 仍存在, 测试覆盖度不足
- [ ] 5.3 真实测试: cuModuleGetFunction 多 kernel handle / `ptxemu_image_kernel_count` 验证 / `ptxemu_image_kernel_name_at` 截断契约
- [ ] 5.4 **实现 + 验证 (Green)**: 全部 placeholder 替换, 测试通过
- [ ] 5.5 **Commit C5**: `test(cudart): multi_kernel_selection 升级 (placeholder → 真实测试, commit C5)`
- [ ] 5.6 **回退验证**: `git revert HEAD` 后 placeholder 恢复 (可接受临时回退)

## 6. Phase C6: ptxemu_image_kernel_name 多 kernel + ABI baseline (P1+P2)

- [ ] 6.1 **测试先行 (TDD Red)**: 在 `tests/integration/cudart/test_libptxemu_device.cpp` 添加 ABI baseline 测试 (v1 binary 加载 + mutation regression)
- [ ] 6.2 **验证失败 (Red)**: kernel_name 仅返回首个, 多 kernel 测试失败
- [ ] 6.3 修改 `src/cudart/cpptlm_module.cpp::get_kernel_name`: 遍历 `manifest.kernels` + 索引访问 (per SC-4)
- [ ] 6.4 单元测试: ABI baseline (v1 binary 加载触发 backward-compat synthesis) + mutation regression
- [ ] 6.5 文档: 在 `docs/architecture/multi-kernel-manifest-gaps-gap-analysis.md` §8 添加 "Data Redundancy" 段落 (声明 `ManifestParam` 为 source of truth)
- [ ] 6.6 **实现 + 验证 (Green)**: ctest ABI baseline 通过
- [ ] 6.7 **Commit C6**: `feat(cpptlm): kernel_name 遍历 + ABI baseline + 文档 (commit C6)`
- [ ] 6.8 **回退验证**: `git revert HEAD` 后 kernel_name 行为恢复 v1 单 kernel

## 7. 跨 Phase 验证门 (所有 commit 后)

- [ ] 7.1 `cmake --build build && ctest --output-on-failure` — 0 failed
- [ ] 7.2 `./scripts/sanity.sh` — 0 errors (per `ptx-lessons-learned` §5)
- [ ] 7.3 `./scripts/regression.sh` — 0 failures
- [ ] 7.4 `nm -D build/lib/libptxemu_device.so` — **无 removed/modified T 符号** (commit 6 ABI 验证)
  - 新增 3 个允许: `ptxemu_image_kernel_count` / `_kernel_name_at` / `_execute_named`
  - `CPPTLM_MODULE_VERSION 1→2` bump 验证
- [ ] 7.5 `nm -D build/lib/libcudart.so` — 仍含 4 个 T 符号 (`cuModuleLoadData` / `cuModuleGetFunction` / `cuLaunchKernel` / `cuModuleUnload`)
- [ ] 7.6 `cpptlm_bridge.h` diff — **空** (5 byte-identical gates hold, **不** 修改 ABI)
- [ ] 7.7 Per-commit git log 检查: 6 个 commit, 每个独立可回退
- [ ] 7.8 `include/ptx_ir/ptxir_format.h` `PTXIR_VERSION` 仍为 4 (Phase 12.4 bump 保留)
- [ ] 7.9 v1 binary 加载后 synthesize 1 entry, `kernel_name` 一致 (per SC-2)
- [ ] 7.10 `KernelEntry.arg_count == ManifestParam.size()` (source of truth 文档化, per 决策 6)
- [ ] 7.11 `cpptlm_module.h` `CPPTLM_MODULE_VERSION 1→2` 验证 (commit C4 必须 bump)
- [ ] 7.12 SC-5 unload-vs-enumerate race 测试通过 (新加测试用例, per Oracle Q3 扩展)
- [ ] 7.13 SC-8 within-module duplicate name first-match 测试通过 (per Oracle Q3 新增)
- [ ] 7.14 SC-6 concurrent thread 隔离测试通过 (2 host thread 并发 `cuModuleGetFunction`)

## 8. 文档同步

- [ ] 8.1 `roadmap.md` §Phase 12.5 状态更新: 4 P0 + 2 P1 + 2 P2 全部 ✅
- [ ] 8.2 `multi-kernel-manifest-gaps-gap-analysis.md` §3 状态列更新: 全部 ⏳ → ✅
- [ ] 8.3 `proposal-approved.md` "已实施" 段添加本提案 (archive 后由 `mark_approved_completed()` 自动迁移)
- [ ] 8.4 `iteration.json` 更新本 change: `status: planned → proposed → shipped`
- [ ] 8.5 ADR-0029 §D4 更新: v1 单 kernel 限制段落移除 (per `ptxir-toolchain-stack.md` v1.4)
- [ ] 8.6 ADR-0025/0027 §v1 限制段落同步更新 (Phase 12.4 已 ship, 需同步)

## 9. 归档 (archive)

- [ ] 9.1 所有 commit 推送 (PR review 通过)
- [ ] 9.2 `openspec archive multi-entry-handle-api` — 同步 delta specs 到 main specs (`openspec/specs/kernel-selection/spec.md`)
- [ ] 9.3 验证 `proposal-approved.md` 自动迁移到 "已实施" 段 (`mark_approved_completed()`)
- [ ] 9.4 `iteration.json` 状态: `shipped` + `archived_at` 时间戳
