## 1. Phase 1: Unit tests and exception cleanup

- [ ] 1.1 **基线 worktree**: 建立 `git worktree add .worktrees/baseline-tcgen05-cp 178457d` 并全量 build 验证 ctest 通过。
- [ ] 1.2 **RED**: 新增 `tests/unit/tcgen05/test_tcgen05_cp.cpp`，编写 `extract_smem_offset_placeholder` 对 immediate / non-shared / register offset 的断言，以及 `cta_group::2` / 缺少 shared memory 的异常断言；确认测试编译失败（handler helper 不可见）。
- [ ] 1.3 **实现**: 将 `extract_smem_offset_placeholder` 和 `throw_cta_group_2` 从匿名 namespace 移到 `ptxsim` namespace；将 `sharedMemSpace == nullptr` 的 `std::runtime_error` 改为 `UnsupportedInstructionException`。
- [ ] 1.4 **GREEN**: 运行 `cmake --build build --target unit_tcgen05_cp` 和 `ctest -R unit_tcgen05_cp -V` 确认新增测试通过。
- [ ] 1.5 **Placeholder 跟踪**: 在 `tcgen05_cp.cpp` 的 `kDestSlot=0`、shape qualifier 注释、register offset 回退处添加 `TODO(Phase 3 of implement-tcgen05-handlers-extended)` 注释。
- [ ] 1.6 **Sanity**: 运行 `cd build && ctest -L "unit;tcgen05" -V` 和 `./tests/ptx/test_all_ptx.sh` 确认无回归。
- [ ] 1.7 **Commit**: `test(tcgen05): add unit tests for tcgen05.cp and unify exception type (ADR-0016)`

## 2. Phase 2: Integration test

- [ ] 2.1 **RED**: 新增 `tests/integration/tcgen05/test_tcgen05_cp.cpp`，编写 128 字节 SMEM → TMEM 拷贝的测试和越界异常测试；确认测试编译失败（未注册 ctest 目标）。
- [ ] 2.2 **实现**: 在 `tests/integration/CMakeLists.txt` 中注册 `integration_tcgen05_cp` 目标；参考 `test_alloc_dealloc_relinquish.cpp` 使用 `ptxsim::testing` 工具构造 warp + CTA + 指令序列。
- [ ] 2.3 **GREEN**: 运行 `cmake --build build --target integration_tcgen05_cp` 和 `ctest -R integration_tcgen05_cp -V` 确认测试通过。
- [ ] 2.4 **Sanity**: 运行 `cd build && ctest -L "integration;tcgen05" -V` 确认无回归。
- [ ] 2.5 **Commit**: `test(tcgen05): add integration test for tcgen05.cp (ADR-0016)`

## 3. Phase 3: E2E and documentation sync

- [ ] 3.1 **E2E 可行性验证**: 尝试 `nvcc -ptx tests/e2e/kernel/test_tcgen05_cp.cu`；若成功则保留 E2E 测试，否则在 `tests/e2e/kernel/CMakeLists.txt` 中显式跳过并注释原因。
- [ ] 3.2 **RED**: 如可行，新增 `tests/e2e/kernel/test_tcgen05_cp.cu` 并编写 kernel；确认编译失败或测试失败。
- [ ] 3.3 **GREEN**: 如新增 E2E，运行 `cmake --build build --target e2e_tcgen05_cp` 和 `ctest -R e2e_tcgen05_cp -V` 确认通过。
- [ ] 3.4 **文档**: 更新 `src/ptxsim/instructions/AGENTS.md` 中 `tcgen05.cp` 的测试覆盖状态；更新根 `AGENTS.md` 已知限制表（如需要）。
- [ ] 3.5 **Sanity**: 运行 `./scripts/sanity.sh` 全量验证。
- [ ] 3.6 **Commit**: `test(tcgen05): add e2e kernel and update AGENTS for tcgen05.cp (ADR-0016)`

## 4. Phase 4: Archive

- [ ] 4.1 **全量验证**: `cd build && ctest --output-on-failure` 和 `./tests/ptx/test_all_ptx.sh` 全部通过。
- [ ] 4.2 **Archive**: 运行 `openspec archive tcgen05-cp-test-coverage-and-exception-cleanup --yes`。
- [ ] 4.3 **Archive 验证**: 检查 `openspec/changes/archive/tcgen05-cp-test-coverage-and-exception-cleanup/` 下 artifact 完整；运行 `git status` 确认无未提交变更。
- [ ] 4.4 **Commit**: `chore(openspec): archive tcgen05-cp-test-coverage-and-exception-cleanup (ADR-0016)`

## 5. Post-Archive Cleanup

- [ ] 5.1 删除基线 worktree：`git worktree remove .worktrees/baseline-tcgen05-cp`。
- [ ] 5.2 在 `docs/dev-process/lessons-learned.md` 追加本 change 的 postmortem（如适用）。
- [ ] 5.3 通知相关 change（如 `implement-tcgen05-handlers-extended`）本测试补充已归档。
