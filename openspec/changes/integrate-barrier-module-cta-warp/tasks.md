## 1. 准备与审计

- [ ] 1.1 全项目 grep 旧 API 使用点：`grep -rn "Wbar\b\|\bwbar\.\|wbars\[\|synchronize_barrier\|bsync_manager_" src/ include/ tests/ > /tmp/barrier_audit.txt`；统计每个 API 的引用次数
- [ ] 1.2 阅读 `src/ptxsim/core/bsync_state.cpp` 全部方法 + `bsync_state.h` 字段；输出 `barrier_audit.md` 列出 `bsync_manager_.release` 是否有 `BarrierModule::release_warp_barrier` 未承担的副作用
- [ ] 1.3 创建 worktree 隔离环境：`git worktree add ../ptx-emu-barrier -b fix/integrate-barrier-module`；后续所有 commit 在 worktree 中
- [ ] 1.4 在 worktree 中建立基线：`./scripts/sanity.sh --quick > baseline.txt`，保存用于对比

## 2. 扩展 BarrierModule（CTA 路径能力补齐）

- [ ] 2.0 前置验证：检查项目 CMake 是否配置 TSan 构建目标；若未配置，Task 3.2 改为使用现有 race detector 或 skip（标注 deferred）；NOTE：当前项目 CMake 中无 `-fsanitize=thread` 配置
- [ ] 2.1 修改 `include/ptxsim/barrier/barrier_module.h`：`release_cta_barrier` 签名增加 `CTAContext* cta_ctx` 参数；增加 forward declaration
- [ ] 2.2 修改 `src/ptxsim/barrier/barrier_module.cpp::release_cta_barrier`：遍历 `arrived_threads_`，对每个 `ThreadContext*` 调用 `set_state(RUN)` + `advance_thread_pc(lane, post_barrier_pc)`；同时清空 `arrived_threads_`
- [ ] 2.2b 修改 `src/ptxsim/barrier/barrier_module.cpp::release_warp_barrier`：在 `set_exec_mask` 前先 `set_active_mask(get_active_mask() | arrived_mask)` 实施 OR 逻辑；遵循 `src/ptxsim/core/AGENTS.md` 不变量（"OR logic must live in the caller"）；必须同时保留 `set_exec_mask(arrived_mask)` 调用（用于 PTX `activemask` 指令）
  - [ ] 2.2b.1 验证：`cd build && ctest -R "release_warp_barrier ORs" -V` PASS；`ctest -R "post_barrier_two_halves" -V` 不回归
- [ ] 2.2c 修改 `src/ptxsim/barrier/warp_barrier.cpp::WarpBarrier::init`：增加 `if (is_initialized_)` 分支，仅更新 metadata（participation_mask、reconvergence_pc、barrier_pc、expected_count、state=Waiting），**不**重置 arrived_mask_/arrived_count_；保持首次 init 路径不变
  - [ ] 2.2c.1 验证：`cd build && ctest -R "WarpBarrier::init preserves" -V` PASS；`ctest -R "post_barrier_reconvergence_simplegemm" -V` 不回归
- [ ] 2.3 修改 `include/ptxsim/cta_context.h` + `src/ptxsim/core/cta_context.cpp`：构造函数初始化 `std::unique_ptr<BarrierModule> barrier_module_`；提供 `BarrierModule& get_barrier_module()` 访问器
- [ ] 2.4 验证：`cmake --build build --target ptxsim` 必须编译通过；MUST NOT 引入新编译错误

## 3. 新增 CTABarrier 完整流程单元测试

- [ ] 3.1 在 `tests/unit/barrier/test_barrier_module.cpp` 增加 TEST_CASE `CTABarrier full arrive-release flow`：init → 多次 arrive → 验证 `is_complete()` 状态转换（Initializing → Waiting → Complete）→ release → 验证 `arrived_threads_.size() == 0` 且 `is_initialized() == false`
- [ ] 3.2 增加 TEST_CASE `CTABarrier mutex concurrent arrives`：用多线程并发调用 `arrive`，验证线程安全的到达计数；MUST 触发 `-fsanitize=thread` 无 race 警告
- [ ] 3.3 增加 TEST_CASE `BarrierModule::release_warp_barrier OR active_mask`：复现 BUG-POSTBARRIER-TWOHALVES 场景，验证 release 后 `active_mask` 是 OR 合并而非覆写
- [ ] 3.4 验证：`cd build && ctest -R unit_barrier_module -V` 全部 PASS；新增 case 必须用 Catch2 BDD-style `[barrier][release]` 标签

## 4. 切换 BarHandler 到 BarrierModule（CTA 路径）

- [ ] 4.1 修改 `src/ptxsim/instructions/barrier.cpp::BarHandler::executeBarrier`：将 `sm_context->synchronize_barrier(barId, context)` 替换为 `cta_ctx->get_barrier_module().arrive_at_cta_barrier(barId, context)`；当返回 true 时调用 `barrier_module.release_cta_barrier(barId, cta_ctx)` 真正释放线程
- [ ] 4.2 关键修复（MUST）：在 release 路径中增加 `advance_thread_pc(lane, post_barrier_pc)` 调用（替换原 `set_next_pc(pc+1)` 的不完整更新）；MUST 覆盖所有 32 lane per warp
- [ ] 4.3 验证：`cmake --build build && ctest -R integration_barrier -V`；integration_cta_barrier_memory_visibility 必须 PASS（仍可暂时保留 work-around）
- [ ] 4.4 删除 `tests/integration/barrier/test_cta_barrier_memory_visibility.cpp:138-184` 的 `advance_thread_pc` work-around；增加新断言 `warp_state.threads[lane].pc == post_barrier_pc`；MUST 仍 PASS（证明 handler bug 已修复）
- [ ] 4.5 跑全量回归：`./scripts/sanity.sh --quick`；任何新增 FAIL 必须立即修复或回滚到 4.1

## 5. 切换 BarWarpSyncHandler 到 BarrierModule（Warp 路径）

- [ ] 5.1 修改 `src/ptxsim/instructions/barrier.cpp::BarWarpSyncHandler::processOperation` 正常路径：替换 `wbar.arrive(lane_id)` 为 `barrier_module.arrive_at_warp_barrier(wbar_id, lane_id)`；替换 `wbar.init()` 为 `barrier_module.init_warp_barrier()`
- [ ] 5.2 修改 force_reconvergence 路径：保留 `if (!wbar.is_initialized) ... else 保留 arrived_mask` 的 BUG-RECONVERGENCE-SIMPLEGEMM 修复逻辑，但改为调用 `BarrierModule::init_warp_barrier`（在 `BarrierModule` 内部实现"已初始化则保留 arrived_mask"逻辑）
- [ ] 5.3 修改 release 路径：替换 `wbar.is_complete()` + `wbar.arrived_mask` + `wbar.participation_mask` 为 `barrier_module.is_warp_barrier_complete()` + `barrier_module.get_warp_barrier(0)->get_arrived_mask()`
- [ ] 5.4 删除 `src/ptxsim/instructions/barrier.cpp` 中所有 `sm_ctx->bsync_manager_.bsync/release` 调用（如果行为已通过 `BarrierModule` 覆盖）；NOTE：先确认 1.2 审计结果再删
- [ ] 5.5 验证：`ctest -R "integration_warp_barrier|integration_barrier_post_barrier|integration_divergence_sync" -V` 全部 PASS；MUST NOT 出现 BUG-POSTBARRIER-TWOHALVES 或 BUG-RECONVERGENCE-SIMPLEGEMM 回归

## 6. 旧代码清理

- [ ] 6.1 删除 `include/ptxsim/wbar.h`；从 `include/ptxsim/warp_state.h` 移除 `#include "ptxsim/wbar.h"`、`std::array<Wbar, 4> wbars`、`int current_wbar_id = -1`、`WarpState::reset()` 中的 wbar reset 循环
- [ ] 6.2 从 `src/ptxsim/core/sm_context.h` + `sm_context.cpp` 删除：
  - `synchronize_barrier()` 方法体
  - `barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_` 字段
  - **`sm_context.cpp:200-260` 周期 barrier 检查代码块**（`exe_once` 内的 `for (auto &[barId, waiting_threads] : barrier_waiting_threads)` 整段）：该逻辑依赖 `barrier_mutex_` 和全局 `barrier_waiting_threads` map；删除后 barrier 同步由 `CTAContext::barrier_module_` 完全接管；MUST NOT 留下孤儿 mutex
- [ ] 6.3 从 `src/CMakeLists.txt` 移除 `ptxsim/barrier/{warp_barrier,cta_barrier,barrier_module}.cpp` 仍保留（已加入）；如有 `bsync_state.cpp` 引用且仅用于 barrier（已审计），删除该 target
- [ ] 6.4 删除遗留备份：`rm src/ptxsim/instructions/barrier.cpp.bak src/ptxsim/instructions/barrier.cpp.orig`；MUST 验证 git history 可恢复（`git log --all --full-history -- src/ptxsim/instructions/barrier.cpp.orig`）
- [ ] 6.5 验证：`cmake --build build` 编译通过；`grep -rn "Wbar\b\|wbar\.\|wbars\[\|synchronize_barrier\|barrier_waiting_threads" src/ include/` 输出为空（MUST 零匹配）

## 7. 文档同步

- [ ] 7.1 重写 `docs/research/barrier-semantics/04-ptx-emu-current-implementation.md`：移除旧 `BarWarpSyncHandler` 描述；改为描述 `BarrierModule` 生产路径，包括 `init_warp_barrier` / `arrive_at_warp_barrier` / `release_warp_barrier` / `arrive_at_cta_barrier` / `release_cta_barrier` 流程图
- [ ] 7.2 更新 `docs/technical_design/barrier_module_design.md`：状态从"草稿"改为"已落地 v1"；补"已集成到生产路径"章节（指向 `barrier.cpp:333` 和 `cta_context.cpp`）；移除 §10 实施计划 TODO（已全部完成）
- [ ] 7.3 在 `docs/adr/0008-barrier-semantics.md` 追加 §"2026-06-17 追加：BarrierModule 集成决策"：描述 `CTAContext` 持有 `BarrierModule` 决策、`release_cta_barrier` 新签名、删除 `bsync_manager_`、新增 ADR 合规检查项
- [ ] 7.4 更新 `src/ptxsim/AGENTS.md`：在 STRUCTURE 块增加 `barrier/           # BarrierModule + WarpBarrier + CTABarrier`；在 KEY FILES 增加 `barrier/barrier_module.cpp`
- [ ] 7.5 更新 `src/ptxsim/instructions/AGENTS.md`：将"barrier.cpp"行改为"barrier.cpp (指令分发入口) → barrier/barrier_module.cpp (实际状态管理)"；标注 BarrierModule 为生产路径
- [ ] 7.6 验证：`grep -l "BarrierModule" docs/research/barrier-semantics/*.md docs/technical_design/*.md docs/adr/0008-*.md` 至少返回 3 个文件

## 8. 验证与发布

- [ ] 8.1 `./scripts/sanity.sh --quick` 全部通过；与 baseline.txt 对比，MUST NOT 新增 FAIL
- [ ] 8.2 `./scripts/sanity.sh` 完整回归通过；e2e 测试 `e2e_barrier_warp_sync` / `e2e_test3_cfg_full` / `e2e_barrier_warp_sync` 全部 PASS
- [ ] 8.3 `./tests/ptx/test_all_ptx.sh` 全部 PTX 语法测试通过
- [ ] 8.4 在 worktree 中创建最终 commit：`git add . && git commit -m "refactor(barrier): integrate BarrierModule into production path + fix BarHandler PC advance bug"`；commit message 引用 issue `BUG-POSTBARRIER-TWOHALVES` + `BUG-RECONVERGENCE-SIMPLEGEMM` + `BUG-HANDLER-PC-ADVANCE`
- [ ] 8.5 合并到主分支：`cd /workspace/project/PTX-EMU && git merge --no-ff fix/integrate-barrier-module -m "Merge branch..."`；清理 worktree
- [ ] 8.6 创建 PR（如适用）：`gh pr create --title "fix(barrier): integrate BarrierModule + fix handler PC advance" --body-file .github/PULL_REQUEST_TEMPLATE.md`；如不适用则在 commit log 中标注
