# SIMT 架构审查报告与测试计划

**日期**: 2026-05-04  
**审查范围**: SIMT 控制流管理数据结构与测试覆盖  
**审查目标**: simt_stack, warp_state, ThreadState, WarpContext, Wbar  
**文档状态**: 索引文档（配合改进计划和测试计划使用）

---

## 文档导航

本文档是 SIMT 架构审查的**索引和摘要文档**，详细内容请参见配套文档：

| 文档 | 路径 | 内容 |
|------|------|------|
| **架构审查报告** | 本文档 | 问题摘要、数据结构图、一致性分析 |
| **架构改进计划** | [simt-architecture-improvement-plan.md](../plans/simt-architecture-improvement-plan.md) | 4 个 Phase 的详细实施计划、代码变更示例 |
| **完整测试计划** | [simt-complete-test-plan.md](../testing/simt-complete-test-plan.md) | 53 个测试用例的详细规范、测试步骤、预期结果 |

---

## 一、SIMT 架构数据结构梳理

### 1.1 核心数据结构关系图

```
WarpContext
├── warp_state: WarpState                    # 单一权威源 (Single Source of Truth)
│   ├── threads[32]: ThreadState             # 每线程状态
│   │   ├── pc: uint32_t                     # 当前 PC (权威源)
│   │   ├── next_pc: uint32_t                # 下一条指令 PC
│   │   ├── status: ThreadStatus             # Active/Blocked/Exited/Yielded
│   │   ├── is_exited: bool
│   │   ├── is_blocked: bool
│   │   └── is_active: bool
│   ├── exec_mask: uint32_t                  # 当前分歧执行掩码 (divergence)
│   ├── wbars[4]: Wbar                       # Warp 级屏障
│   ├── current_wbar_id: int
│   └── warp_pc: uint32_t                    # 向后兼容 (deprecated)
│
├── active_mask[32]: bool                    # 线程存活掩码 (liveness)
│   └── (由 update_active_mask() 从 ThreadContext 刷新)
│
├── simt_stack: SIMTStack                    # SIMT 控制流栈
│   └── entries_: vector<SIMTStackEntry>
│       ├── branch_pc: int
│       ├── reconvergence_pc: int
│       ├── active_mask: uint32_t
│       ├── return_mask: uint32_t
│       └── return_pc: int
│
├── pc_stacks[32]: vector<int>               # 每线程 PC 栈 (传统分支，待清理)
│
└── threads[32]: unique_ptr<ThreadContext>   # 线程上下文 (执行容器)
    ├── warp_context_: WarpContext*           # 回指 warp
    ├── lane_id_: int                         # Lane ID
    └── (通过 warp_context_ 访问 warp_state)
```

> **关键区分**: `active_mask[32]` (bool 数组, 跟踪线程存活/退出) 与 `warp_state.exec_mask` (uint32_t, 跟踪当前分歧路径) 是**两个独立机制**。前者由 `update_active_mask()` 管理，后者由 `handle_branch()` 和 SIMT 栈收敛管理。它们在分歧执行期间的值**不应相同**。

### 1.2 数据流向

```
分支指令 (handle_branch)              [warp_context.cpp:9-82]
  → 检测分歧 (taken_mask vs not_taken_mask)
  → 推入 SIMT 栈 {branch_pc, reconvergence_pc, return_mask}
  → 设置 warp_state.threads[i].pc/next_pc (按路径分流)
  → 设置 warp_state.exec_mask = taken_mask (分歧时)
        ↓
指令执行 (execute_warp_instruction)   [warp_context.cpp:146-196]
  → lane 活跃 + PC 匹配检查 (使用 active_mask + warp_state.pc)
  → sync_from_warp_state() → ThreadContext (WarpState → ThreadContext)
  → BAR_SYNC fallback → synchronize_barrier() (如果线程阻塞)
  → execute_thread_instruction() → InstructionFactory 分派
  → sync_to_warp_state() → warp_state (ThreadContext → WarpState)
  → update_active_mask() → active_mask[32] (ThreadContext 退出/阻塞 → bool[])
        ↓
收敛检查 (check_reconvergence)        [warp_context.cpp:98-101]
  → simt_stack.check_reconvergence(threads)
  → 如果收敛 → pop SIMT 栈 (⚠ 当前未恢复 exec_mask, BUG-001)
  → exec_mask 应恢复为 0xFFFFFFFF 或外层 return_mask
        ↓
屏障同步 (bar.warp.sync)             [barrier.cpp:95-182]
  → wbar.arrive(lane_id)
  → 如果完成 → 设置所有线程 PC = reconvergence_pc
  → 更新状态 + wbar.reset()
  → (⚠ 屏障完成后未检查 SIMT 栈收敛, BUG-003)
```

---

## 二、设计问题摘要

### 🔴 严重问题 (P0 - 立即修复)

| ID | 问题 | 位置 | 影响 | 修复 Phase |
|----|------|------|------|-----------|
| BUG-001 | exec_mask 收敛后未恢复 | warp_context.cpp:98-101 | 所有分歧分支场景 | Phase 1.1 |
| BUG-002 | 退出线程导致 SIMT 栈永久阻塞 | simt_stack.cpp:8-16 | warp 挂起 | Phase 1.2 |

### 🟡 中等问题 (P1 - 短期修复)

| ID | 问题 | 位置 | 影响 | 修复 Phase |
|----|------|------|------|-----------|
| BUG-003 | 屏障完成未清理 SIMT 栈 | barrier.cpp:144-175 | 栈残留 | Phase 4.1 |
| BUG-004 | sync 覆盖分支设置的 PC | thread_context.cpp:723-754 | PC 不一致 | 已修复，待验证 |
| BUG-005 | SIMT 栈无深度限制 | simt_stack.cpp:28-29 | 内存风险 | Phase 2.2 |

### 🟢 轻微问题 (P2 - 代码清理)

| ID | 问题 | 位置 | 影响 | 修复 Phase |
|----|------|------|------|-----------|
| ISSUE-001 | WarpState::pc_stack 未使用 | warp_state.h:20-21 | 内存浪费、混淆 | Phase 1.3 |
| ISSUE-002 | 双重 PC 管理机制 | warp_context.h | 同步风险 | Phase 2.1 |
| ISSUE-003 | reset() 不清空 pc_stack | warp_state.h:34 | 调试脏数据 | Phase 1.3 |
| ISSUE-004 | active_mask 双源不一致 | warp_context.cpp:198-210 | 状态不同步 | Phase 2 (新增) |
| ISSUE-005 | is_lane_active vs is_lane_schedulable 双源 | warp_context.h:128,103 | 方法返回不一致 | Phase 2 (新增) |
| ISSUE-006 | WarpContext::pc 向后兼容字段残留 | warp_context.h:57-60 | 混淆权威源 | Phase 2 (新增) |

> **ISSUE-004 详情**: `update_active_mask()` (warp_context.cpp:198-210) 从 `ThreadContext::is_exited()` 读取退出状态，但从 `warp_state.threads[i].is_blocked` 读取阻塞状态——两个不同数据源。修复 Phase 1.2 后，`warp_state.threads[i].is_exited` 是退出状态的权威源，`active_mask` 应统一从 `warp_state` 读取。
> 
> **ISSUE-005 详情**: `is_lane_active(lane)` 使用 `active_mask[lane_id]` (WarpContext 成员)，而 `is_lane_schedulable(lane)` 使用 `warp_state.threads[lane].is_schedulable()` (WarpState 成员)。两者可能返回不同结果，且 `execute_warp_instruction()` 同时使用两者（line 153-158），存在竞态语义风险。

> **详细分析**: 包括问题描述、代码片段、影响分析、修复建议，请参见本文档"二、设计问题检查"章节的原始内容（下方保留）。

---

## 三、数据一致性分析

### 3.1 PC 一致性

| 操作 | warp_state.threads[i].pc | ThreadContext (通过 get_pc) | 一致性 |
|------|-------------------------|---------------------------|--------|
| 初始化 | set | set (via init) | ✅ |
| handle_branch | set | N/A (直接设置 warp_state) | ⚠️ 需后续同步 |
| barrier 完成 | set | set (via force_set_pc) | ✅ |
| commit_pc | set (pc ← next_pc) | 读取 warp_state | ✅ |
| sync_from_warp_state | 权威源 | 读取 warp_state | ✅ |
| sync_to_warp_state | 读取 next_pc | 写入 warp_state.next_pc | ✅ |

**结论**: 当前架构以 `warp_state.threads[i].pc` 为单一权威源，一致性较好。

### 3.2 状态一致性

| 操作 | warp_state.status | ThreadContext.state | 一致性 |
|------|------------------|---------------------|--------|
| 初始化 | Active | RUN | ✅ |
| barrier 等待 | Blocked | BAR_SYNC (via sync_from) | ✅ |
| barrier 完成 | Active | RUN (via force_set_pc) | ⚠️ 需显式更新 |
| 线程退出 | Exited | EXIT (via sync_to) | ✅ |

### 3.3 SIMT Stack 一致性

| 操作 | simt_stack | warp_state.exec_mask | 一致性 |
|------|-----------|---------------------|--------|
| handle_branch (divergent) | push | 更新为 taken_mask | ✅ |
| check_reconvergence | pop if converged | 应恢复 return_mask | ⚠️ **未实现 (BUG-001)** |
| barrier 完成 | 不变 | 更新为 arrived_mask | ⚠️ 不关联 |

**关键问题**: `check_reconvergence()` 弹出栈后**没有恢复 exec_mask**！

### 3.4 active_mask 与 warp_state.threads 一致性 (新增)

| 操作 | active_mask[32] | warp_state.threads[i].is_active | 一致性 |
|------|----------------|-------------------------------|--------|
| 线程退出 | set false (via ThreadContext::is_exited) | set false (via sync_to_warp_state) | ⚠️ 不同步回 active_mask |
| 线程阻塞 | set false (via warp_state.is_blocked) | set true (via sync_to) | ⚠️ 可能不同步 |
| warp reset() | reset to true | reset to true | ✅ |
| handle_branch | 不变 | 不变 | ✅ (不管理存活) |

**关键问题**: `update_active_mask()` 同时从 `ThreadContext` 和 `warp_state` 读取，但只写入 `active_mask`——不写回 `warp_state.threads[i].is_active`。导致两个 `is_active` 源可能不一致 (ISSUE-004)。

---

## 四、架构改进计划索引

> **详细计划**: 包括代码变更示例、测试验证、实施步骤，请参见 [simt-architecture-improvement-plan.md](../plans/simt-architecture-improvement-plan.md)

### 4.1 改进 Phase 概览

| Phase | 类型 | 改动范围 | 风险 | 预计影响 |
|-------|------|---------|------|---------|
| **Phase 1** | Bug 修复 | 3 文件 | 🟢 低 | 修复 3 个逻辑缺陷 |
| **Phase 2** | 代码清理 | 5 文件 | 🟢 低 | 删除未使用代码 |
| **Phase 3** | 测试补充 | 9 新文件 | 🟢 无 | 覆盖率 15% → 90% |
| **Phase 4** | 架构增强 | 3 文件 | 🟡 中 | 深度限制、屏障集成 |

### 4.2 Phase 1: 关键 Bug 修复 (P0)

- **1.1** 修复 exec_mask 收敛后未恢复 → `warp_context.cpp`
- **1.2** 修复退出线程导致 SIMT 栈永久阻塞 → `simt_stack.cpp`
- **1.3** 删除 WarpState 未使用字段 → `warp_state.h`

### 4.3 Phase 2: 代码清理与统一 (P1)

- **2.1** 评估并清理双重 PC 管理
- **2.2** 添加 SIMT 栈最大深度限制 (MAX_DEPTH = 10)

### 4.4 Phase 3: 测试补充 (P0-P1)

创建 9 个新测试文件，53 个测试用例（详见测试计划索引）

### 4.5 Phase 4: 架构增强 (P1-P2)

- **4.1** 屏障完成后 SIMT 栈清理
- **4.2** 添加 SIMT 调试工具
- **4.3** 性能优化

### 4.6 实施里程碑

| 里程碑 | 内容 | 验收标准 |
|--------|------|---------|
| Milestone 1 | Bug 修复完成 | ctest 全部通过，新增 2 个 bug 修复测试通过 |
| Milestone 2 | 代码清理完成 | clang-tidy 无警告，代码行数减少 10% |
| Milestone 3 | 核心测试完成 | ctest -L simt 全部通过，覆盖率 ≥ 85% |
| Milestone 4 | 集成测试与文档 | 覆盖率 ≥ 90%，架构文档更新 |

### 4.7 风险控制

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Bug 修复引入回归 | 中 | 高 | 每步运行全量测试 |
| 删除 pc_stacks 影响未知调用 | 低 | 高 | 全局搜索 + 编译验证 |
| 测试用例设计错误 | 中 | 中 | 代码审查 + 手动验证 |
| SIMT 栈深度限制影响现有内核 | 低 | 中 | 先跑 benchmark 检测 |

---

## 五、测试计划索引

> **详细计划**: 包括 53 个测试用例的详细规范、测试步骤、预期结果、CMakeLists.txt 集成，请参见 [simt-complete-test-plan.md](../testing/simt-complete-test-plan.md)

### 5.1 测试目标

| 目标 | 当前状态 | 目标状态 | 度量方式 |
|------|---------|---------|---------|
| 代码覆盖率 (SIMT 分支/收敛路径) | ~40% (已有 22+ Catch2 测试) | ≥ 90% | gcov/lcov |
| 测试用例数 | 22 (Catch2 SIMT) + 3 (warp standalone) | 75+ | ctest --list-tests |
| Bug 检测率 | 部分 (3/5 已知 bug 有测试) | 100% | 已知 bug 清单 |
| 回归测试 | 部分 (ctest -L simt) | 全部 | ctest 全量通过 |

> **注**: 原报告估计覆盖率 ~15% 过于保守。实际代码探索发现已有 22+ Catch2 测试覆盖 barrier、divergence、PC 管理、调度器等 SIMT 关键路径。但 **exec_mask 管理** 和 **SIMT 栈收敛恢复** 路径确实缺乏直接测试。

### 5.2 测试组概览 (9 组，53 用例)

| 组 | 文件 | 用例数 | 优先级 | Bug 检测 | 依赖 Phase |
|----|------|--------|--------|---------|-----------|
| A | test_simt_stack_catch2.cpp | 8 | P0 | 中 | Phase 2 (A8) |
| B | test_simt_stack_entry.cpp | 6 | P0 | **高** | Phase 1 (B4) |
| C | test_warp_state.cpp | 7 | P0 | 低 | Phase 1 (C7) |
| D | test_handle_branch_integration.cpp | 5 | P0 | **高** | Phase 1 (D5) |
| E | test_barrier_simt_integration.cpp | 4 | P1 | **高** | Phase 4 (E2,E3) |
| F | test_exec_mask.cpp | 6 | P1 | **高** | Phase 1 (F3) |
| G | test_pc_management_advanced.cpp | 5 | P1 | 中 | Phase 2 |
| H | test_sync_mechanism.cpp | 6 | P1 | **高** | 无 |
| I | test_simt_integration.cpp | 6 | P1 | **高** | Phase 1,4 |
| **总计** | | **53** | | | |

### 5.3 Bug 检测映射

| Bug ID | 描述 | 测试用例 | 严重性 | 检测阶段 |
|--------|------|---------|--------|---------|
| BUG-001 | exec_mask 收敛后未恢复 | F3, D5, I1 | 🔴 高 | Phase 1 |
| BUG-002 | 退出线程阻塞 SIMT 栈 | B4, I4 | 🔴 高 | Phase 1 |
| BUG-003 | 屏障完成未清理 SIMT 栈 | E2, E3 | 🟡 中 | Phase 4 |
| BUG-004 | sync 覆盖分支 PC | H3 | 🟡 中 | 无 |
| BUG-005 | SIMT 栈无深度限制 | A8 | 🟢 低 | Phase 2 |

### 5.4 测试执行顺序

```
Phase 1 执行:
1. B (SIMTStackEntry) - 检测退出线程 bug
2. F (exec_mask) - 检测恢复 bug
3. D (handle_branch) - 验证分支集成
4. C (WarpState) - 基础验证

Phase 2 执行:
5. A (SIMT Stack) - 基础操作
6. G (PC Management) - 高级 PC

Phase 3 执行:
7. H (Sync) - 同步机制
8. E (Barrier + SIMT) - 屏障交互

Phase 4 执行:
9. I (Integration) - 完整场景
```

### 5.5 测试运行命令

```bash
# 运行所有 SIMT 测试
cd build && ctest -L simt -V

# 运行特定测试组
ctest -R test_simt_stack_catch2 -V
ctest -R test_exec_mask -V

# 运行 barrier 相关测试
ctest -L barrier -V

# 运行集成测试
ctest -L integration -V

# 运行关键 Bug 检测测试
ctest -R "test_simt_stack_entry|test_exec_mask" -V
```

### 5.6 测试环境配置

```bash
# Debug 构建 (启用覆盖率)
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="--coverage" \
  -DCMAKE_C_FLAGS="--coverage"

# 生成覆盖率报告
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_report
```

---

## 六、验收标准

### 6.1 功能验收

- [ ] 所有 7 个架构改进实施完成
- [ ] 53 个新测试用例全部通过
- [ ] 现有测试 100% 通过 (无回归)
- [ ] clang-format + clang-tidy 无警告

### 6.2 质量验收

| 指标 | 目标 | 测量方式 |
|------|------|---------|
| 代码覆盖率 (SIMT) | ≥ 90% | lcov |
| 分支覆盖率 (SIMT) | ≥ 85% | gcov --branches |
| 测试执行时间 | < 30s | ctest --timeout |
| 内存泄漏 | 0 | valgrind |
| 代码重复率 | < 5% | cpplint |

### 6.3 性能验收

| 基准 | 当前 | 目标 | 测量方式 |
|------|------|------|---------|
| SIMT push 操作 | ~5ns | < 10ns | benchmark |
| check_reconvergence | ~50ns | < 100ns | benchmark |
| handle_branch | ~200ns | < 500ns | benchmark |

---

## 七、文件变更清单

| 文件 | 变更类型 | Phase | 详细说明 |
|------|---------|-------|---------|
| `include/ptxsim/warp_state.h` | 删除字段 | 1 | 改进计划 §1.3 |
| `include/ptxsim/warp_context.h` | 删除方法 | 2 | 改进计划 §2.1 |
| `include/ptxsim/simt_stack.h` | 添加常量 | 2 | 改进计划 §2.2 |
| `src/ptxsim/core/warp_context.cpp` | 修复逻辑 | 1 | 改进计划 §1.1 |
| `src/ptxsim/core/simt_stack.cpp` | 修复逻辑 | 1 | 改进计划 §1.2 |
| `src/ptxsim/core/sm_context.cpp` | 扩展检查 | 4 | 改进计划 §4.1 |
| `src/ptxsim/instructions/barrier.cpp` | 可选修改 | 4 | 改进计划 §4.1 |
| `tests/CMakeLists.txt` | 添加测试 | 3 | 测试计划 §四 |
| `tests/test_*.cpp` (9 文件) | 新增 | 3 | 测试计划 §二 |

---

## 八、总结

### 发现的设计问题

| 类别 | 数量 | 严重性 |
|------|------|--------|
| 未使用字段 | 3 | 🟢 低 |
| 数据不一致风险 | 6 | 🟡 中 |
| 逻辑缺陷 | 4 | 🔴 高 |
| **总计** | **11** | |

> **新增问题** (2026-05-04 评审): ISSUE-004 (active_mask 双源)、ISSUE-005 (is_lane_active 双源)、ISSUE-006 (WarpContext::pc 残留)。均在 Phase 2 代码清理阶段处理。

### 测试覆盖缺口

| 模块 | 当前覆盖 | 目标覆盖 | 差距 |
|------|---------|---------|------|
| SIMT Stack 基础操作 | ~50% (push/pop 逻辑通过集成测试间接覆盖) | 95% | 45% |
| exec_mask 恢复路径 | **0%** (直接测试缺失) | 90% | 90% |
| Barrier + SIMT 栈交互 | ~30% (barrier 完成已测试，但栈清理未测试) | 85% | 55% |
| sync 机制 | ~50% (sync_from/to 通过集成测试覆盖) | 90% | 40% |
| 集成场景 | ~40% (test3_reproduction 等覆盖核心场景) | 80% | 40% |
| WarpContext 核心 (新增关注) | ~60% (active_mask、调度、执行已覆盖) | 90% | 30% |

### 关键行动项

1. ✅ 立即修复 exec_mask 恢复逻辑 (Phase 1.1)
2. ✅ 立即修复退出线程收敛检查 (Phase 1.2)
3. ✅ 创建 53 个新测试用例 (Phase 3)
4. 🔄 清理未使用字段和代码 (Phase 1.3, 2.1)
5. 🔄 统一 active_mask 数据源 (Phase 2 - ISSUE-004)
6. 🔄 废弃 WarpContext::pc 并统一 PC 访问 (Phase 2 - ISSUE-006)
7. 🔄 迁移旧测试到 Catch2 框架 (Phase 3)
8. 🔄 添加 SIMT 栈深度限制 (Phase 2.2)
9. 🔄 屏障完成后 SIMT 栈清理 (Phase 4.1)
10. 🔄 修复 is_lane_active/is_lane_schedulable 双源 (Phase 2 - ISSUE-005)

---

**报告生成时间**: 2026-05-04  
**审查者**: AI Architecture Review  
**文档角色**: 索引和摘要（配合改进计划和测试计划使用）  
**下一步**: 开始 Phase 1 实施（详见 [改进计划](../plans/simt-architecture-improvement-plan.md)）

---

## 附录：原始设计问题详细分析

> 以下为审查报告中各问题的详细分析，保留供参考。

### 问题 1: exec_mask 收敛后未恢复 (BUG-001) 🔴

**位置**: `src/ptxsim/core/warp_context.cpp:98-101`

```cpp
void WarpContext::check_reconvergence() {
    if (simt_stack.empty()) return;
    simt_stack.check_reconvergence(warp_state.threads);
    // ⚠️ 没有恢复 exec_mask
}
```

**影响**: 收敛后执行掩码错误，影响后续指令调度

**修复**: 见 [改进计划 §1.1](../plans/simt-architecture-improvement-plan.md#11-修复-exec_mask-收敛后未恢复)

---

### 问题 2: WarpState::pc_stack 和 pc_stack_depth 未使用 (ISSUE-001) 🟢

**位置**: `include/ptxsim/warp_state.h:20-21`

```cpp
std::array<int, 16> pc_stack;
int pc_stack_depth = 0;
```

**问题**:
- 这两个字段在 WarpState 中定义，但**没有任何代码读写它们**
- 与 WarpContext::pc_stacks[32] 重复且混乱

**影响**:
- 内存浪费 (16 * 4 = 64 字节/warp)
- 代码维护困惑

**修复**: 见 [改进计划 §1.3](../plans/simt-architecture-improvement-plan.md#13-删除-warpstate-未使用字段)

---

### 问题 3: 双重 PC 管理机制存在不一致风险 (ISSUE-002) 🟢

**位置**: 
- `WarpContext::pc_stacks[32]` (传统分支)
- `warp_state.threads[32].pc` (SIMT v2.0)

**当前代码路径**:
- `handle_branch()` → 使用 `warp_state.threads[i].pc` ✅
- `handle_branch_divergence()` → 使用 `pc_stacks[i]` ⚠️
- `update_pc_stack()` → 操作 `pc_stacks[i]` ⚠️

**修复**: 见 [改进计划 §2.1](../plans/simt-architecture-improvement-plan.md#21-评估并清理双重-pc-管理)

---

### 问题 4: SIMT Stack 未在屏障完成后清理 (BUG-003) 🟡

**位置**: `src/ptxsim/instructions/barrier.cpp:144-175`

**场景**:
```
PC=11: bra.cond target (divergent) → push SIMT stack {reconv=20}
PC=12: ... (taken path)
PC=16: bar.warp.sync 0
PC=21: ← reconvergence point (should pop SIMT stack)
```

**修复**: 见 [改进计划 §4.1](../plans/simt-architecture-improvement-plan.md#41-屏障完成后-simt-栈清理)

---

### 问题 5: sync_to_warp_state 可能覆盖分支设置的 PC (BUG-004) 🟡

**位置**: `src/ptxsim/core/thread_context.cpp:723-754`

**状态**: 已修复 (通过架构调整)，但需要测试验证

**测试**: 见 [测试计划 §H3](../testing/simt-complete-test-plan.md#h3-sync-不覆盖分支设置的-pc)

---

### 问题 6: SIMTStackEntry::is_converged 检查不完整 (BUG-002) 🔴

**位置**: `src/ptxsim/core/simt_stack.cpp:8-16`

**场景**:
```
1. 分支: return_mask = 0xFFFFFFFF (所有线程)
2. taken 路径: 线程 0-15 执行 exit
3. 收敛检查: 线程 0-15 永远不会到达 reconvergence_pc (已退出)
4. 结果: is_converged 永远返回 false → SIMT 栈永远不弹出
```

**修复**: 见 [改进计划 §1.2](../plans/simt-architecture-improvement-plan.md#12-修复退出线程导致-simt-栈永久阻塞)

---

### 问题 7: SIMTStack 无最大深度限制 (BUG-005) 🟢

**位置**: `src/ptxsim/core/simt_stack.cpp:28-29`

```cpp
void SIMTStack::push(const SIMTStackEntry& entry) {
    entries_.push_back(entry);  // 无限制增长
}
```

**建议**: 添加最大深度检查 (如 10 层)

**修复**: 见 [改进计划 §2.2](../plans/simt-architecture-improvement-plan.md#22-添加-simt-栈最大深度限制)

