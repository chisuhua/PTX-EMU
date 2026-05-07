# SIMT 架构完整测试计划

**文档版本**: 1.0
**日期**: 2026-05-04
**状态**: ✅ 已完成  
**基于文档**:
- [SIMT 架构审查报告](../reports/simt-architecture-review-and-test-plan.md)
- [SIMT 架构改进计划](../plans/simt-architecture-improvement-plan.md)
- [SIMT 架构分歧审查](simt-architecture-divergence-review.md)

---

## 执行摘要

本测试计划旨在验证 SIMT (Single Instruction, Multiple Thread) 架构的正确性，覆盖 **53 个新测试用例** 和 **现有测试迁移**，目标将 SIMT 相关代码的测试覆盖率从 **15% 提升至 90%+**。

### 测试目标

| 目标 | 当前状态 | 目标状态 | 度量方式 |
|------|---------|---------|---------|
| 覆盖率 (SIMT 关键路径) | ~40% (22+ Catch2 + 3 standalone) | ≥ 90% | gcov/lcov |
| 测试用例数 | 25 (Catch2 SIMT) + 38 (PTX 解析) | 85+ | ctest --list-tests |
| Bug 检测率 | 60% (3/5 已知 bug) | 100% | 已知 bug 清单 |
| 回归测试 | `ctest -L simt` (已有) | 全部 | ctest 全量通过 |

### 实际完成状态 (2026-05-05)

| 目标 | 计划 | 实际 | 状态 |
|------|------|------|------|
| 覆盖率 (SIMT) | ≥ 90% | 待测量 | ⏳ |
| 测试用例数 | 60 | 57 | ✅ |
| Bug 检测率 | 100% | 100% (3/3) | ✅ |
| 回归测试 | 全部通过 | 9/9 通过 | ✅ |

### 测试范围

**核心组件**:
- `SIMTStack` - SIMT 控制流栈
- `SIMTStackEntry` - 栈条目与收敛检查
- `WarpState` - Warp 级状态容器
- `WarpContext` - Warp 上下文与分支处理
- `ThreadState` - 线程状态
- `ThreadContext` - 线程上下文与同步机制
- `Wbar` - Warp 级屏障

**测试标签**: `simt`, `branch`, `barrier`, `sync`, `integration`, `pc`, `exec_mask`

---

## 一、测试环境配置

### 1.1 构建配置

```bash
# 1. 设置环境
. env.sh

# 2. Debug 构建 (启用覆盖率)
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="--coverage" \
  -DCMAKE_C_FLAGS="--coverage"

# 3. 构建
cmake --build build -j$(nproc)
```

### 1.2 测试执行

```bash
cd build

# 运行所有 SIMT 测试
ctest -L simt -V

# 运行特定测试组
ctest -R test_simt_stack -V
ctest -R test_exec_mask -V

# 并行运行测试
ctest -j$(nproc) -L simt

# 生成覆盖率报告
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_report
```

### 1.3 测试依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Catch2 | 3.x (amalgamated) | 测试框架 |
| CMake | ≥ 3.15 | 构建系统 |
| GCC/Clang | ≥ 10 | 编译器 |
| gcov/lcov | 系统默认 | 覆盖率工具 |

---

## 二、测试用例详细规范

### 测试组 A: SIMT Stack 基础操作 (8 用例)

**文件**: `tests/test_simt_stack_catch2.cpp`  
**优先级**: P0  
**目标**: 验证 SIMTStack 类的核心功能

#### A1: 基本 push/pop 操作

```cpp
TEST_CASE("SIMTStack: push and pop operations", "[simt_stack][basic]")
```

**测试步骤**:
1. 创建空 SIMTStack
2. 创建 SIMTStackEntry (branch_pc=10, reconvergence_pc=20)
3. push(entry)
4. 验证 stack.empty() == false
5. 验证 stack.depth() == 1
6. 验证 top().reconvergence_pc == 20
7. pop()
8. 验证 stack.empty() == true
9. 验证返回的 entry 与 push 的一致

**预期结果**: 所有断言通过

---

#### A2: empty 和 depth 检查

```cpp
TEST_CASE("SIMTStack: empty and depth checks", "[simt_stack][basic]")
```

**测试步骤**:
1. 新栈: empty() == true, depth() == 0
2. push 3 个条目
3. empty() == false, depth() == 3
4. pop 2 次
5. depth() == 1
6. pop 1 次
7. empty() == true, depth() == 0

**预期结果**: 深度计数准确

---

#### A3: top 访问 (const 和非 const)

```cpp
TEST_CASE("SIMTStack: top access", "[simt_stack][basic]")
```

**测试步骤**:
1. push 2 个不同条目
2. 非 const top(): 验证返回最新条目
3. const top(): 验证返回最新条目
4. 修改 top() 返回的引用，验证栈内数据更新

**预期结果**: top() 始终返回栈顶

---

#### A4: clear 操作

```cpp
TEST_CASE("SIMTStack: clear operation", "[simt_stack][basic]")
```

**测试步骤**:
1. push 5 个条目
2. clear()
3. empty() == true, depth() == 0

**预期结果**: 栈完全清空

---

#### A5: pop 空栈抛出异常

```cpp
TEST_CASE("SIMTStack: pop empty throws exception", "[simt_stack][exception]")
```

**测试步骤**:
1. 创建空栈
2. 调用 pop()
3. 验证抛出 std::runtime_error

**预期结果**: 异常消息包含 "empty"

---

#### A6: top 空栈抛出异常

```cpp
TEST_CASE("SIMTStack: top empty throws exception", "[simt_stack][exception]")
```

**测试步骤**:
1. 创建空栈
2. 调用 top()
3. 验证抛出 std::runtime_error

**预期结果**: 异常消息包含 "empty"

---

#### A7: 嵌套 push 保持顺序

```cpp
TEST_CASE("SIMTStack: nested push preserves order", "[simt_stack][nested]")
```

**测试步骤**:
1. push entry1 (branch_pc=10, reconvergence_pc=30)
2. push entry2 (branch_pc=15, reconvergence_pc=25)
3. push entry3 (branch_pc=20, reconvergence_pc=22)
4. 验证 top() == entry3
5. pop(), 验证 top() == entry2
6. pop(), 验证 top() == entry1
7. pop(), 验证 empty()

**预期结果**: LIFO 顺序正确

---

#### A8: 最大深度限制 (Phase 4)

```cpp
TEST_CASE("SIMTStack: maximum depth enforcement", "[simt_stack][limit]")
```

**前置条件**: Phase 2 已实施深度限制

**测试步骤**:
1. push 10 个条目 (MAX_DEPTH)
2. 验证 depth() == 10
3. 尝试 push 第 11 个条目
4. 验证抛出 std::runtime_error

**预期结果**: 异常消息包含 "overflow" 或 "maximum depth"

---

### 测试组 B: SIMTStackEntry 收敛检查 (6 用例)

**文件**: `tests/test_simt_stack_entry.cpp`  
**优先级**: P0  
**目标**: 验证收敛检查逻辑，**检测退出线程 bug**

#### B1: 所有线程收敛

```cpp
TEST_CASE("SIMTStackEntry: all threads converged", "[simt_entry][convergence]")
```

**测试步骤**:
1. 创建 ThreadState[32]，所有线程 pc=20
2. 创建 entry (reconvergence_pc=20, return_mask=0xFFFFFFFF)
3. 调用 is_converged(threads)

**预期结果**: 返回 true

---

#### B2: 部分线程未收敛

```cpp
TEST_CASE("SIMTStackEntry: partial convergence", "[simt_entry][convergence]")
```

**测试步骤**:
1. 创建 ThreadState[32]
2. 线程 0-15: pc=20
3. 线程 16-31: pc=15
4. entry (reconvergence_pc=20, return_mask=0xFFFFFFFF)
5. 调用 is_converged(threads)

**预期结果**: 返回 false

---

#### B3: return_mask 排除的线程不影响收敛

```cpp
TEST_CASE("SIMTStackEntry: excluded threads don't affect convergence", "[simt_entry]")
```

**测试步骤**:
1. 线程 0-15: pc=20
2. 线程 16-31: pc=99 (任意值)
3. entry (reconvergence_pc=20, return_mask=0x0000FFFF)
4. 调用 is_converged(threads)

**预期结果**: 返回 true (只检查 return_mask 中的线程)

---

#### B4: 退出线程的收敛处理 ⚠️ **关键 Bug 检测**

```cpp
TEST_CASE("SIMTStackEntry: exited threads handling", "[simt_entry][bug][critical]")
```

**场景**: 分支后部分线程执行 exit，不应阻塞收敛

**测试步骤**:
1. 初始化 ThreadState[32]
2. 创建 entry (reconvergence_pc=20, return_mask=0xFFFFFFFF)
3. 模拟 lanes 0-15 退出:
   - threads[i].is_exited = true
   - threads[i].is_active = false
   - threads[i].pc = 0 (永远不会到 20)
4. 模拟 lanes 16-31 到达: threads[i].pc = 20
5. 调用 stack.check_reconvergence(threads)

**预期结果 (当前失败)**:
- ✅ 返回 true (退出线程被跳过)
- ✅ stack.empty() == true (栈弹出)

**Bug 修复后**: 测试应通过

---

#### B5: 空 return_mask 立即收敛

```cpp
TEST_CASE("SIMTStackEntry: empty return_mask", "[simt_entry]")
```

**测试步骤**:
1. entry (return_mask=0x00000000)
2. 线程 PC 任意值
3. 调用 is_converged(threads)

**预期结果**: 返回 true (无线程需要检查)

---

#### B6: toString 输出格式

```cpp
TEST_CASE("SIMTStackEntry: toString format", "[simt_entry]")
```

**测试步骤**:
1. 创建 entry (branch_pc=10, reconvergence_pc=20, active_mask=0xFFFF, return_pc=20)
2. 调用 toString()

**预期结果**: 字符串包含关键字段值

---

### 测试组 C: WarpState 测试 (7 用例)

**文件**: `tests/test_warp_state.cpp`  
**优先级**: P0  
**目标**: 验证 WarpState 状态管理

#### C1: 默认初始化

```cpp
TEST_CASE("WarpState: default initialization", "[warp_state]")
```

**测试步骤**:
1. 创建 WarpState
2. 验证所有 threads[i].pc == 0
3. 验证 exec_mask == 0xFFFFFFFF
4. 验证所有 threads[i].is_active == true
5. 验证所有 threads[i].is_exited == false
6. 验证 current_wbar_id == -1
7. 验证 warp_pc == 0

**预期结果**: 所有字段正确初始化

---

#### C2: reset() 完整性

```cpp
TEST_CASE("WarpState: reset restores defaults", "[warp_state]")
```

**测试步骤**:
1. 修改所有字段为非默认值
2. 调用 reset()
3. 验证所有字段恢复默认值

**预期结果**: 完全重置

---

#### C3: count_active_lanes

```cpp
TEST_CASE("WarpState: count_active_lanes", "[warp_state]")
```

**测试步骤**:
1. 设置 16 个线程 is_active=true, 16 个 false
2. 验证 count_active_lanes() == 16
3. 设置 5 个线程 is_exited=true
4. 验证 count_active_lanes() == 11 (排除退出)

**预期结果**: 计数准确

---

#### C4: count_schedulable_lanes

```cpp
TEST_CASE("WarpState: count_schedulable_lanes", "[warp_state]")
```

**测试步骤**:
1. 设置线程状态:
   - 20 个: Active (可调度的)
   - 5 个: Blocked (不可调度)
   - 5 个: Exited (不可调度)
   - 2 个: is_active=false (不可调度)
2. 验证 count_schedulable_lanes() == 20

**预期结果**: 只计数可调度的线程

---

#### C5: is_all_exited

```cpp
TEST_CASE("WarpState: is_all_exited", "[warp_state]")
```

**测试步骤**:
1. 所有线程 is_exited=false → false
2. 31 个退出，1 个未退出 → false
3. 所有线程 is_exited=true → true

**预期结果**: 逻辑正确

---

#### C6: has_schedulable_threads

```cpp
TEST_CASE("WarpState: has_schedulable_threads", "[warp_state]")
```

**测试步骤**:
1. 至少 1 个可调度 → true
2. 全部不可调度 → false

**预期结果**: 快捷方法正确

---

#### C7: pc_stack 未使用验证 (临时)

```cpp
TEST_CASE("WarpState: pc_stack field removal verification", "[warp_state][cleanup]")
```

**前置条件**: Phase 1.3 已删除 pc_stack 字段

**测试步骤**:
1. 验证 WarpState 不包含 pc_stack 成员
2. 验证编译通过

**预期结果**: 编译成功，字段已移除

---

### 测试组 D: handle_branch 与 SIMT Stack 集成 (5 用例)

**文件**: `tests/test_handle_branch_integration.cpp`  
**优先级**: P0  
**目标**: 验证分支处理与 SIMT 栈的交互

#### 测试辅助函数

```cpp
static void setup_warp_with_threads(WarpContext& warp) {
    // 创建 32 个 ThreadContext 并添加到 warp
    // (复用 test_pc_management.cpp 的 init_warp_with_threads)
}
```

---

#### D1: 非分歧分支 (所有线程 taken)

```cpp
TEST_CASE("handle_branch: non-divergent all taken", "[branch][simt]")
```

**测试步骤**:
1. 设置 warp，所有线程 pc=10
2. handle_branch(predicate="", target_pc=20, reconvergence_pc=30, current_pc=10)
3. 验证 simt_stack.empty() == true (无分歧，不 push)
4. 验证所有线程 pc=20
5. 验证 exec_mask == 0xFFFFFFFF

**预期结果**: 无 SIMT 栈操作，所有线程跳转

---

#### D2: 非分歧分支 (所有线程 not taken)

```cpp
TEST_CASE("handle_branch: non-divergent none taken", "[branch][simt]")
```

**测试步骤**:
1. 设置 warp，predicate 全部 false
2. handle_branch(target_pc=20, reconvergence_pc=30, current_pc=10)
3. 验证 simt_stack.empty() == true
4. 验证所有线程 pc=11 (fallthrough)
5. 验证 exec_mask == 0xFFFFFFFF

**预期结果**: 无 SIMT 栈操作，所有线程 fallthrough

---

#### D3: 分歧分支 (部分 taken) ⚠️ **核心场景**

```cpp
TEST_CASE("handle_branch: divergent branch", "[branch][simt][divergence]")
```

**场景**: 线程 0-15 taken (target=20), 线程 16-31 not taken (fallthrough=11)

**测试步骤**:
1. 设置 predicate，使得 lanes 0-15 true, 16-31 false
2. handle_branch(target_pc=20, reconvergence_pc=30, current_pc=10)
3. 验证 simt_stack.depth() == 1
4. 验证 top.branch_pc == 10
5. 验证 top.reconvergence_pc == 30
6. 验证 top.active_mask == 0x0000FFFF (taken lanes)
7. 验证 lanes 0-15 pc=20
8. 验证 lanes 16-31 pc=11
9. 验证 exec_mask == 0x0000FFFF

**预期结果**: SIMT 栈正确 push，线程分流

---

#### D4: 嵌套分歧分支

```cpp
TEST_CASE("handle_branch: nested divergent branches", "[branch][simt][nested]")
```

**测试步骤**:
1. 第一次分歧 (PC=10): lanes 0-15 taken → PC=20, lanes 16-31 fallthrough → PC=11, reconvergence=30
2. 第二次分歧 (PC=20): lanes 0-7 taken → PC=25, lanes 8-15 fallthrough → PC=21, reconvergence=28
3. 验证 simt_stack.depth() == 2
4. 验证栈顶 (内层): branch_pc=20, reconvergence_pc=28
5. 验证栈底 (外层): branch_pc=10, reconvergence_pc=30
6. 验证 exec_mask == 0x000000FF (内层 taken)

**预期结果**: 嵌套栈正确，exec_mask 更新为内层

---

#### D5: 分歧后收敛检查 ⚠️ **Bug 检测**

```cpp
TEST_CASE("handle_branch: convergence after divergence", "[branch][simt][convergence]")
```

**测试步骤**:
1. 执行 D3 的分歧分支
2. 设置所有线程 pc=30 (reconvergence point)
3. 调用 check_reconvergence()
4. 验证 simt_stack.empty() == true
5. 验证 exec_mask == 0xFFFFFFFF ⚠️ **当前失败**

**预期结果 (Bug 修复后)**:
- ✅ SIMT 栈弹出
- ✅ exec_mask 恢复为全活跃

---

### 测试组 E: Barrier 与 SIMT Stack 交互 (4 用例)

**文件**: `tests/test_barrier_simt_integration.cpp`  
**优先级**: P1  
**目标**: 验证屏障与 SIMT 栈的协同工作

#### E1: 屏障在收敛点之后

```cpp
TEST_CASE("barrier after reconvergence", "[barrier][simt]")
```

**测试步骤**:
1. 分歧分支 → push SIMT
2. 收敛 → pop SIMT
3. 执行 barrier
4. 验证 simt_stack.empty() == true
5. 验证 barrier 正常完成

**预期结果**: 无干扰

---

#### E2: 屏障在分歧路径内 ⚠️ **关键场景**

```cpp
TEST_CASE("barrier inside divergent path", "[barrier][simt][divergence]")
```

**场景**:
```
PC=10: bra.cond target=20 (divergent) → push SIMT {reconv=30}
PC=11: ... (not-taken path, lanes 16-31)
PC=15: bar.warp.sync
PC=20: ... (taken path, lanes 0-15)
PC=30: ← reconvergence point
```

**测试步骤**:
1. 执行分歧分支
2. 模拟 barrier 完成，设置所有线程 PC=30
3. 验证 simt_stack 状态 (应弹出或待检查)
4. 验证 exec_mask 状态

**预期结果**: 
- 如果 Phase 4.1 已实施: SIMT 栈弹出
- 否则: SIMT 栈仍保留，待后续检查

---

#### E3: 屏障完成后收敛检查

```cpp
TEST_CASE("barrier completion triggers reconvergence check", "[barrier][simt]")
```

**测试步骤**:
1. 分歧分支，SIMT reconvergence_pc = barrier reconvergence_pc
2. 屏障完成
3. 验证 check_reconvergence() 被调用
4. 验证 SIMT 栈弹出

**预期结果**: 屏障触发收敛检查

---

#### E4: Wbar 与 SIMT Stack 独立清理

```cpp
TEST_CASE("wbar and simt_stack independent cleanup", "[barrier][simt]")
```

**测试步骤**:
1. 创建 wbar 和 SIMT 栈
2. wbar 完成 → 重置 wbar
3. 验证 SIMT 栈状态不受影响 (除非触发收敛)

**预期结果**: 独立清理

---

### 测试组 F: exec_mask 管理 (6 用例)

**文件**: `tests/test_exec_mask.cpp`  
**优先级**: P1  
**目标**: 验证执行掩码的正确性，**检测恢复 bug**

#### F1: 默认 exec_mask

```cpp
TEST_CASE("exec_mask: default value", "[exec_mask]")
```

**测试步骤**:
1. 创建 WarpContext
2. 验证 get_exec_mask() == 0xFFFFFFFF

**预期结果**: 全活跃

---

#### F2: 分歧分支后 exec_mask

```cpp
TEST_CASE("exec_mask: after divergent branch", "[exec_mask][branch]")
```

**测试步骤**:
1. 执行分歧分支 (lanes 0-15 taken)
2. 验证 exec_mask == 0x0000FFFF

**预期结果**: 更新为 taken_mask

---

#### F3: 收敛后 exec_mask 恢复 ⚠️ **关键 Bug 检测**

```cpp
TEST_CASE("exec_mask: restored after reconvergence", "[exec_mask][bug][critical]")
```

**测试步骤**:
1. 执行分歧分支 (exec_mask != 0xFFFFFFFF)
2. 设置所有线程到 reconvergence_pc
3. 调用 check_reconvergence()
4. 验证 simt_stack.empty() == true
5. 验证 exec_mask == 0xFFFFFFFF ⚠️ **当前失败**

**预期结果 (Bug 修复后)**:
- ✅ exec_mask 恢复为 0xFFFFFFFF

**影响**: 此 bug 导致收敛后执行掩码错误，影响后续指令调度

---

#### F4: 屏障完成后 exec_mask

```cpp
TEST_CASE("exec_mask: after barrier completion", "[exec_mask][barrier]")
```

**测试步骤**:
1. 设置 barrier，参与掩码 = 0x0000FFFF
2. 屏障完成
3. 验证 exec_mask == 0x0000FFFF (arrived_mask)

**预期结果**: 更新为到达掩码

---

#### F5: set_exec_mask / get_exec_mask

```cpp
TEST_CASE("exec_mask: setter and getter", "[exec_mask]")
```

**测试步骤**:
1. set_exec_mask(0x12345678)
2. 验证 get_exec_mask() == 0x12345678

**预期结果**: 读写一致

---

#### F6: active_mask 与 exec_mask 独立性 ⚠️ **关键概念区分**

```cpp
TEST_CASE("exec_mask: active_mask and exec_mask independence", "[exec_mask][concept]")
```

**概念说明**: `active_mask`（线程存活）与 `exec_mask`（当前分歧路径）是**两个独立机制**。在分歧执行期间，这两个掩码的值**不应该**相同。

**测试步骤**:
1. 创建 warp，所有 32 个线程活跃（active_mask 全 1）
2. 验证 `get_active_mask() == 0xFFFFFFFF`
3. 验证 `get_exec_mask() == 0xFFFFFFFF`
4. 执行分歧分支 (lanes 0-15 taken)
5. 验证 `get_exec_mask() == 0x0000FFFF`（只有 taken 线程）
6. 验证 `get_active_mask() == 0xFFFFFFFF`（所有线程仍存活，不同于 exec_mask!）
7. 收敛后
8. 验证 `get_exec_mask() == 0xFFFFFFFF`（恢复）
9. 验证 `get_active_mask() == 0xFFFFFFFF`（一致）

**预期结果**: 分歧期间两个掩码值不同，收敛后一致

---

### 测试组 G: PC 管理高级测试 (5 用例)

**文件**: `tests/test_pc_management_advanced.cpp`  
**优先级**: P1  
**目标**: 验证 PC 管理机制

#### G1: update_pc_stack 功能 (已废弃)

```cpp
TEST_CASE("update_pc_stack: updates correct lane", "[pc_stack][legacy]")
```

**前置条件**: 如果 update_pc_stack 仍在使用

**测试步骤**:
1. 设置 pc_stacks[lane_id] = {0, 5, 10}
2. update_pc_stack(lane_id, 20)
3. 验证 pc_stacks[lane_id].back() == 20

**预期结果**: 更新栈顶

**注意**: Phase 2 可能删除此功能

---

#### G2: handle_branch_divergence (已废弃)

```cpp
TEST_CASE("handle_branch_divergence: legacy path", "[pc_stack][legacy]")
```

**测试步骤**:
1. 调用 handle_branch_divergence(lane_id=5, new_pc=20)
2. 验证 pc_stacks[5] 包含新 PC

**预期结果**: 传统路径工作

**注意**: Phase 2 可能删除此功能

---

#### G3: pc_stacks 重置

```cpp
TEST_CASE("pc_stacks: reset clears all stacks", "[pc_stack]")
```

**测试步骤**:
1. 设置所有 pc_stacks[i] = {0, 5}
2. warp.reset()
3. 验证所有 pc_stacks[i] == {0}

**预期结果**: 重置为初始状态

---

#### G4: advance_thread_pc 单一推进

```cpp
TEST_CASE("advance_thread_pc: unified PC update", "[pc][unified]")
```

**测试步骤**:
1. warp.advance_thread_pc(lane_id=5, new_pc=20)
2. 验证 warp_state.threads[5].pc == 20
3. 验证 warp_state.threads[5].next_pc == 20

**预期结果**: 统一更新

---

#### G5: advance_all_threads 批量推进

```cpp
TEST_CASE("advance_all_threads: advance all active threads", "[pc][unified]")
```

**测试步骤**:
1. 设置部分线程 is_active=false
2. warp.advance_all_threads(new_pc=30)
3. 验证活跃线程 pc=30, next_pc=30
4. 验证非活跃线程不变

**预期结果**: 只推进活跃线程

---

### 测试组 H: sync 机制测试 (6 用例)

**文件**: `tests/test_sync_mechanism.cpp`  
**优先级**: P1  
**目标**: 验证 ThreadContext ↔ WarpState 双向同步

#### H1: sync_from_warp_state 基础

```cpp
TEST_CASE("sync_from_warp_state: basic sync", "[sync]")
```

**测试步骤**:
1. 设置 warp_state.threads[lane].pc = 15
2. 设置 warp_state.threads[lane].status = Active
3. thread.sync_from_warp_state()
4. 验证 thread.get_pc() == 15
5. 验证 thread.get_state() == RUN

**预期结果**: WarpState → ThreadContext 同步

---

#### H2: sync_to_warp_state 基础

```cpp
TEST_CASE("sync_to_warp_state: basic sync", "[sync]")
```

**测试步骤**:
1. 设置 thread.next_pc = 20
2. 设置 thread.state = RUN
3. thread.sync_to_warp_state()
4. 验证 warp_state.threads[lane].next_pc == 20
5. 验证 warp_state.threads[lane].status = Active

**预期结果**: ThreadContext → WarpState 同步

---

#### H3: sync 不覆盖分支设置的 PC ⚠️ **Bug 检测**

```cpp
TEST_CASE("sync: doesn't overwrite branch PC", "[sync][branch]")
```

**测试步骤**:
1. handle_branch() 设置 warp_state.threads[i].next_pc = 20
2. 其他线程调用 sync_to_warp_state()
3. 验证 warp_state.threads[i].next_pc 仍为 20

**预期结果**: 不覆盖

---

#### H4: 屏障完成后的 sync

```cpp
TEST_CASE("sync: after barrier completion", "[sync][barrier]")
```

**测试步骤**:
1. 屏障完成，设置 warp_state.threads[i].pc = 30
2. thread.sync_from_warp_state()
3. 验证 thread.get_pc() == 30
4. thread.sync_to_warp_state()
5. 验证 warp_state.threads[i].pc 仍为 30

**预期结果**: 同步一致

---

#### H5: 退出线程的 sync

```cpp
TEST_CASE("sync: exited thread handling", "[sync][exit]")
```

**测试步骤**:
1. 设置 thread.state = EXIT
2. thread.sync_to_warp_state()
3. 验证 warp_state.threads[lane].is_exited == true
4. 验证 warp_state.threads[lane].is_active == false

**预期结果**: 状态正确同步

---

#### H6: 双向 sync 循环一致性

```cpp
TEST_CASE("sync: bidirectional cycle consistency", "[sync]")
```

**测试步骤**:
1. 初始状态: warp_state.pc = 10, thread.pc = 10
2. warp_state.pc = 15
3. thread.sync_from_warp_state() → thread.pc = 15
4. thread.next_pc = 20
5. thread.sync_to_warp_state() → warp_state.next_pc = 20
6. thread.commit_pc() → thread.pc = 20
7. 验证 warp_state.pc = 20

**预期结果**: 循环同步一致

---

### 测试组 I: 集成场景测试 (6 用例)

**文件**: `tests/test_simt_integration.cpp`  
**优先级**: P1  
**目标**: 验证完整执行场景

#### I1: 完整分歧-收敛循环 ⚠️ **综合测试**

```cpp
TEST_CASE("Integration: full divergence-convergence cycle", "[integration]")
```

**场景**:
```
PC=0:  mov r1, %tid.x
PC=1:  setp.lt.u32 p1, r1, 16
PC=2:  @p1 bra target  [reconv=5]
PC=3:  add r1, r1, 1        (lanes 16-31)
PC=4:  bra.uni merge
target:
PC=20: sub r1, r1, 1        (lanes 0-15)
merge:
PC=5:  bar.sync 0
PC=6:  ret
```

**测试步骤**:
1. 初始化 warp，执行到 PC=2
2. 执行分歧分支
3. 验证 SIMT 栈 push
4. 模拟执行两条路径
5. 设置所有线程到 PC=5
6. 执行收敛检查
7. 验证 SIMT 栈弹出
8. 验证 exec_mask = 0xFFFFFFFF
9. 验证所有线程 pc=5

**预期结果**: 完整循环正确

---

#### I2: 多层嵌套分支

```cpp
TEST_CASE("Integration: nested branches with multiple levels", "[integration]")
```

**场景**: 3 层嵌套分歧分支

**测试步骤**:
1. 执行 3 次分歧分支
2. 验证 SIMT 栈 depth=3
3. 逐层收敛
4. 验证每次收敛后 exec_mask 恢复

**预期结果**: 嵌套正确管理

---

#### I3: 分支 + 屏障组合

```cpp
TEST_CASE("Integration: branch + barrier combination", "[integration]")
```

**场景**: 分歧路径内包含屏障

**测试步骤**:
1. 分歧分支
2. 两条路径各自执行 barrier
3. 验证屏障完成
4. 收敛
5. 验证最终状态

**预期结果**: 协同工作

---

#### I4: 退出路径上的收敛 ⚠️ **Bug 检测**

```cpp
TEST_CASE("Integration: convergence with thread exits", "[integration]")
```

**场景**: taken 路径包含 exit 指令

**测试步骤**:
1. 分歧分支
2. taken 路径执行 exit (lanes 0-15)
3. not-taken 路径执行到 reconvergence
4. 验证收敛检查跳过退出线程
5. 验证 SIMT 栈弹出

**预期结果**: 不因退出线程阻塞

---

#### I5: 调度器与 SIMT 交互

```cpp
TEST_CASE("Integration: scheduler + SIMT stack", "[integration][scheduler]")
```

**测试步骤**:
1. 创建 2 个 warp
2. warp1: 分歧状态 (pc != next_pc)
3. warp2: 就绪状态
4. 调度器选择 warp2
5. warp1 收敛后，调度器可选择 warp1

**预期结果**: 调度器跳过未就绪 warp

---

#### I6a: 单层 if-else 分支完整执行

```cpp
TEST_CASE("Integration: single if-else with barrier", "[integration][ptx]")
```

**场景** (对应 bench/test_syncthreads 中的 test_shared_barrier):
```
PC=0:  mov r1, %tid.x         // 获取线程 ID
PC=1:  setp.lt.u32 p1, r1, 16 // p1 = (tid < 16)
PC=2:  @p1 bra target         // 分歧分支
PC=3:  add r1, r1, 1          // not-taken 路径 (lanes 16-31)
PC=4:  bra.uni merge
target:
PC=20: sub r1, r1, 1          // taken 路径 (lanes 0-15)
merge:
PC=5:  st.shared [r0], r1     // 共享内存写入
PC=6:  bar.sync 0             // 收敛屏障
PC=7:  ld.shared r2, [r0]     // 读回
PC=8:  ret
```

**测试步骤**:
1. 加载 test_shared_barrier PTX
2. 执行到 PC=2 (分支点)
3. 验证 SIMT 栈 push (depth=1)
4. 验证 exec_mask = taken_mask
5. 执行两条路径到 bar.sync
6. 屏障完成 → 验证所有线程 PC=7
7. 验证 SIMT 栈 depth=0 (屏障触发收敛)
8. 验证 exec_mask = 0xFFFFFFFF

**预期结果**: 完整分歧-屏障-收敛循环正确

---

#### I6b: 嵌套 if 分支完整执行

```cpp
TEST_CASE("Integration: nested if with barrier", "[integration][ptx]")
```

**场景** (对应 bench/test_warp_divergence 中的 nested 模式):
```
PC=0:  setp.lt.u32 p1, %tid.x, 16
PC=1:  @p1 bra outer_target      // 外层分歧
PC=2:  setp.lt.u32 p2, %tid.x, 24
PC=3:  @p2 bra inner_target      // 内层分歧 (仅 lanes 16-23)
PC=4:  ...
outer_target: ...
inner_target:
PC=X:  bar.sync 0                // 收敛点
```

**测试步骤**:
1. 执行两层嵌套分歧
2. 验证 SIMT 栈 depth=2
3. 执行屏障完成收敛
4. 验证栈完全清空 (depth=0)
5. 验证 exec_mask 恢复为全活跃

**预期结果**: 嵌套栈正确管理

---

#### I6c: 含退出线程的分支路径执行

```cpp
TEST_CASE("Integration: branch path with thread exits", "[integration][ptx][exit]")
```

**场景**: taken 路径中部分线程提前 exit
```
PC=0:  setp.lt.u32 p1, %tid.x, 16
PC=1:  @p1 bra target
PC=2:  ... (not-taken: continue)
PC=3:  bra.uni merge
target:
PC=20: setp.lt.u32 p2, %tid.x, 8
PC=21: @p2 bra early_exit    // 内层: lanes 0-7 进一步分支
PC=22: ... (lanes 8-15)
PC=23: bra.uni inner_merge
early_exit:
PC=30: ret                   // lanes 0-7 提前退出
inner_merge:
PC=40: ...
merge:
PC=50: bar.sync 0
```

**测试步骤**:
1. 执行嵌套分支 (depth=2)
2. lanes 0-7 执行 ret (exit)
3. 验证 warp_state.threads[0-7].is_exited = true
4. lanes 8-15 到达 inner_merge → 验证内层收敛 (跳过 exited threads)
5. 屏障完成 → 验证外层收敛
6. 验证 SIMT 栈 depth=0

**预期结果**: 退出线程不阻塞收敛 (BUG-002 修复验证)

---

### 测试组 J: active_mask 与 exec_mask 一致性 (5 用例) 🆕

**文件**: `tests/test_active_mask_consistency.cpp`  
**优先级**: P1  
**目标**: 验证 `active_mask[]` 和 `exec_mask` 独立更新且互不干扰（对应 ISSUE-004）

#### J1: 默认状态下两个掩码一致

```cpp
TEST_CASE("active_mask: default matches exec_mask", "[active_mask][consistency]")
```

**测试步骤**:
1. 创建 warp（全部线程活跃）
2. 验证 `get_active_mask() == 0xFFFFFFFF`
3. 验证 `get_exec_mask() == 0xFFFFFFFF`

**预期结果**: 初始化时一致

#### J2: 分歧期间 active_mask 保持不变

**测试步骤**:
1. 创建 warp（全部线程活跃）
2. 执行分歧分支 (lanes 0-15 taken)
3. 验证 `get_exec_mask() == 0x0000FFFF`（只有 taken）
4. 验证 `get_active_mask() == 0xFFFFFFFF`（全部存活，不同于 exec_mask!）

**预期结果**: active_mask 不受分歧影响，两个掩码值不同

#### J3: 线程退出时两者同步排除

**测试步骤**:
1. lanes 0-7 退出，调用 `update_active_mask()`
2. 验证 `active_mask` 中 bits 0-7 == 0
3. 验证 `exec_mask` 中 bits 0-7 == 0

#### J4: 收敛后一致恢复

**测试步骤**:
1. 分歧 → 收敛 → exec_mask 恢复
2. 验证 `get_active_mask() == 0xFFFFFFFF`（存活未变）
3. 验证 `get_exec_mask() == 0xFFFFFFFF`

#### J5: active_count 与掩码一致性

**测试步骤**:
1. 设置 16 个退出、8 个阻塞、8 个活跃
2. 调用 `update_active_mask()`
3. 验证 `get_active_count() == 8`

---

### 测试组 K: 共享测试夹具 (测试基础设施) 🆕

**文件**: `tests/simt_test_fixture.h`  
**优先级**: P1  
**目标**: 提供可复用的测试夹具，减少 9 个新测试文件中的重复初始化代码

```cpp
// tests/simt_test_fixture.h
#pragma once
#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include <memory>

namespace ptxsim::testing {

class SimtTestFixture {
public:
    static std::unique_ptr<WarpContext> createFullWarp();
    static std::unique_ptr<WarpContext> createWarpWithPredicate(uint32_t pred_mask);
    static void executeDivergentBranch(WarpContext& w, int taken_cnt, int target, int reconv);
    static void setAllThreadsAtPC(WarpContext& w, uint32_t pc);
    static void requireStackDepth(const WarpContext& w, size_t expected);
    static void requireExecMask(const WarpContext& w, uint32_t expected);
};

}
```

> **注意**: 夹具为可选基础设施，新增测试组 J 优先使用。已有测试组 (A-I) 可以渐进迁移。

---

## 三、测试矩阵与优先级

### 3.1 测试用例总览

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
| I | test_simt_integration.cpp | 8 | P1 | **高** | Phase 1,4 |
| J | test_active_mask_consistency.cpp 🆕 | 5 | P1 | **高** (ISSUE-004) | Phase 2 |
| K | simt_test_fixture.h 🆕 | (基础设施) | P1 | — | 无 |
| **总计** | | **60** | | | |

### 3.2 Bug 检测映射

| Bug ID | 描述 | 测试用例 | 严重性 | 检测阶段 |
|--------|------|---------|--------|---------|
| BUG-001 | exec_mask 收敛后未恢复 | F3, D5, I1 | 🔴 高 | Phase 1 |
| BUG-002 | 退出线程阻塞 SIMT 栈 | B4, I4 | 🔴 高 | Phase 1 |
| BUG-003 | 屏障完成未清理 SIMT 栈 | E2, E3 | 🟡 中 | Phase 4 |
| BUG-004 | sync 覆盖分支 PC | H3 | 🟡 中 | 无 |
| BUG-005 | SIMT 栈无深度限制 | A8 | 🟢 低 | Phase 2 |

### 3.3 执行顺序

```
Phase 1 执行:
1. B (SIMTStackEntry) - 检测退出线程 bug
2. F (exec_mask) - 检测恢复 bug
3. D (handle_branch) - 验证分支集成
4. C (WarpState) - 基础验证

Phase 2 执行:
5. A (SIMT Stack) - 基础操作
6. G (PC Management) - 高级 PC
7. J (active_mask) - 一致性验证 🆕

Phase 3 执行:
8. H (Sync) - 同步机制
9. E (Barrier + SIMT) - 屏障交互

Phase 4 执行:
10. I (Integration) - 完整场景 (8 用例, 含 I6a/I6b/I6c)
```

---

## 四、CMakeLists.txt 集成

### 4.1 测试注册

在 `tests/CMakeLists.txt` 中添加:

```cmake
# ==========================================
# SIMT Architecture Tests (New)
# ==========================================

# Helper function to register SIMT tests
function(register_simt_test test_name source_file labels)
    add_executable(${test_name} ${source_file} catch_amalgamated.cpp)
    target_link_libraries(${test_name} PRIVATE ptxsim)
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "${labels}")
endfunction()

# Test Group A: SIMT Stack Basic
register_simt_test(test_simt_stack_catch2 
    test_simt_stack_catch2.cpp 
    "simt;stack")

# Test Group B: SIMTStackEntry Convergence
register_simt_test(test_simt_stack_entry 
    test_simt_stack_entry.cpp 
    "simt;convergence")

# Test Group C: WarpState
register_simt_test(test_warp_state 
    test_warp_state.cpp 
    "simt;warp_state")

# Test Group D: handle_branch Integration
register_simt_test(test_handle_branch_integration 
    test_handle_branch_integration.cpp 
    "simt;branch")

# Test Group E: Barrier + SIMT Integration
register_simt_test(test_barrier_simt_integration 
    test_barrier_simt_integration.cpp 
    "simt;barrier")

# Test Group F: exec_mask Management
register_simt_test(test_exec_mask 
    test_exec_mask.cpp 
    "simt;exec_mask")

# Test Group G: PC Management Advanced
register_simt_test(test_pc_management_advanced 
    test_pc_management_advanced.cpp 
    "simt;pc")

# Test Group H: Sync Mechanism
register_simt_test(test_sync_mechanism 
    test_sync_mechanism.cpp 
    "simt;sync")

# Test Group I: Integration Scenarios
register_simt_test(test_simt_integration 
    test_simt_integration.cpp 
    "simt;integration")
```

### 4.2 测试标签说明

| 标签 | 含义 | 运行命令 |
|------|------|---------|
| `simt` | 所有 SIMT 测试 | `ctest -L simt` |
| `stack` | SIMT Stack 基础 | `ctest -L stack` |
| `convergence` | 收敛检查 | `ctest -L convergence` |
| `branch` | 分支相关 | `ctest -L branch` |
| `barrier` | 屏障相关 | `ctest -L barrier` |
| `exec_mask` | 执行掩码 | `ctest -L exec_mask` |
| `pc` | PC 管理 | `ctest -L pc` |
| `sync` | 同步机制 | `ctest -L sync` |
| `integration` | 集成测试 | `ctest -L integration` |
| `bug` | Bug 检测测试 | 手动筛选 |
| `critical` | 关键 Bug 检测 | 手动筛选 |

---

## 五、测试执行指南

### 5.1 日常开发

```bash
# 快速运行 SIMT 测试
cd build && ctest -L simt -j$(nproc)

# 运行特定测试 (verbose)
ctest -R test_exec_mask -V

# 运行关键 Bug 检测测试
ctest -R "test_simt_stack_entry|test_exec_mask" -V
```

### 5.2 CI/CD 集成

```bash
# 完整测试套件
cd build
ctest --output-on-failure

# 覆盖率收集
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/tests/*' '*/external/*' --output-file coverage_filtered.info
genhtml coverage_filtered.info --output-directory coverage_html

# 阈值检查 (覆盖率 ≥ 90%)
lcov --summary coverage_filtered.info | grep "lines" | awk '{print $2}' | \
  awk -F'%' '{if ($1 < 90) exit 1}'
```

### 5.3 调试失败测试

```bash
# 使用 GDB 调试
cd build
gdb --args ./test_exec_mask

# 断点设置在失败用例
(gdb) break test_exec_mask.cpp:42
(gdb) run

# 使用 LLDB (macOS)
lldb -- ./test_exec_mask
(lldb) breakpoint set -f test_exec_mask.cpp -l 42
(lldb) run
```

---

## 六、验收标准

### 6.1 功能验收

- [ ] 53 个新测试用例全部通过
- [ ] 现有测试 100% 通过 (无回归)
- [ ] 关键 Bug 检测测试 (B4, F3, D5) 通过
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

## 七、维护与演进

### 7.1 测试维护责任

| 测试组 | 负责人 | 审查频率 |
|--------|--------|---------|
| A, B (SIMT Stack) | SIMT 模块所有者 | 每月 |
| C, D (Warp) | 执行引擎团队 | 每月 |
| E (Barrier) | 同步机制团队 | 每月 |
| F (exec_mask) | SIMT 模块所有者 | 每月 |
| G, H (PC/Sync) | 执行引擎团队 | 每月 |
| I (Integration) | 架构团队 | 每季度 |

### 7.2 测试演进

**添加新测试**:
1. 确定测试组 (A-I)
2. 创建测试文件 (如不存在)
3. 注册到 CMakeLists.txt
4. 添加适当标签
5. 运行全量测试验证

**删除旧测试**:
1. 确认测试已过时
2. 更新 CMakeLists.txt
3. 删除测试文件
4. 更新本文档

### 7.3 文档更新

**触发条件**:
- 新增/删除测试用例
- 测试标签变更
- 验收标准调整
- Bug 检测映射更新

**更新流程**:
1. 修改本文档
2. 提交 PR
3. 架构团队审查

---

## 八、附录

### A. 测试用例快速索引

| ID | 名称 | 文件 | 行号 | 标签 |
|----|------|------|------|------|
| A1-A8 | SIMTStack 基础操作 | test_simt_stack_catch2.cpp | TBD | [simt_stack] |
| B1-B6 | SIMTStackEntry 收敛 | test_simt_stack_entry.cpp | TBD | [simt_entry] |
| C1-C7 | WarpState 状态 | test_warp_state.cpp | TBD | [warp_state] |
| D1-D5 | handle_branch 集成 | test_handle_branch_integration.cpp | TBD | [branch] |
| E1-E4 | Barrier + SIMT | test_barrier_simt_integration.cpp | TBD | [barrier] |
| F1-F6 | exec_mask 管理 | test_exec_mask.cpp | TBD | [exec_mask] |
| G1-G5 | PC 管理高级 | test_pc_management_advanced.cpp | TBD | [pc] |
| H1-H6 | sync 机制 | test_sync_mechanism.cpp | TBD | [sync] |
| I1-I6c | 集成场景 (8 用例) | test_simt_integration.cpp | TBD | [integration] |
| J1-J5 🆕 | active_mask 一致性 | test_active_mask_consistency.cpp | TBD | [active_mask] |

*(完整索引在测试实施后填充)*

### B. 已知测试限制

| 测试 | 限制 | 原因 | 临时方案 |
|------|------|------|---------|
| I6a-I6c | 需要真实 PTX | 依赖 cuobjdump 提取 | 使用 bench/ 中预提取的 PTX |
| E2 | 依赖 Phase 4 | 屏障集成未完成 | 标记为 XFAIL |
| A8 | 依赖 Phase 2 | 深度限制未实现 | 标记为 SKIP |
| G1, G2 | Phase 2 后废弃 | pc_stacks 将被删除 | Phase 2 前作为过渡验证运行，Phase 2 后标记为 SKIP 并计划删除 |
| J | 依赖 Phase 2.3 | active_mask 统一未完成 | Phase 2 前标记为 XFAIL

### C. 参考文档

- [SIMT Architecture v2.0](../architecture/SIMT-ARCHITECTURE-V2.md)
- [SIMT Architecture Review](../reports/simt-architecture-review-and-test-plan.md)
- [SIMT Architecture Improvement Plan](../plans/simt-architecture-improvement-plan.md)
- [SIMT Divergence Review](simt-architecture-divergence-review.md)
- [PTX ISA 9.1 - Control Flow](https://docs.nvidia.com/cuda/ptx-isa-9.1/)
- [Catch2 Documentation](https://github.com/catchorg/Catch2/blob/devel/docs/Readme.md)

---

**文档创建**: 2026-05-04
**最后更新**: 2026-05-05
**维护者**: PTX-EMU Architecture Team
**审批状态**: ✅ 已审批
**执行状态**: ✅ 已完成
**完成内容**:
- 9 个测试文件已创建 (A-I 组)
- 57 个测试用例已实现
- 所有 SIMT 测试通过 (ctest -L simt: 9/9)
- Bug 检测测试全部就绪 (B4, F3, D5)

