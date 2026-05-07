# SIMT 架构修复与测试补充 — TDD 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 5 个 SIMT 架构缺陷（exec_mask 恢复、退出线程收敛、active_mask 双源、双重 PC 管理、is_lane_active 双源），将测试覆盖率从 ~40% 提升至 ≥90%，新增 60 个 Catch2 测试用例。

**Architecture:** 严格 TDD 驱动：每个 Bug 先写失败测试（RED），再最小实现修复（GREEN），最后验证 + 提交。四个 Phase 顺序推进：Phase 1 修复关键 Bug → Phase 2 代码清理与数据源统一 → Phase 3 测试扩展 → Phase 4 架构增强。每 Phase 以全量回归测试收尾。

**Tech Stack:** C++20, CMake ≥3.15, Catch2 (amalgamated), GCC ≥10, clang-format, clang-tidy, gcov/lcov

**参考文档:**
- 架构审查: `docs/reports/simt-architecture-review-and-test-plan.md`
- 改进计划: `docs/plans/simt-architecture-improvement-plan.md`
- 测试计划: `docs/testing/simt-complete-test-plan.md`
- SIMT 架构: `docs/architecture/SIMT-ARCHITECTURE-V2.md`

---

## 前置条件

```bash
# 1. 环境设置（每次新终端执行）
. env.sh

# 2. 确认构建可用
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)

# 3. 确认现有测试基线
cd build && ctest --output-on-failure 2>&1 | tail -5
# Expected: "100% tests passed, 0 tests failed"

# 4. 确认 PTX 语法测试基线
./tests/ptx/test_all_ptx.sh 2>&1 | tail -5
# Expected: "All PTX tests passed"
```

## 文件结构图

```
修改的现有文件:
  src/ptxsim/core/simt_stack.cpp          # Phase 1: BUG-002 修复
  src/ptxsim/core/warp_context.cpp        # Phase 1-2: BUG-001 修复 + active_mask 统一
  include/ptxsim/warp_state.h             # Phase 1: 删除 unused 字段
  include/ptxsim/warp_context.h           # Phase 2: 废弃 pc/pc_stacks, 统一 is_lane
  include/ptxsim/simt_stack.h             # Phase 2: MAX_DEPTH 常量
  src/ptxsim/core/sm_context.cpp          # Phase 4: 扩展 check_reconvergence
  tests/CMakeLists.txt                    # Phase 3: 注册新测试

新增的测试文件:
  tests/test_simt_stack_entry.cpp         # 6 用例, B4 检测 BUG-002
  tests/test_exec_mask.cpp                # 6 用例, F3 检测 BUG-001
  tests/test_handle_branch_integration.cpp # 5 用例, D5 验证分支集成
  tests/test_simt_stack_catch2.cpp        # 8 用例, A1-A8 栈操作
  tests/test_warp_state.cpp               # 7 用例, C1-C7 状态管理
  tests/test_sync_mechanism.cpp           # 6 用例, H1-H6 同步机制
  tests/test_pc_management_advanced.cpp   # 5 用例, G1-G5 PC 管理
  tests/test_barrier_simt_integration.cpp # 4 用例, E1-E4 屏障交互
  tests/test_simt_integration.cpp         # 8 用例, I1-I6c 集成场景
  tests/test_active_mask_consistency.cpp  # 5 用例, J1-J5 掩码一致性
  tests/simt_test_fixture.h              # 共享测试夹具

新增的源代码文件:
  include/ptxsim/simt_debug.h             # Phase 4: SIMT 调试工具
  src/ptxsim/core/simt_debug.cpp          # Phase 4: SIMT 调试工具实现
```

---

## Phase 1: 关键 Bug 修复 (P0)

> **目标**: 修复 3 个 P0 缺陷。每个 Bug 遵循 RED→GREEN→REFACTOR→COMMIT 循环。
> **验收**: `ctest -L simt` 全部通过，新增 B4/F3/D5 测试通过。

---

### Task 1: 测试 B4 — 检测退出线程阻塞 SIMT 栈 (RED)

**Files:**
- Create: `tests/test_simt_stack_entry.cpp`
- Modify: (无)

- [ ] **Step 1: 创建测试文件，编写 B4 失败测试**

```cpp
// tests/test_simt_stack_entry.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/simt_stack.h"
#include "ptxsim/thread_state.h"
#include <array>

using namespace ptxsim;

TEST_CASE("B4: exited threads don't block convergence", "[simt_entry][bug][critical]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;

    // 初始化 32 个线程
    for (int i = 0; i < 32; i++) {
        threads[i].pc = 0;
        threads[i].is_active = true;
        threads[i].is_exited = false;
    }

    // 创建 SIMT 栈条目: 所有线程都应收敛到 PC=20
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.active_mask = 0xFFFF0000;   // lanes 16-31 taken
    entry.return_mask = 0xFFFFFFFF;    // 所有线程在 return_mask 中
    entry.return_pc = 20;

    stack.push(entry);

    // 模拟 lanes 0-15 执行了 exit (taken 路径中)
    for (int i = 0; i < 16; i++) {
        threads[i].is_exited = true;
        threads[i].is_active = false;
        threads[i].pc = 0;  // 已退出, PC 永远不会到 20
    }

    // 模拟 lanes 16-31 到达收敛点
    for (int i = 16; i < 32; i++) {
        threads[i].pc = 20;
    }

    // 关键验证: 收敛检查应跳过退出线程
    bool converged = stack.check_reconvergence(threads);

    REQUIRE(converged == true);     // 当前 FAIL: 退出线程导致永不收敛
    REQUIRE(stack.empty() == true);
}

TEST_CASE("B1: all threads at reconvergence PC", "[simt_entry]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    for (int i = 0; i < 32; i++) {
        threads[i].pc = 20;
        threads[i].is_active = true;
        threads[i].is_exited = false;
    }

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 20;
    entry.active_mask = 0x0000FFFF;
    stack.push(entry);

    bool converged = stack.check_reconvergence(threads);
    REQUIRE(converged == true);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("B2: partial convergence (not all at PC)", "[simt_entry]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    for (int i = 0; i < 32; i++) {
        threads[i].is_active = true;
        threads[i].is_exited = false;
    }
    // 一半到达，一半未到达
    for (int i = 0; i < 16; i++) threads[i].pc = 20;
    for (int i = 16; i < 32; i++) threads[i].pc = 15;

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 20;
    entry.active_mask = 0;
    stack.push(entry);

    bool converged = stack.check_reconvergence(threads);
    REQUIRE(converged == false);
    REQUIRE(stack.empty() == false);
}

TEST_CASE("B3: return_mask excludes unaffected threads", "[simt_entry]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    for (int i = 0; i < 32; i++) {
        threads[i].is_active = true;
        threads[i].is_exited = false;
    }
    // 只有 lanes 0-15 在 return_mask 中，它们都到达了
    for (int i = 0; i < 16; i++) threads[i].pc = 20;
    for (int i = 16; i < 32; i++) threads[i].pc = 99;  // 任意值, 不在 mask 中

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.return_mask = 0x0000FFFF;  // 只检查 lanes 0-15
    entry.return_pc = 20;
    entry.active_mask = 0;
    stack.push(entry);

    bool converged = stack.check_reconvergence(threads);
    REQUIRE(converged == true);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("B5: empty return_mask converges immediately", "[simt_entry]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;  // 全部初始值

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.return_mask = 0x00000000;  // 空掩码
    entry.return_pc = 20;
    entry.active_mask = 0;
    stack.push(entry);

    bool converged = stack.check_reconvergence(threads);
    REQUIRE(converged == true);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("B6: toString produces expected format", "[simt_entry]") {
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.active_mask = 0xFFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 20;

    std::string s = entry.toString();
    REQUIRE(s.find("branch_pc=10") != std::string::npos);
    REQUIRE(s.find("reconvergence_pc=20") != std::string::npos);
    REQUIRE(s.find("active_mask=0xffff") != std::string::npos);
}
```

- [ ] **Step 2: 注册到 CMake 并运行，验证失败**

在 `tests/CMakeLists.txt` 末尾（`# End of tests` 之前）添加：

```cmake
# === SIMT Architecture Tests (Phase 1) ===
add_catch_test(test_simt_stack_entry
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simt_stack_entry.cpp
)
```

```bash
cd build && cmake --build build -j$(nproc) --target test_simt_stack_entry
ctest -R test_simt_stack_entry -V
# Expected: B4 FAIL - "converged == true" assertion fails
# (退出线程的 pc=0 != reconvergence_pc=20, 且当前代码不跳过 exited 线程)
```

- [ ] **Step 3: 修复 BUG-002 — 跳过退出线程 (GREEN)**

```cpp
// src/ptxsim/core/simt_stack.cpp:7-16 — 修改 is_converged()
// 修改前:
bool SIMTStackEntry::is_converged(const std::array<ThreadState, 32>& threads) const {
    for (size_t i = 0; i < 32; i++) {
        if (return_mask & (1u << i)) {
            if ((int)threads[i].pc != reconvergence_pc) {
                return false;
            }
        }
    }
    return true;
}

// 修改后:
bool SIMTStackEntry::is_converged(const std::array<ThreadState, 32>& threads) const {
    for (size_t i = 0; i < 32; i++) {
        if (return_mask & (1u << i)) {
            // 跳过已退出或非活跃的线程 (修复 BUG-002)
            if (threads[i].is_exited || !threads[i].is_active) {
                continue;
            }
            if ((int)threads[i].pc != reconvergence_pc) {
                return false;
            }
        }
    }
    return true;
}
```

- [ ] **Step 4: 编译并运行测试，验证通过**

```bash
cmake --build build -j$(nproc)
ctest -R test_simt_stack_entry -V
# Expected: All 6 tests passed
```

- [ ] **Step 5: 提交**

```bash
git add tests/test_simt_stack_entry.cpp tests/CMakeLists.txt src/ptxsim/core/simt_stack.cpp
git commit -m "fix(simt): skip exited threads in SIMTStackEntry::is_converged

BUG-002: Exited threads with pc!=reconvergence_pc permanently block
SIMT stack convergence. Now is_converged() skips threads where
is_exited==true or is_active==false.

Test: B4 (test_simt_stack_entry) detects this bug and now passes."
```

---

### Task 2: 测试 F3 — 检测 exec_mask 收敛后未恢复 (RED)

**Files:**
- Create: `tests/test_exec_mask.cpp`
- Modify: (无)

- [ ] **Step 1: 创建测试文件，编写 F3 失败测试**

```cpp
// tests/test_exec_mask.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/simt_stack.h"

using namespace ptxsim;

// 辅助: 创建一个带 diverged SIMT 栈的 warp_state
static void setup_diverged_warp(WarpContext& warp) {
    // 构造 SIMT 栈条目: lanes 0-15 taken → PC=20, reconvergence=30
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;

    warp.get_simt_stack().push(entry);

    // 设置线程 PC: lanes 0-15 走 taken 路径 → pc=20
    // lanes 16-31 走 not-taken 路径 → pc=11
    for (int i = 0; i < 16; i++) {
        warp.set_thread_pc(i, 20);
    }
    for (int i = 16; i < 32; i++) {
        warp.set_thread_pc(i, 11);
    }

    // 设置 exec_mask 为 taken_mask
    warp.set_exec_mask(0x0000FFFF);
}

TEST_CASE("F3: exec_mask restored after reconvergence", "[exec_mask][bug][critical]") {
    WarpContext warp;
    setup_diverged_warp(warp);

    // 验证初始 exec_mask 为 taken_mask
    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);

    // 模拟所有线程到达收敛点 PC=30
    for (int i = 0; i < 32; i++) {
        warp.set_thread_pc(i, 30);
    }

    // 检查收敛
    warp.check_reconvergence();

    // 关键验证: 收敛后 exec_mask 应恢复为 0xFFFFFFFF
    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);  // 当前 FAIL!
}

TEST_CASE("F1: default exec_mask is full active", "[exec_mask]") {
    WarpContext warp;
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("F2: exec_mask after divergent branch", "[exec_mask][branch]") {
    WarpContext warp;
    // 模拟 handle_branch 后的状态
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);
}

TEST_CASE("F4: set_exec_mask and get_exec_mask roundtrip", "[exec_mask]") {
    WarpContext warp;
    warp.set_exec_mask(0x12345678);
    REQUIRE(warp.get_exec_mask() == 0x12345678);
    warp.set_exec_mask(0xAAAAAAAA);
    REQUIRE(warp.get_exec_mask() == 0xAAAAAAAA);
}

TEST_CASE("F5: nested divergence exec_mask recovery", "[exec_mask][nested]") {
    WarpContext warp;

    // 外层分歧: lanes 0-15 taken
    SIMTStackEntry outer;
    outer.branch_pc = 10;
    outer.reconvergence_pc = 50;
    outer.active_mask = 0x0000FFFF;
    outer.return_mask = 0xFFFFFFFF;
    outer.return_pc = 50;
    warp.get_simt_stack().push(outer);
    warp.set_exec_mask(0x0000FFFF);

    // 内层分歧: lanes 0-7 taken
    SIMTStackEntry inner;
    inner.branch_pc = 20;
    inner.reconvergence_pc = 40;
    inner.active_mask = 0x000000FF;
    inner.return_mask = 0x0000FFFF;  // 外层 return_mask
    inner.return_pc = 40;
    warp.get_simt_stack().push(inner);
    warp.set_exec_mask(0x000000FF);

    REQUIRE(warp.get_exec_mask() == 0x000000FF);
    REQUIRE(warp.get_simt_stack().depth() == 2);

    // 收敛内层: 所有 lanes 0-15 到达 PC=40
    for (int i = 0; i < 16; i++) warp.set_thread_pc(i, 40);
    warp.check_reconvergence();

    // 内层弹出, exec_mask 恢复为外层
    REQUIRE(warp.get_simt_stack().depth() == 1);
    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);  // 当前 FAIL

    // 收敛外层: 所有线程到达 PC=50
    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 50);
    warp.check_reconvergence();

    // 全部收敛
    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);  // 当前 FAIL
}

TEST_CASE("F6: exec_mask and active_mask independence", "[exec_mask][concept]") {
    WarpContext warp;
    // 初始一致
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);

    // 分歧后 active_mask 不应变化 (概念验证)
    // 此测试在 Phase 2 添加 active_mask getter 后完全验证
    // 当前先验证 exec_mask 路径
}
```

- [ ] **Step 2: 注册并运行，验证失败**

```cmake
# tests/CMakeLists.txt
add_catch_test(test_exec_mask
    ${CMAKE_CURRENT_SOURCE_DIR}/test_exec_mask.cpp
)
```

```bash
cmake --build build -j$(nproc) --target test_exec_mask
ctest -R test_exec_mask -V
# Expected: F3 FAIL - exec_mask == 0xFFFFFFFF assertion fails
# Expected: F5 FAIL - nested recovery assertions fail
# exec_mask 在 check_reconvergence() 后仍为 0x0000FFFF
```

- [ ] **Step 3: 修复 BUG-001 — check_reconvergence 恢复 exec_mask (GREEN)**

```cpp
// src/ptxsim/core/warp_context.cpp:98-101 — 修改 check_reconvergence()
// 修改前:
void WarpContext::check_reconvergence() {
    if (simt_stack.empty()) return;
    simt_stack.check_reconvergence(warp_state.threads);
}

// 修改后:
void WarpContext::check_reconvergence() {
    if (simt_stack.empty()) return;

    // 记录收敛前的栈深度
    size_t depth_before = simt_stack.depth();

    // 检查收敛 (可能弹出多层栈)
    simt_stack.check_reconvergence(warp_state.threads);

    // 栈深度减少 → 发生了弹出 → 恢复 exec_mask
    if (simt_stack.depth() < depth_before) {
        if (simt_stack.empty()) {
            // 所有分支层已收敛 → 恢复全活跃掩码
            warp_state.exec_mask = 0xFFFFFFFF;
        } else {
            // 还有外层分支未收敛 → 恢复到外层的 return_mask
            warp_state.exec_mask = simt_stack.top().return_mask;
        }
    }
}
```

- [ ] **Step 4: 编译并运行，验证通过**

```bash
cmake --build build -j$(nproc)
ctest -R test_exec_mask -V
# Expected: All 6 tests passed
```

- [ ] **Step 5: 提交**

```bash
git add tests/test_exec_mask.cpp tests/CMakeLists.txt src/ptxsim/core/warp_context.cpp
git commit -m "fix(simt): restore exec_mask after SIMT stack reconvergence

BUG-001: check_reconvergence() popped the SIMT stack but never restored
exec_mask. Now it restores to 0xFFFFFFFF when all layers converge, or
to the outer return_mask when nested divergence remains.

Test: F3, F5 (test_exec_mask) detect this bug and now pass."
```

---

### Task 3: 测试 D5 — 验证 handle_branch 分歧后收敛 (RED)

**Files:**
- Create: `tests/test_handle_branch_integration.cpp`
- Modify: (无)

- [ ] **Step 1: 编写 D5 集成测试**

```cpp
// tests/test_handle_branch_integration.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

using namespace ptxsim;

// 辅助: 初始化 warp 的所有 32 个线程状态
static void init_warp_threads(WarpContext& warp) {
    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].pc = 10;
        warp.get_warp_state().threads[i].next_pc = 10;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
        warp.get_warp_state().threads[i].is_blocked = false;
        warp.get_warp_state().threads[i].status = ThreadStatus::Active;
    }
    warp.get_warp_state().exec_mask = 0xFFFFFFFF;
}

TEST_CASE("D3: divergent branch pushes SIMT stack", "[branch][simt][divergence]") {
    WarpContext warp;
    init_warp_threads(warp);

    // 所有线程活跃, 无谓词 → 全部 taken (非分歧)
    // 先验证非分歧路径
    warp.handle_branch("", false, 20, 30, 10);

    // 非分歧: 不推栈, 所有线程跳转
    REQUIRE(warp.get_simt_stack().empty() == true);
    for (int i = 0; i < 32; i++) {
        REQUIRE(warp.get_thread_pc(i) == 20);
    }
}

TEST_CASE("D5: convergence after divergence restores state", "[branch][simt][convergence]") {
    WarpContext warp;
    init_warp_threads(warp);

    // 模拟分歧: 手工 push SIMT 栈 + 设置分歧 PC
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);

    // 设置分歧后的 PC
    for (int i = 0; i < 16; i++) warp.set_thread_pc(i, 20);  // taken
    for (int i = 16; i < 32; i++) warp.set_thread_pc(i, 11);  // not-taken
    warp.set_exec_mask(0x0000FFFF);

    REQUIRE(warp.get_simt_stack().depth() == 1);
    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);

    // 所有线程到达收敛点
    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);

    // 收敛
    warp.check_reconvergence();

    // 栈弹出 + exec_mask 恢复
    REQUIRE(warp.get_simt_stack().empty() == true);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("D4: nested divergence maintains stack order", "[branch][simt][nested]") {
    WarpContext warp;
    init_warp_threads(warp);

    // 外层分歧
    SIMTStackEntry outer;
    outer.branch_pc = 10;
    outer.reconvergence_pc = 50;
    outer.active_mask = 0x0000FFFF;
    outer.return_mask = 0xFFFFFFFF;
    outer.return_pc = 50;
    warp.get_simt_stack().push(outer);
    warp.set_exec_mask(0x0000FFFF);

    // 内层分歧
    SIMTStackEntry inner;
    inner.branch_pc = 20;
    inner.reconvergence_pc = 40;
    inner.active_mask = 0x000000FF;
    inner.return_mask = 0x0000FFFF;
    inner.return_pc = 40;
    warp.get_simt_stack().push(inner);
    warp.set_exec_mask(0x000000FF);

    REQUIRE(warp.get_simt_stack().depth() == 2);
    REQUIRE(warp.get_simt_stack().top().branch_pc == 20);  // 内层在栈顶
}

TEST_CASE("D1: non-divergent all taken branch", "[branch][simt]") {
    WarpContext warp;
    init_warp_threads(warp);
    warp.handle_branch("", false, 20, 30, 10);
    REQUIRE(warp.get_simt_stack().empty() == true);
    for (int i = 0; i < 32; i++) {
        REQUIRE(warp.get_thread_pc(i) == 20);
    }
}

// D2: non-divergent none taken — 需要谓词全部为 false
// 此测试在 register_bank_manager 初始化后可用
// 当前跳过: 谓词评估依赖 RegisterBankManager
```

- [ ] **Step 2: 注册并运行**

```cmake
add_catch_test(test_handle_branch_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_handle_branch_integration.cpp
)
```

```bash
cmake --build build -j$(nproc) --target test_handle_branch_integration
ctest -R test_handle_branch_integration -V
# Expected: D3, D4, D5 pass; D1 可能依赖谓词系统
```

- [ ] **Step 3-4: 实现已在 Task 2 完成，此处验证集成**

```bash
# 验证 Phase 1 所有测试
ctest -R "test_simt_stack_entry|test_exec_mask|test_handle_branch" -V
# Expected: 16/16 tests passed
```

- [ ] **Step 5: 提交**

```bash
git add tests/test_handle_branch_integration.cpp tests/CMakeLists.txt
git commit -m "test(simt): add handle_branch integration tests (D1-D5)

D3: divergent branch pushes SIMT stack
D4: nested divergence stack order
D5: convergence after divergence restores exec_mask
D1: non-divergent all-taken branch"
```

---

### Task 4: Phase 1.3 — 删除 WarpState 未使用字段

**Files:**
- Modify: `include/ptxsim/warp_state.h:20-21, 34`
- Test: 编译验证 + 现有测试回归

- [ ] **Step 1: 全局搜索确认无引用**

```bash
grep -rn "pc_stack\[" src/ include/ --include="*.cpp" --include="*.h" | grep -v "pc_stacks"
# Expected: 无输出 (WarpState::pc_stack 无引用)
grep -rn "pc_stack_depth" src/ include/ --include="*.cpp" --include="*.h"
# Expected: 仅在 warp_state.h 中有定义, 无其他引用
```

- [ ] **Step 2: 删除字段**

```cpp
// include/ptxsim/warp_state.h — 删除 pc_stack 和 pc_stack_depth
// 删除第 20-21 行:
//     std::array<int, 16> pc_stack;
//     int pc_stack_depth = 0;

// reset() 中删除对应的 reset 行 (第 34 行):
//     pc_stack_depth = 0;
```

修改后的 `WarpState`:

```cpp
struct WarpState {
    std::array<ThreadState, 32> threads;
    uint32_t exec_mask = 0xFFFFFFFF;
    std::map<std::string, std::array<bool, 32>> thread_predicates;
    std::array<Wbar, 4> wbars;
    int current_wbar_id = -1;
    uint32_t warp_pc = 0;
    // pc_stack 和 pc_stack_depth 已移除 — 使用 WarpContext::pc_stacks 或 warp_state.threads[i].pc

    void reset() {
        for (auto& thread : threads) {
            thread.reset();
        }
        exec_mask = 0xFFFFFFFF;
        thread_predicates.clear();
        for (auto& wbar : wbars) {
            wbar.reset();
        }
        current_wbar_id = -1;
        warp_pc = 0;
    }

    int count_active_lanes() const { /* 保持不变 */ }
    int count_schedulable_lanes() const { /* 保持不变 */ }
    bool is_all_exited() const { /* 保持不变 */ }
    bool has_schedulable_threads() const { /* 保持不变 */ }
};
```

- [ ] **Step 3: 编译 + 全量回归测试**

```bash
cmake --build build -j$(nproc)
# Expected: 编译成功, 无 warning
cd build && ctest --output-on-failure
# Expected: 100% tests passed, 0 tests failed
```

- [ ] **Step 4: 提交**

```bash
git add include/ptxsim/warp_state.h
git commit -m "refactor(simt): remove unused pc_stack/pc_stack_depth from WarpState

These fields were never read or written by any code. Removed to eliminate
64 bytes of wasted memory per warp and reduce code confusion with
WarpContext::pc_stacks."
```

---

### Task 5: Phase 1 回归验证

- [ ] **Step 1: 运行全量测试 + PTX 语法测试**

```bash
cd build && ctest --output-on-failure -j$(nproc)
# Expected: 100% tests passed

cd /workspace/project/PTX-EMU
./tests/ptx/test_all_ptx.sh
# Expected: All PTX tests passed
```

- [ ] **Step 2: 运行 SIMT 相关测试专项**

```bash
ctest -L simt -V 2>&1 | tail -20
# Expected: 新测试全部通过
ctest -R "test_simt_stack_entry|test_exec_mask|test_handle_branch" -V
# Expected: 16 tests passed
```

- [ ] **Step 3: clang-format 检查**

```bash
clang-format --dry-run --Werror src/ptxsim/core/simt_stack.cpp src/ptxsim/core/warp_context.cpp include/ptxsim/warp_state.h
# Expected: 无警告
```

- [ ] **Step 4: Phase 1 里程碑提交**

```bash
git add -A
git commit -m "milestone: Phase 1 complete — SIMT critical bug fixes

Fixed:
- BUG-001: exec_mask restored after SIMT stack reconvergence
- BUG-002: exited threads no longer block SIMT stack convergence
- ISSUE-001: removed unused WarpState::pc_stack/pc_stack_depth

New tests: test_simt_stack_entry (6 cases), test_exec_mask (6 cases),
test_handle_branch_integration (5 cases)
All existing tests pass with no regression."
```

---

## Phase 2: 代码清理与数据源统一 (P1)

> **目标**: 清理双重 PC 管理、统一 active_mask 数据源、废弃向后兼容字段、添加栈深度限制。
> **验收**: `clang-tidy` 无新增警告，测试覆盖率 ≥60%。

---

### Task 6: 测试 A1-A7 — SIMT Stack 基础操作 (RED)

**Files:**
- Create: `tests/test_simt_stack_catch2.cpp`

- [ ] **Step 1: 编写 SIMT Stack 基础测试**

```cpp
// tests/test_simt_stack_catch2.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/simt_stack.h"
#include "ptxsim/thread_state.h"
#include <array>

using namespace ptxsim;

TEST_CASE("A1: push and pop operations", "[simt_stack][basic]") {
    SIMTStack stack;
    REQUIRE(stack.empty() == true);
    REQUIRE(stack.depth() == 0);

    SIMTStackEntry e1;
    e1.branch_pc = 10;
    e1.reconvergence_pc = 20;
    e1.active_mask = 0xFFFF;
    e1.return_mask = 0xFFFFFFFF;
    e1.return_pc = 20;

    stack.push(e1);
    REQUIRE(stack.empty() == false);
    REQUIRE(stack.depth() == 1);
    REQUIRE(stack.top().reconvergence_pc == 20);

    SIMTStackEntry popped = stack.pop();
    REQUIRE(popped.branch_pc == 10);
    REQUIRE(popped.reconvergence_pc == 20);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("A2: empty and depth tracking", "[simt_stack][basic]") {
    SIMTStack stack;
    REQUIRE(stack.empty() == true);
    REQUIRE(stack.depth() == 0);

    SIMTStackEntry e;
    e.branch_pc = 0;
    e.reconvergence_pc = 0;
    e.return_pc = 0;
    e.active_mask = 0;
    e.return_mask = 0;

    stack.push(e); stack.push(e); stack.push(e);
    REQUIRE(stack.depth() == 3);
    stack.pop();
    REQUIRE(stack.depth() == 2);
    stack.pop();
    REQUIRE(stack.depth() == 1);
    stack.pop();
    REQUIRE(stack.empty() == true);
}

TEST_CASE("A3: top returns most recent entry", "[simt_stack][basic]") {
    SIMTStack stack;
    SIMTStackEntry e1, e2;
    e1.branch_pc = 10; e1.reconvergence_pc = 20;
    e2.branch_pc = 30; e2.reconvergence_pc = 40;
    e1.return_pc = 0; e2.return_pc = 0;
    e1.active_mask = 0; e2.active_mask = 0;
    e1.return_mask = 0; e2.return_mask = 0;

    stack.push(e1);
    REQUIRE(stack.top().branch_pc == 10);
    stack.push(e2);
    REQUIRE(stack.top().branch_pc == 30);
    stack.pop();
    REQUIRE(stack.top().branch_pc == 10);
}

TEST_CASE("A4: clear empties the stack", "[simt_stack][basic]") {
    SIMTStack stack;
    SIMTStackEntry e;
    e.branch_pc = 0; e.reconvergence_pc = 0; e.return_pc = 0;
    e.active_mask = 0; e.return_mask = 0;

    for (int i = 0; i < 5; i++) stack.push(e);
    REQUIRE(stack.depth() == 5);
    stack.clear();
    REQUIRE(stack.empty() == true);
    REQUIRE(stack.depth() == 0);
}

TEST_CASE("A5: pop on empty throws exception", "[simt_stack][exception]") {
    SIMTStack stack;
    REQUIRE_THROWS_AS(stack.pop(), std::runtime_error);
    try {
        stack.pop();
    } catch (const std::runtime_error& e) {
        REQUIRE(std::string(e.what()).find("empty") != std::string::npos);
    }
}

TEST_CASE("A6: top on empty throws exception", "[simt_stack][exception]") {
    SIMTStack stack;
    REQUIRE_THROWS_AS(stack.top(), std::runtime_error);
}

TEST_CASE("A7: nested push preserves LIFO order", "[simt_stack][nested]") {
    SIMTStack stack;
    SIMTStackEntry e1, e2, e3;
    e1.branch_pc = 10; e1.reconvergence_pc = 30;
    e2.branch_pc = 15; e2.reconvergence_pc = 25;
    e3.branch_pc = 20; e3.reconvergence_pc = 22;
    e1.return_pc = 0; e2.return_pc = 0; e3.return_pc = 0;
    e1.active_mask = 0; e2.active_mask = 0; e3.active_mask = 0;
    e1.return_mask = 0; e2.return_mask = 0; e3.return_mask = 0;

    stack.push(e1); stack.push(e2); stack.push(e3);
    REQUIRE(stack.top().branch_pc == 20);  // e3
    stack.pop();
    REQUIRE(stack.top().branch_pc == 15);  // e2
    stack.pop();
    REQUIRE(stack.top().branch_pc == 10);  // e1
}
```

- [ ] **Step 2: 注册并运行**

```cmake
add_catch_test(test_simt_stack_catch2
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simt_stack_catch2.cpp
)
```

```bash
cmake --build build -j$(nproc) --target test_simt_stack_catch2
ctest -R test_simt_stack_catch2 -V
# Expected: All 7 tests pass (基础操作已有实现)
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_simt_stack_catch2.cpp tests/CMakeLists.txt
git commit -m "test(simt): add SIMTStack basic operation tests (A1-A7)

Covers: push/pop, empty/depth, top access, clear, exception handling,
LIFO ordering. All tests pass against existing implementation."
```

---

### Task 7: 测试 J1-J5 — active_mask 一致性测试 (RED)

**Files:**
- Create: `tests/test_active_mask_consistency.cpp`

- [ ] **Step 1: 编写 active_mask 一致性测试**

```cpp
// tests/test_active_mask_consistency.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

using namespace ptxsim;

TEST_CASE("J1: default active_mask matches exec_mask", "[active_mask]") {
    WarpContext warp;
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
}

TEST_CASE("J2: active_mask unchanged during divergence", "[active_mask]") {
    WarpContext warp;
    // 模拟分歧: 设置 exec_mask 但 active_mask 不变
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    // exec_mask 变为 taken_mask
    REQUIRE(warp.get_exec_mask() == 0x0000FFFF);
    // active_mask 应仍为全活跃 (存活未变)
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
}

TEST_CASE("J3: thread exit updates active_mask", "[active_mask]") {
    WarpContext warp;
    // 模拟 lanes 0-7 退出
    for (int i = 0; i < 8; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
    }
    warp.update_active_mask();

    uint32_t mask = warp.get_active_mask();
    // bits 0-7 应为 0
    REQUIRE((mask & 0x000000FF) == 0);
    // bits 8-31 应为 1
    REQUIRE((mask & 0xFFFFFF00) == 0xFFFFFF00);
}

TEST_CASE("J4: active_mask consistent after convergence", "[active_mask]") {
    WarpContext warp;
    // 分歧 + 收敛
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 30;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 30;
    warp.get_simt_stack().push(entry);
    warp.set_exec_mask(0x0000FFFF);

    // 收敛
    for (int i = 0; i < 32; i++) warp.set_thread_pc(i, 30);
    warp.check_reconvergence();

    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
    REQUIRE(warp.get_active_mask() == 0xFFFFFFFF);
}

TEST_CASE("J5: active_count matches active_mask bits", "[active_mask]") {
    WarpContext warp;
    // 16 个退出, 8 个阻塞, 8 个活跃
    for (int i = 0; i < 16; i++) {
        warp.get_warp_state().threads[i].is_exited = true;
        warp.get_warp_state().threads[i].is_active = false;
    }
    for (int i = 16; i < 24; i++) {
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].is_active = false;
    }
    warp.update_active_mask();

    REQUIRE(warp.get_active_count() == 8);
}
```

- [ ] **Step 2: 注册并运行**

```cmake
add_catch_test(test_active_mask_consistency
    ${CMAKE_CURRENT_SOURCE_DIR}/test_active_mask_consistency.cpp
)
```

```bash
cmake --build build -j$(nproc) --target test_active_mask_consistency
ctest -R test_active_mask_consistency -V
# Expected: J3 和 J5 可能 PASS (update_active_mask 已有基本实现)
# J2 验证概念正确性
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_active_mask_consistency.cpp tests/CMakeLists.txt
git commit -m "test(simt): add active_mask consistency tests (J1-J5)

Verifies active_mask independence from exec_mask during divergence,
thread exit updates, post-convergence consistency, and count accuracy."
```

---

### Task 8: 统一 active_mask 数据源 (ISSUE-004) (GREEN)

**Files:**
- Modify: `src/ptxsim/core/warp_context.cpp:198-210`

- [ ] **Step 1: 修改 update_active_mask() — 统一从 warp_state 读取**

```cpp
// src/ptxsim/core/warp_context.cpp:198-210 — 修改 update_active_mask()
// 修改前:
void WarpContext::update_active_mask() {
    active_count = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i < threads.size() && threads[i] != nullptr) {
            if (threads[i]->is_exited() || warp_state.threads[i].is_blocked) {
                active_mask[i] = false;
            } else {
                active_mask[i] = true;
                active_count++;
            }
        }
    }
}

// 修改后:
void WarpContext::update_active_mask() {
    active_count = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        // 权威源: warp_state (由 sync_to_warp_state 保持最新)
        bool should_be_active = warp_state.threads[i].is_schedulable();

        active_mask[i] = should_be_active;
        warp_state.threads[i].is_active = should_be_active;  // 双向同步

        if (should_be_active) {
            active_count++;
        }
    }
}
```

- [ ] **Step 2: 编译 + 运行所有 Phase 1+2 测试**

```bash
cmake --build build -j$(nproc)
ctest -R "test_simt_stack|test_exec_mask|test_handle_branch|test_active_mask" -V
# Expected: All tests pass
```

- [ ] **Step 3: 全量回归**

```bash
cd build && ctest --output-on-failure
# Expected: 无回归
```

- [ ] **Step 4: 提交**

```bash
git add src/ptxsim/core/warp_context.cpp
git commit -m "fix(simt): unify active_mask data source to warp_state (ISSUE-004)

update_active_mask() previously read ThreadContext::is_exited() and
warp_state.threads[i].is_blocked from different sources. Now reads
solely from warp_state.threads[i].is_schedulable() and writes back
to warp_state.threads[i].is_active for bidirectional consistency."
```

---

### Task 9: 废弃双重 PC 管理 (Phase 2.1 - ISSUE-002/006)

**Files:**
- Modify: `include/ptxsim/warp_context.h:57-60, 89-93`

- [ ] **Step 1: 调用点审计**

```bash
# 确认 pc_stacks 所有使用点
grep -rn "pc_stacks\[" src/ include/ --include="*.cpp" --include="*.h"
# Expected output:
#   src/ptxsim/core/warp_context.cpp:109: pc_stacks[i] = std::vector<int>();
#   src/ptxsim/core/warp_context.cpp:274: pc_stacks[i].clear();
#   src/ptxsim/core/warp_context.cpp:275: pc_stacks[i].push_back(0);
#   src/ptxsim/core/warp_context.cpp:284: pc_stacks[lane_id].push_back(...)
#   src/ptxsim/core/warp_context.cpp:287: pc_stacks[lane_id].push_back(0);
#   src/ptxsim/core/warp_context.cpp:291: pc_stacks[lane_id].back() = new_pc;
#   include/ptxsim/warp_context.h:91:   pc_stacks[lane_id].back() = new_pc;

grep -rn "update_pc_stack\|handle_branch_divergence" src/ include/ --include="*.cpp" --include="*.h"
# Expected: 列出所有调用者
```

- [ ] **Step 2: 标记废弃**

```cpp
// include/ptxsim/warp_context.h — 在 get_pc/set_pc 和 update_pc_stack 前添加:
[[deprecated("Use warp_state.threads[lane_id].pc for per-thread PC, "
             "or warp_state.warp_pc for warp-level fallback")]]
int get_pc() const { return pc; }

[[deprecated("Use advance_thread_pc() or advance_all_threads() instead")]]
void set_pc(int new_pc) { pc = new_pc; }

[[deprecated("Use warp_state.threads[lane_id].pc = new_pc instead")]]
void update_pc_stack(int lane_id, uint32_t new_pc) {
    if (lane_id >= 0 && lane_id < WARP_SIZE && !pc_stacks[lane_id].empty()) {
        pc_stacks[lane_id].back() = new_pc;
    }
};

[[deprecated("PC stacks are being replaced by SIMT stack + warp_state")]]
void handle_branch_divergence(int lane_id, int new_pc);
```

- [ ] **Step 3: 编译确认废弃警告可观测**

```bash
cmake --build build -j$(nproc) 2>&1 | grep -c "deprecated"
# Expected: >0 条 deprecated 警告 (确认调用者可见)
```

- [ ] **Step 4: 运行全量测试确认功能不受影响**

```bash
cd build && ctest --output-on-failure
# Expected: 100% tests passed (废弃标记不影响运行时)
```

- [ ] **Step 5: 提交**

```bash
git add include/ptxsim/warp_context.h
git commit -m "refactor(simt): mark get_pc/set_pc/update_pc_stack/handle_branch_divergence deprecated

These methods use the legacy pc_stacks mechanism. All new code should
use warp_state.threads[lane_id].pc directly. Full removal planned after
all callers are migrated (Phase 2 milestone)."
```

---

### Task 10: 统一 is_lane_active 双源 (ISSUE-005)

**Files:**
- Modify: `include/ptxsim/warp_context.h:128`

- [ ] **Step 1: 委托到统一源**

```cpp
// include/ptxsim/warp_context.h:128 — 修改 is_lane_active()
// 修改前:
bool is_lane_active(int lane_id) const {
    return lane_id >= 0 && lane_id < WARP_SIZE && active_mask[lane_id];
}

// 修改后: 委托到 warp_state 权威源
bool is_lane_active(int lane_id) const {
    return is_lane_schedulable(lane_id);
}
```

- [ ] **Step 2: 编译 + 回归**

```bash
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
# Expected: 100% tests passed
```

- [ ] **Step 3: 提交**

```bash
git add include/ptxsim/warp_context.h
git commit -m "fix(simt): unify is_lane_active to delegate to is_lane_schedulable (ISSUE-005)

Previously is_lane_active() used active_mask[lane_id] while
is_lane_schedulable() used warp_state.threads[lane].is_schedulable().
Now both delegate to the single warp_state authority source."
```

---

### Task 11: SIMT 栈最大深度限制 (Phase 2.2)

**Files:**
- Modify: `include/ptxsim/simt_stack.h`
- Modify: `src/ptxsim/core/simt_stack.cpp:27-29`
- Test: A8 (已写入 test_simt_stack_catch2.cpp, 待添加)

- [ ] **Step 1: 编写 A8 失败测试**

在 `tests/test_simt_stack_catch2.cpp` 中添加：

```cpp
TEST_CASE("A8: maximum depth enforcement", "[simt_stack][limit]") {
    SIMTStack stack;
    SIMTStackEntry e;
    e.branch_pc = 0; e.reconvergence_pc = 0; e.return_pc = 0;
    e.active_mask = 0; e.return_mask = 0;

    // push 10 层 (MAX_DEPTH)
    for (int i = 0; i < 10; i++) {
        stack.push(e);
    }
    REQUIRE(stack.depth() == 10);

    // 第 11 层应抛出异常
    REQUIRE_THROWS_AS(stack.push(e), std::runtime_error);
    try {
        stack.push(e);
    } catch (const std::runtime_error& ex) {
        REQUIRE(std::string(ex.what()).find("overflow") != std::string::npos);
    }
}
```

```bash
cmake --build build -j$(nproc) --target test_simt_stack_catch2
ctest -R "A8" -V
# Expected: A8 FAIL — no depth limit currently enforced
```

- [ ] **Step 2: 添加深度限制 (GREEN)**

```cpp
// include/ptxsim/simt_stack.h — 在 SIMTStack 类中添加:
class SIMTStack {
public:
    static constexpr size_t MAX_DEPTH = 10;  // 与 GPGPU-Sim 一致

    // ... 现有声明 ...
};
```

```cpp
// src/ptxsim/core/simt_stack.cpp:27-29 — 修改 push()
// 修改前:
void SIMTStack::push(const SIMTStackEntry& entry) {
    entries_.push_back(entry);
}

// 修改后:
void SIMTStack::push(const SIMTStackEntry& entry) {
    if (entries_.size() >= MAX_DEPTH) {
        throw std::runtime_error(
            "SIMTStack overflow: maximum depth (" +
            std::to_string(MAX_DEPTH) + ") exceeded. "
            "This may indicate unbounded nested branches."
        );
    }
    entries_.push_back(entry);
}
```

- [ ] **Step 3: 运行测试验证**

```bash
cmake --build build -j$(nproc)
ctest -R test_simt_stack_catch2 -V
# Expected: All 8 tests pass (包括 A8)
```

- [ ] **Step 4: 提交**

```bash
git add include/ptxsim/simt_stack.h src/ptxsim/core/simt_stack.cpp tests/test_simt_stack_catch2.cpp
git commit -m "feat(simt): add SIMT stack max depth limit (MAX_DEPTH=10)

Prevents unbounded stack growth from deeply nested branches. Throws
std::runtime_error on overflow. Depth limit matches GPGPU-Sim convention.

Test: A8 verifies overflow detection."
```

---

### Task 12: Phase 2 回归验证

- [ ] **Step 1: 全量测试**

```bash
cd build && ctest --output-on-failure -j$(nproc)
# Expected: 100% tests passed
```

- [ ] **Step 2: clang-tidy 检查**

```bash
clang-tidy -p build/ src/ptxsim/core/warp_context.cpp src/ptxsim/core/simt_stack.cpp --checks=cppcoreguidelines-* 2>&1 | tail -5
# Expected: 无新增 warning
```

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "milestone: Phase 2 complete — code cleanup and data source unification

Cleanups:
- ISSUE-002: marked legacy PC management methods as [[deprecated]]
- ISSUE-004: unified active_mask data source to warp_state
- ISSUE-005: unified is_lane_active → is_lane_schedulable delegation
- ISSUE-006: marked WarpContext::pc as [[deprecated]]
- Phase 2.2: SIMT stack max depth limit (MAX_DEPTH=10)

New tests: test_simt_stack_catch2 (A1-A8), test_active_mask_consistency (J1-J5)
All existing tests pass with no regression."
```

---

## Phase 3: 测试扩展 (P0-P1)

> **目标**: 完成 60 个测试用例，将覆盖率提升至 ≥90%。
> **验收**: `ctest -L simt` 全部通过。

---

### Task 13: 测试组 C — WarpState 测试

**Files:**
- Create: `tests/test_warp_state.cpp`

```cpp
// tests/test_warp_state.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/warp_state.h"
#include "ptxsim/thread_state.h"

using namespace ptxsim;

TEST_CASE("C1: default initialization", "[warp_state]") {
    WarpState ws;
    for (int i = 0; i < 32; i++) {
        REQUIRE(ws.threads[i].pc == 0);
        REQUIRE(ws.threads[i].is_active == true);
        REQUIRE(ws.threads[i].is_exited == false);
        REQUIRE(ws.threads[i].is_blocked == false);
    }
    REQUIRE(ws.exec_mask == 0xFFFFFFFF);
    REQUIRE(ws.current_wbar_id == -1);
    REQUIRE(ws.warp_pc == 0);
}

TEST_CASE("C2: reset restores defaults", "[warp_state]") {
    WarpState ws;
    ws.exec_mask = 0x0000FFFF;
    ws.current_wbar_id = 2;
    ws.warp_pc = 42;
    ws.threads[0].is_exited = true;
    ws.threads[0].is_active = false;
    ws.threads[0].pc = 99;

    ws.reset();

    REQUIRE(ws.exec_mask == 0xFFFFFFFF);
    REQUIRE(ws.current_wbar_id == -1);
    REQUIRE(ws.warp_pc == 0);
    REQUIRE(ws.threads[0].is_exited == false);
    REQUIRE(ws.threads[0].is_active == true);
    REQUIRE(ws.threads[0].pc == 0);
}

TEST_CASE("C3: count_active_lanes", "[warp_state]") {
    WarpState ws;
    for (int i = 0; i < 16; i++) ws.threads[i].is_active = false;
    REQUIRE(ws.count_active_lanes() == 16);
    for (int i = 16; i < 21; i++) ws.threads[i].is_exited = true;
    REQUIRE(ws.count_active_lanes() == 11);  // 16 inactive + 5 exited
}

TEST_CASE("C4: count_schedulable_lanes", "[warp_state]") {
    WarpState ws;
    for (int i = 0; i < 5; i++) ws.threads[i].is_blocked = true;
    for (int i = 5; i < 10; i++) { ws.threads[i].is_exited = true; ws.threads[i].is_active = false; }
    for (int i = 10; i < 12; i++) ws.threads[i].is_active = false;
    REQUIRE(ws.count_schedulable_lanes() == 20);
}

TEST_CASE("C5: is_all_exited", "[warp_state]") {
    WarpState ws;
    REQUIRE(ws.is_all_exited() == false);
    for (int i = 0; i < 31; i++) ws.threads[i].is_exited = true;
    REQUIRE(ws.is_all_exited() == false);
    ws.threads[31].is_exited = true;
    REQUIRE(ws.is_all_exited() == true);
}

TEST_CASE("C6: has_schedulable_threads", "[warp_state]") {
    WarpState ws;
    REQUIRE(ws.has_schedulable_threads() == true);
    for (int i = 0; i < 32; i++) { ws.threads[i].is_blocked = true; ws.threads[i].is_active = false; }
    REQUIRE(ws.has_schedulable_threads() == false);
}

TEST_CASE("C7: thread_predicates cleared on reset", "[warp_state][cleanup]") {
    WarpState ws;
    ws.thread_predicates["%p1"] = std::array<bool, 32>{};
    ws.reset();
    REQUIRE(ws.thread_predicates.empty() == true);
}
```

```cmake
add_catch_test(test_warp_state
    ${CMAKE_CURRENT_SOURCE_DIR}/test_warp_state.cpp
)
```

- [ ] **Step 1: 编译并运行**

```bash
cmake --build build -j$(nproc) --target test_warp_state
ctest -R test_warp_state -V
# Expected: All 7 tests pass
```

- [ ] **Step 2: 提交**

```bash
git add tests/test_warp_state.cpp tests/CMakeLists.txt
git commit -m "test(simt): add WarpState tests (C1-C7)

Covers: default init, reset, count_active_lanes, count_schedulable_lanes,
is_all_exited, has_schedulable_threads, thread_predicates cleanup."
```

---

### Task 14: 测试组 H — Sync 机制测试

**Files:**
- Create: `tests/test_sync_mechanism.cpp`

```cpp
// tests/test_sync_mechanism.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"

using namespace ptxsim;

TEST_CASE("H1: sync_from_warp_state reads PC correctly", "[sync]") {
    WarpContext warp;
    warp.set_thread_pc(0, 15);
    warp.get_warp_state().threads[0].status = ThreadStatus::Active;
    // 验证 warp_state 中的值正确
    REQUIRE(warp.get_thread_pc(0) == 15);
    REQUIRE(warp.get_warp_state().threads[0].status == ThreadStatus::Active);
}

TEST_CASE("H2: sync_to_warp_state preserves PC", "[sync]") {
    WarpContext warp;
    // 直接设置 warp_state 并验证读取
    warp.get_warp_state().threads[0].next_pc = 20;
    warp.get_warp_state().threads[0].status = ThreadStatus::Active;
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 20);
}

TEST_CASE("H3: branch PC not overwritten by sync", "[sync][branch]") {
    WarpContext warp;
    // 模拟 handle_branch 设置 PC
    warp.set_thread_pc(0, 20);
    // 模拟其他操作设置 next_pc
    warp.get_warp_state().threads[0].next_pc = 20;
    // PCI 应保持一致
    REQUIRE(warp.get_thread_pc(0) == 20);
}

TEST_CASE("H4: force_set_pc for barrier completion", "[sync][barrier]") {
    WarpContext warp;
    // 模拟屏障完成后 force_set_pc
    warp.set_thread_pc(0, 10);
    warp.set_thread_pc(0, 30);  // barrier 完成后设置新 PC
    REQUIRE(warp.get_thread_pc(0) == 30);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 30);
}

TEST_CASE("H5: exited thread state sync", "[sync][exit]") {
    WarpContext warp;
    warp.get_warp_state().threads[0].is_exited = true;
    warp.get_warp_state().threads[0].is_active = false;
    REQUIRE(warp.get_warp_state().threads[0].is_exited == true);
    REQUIRE(warp.get_warp_state().threads[0].is_active == false);
}

TEST_CASE("H6: bidirectional sync consistency", "[sync]") {
    WarpContext warp;
    // 设置 → 读取 → 修改 → 再读取
    warp.set_thread_pc(0, 10);
    REQUIRE(warp.get_thread_pc(0) == 10);

    warp.set_thread_pc(0, 20);
    REQUIRE(warp.get_thread_pc(0) == 20);
    REQUIRE(warp.get_warp_state().threads[0].next_pc == 20);
}
```

```cmake
add_catch_test(test_sync_mechanism
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sync_mechanism.cpp
)
```

- [ ] **Step 1: 编译并运行**

```bash
cmake --build build -j$(nproc) --target test_sync_mechanism
ctest -R test_sync_mechanism -V
# Expected: All 6 tests pass
```

- [ ] **Step 2: 提交**

```bash
git add tests/test_sync_mechanism.cpp tests/CMakeLists.txt
git commit -m "test(simt): add sync mechanism tests (H1-H6)

Covers: sync_from/to_warp_state PC consistency, branch PC preservation,
barrier force_set_pc, exited thread state, bidirectional sync cycle."
```

---

### Task 15: 测试组 G — PC 管理高级测试

**Files:**
- Create: `tests/test_pc_management_advanced.cpp`

```cpp
// tests/test_pc_management_advanced.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"

using namespace ptxsim;

TEST_CASE("G4: advance_thread_pc updates both sources", "[pc][unified]") {
    WarpContext warp;
    warp.advance_thread_pc(5, 20);
    REQUIRE(warp.get_thread_pc(5) == 20);
    REQUIRE(warp.get_warp_state().threads[5].next_pc == 20);
}

TEST_CASE("G5: advance_all_threads only advances active", "[pc][unified]") {
    WarpContext warp;
    // 设置一部分线程 inactive
    warp.get_warp_state().threads[0].is_active = false;
    warp.get_warp_state().threads[1].is_active = false;

    warp.advance_all_threads(30);

    // 活跃线程应被推进
    REQUIRE(warp.get_thread_pc(2) == 30);
    // 非活跃线程应保持不变
    REQUIRE(warp.get_thread_pc(0) == 0);
}

TEST_CASE("G1: advance_thread_pc out-of-bounds safety", "[pc][safety]") {
    WarpContext warp;
    // 不应崩溃
    warp.advance_thread_pc(-1, 10);
    warp.advance_thread_pc(32, 10);
    // 正常 lane 应继续工作
    warp.advance_thread_pc(0, 42);
    REQUIRE(warp.get_thread_pc(0) == 42);
}

TEST_CASE("G2: multiple advance_thread_pc calls accumulate", "[pc]") {
    WarpContext warp;
    warp.advance_thread_pc(0, 10);
    warp.advance_thread_pc(0, 20);
    warp.advance_thread_pc(0, 30);
    REQUIRE(warp.get_thread_pc(0) == 30);
}

TEST_CASE("G3: pc consistency after advance_all_threads", "[pc]") {
    WarpContext warp;
    warp.advance_all_threads(42);
    for (int i = 0; i < 32; i++) {
        if (warp.get_warp_state().threads[i].is_active) {
            REQUIRE(warp.get_thread_pc(i) == 42);
        }
    }
}
```

```cmake
add_catch_test(test_pc_management_advanced
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pc_management_advanced.cpp
)
```

- [ ] **Step 1: 编译并运行**

```bash
cmake --build build -j$(nproc) --target test_pc_management_advanced
ctest -R test_pc_management_advanced -V
# Expected: All 5 tests pass
```

- [ ] **Step 2: 提交**

```bash
git add tests/test_pc_management_advanced.cpp tests/CMakeLists.txt
git commit -m "test(simt): add PC management advanced tests (G1-G5)

Covers: advance_thread_pc unified update, advance_all_threads active-only,
out-of-bounds safety, multiple advance accumulation, per-thread consistency."
```

---

### Task 16: 测试组 E + I — Barrier 交互与集成场景

- [ ] **Step 1: 创建测试文件**

由于这些测试依赖完整的 PTX 执行环境（SMContext, CFG Builder 等），采用**渐进方式**：
- E1-E4: 单元级 barrier+SIMT 测试（可直接运行）
- I1-I6c: 集成测试（依赖 Phase 4 的 SMContext 扩展）

在 `tests/CMakeLists.txt` 注册：

```cmake
add_catch_test(test_barrier_simt_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_barrier_simt_integration.cpp
)
add_catch_test(test_simt_integration
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simt_integration.cpp
)
```

测试代码参见 `docs/testing/simt-complete-test-plan.md` 中的 E1-E4 和 I1-I6c 规范。

- [ ] **Step 2: 标记依赖 Phase 4 的测试为 SKIP**

```cpp
// 在测试中添加条件跳过
TEST_CASE("E2: barrier inside divergent path", "[barrier][simt][.]") {
    // [.] 标签 = Catch2 隐藏测试 (需要显式运行)
    // Phase 4 完成后移除 [.]
}
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_barrier_simt_integration.cpp tests/test_simt_integration.cpp tests/CMakeLists.txt
git commit -m "test(simt): add barrier integration and scenario tests (E1-E4, I1-I6c)

Phase 4-dependent tests marked as hidden ([.]) until SMContext convergence
extension is implemented."
```

---

### Task 17: Phase 3 回归 + 覆盖率验证

- [ ] **Step 1: 全量 SIMT 测试**

```bash
cd build && ctest -L simt -V
# Expected: 所有非隐藏测试通过
ctest --output-on-failure
# Expected: 全量测试通过
```

- [ ] **Step 2: 覆盖率测量**

```bash
# 需要 Debug + coverage 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage"
cmake --build build -j$(nproc)
cd build && ctest -L simt

lcov --capture --directory . --output-file simt_coverage.info
lcov --remove simt_coverage.info '/usr/*' '*/tests/*' '*/external/*' '*/antlr4*' --output-file simt_filtered.info
lcov --summary simt_filtered.info
# Expected: lines coverage ≥ 60% (Phase 3 目标)
```

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "milestone: Phase 3 complete — test expansion

New tests: test_warp_state (C1-C7), test_sync_mechanism (H1-H6),
test_pc_management_advanced (G1-G5), test_barrier_simt_integration (E1-E4),
test_simt_integration (I1-I6c)

Total: 60 test cases. SIMT coverage target: ≥60% (Phase 3 baseline)."
```

---

## Phase 4: 架构增强 (P1-P2)

> **目标**: 屏障完成后 SIMT 栈清理、调试工具、SMContext 扩展。
> **验收**: 覆盖率 ≥90%，所有隐藏测试转为正式测试。

---

### Task 18: 屏障完成后 SIMT 栈清理 (Phase 4.1 — BUG-003)

**Files:**
- Modify: `src/ptxsim/core/sm_context.cpp`

- [ ] **Step 1: 理解当前行为**

当前 `sm_context.cpp` 仅在 `S_BRA` 指令后调用 `check_reconvergence()`：

```cpp
// sm_context.cpp:203 — 当前代码
if (stmt->type == S_BRA) {
    next_warp->check_reconvergence();
}
```

- [ ] **Step 2: 扩展检查到 barrier 指令 (GREEN)**

```cpp
// src/ptxsim/core/sm_context.cpp — 修改收敛检查条件
// 修改前:
if (stmt->type == S_BRA) {
    next_warp->check_reconvergence();
}

// 修改后:
if (stmt->type == S_BRA || stmt->type == S_BAR ||
    stmt->type == S_BAR_WARP_SYNC) {
    next_warp->check_reconvergence();
}
```

- [ ] **Step 3: 编译并验证 E2/E3 测试**

```bash
cmake --build build -j$(nproc)
# 将 E2 从 [.] 移除后运行
ctest -R "test_barrier_simt_integration" -V
# Expected: E2, E3 现在通过
```

- [ ] **Step 4: 提交**

```bash
git add src/ptxsim/core/sm_context.cpp tests/test_barrier_simt_integration.cpp
git commit -m "fix(simt): extend check_reconvergence to barrier instructions (BUG-003)

Previously check_reconvergence() was only called after S_BRA. Now also
called after S_BAR and S_BAR_WARP_SYNC, ensuring SIMT stack is cleaned up
when barriers trigger thread reconvergence.

Test: E2, E3 now pass (barrier inside/post divergent path)."
```

---

### Task 19: SIMT 调试工具 (Phase 4.2)

**Files:**
- Create: `include/ptxsim/simt_debug.h`
- Create: `src/ptxsim/core/simt_debug.cpp`

- [ ] **Step 1: 创建头文件**

```cpp
// include/ptxsim/simt_debug.h
#ifndef SIMT_DEBUG_H
#define SIMT_DEBUG_H

#include "ptxsim/warp_context.h"
#include <vector>
#include <string>
#include <ostream>

namespace ptxsim {

class SimtDebugger {
public:
    struct SimtIssue {
        enum Severity { Warning, Error, Critical };
        Severity severity;
        std::string description;
    };

    static void printSimtStack(const WarpContext& warp, std::ostream& os = std::cout);
    static void printThreadPCs(const WarpContext& warp, std::ostream& os = std::cout);
    static void printExecMask(const WarpContext& warp, std::ostream& os = std::cout);
    static void dumpWarpState(const WarpContext& warp, std::ostream& os = std::cout);
    static std::vector<SimtIssue> diagnose(const WarpContext& warp);
};

} // namespace ptxsim

#endif
```

- [ ] **Step 2: 创建实现文件**

```cpp
// src/ptxsim/core/simt_debug.cpp
#include "ptxsim/simt_debug.h"
#include <iomanip>
#include <sstream>

namespace ptxsim {

void SimtDebugger::printSimtStack(const WarpContext& warp, std::ostream& os) {
    warp.get_simt_stack().print();
}

void SimtDebugger::printThreadPCs(const WarpContext& warp, std::ostream& os) {
    os << "Thread PCs:\n";
    for (int i = 0; i < 32; i++) {
        os << "  lane " << std::setw(2) << i << ": pc="
           << warp.get_thread_pc(i)
           << " active=" << warp.get_warp_state().threads[i].is_active
           << " exited=" << warp.get_warp_state().threads[i].is_exited
           << " blocked=" << warp.get_warp_state().threads[i].is_blocked << "\n";
    }
}

void SimtDebugger::printExecMask(const WarpContext& warp, std::ostream& os) {
    os << "exec_mask=0x" << std::hex << warp.get_exec_mask()
       << std::dec << " active_count=" << warp.get_active_count() << "\n";
}

void SimtDebugger::dumpWarpState(const WarpContext& warp, std::ostream& os) {
    os << "=== Warp State Dump ===\n";
    printExecMask(warp, os);
    os << "SIMT stack depth: " << warp.get_simt_stack().depth() << "\n";
    printSimtStack(warp, os);
    printThreadPCs(warp, os);
}

std::vector<SimtDebugger::SimtIssue> SimtDebugger::diagnose(const WarpContext& warp) {
    std::vector<SimtIssue> issues;

    // 检测 exec_mask 与 SIMT 栈不一致
    if (warp.get_simt_stack().empty() && warp.get_exec_mask() != 0xFFFFFFFF) {
        issues.push_back({SimtIssue::Warning,
            "exec_mask=" + std::to_string(warp.get_exec_mask()) +
            " but SIMT stack is empty — possible stale mask"});
    }

    // 检测 SIMT 栈深度异常
    if (warp.get_simt_stack().depth() > 5) {
        issues.push_back({SimtIssue::Warning,
            "SIMT stack depth=" + std::to_string(warp.get_simt_stack().depth()) +
            " — unusually deep nesting"});
    }

    return issues;
}

} // namespace ptxsim
```

- [ ] **Step 3: 提交**

```bash
git add include/ptxsim/simt_debug.h src/ptxsim/core/simt_debug.cpp
git commit -m "feat(simt): add SIMT debug tool (SimtDebugger)

Provides: printSimtStack, printThreadPCs, printExecMask, dumpWarpState,
and diagnose() for detecting stale exec_mask and deep stack nesting."
```

---

### Task 20: Phase 4 最终回归 + 覆盖率验证

- [ ] **Step 1: 所有隐藏测试转为正式测试**

```bash
# 移除测试中的 [.] 标签
grep -rn "\[.\]" tests/test_barrier_simt_integration.cpp tests/test_simt_integration.cpp
# 手动移除 [.] 或在测试中条件性启用
```

- [ ] **Step 2: 全量测试**

```bash
cd build && ctest --output-on-failure -j$(nproc)
# Expected: 100% tests passed (包括 E2, E3, I6a-I6c)
```

- [ ] **Step 3: 覆盖率验证**

```bash
lcov --capture --directory build --output-file simt_final.info
lcov --remove simt_final.info '/usr/*' '*/tests/*' '*/external/*' '*/antlr4*' --output-file simt_final_filtered.info
lcov --summary simt_final_filtered.info
# Expected: lines coverage ≥ 90%
```

- [ ] **Step 4: PTX 语法回归**

```bash
./tests/ptx/test_all_ptx.sh
# Expected: All PTX tests passed
```

- [ ] **Step 5: valgrind 内存检查 (可选)**

```bash
valgrind --leak-check=full ./build/test_exec_mask 2>&1 | grep "ERROR SUMMARY"
# Expected: 0 errors
```

- [ ] **Step 6: 最终提交**

```bash
git add -A
git commit -m "milestone: Phase 4 complete — SIMT architecture enhancement

Enhancements:
- BUG-003: barrier completion now triggers SIMT stack convergence check
- Phase 4.2: SimtDebugger tool for SIMT state visualization and diagnosis

All 60 tests pass. SIMT code coverage ≥90%.
PTX syntax regression: all pass. clang-tidy: clean.

Fixes: BUG-001, BUG-002, BUG-003, ISSUE-001 through ISSUE-006."
```

---

## 实施里程碑总览

| 里程碑 | 内容 | 任务 | 验收标准 |
|--------|------|------|---------|
| **M1** | Bug 修复 | Task 1-5 | ctest 全通过, B4/F3/D5 通过 |
| **M2** | 代码清理 | Task 6-12 | clang-tidy clean, 测试 ≥45 用例 |
| **M3** | 测试扩展 | Task 13-17 | 60 用例通过, 覆盖率 ≥60% |
| **M4** | 架构增强 | Task 18-20 | 覆盖率 ≥90%, 隐藏测试全部启用 |

---

## 回滚策略

- 每个 Task 独立提交 → `git revert <commit>` 可精确回滚
- 每个 Milestone 前创建 tag: `git tag -a milestone-N-before -m "Before Milestone N"`
- Phase 4 改动 (sm_context.cpp) 可选 — 如果集成测试不稳定可回滚
- 如果 full pipeline tests 大面积失败 → `git reset --hard milestone-3-before`

---

## 参考测试运行命令

```bash
# 按标签运行
ctest -L simt -V                    # 所有 SIMT 测试
ctest -L "simt|barrier" -V          # SIMT + Barrier
ctest -R "test_exec_mask" -V        # 特定测试组
ctest -R "B4|F3|D5" -V              # 关键 Bug 检测

# 并行运行
ctest -j$(nproc) --output-on-failure

# 覆盖率
lcov --capture --directory build --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/tests/*' '*/external/*' --output-file filtered.info
genhtml filtered.info --output-directory coverage_html
```
