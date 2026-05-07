# SIMT 架构改进与完善计划

**日期**: 2026-05-04  
**基于**: [SIMT 架构审查报告](simt-architecture-review-and-test-plan.md)  
**目标**: 修复设计缺陷、统一数据管理、提升测试覆盖率至 90%+

---

## 执行摘要

本计划提出 **10 个架构改进** 和 **53 个测试用例**，分 4 个阶段实施：

| 阶段 | 类型 | 改动范围 | 风险 | 预计影响 |
|------|------|---------|------|---------|
| Phase 1 | Bug 修复 | 3 文件 | 🟢 低 | 修复 3 个逻辑缺陷 |
| Phase 2 | 代码清理 | 5 文件 | 🟢 低 | 删除未使用代码，统一数据源 |
| Phase 3 | 测试补充 | 9 新文件 | 🟢 无 | 覆盖率 40% → 90% |
| Phase 4 | 架构增强 | 3 文件 | 🟡 中 | 深度限制、屏障集成、调试工具 |

---

## Phase 1: 关键 Bug 修复 (P0 - 立即执行)

### 1.1 修复 exec_mask 收敛后未恢复

**问题**: `check_reconvergence()` 弹出 SIMT 栈后没有恢复 exec_mask  
**严重性**: 🔴 高 - 导致收敛后执行掩码错误  
**影响范围**: 所有分歧分支场景

#### 修改文件
- `src/ptxsim/core/warp_context.cpp` (第 98-101 行)

#### 代码变更

```cpp
// 当前代码 (错误):
void WarpContext::check_reconvergence() {
    if (simt_stack.empty()) return;
    simt_stack.check_reconvergence(warp_state.threads);
    // ⚠️ 没有恢复 exec_mask
}

// 修复后代码:
void WarpContext::check_reconvergence() {
    if (simt_stack.empty()) return;
    
    // 记录收敛前的栈深度
    size_t depth_before = simt_stack.depth();
    
    // 检查收敛 (可能弹出多层栈)
    simt_stack.check_reconvergence(warp_state.threads);
    
    // 如果栈深度减少 (发生弹出)，恢复 exec_mask
    if (simt_stack.depth() < depth_before) {
        if (simt_stack.empty()) {
            // 所有分支层已收敛 → 恢复全活跃掩码
            warp_state.exec_mask = 0xFFFFFFFF;
        } else {
            // 还有外层分支未收敛 → 恢复到外层的 return_mask
            warp_state.exec_mask = simt_stack.top().return_mask;
        }
        
        PTX_DEBUG_EMU("[SIMT:RECONV] Restored exec_mask=0x%X after reconvergence (depth: %zu→%zu)", 
                      warp_state.exec_mask, depth_before, simt_stack.depth());
    }
}
```

> **设计说明**: 使用 `depth_before` 对比而非返回值的 `converged` 标志，因为 `check_reconvergence()` 在栈原本为空时也返回 `true`（simt_stack.cpp:68），单靠返回值无法区分"原本就空"和"弹出后变空"。对比 `depth()` 变化可以精确判断是否发生了弹出。

#### 测试验证
- 新增测试: `test_exec_mask.cpp` → F3 用例
- 验证点: 分歧 → 收敛后 exec_mask = 0xFFFFFFFF

---

### 1.2 修复退出线程导致 SIMT 栈永久阻塞

**问题**: `is_converged()` 不检查线程是否已退出  
**严重性**: 🔴 高 - 导致 SIMT 栈永不弹出，warp 挂起  
**影响范围**: 分支路径中有 exit 指令的场景

#### 修改文件
- `src/ptxsim/core/simt_stack.cpp` (第 7-16 行)

#### 代码变更

```cpp
// 当前代码 (错误):
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

// 修复后代码:
bool SIMTStackEntry::is_converged(const std::array<ThreadState, 32>& threads) const {
    for (size_t i = 0; i < 32; i++) {
        // 只检查 return_mask 中仍在执行的线程
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

> **边缘情况说明**: 如果 `reconvergence_pc == 0`，未初始化的退出线程 (`pc=0`) 会被动误判为收敛。但当前架构中 `reconvergence_pc` 由 CFG 分析计算且恒为正（PTX 入口 PC ≥ 1），因此此边缘情况暂时安全。若未来支持 PC=0 入口，需额外添加 `reconvergence_pc > 0` 的断言或改用 `pc >= reconvergence_pc` 语义。

#### 测试验证
- 新增测试: `test_simt_stack_entry.cpp` → B4 用例
- 验证点: 分支后部分线程 exit，剩余线程能正常收敛

---

### 1.3 删除 WarpState 未使用字段

**问题**: `pc_stack` 和 `pc_stack_depth` 从未使用  
**严重性**: 🟢 低 - 代码清洁度问题  
**影响范围**: 内存浪费 (64 字节/warp)，代码混淆

#### 修改文件
- `include/ptxsim/warp_state.h` (第 20-21, 34 行)

#### 代码变更

```cpp
// 删除前:
struct WarpState {
    std::array<ThreadState, 32> threads;
    uint32_t exec_mask = 0xFFFFFFFF;
    std::array<Wbar, 4> wbars;
    int current_wbar_id = -1;
    uint32_t warp_pc = 0;
    std::array<int, 16> pc_stack;         // ❌ 删除
    int pc_stack_depth = 0;               // ❌ 删除
    
    void reset() {
        // ...
        pc_stack_depth = 0;               // ❌ 删除
    }
};

// 删除后:
struct WarpState {
    std::array<ThreadState, 32> threads;
    uint32_t exec_mask = 0xFFFFFFFF;
    std::array<Wbar, 4> wbars;
    int current_wbar_id = -1;
    uint32_t warp_pc = 0;
    // pc_stack 和 pc_stack_depth 已移除 - 使用 WarpContext::pc_stacks 替代
    
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
};
```

#### 验证
- 全局搜索确认无引用: `grep -r "pc_stack" src/ include/`
- 编译通过，测试无回归

---

## Phase 2: 代码清理与统一 (P1 - 短期)

### 2.1 评估并清理双重 PC 管理

**问题**: `WarpContext::pc_stacks[32]` 和 `warp_state.threads[i].pc` 并存  
**目标**: 统一到单一权威源 `warp_state.threads[i].pc`

#### 调用点完整审计 (必须先执行!)

| 使用点 | 文件 | 行号 | 使用方式 | 替代方案 |
|--------|------|------|---------|---------|
| `handle_branch_divergence()` | warp_context.cpp | 281-295 | pc_stacks[lane_id] push+set | `warp_state.threads[i].pc = new_pc` |
| `update_pc_stack()` | warp_context.h | 89-93 | pc_stacks[lane_id].back() = new_pc | `warp_state.threads[i].pc = new_pc` |
| `reset()` | warp_context.cpp | 274-275 | pc_stacks[i].clear(); push_back(0) | `warp_state.reset()` (已覆盖) |
| `barrier.cpp` | barrier.cpp | ~164 | update_pc_stack(i, reconvergence_pc) | `warp_state.threads[i].pc = reconvergence_pc` |

> **审计结论**: 所有 4 个调用点均可直接使用 `warp_state.threads[i].pc` 替代。`pc_stacks` 的 push/pop 语义已由 `simt_stack` (SIMTStack) 覆盖。

#### 清理计划

**步骤 1**: 标记废弃方法

```cpp
// warp_context.h - 添加废弃标记
[[deprecated("Use warp_state.threads[lane].pc directly")]]
void handle_branch_divergence(int lane_id, int new_pc);

[[deprecated("Use warp_state.threads[lane].pc = new_pc instead")]]
void update_pc_stack(int lane_id, uint32_t new_pc);
```

**步骤 2**: 更新调用点

```cpp
// barrier.cpp:164 - 修改前:
warp_ctx->update_pc_stack(i, reconvergence_pc);

// 修改后:
warp_ctx->get_warp_state().threads[i].pc = reconvergence_pc;
```

**步骤 3**: 删除 pc_stacks 成员 (确认无引用后)

```cpp
// warp_context.h - 删除:
std::vector<int> pc_stacks[WARP_SIZE];
```

**步骤 4**: 删除废弃方法 (确认无调用后)

```cpp
// warp_context.cpp - 删除:
void WarpContext::handle_branch_divergence(int lane_id, int new_pc) { ... }

// warp_context.h - 删除声明
```

#### 风险控制
- 每步完成后运行测试套件
- 保留 git 分支以便回滚

---

### 2.2 添加 SIMT 栈最大深度限制

**问题**: SIMT 栈无限制增长，可能导致内存问题  
**目标**: 限制为 10 层 (与 GPGPU-Sim 一致)

#### 修改文件
- `include/ptxsim/simt_stack.h`
- `src/ptxsim/core/simt_stack.cpp`

#### 代码变更

```cpp
// simt_stack.h - 添加常量:
class SIMTStack {
public:
    static constexpr size_t MAX_DEPTH = 10;  // 硬件限制
    
    void push(const SIMTStackEntry& entry);
    // ...
};

// simt_stack.cpp - 添加检查:
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

#### 测试验证
- 新增测试: `test_simt_stack_catch2.cpp` → A8 用例
- 验证点: push 11 层抛出异常

---

### 2.3 统一 active_mask 数据源 (ISSUE-004)

**问题**: `update_active_mask()` 同时从 `ThreadContext` 和 `warp_state` 读取状态，只写入 `active_mask[]`，不完全同步  
**目标**: `active_mask[]` 和 `warp_state.threads[i].is_active` 双向一致

#### 修改文件
- `src/ptxsim/core/warp_context.cpp` (第 198-210 行)

#### 代码变更

```cpp
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

// 修改后: 统一从 warp_state 读取，并同步写回
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

### 2.4 废弃 WarpContext::pc 向后兼容字段 (ISSUE-006)

**问题**: `WarpContext::pc` 字段在 `warp_state.threads[i].pc` 成为权威源后仍存在  
**目标**: 标记废弃，所有引用改用 `warp_state.warp_pc` 或直接使用 per-thread PC

#### 修改文件
- `include/ptxsim/warp_context.h` (第 57-60 行)

#### 代码变更

```cpp
// 标记废弃:
[[deprecated("Use warp_state.threads[lane_id].pc for per-thread PC, "
             "or warp_state.warp_pc for warp-level fallback")]]
int get_pc() const { return pc; }

[[deprecated("Use advance_thread_pc() or advance_all_threads() instead")]]
void set_pc(int new_pc) { pc = new_pc; }
```

### 2.5 统一 is_lane_active / is_lane_schedulable (ISSUE-005)

**问题**: 两个方法使用不同的数据源，可能返回不同结果  
**目标**: `is_lane_active()` 改为委托给 `is_lane_schedulable()` (后者使用 warp_state 权威源)

#### 修改文件
- `include/ptxsim/warp_context.h` (第 128 行)

#### 代码变更

```cpp
// 修改后: 委托到统一源
bool is_lane_active(int lane_id) const {
    return is_lane_schedulable(lane_id);
}
```

> **注意**: 此修改可能影响 `execute_warp_instruction()` 的行为 (line 153-158)。需要验证 BAR_SYNC fallback 路径不受影响。

---

## Phase 3: 测试补充 (P0-P1 - 并行执行)

### 3.1 测试文件组织结构

```
tests/
├── test_simt_stack_catch2.cpp          # A: SIMT Stack 基础 (8 用例)
├── test_simt_stack_entry.cpp           # B: SIMTStackEntry 收敛 (6 用例)
├── test_warp_state.cpp                 # C: WarpState 状态 (7 用例)
├── test_handle_branch_integration.cpp  # D: handle_branch 集成 (5 用例)
├── test_barrier_simt_integration.cpp   # E: Barrier + SIMT (4 用例)
├── test_exec_mask.cpp                  # F: exec_mask 管理 (6 用例)
├── test_pc_management_advanced.cpp     # G: PC 管理高级 (5 用例)
├── test_sync_mechanism.cpp             # H: sync 机制 (6 用例)
└── test_simt_integration.cpp           # I: 集成场景 (6 用例)
```

### 3.2 CMakeLists.txt 更新

> **统一模式**: 使用 `register_simt_test()` CMake 函数封装，与测试计划 §4.1 一致。

```cmake
# tests/CMakeLists.txt - 在现有测试后添加:

# === SIMT Architecture Tests ===

# Helper function for registering SIMT tests
function(register_simt_test test_name source_file labels)
    add_executable(${test_name} ${source_file} catch_amalgamated.cpp)
    target_link_libraries(${test_name} PRIVATE ptxsim)
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "${labels}")
endfunction()

# Group A: SIMT Stack Basic Operations
register_simt_test(test_simt_stack_catch2 
    test_simt_stack_catch2.cpp 
    "simt")

# Group B: SIMTStackEntry Convergence
register_simt_test(test_simt_stack_entry 
    test_simt_stack_entry.cpp 
    "simt")

# Group C: WarpState
register_simt_test(test_warp_state 
    test_warp_state.cpp 
    "simt")

# Group D: handle_branch Integration
register_simt_test(test_handle_branch_integration 
    test_handle_branch_integration.cpp 
    "simt")

# Group E: Barrier + SIMT Integration
register_simt_test(test_barrier_simt_integration 
    test_barrier_simt_integration.cpp 
    "simt;barrier")

# Group F: exec_mask Management
register_simt_test(test_exec_mask 
    test_exec_mask.cpp 
    "simt")

# Group G: PC Management Advanced
register_simt_test(test_pc_management_advanced 
    test_pc_management_advanced.cpp 
    "simt;pc")

# Group H: Sync Mechanism
register_simt_test(test_sync_mechanism 
    test_sync_mechanism.cpp 
    "simt;sync")

# Group I: Integration Scenarios
register_simt_test(test_simt_integration 
    test_simt_integration.cpp 
    "simt;integration")
```

### 3.3 测试运行命令

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
```

---

### 3.4 关键测试实现示例

#### 测试 F3: exec_mask 收敛后恢复 (检测关键 Bug)

```cpp
// test_exec_mask.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include <memory>

using namespace ptxsim;

static void setup_warp_with_threads(WarpContext& warp) {
    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    
    std::vector<StatementContext> statements;
    StatementContext stmt;
    stmt.type = S_MOV;
    GenericInstr instr;
    stmt.data = instr;
    statements.push_back(stmt);
    
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;
    
    for (int i = 0; i < 32; i++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)i, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym, label2pc);
        warp.add_thread(std::move(thread), i);
    }
}

TEST_CASE("exec_mask: restored after reconvergence", "[exec_mask][bug]") {
    WarpContext warp;
    setup_warp_with_threads(warp);
    
    // 初始状态: 全活跃
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);
    
    // 模拟分歧分支
    // PC=10: bra.cond target=20, reconvergence=30
    warp.handle_branch(
        "%p1", false,    // predicate
        20,              // target_pc
        30,              // reconvergence_pc
        10               // current_inst_pc
    );
    
    // 分歧后: exec_mask 应为 taken_mask (假设线程 0-15 taken)
    uint32_t exec_mask_after_branch = warp.get_exec_mask();
    REQUIRE(exec_mask_after_branch != 0xFFFFFFFF);
    REQUIRE(exec_mask_after_branch == 0xFFFF);  // 假设 lanes 0-15 taken
    
    // 模拟所有线程到达收敛点
    for (int i = 0; i < 32; i++) {
        warp.set_thread_pc(i, 30);
    }
    
    // 检查收敛
    warp.check_reconvergence();
    
    // ⚠️ 关键验证: 收敛后 exec_mask 应恢复为 0xFFFFFFFF
    REQUIRE(warp.get_simt_stack().empty());
    REQUIRE(warp.get_exec_mask() == 0xFFFFFFFF);  // 当前失败!
}
```

#### 测试 B4: 退出线程的收敛处理 (检测关键 Bug)

```cpp
// test_simt_stack_entry.cpp
#include "catch_amalgamated.hpp"
#include "ptxsim/simt_stack.h"
#include "ptxsim/thread_state.h"
#include <array>

using namespace ptxsim;

TEST_CASE("SIMTStackEntry: exited threads handling", "[simt_entry][bug]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    
    // 初始化线程
    for (int i = 0; i < 32; i++) {
        threads[i].pc = 0;
        threads[i].is_active = true;
        threads[i].is_exited = false;
    }
    
    // 创建 SIMT 栈条目: 所有线程都应收敛到 PC=20
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.active_mask = 0xFFFF0000;  // lanes 16-31 taken
    entry.return_mask = 0xFFFFFFFF;  // 所有线程
    entry.return_pc = 20;
    
    stack.push(entry);
    
    // 模拟: lanes 0-15 执行了 exit (taken 路径)
    for (int i = 0; i < 16; i++) {
        threads[i].is_exited = true;
        threads[i].is_active = false;
        // 这些线程的 PC 仍然是 0，永远不会到 20
    }
    
    // 模拟: lanes 16-31 到达收敛点
    for (int i = 16; i < 32; i++) {
        threads[i].pc = 20;
    }
    
    // ⚠️ 关键验证: 应该收敛 (退出线程不应阻塞)
    bool converged = stack.check_reconvergence(threads);
    
    REQUIRE(converged == true);  // 当前失败!
    REQUIRE(stack.empty() == true);
}
```

---

## Phase 4: 架构增强 (P1-P2 - 中期)

### 4.1 屏障完成后 SIMT 栈清理

**问题**: 屏障完成后未检查 SIMT 栈收敛  
**严重性**: 🟡 中 - 可能导致栈残留

#### 分析

当前屏障完成流程 (`barrier.cpp:143-175`):
```
1. wbar.arrive(lane_id)
2. if wbar.is_complete():
   - 设置所有线程 PC = reconvergence_pc
   - update_pc_stack()
   - 重置状态
   - wbar.reset()
   - ❌ 没有调用 check_reconvergence()
```

#### 修改方案

**方案 A**: 在屏障 handler 中调用 (推荐)

```cpp
// barrier.cpp:169 - 在 barrier 完成后添加:

// 屏障完成后，检查 SIMT 栈收敛
if (warp_ctx->get_simt_stack().empty() == false) {
    warp_ctx->check_reconvergence();
    PTX_DEBUG_EMU("bar.warp.sync: Checked SIMT stack reconvergence after barrier");
}
```

**方案 B**: 在 warp 调度器中统一检查 (已在 sm_context.cpp:204,229 实现)

当前 `sm_context.cpp` 已经在指令执行后调用 `check_reconvergence()`，
但仅限于 `S_BRA` 指令。需要扩展到 barrier 指令。

```cpp
// sm_context.cpp:203 - 修改前:
if (stmt->type == S_BRA) {
    next_warp->check_reconvergence();
}

// 修改后:
if (stmt->type == S_BRA || stmt->type == S_BAR || 
    stmt->type == S_BAR_WARP_SYNC) {
    next_warp->check_reconvergence();
}
```

**推荐**: 方案 B (更统一，减少代码重复)

---

### 4.2 添加 SIMT 调试工具

**目标**: 提供 SIMT 栈可视化工具，辅助调试

#### 新增文件
- `include/ptxsim/simt_debug.h`
- `src/ptxsim/core/simt_debug.cpp`

#### API 设计

```cpp
// simt_debug.h
namespace ptxsim {

class SimtDebugger {
public:
    // 打印 SIMT 栈状态
    static void print_simt_stack(const WarpContext& warp);
    
    // 打印线程 PC 分布
    static void print_thread_pcs(const WarpContext& warp);
    
    // 打印执行掩码
    static void print_exec_mask(const WarpContext& warp);
    
    // 完整状态快照
    static void dump_warp_state(const WarpContext& warp, std::ostream& os);
    
    // 检测常见问题
    struct SimtIssue {
        enum Severity { Warning, Error, Critical };
        Severity severity;
        std::string description;
    };
    static std::vector<SimtIssue> diagnose(const WarpContext& warp);
};

}
```

#### 使用示例

```cpp
// 在 sm_context.cpp 中:
if (ptxsim::DebugConfig::get().is_simt_debug_enabled()) {
    ptxsim::SimtDebugger::print_simt_stack(*next_warp);
    auto issues = ptxsim::SimtDebugger::diagnose(*next_warp);
    for (const auto& issue : issues) {
        PTX_WARN_EMU("SIMT Issue: %s", issue.description.c_str());
    }
}
```

---

### 4.3 性能优化 (长期)

#### 4.3.1 减少 execute_warp_instruction() 中的冗余同步

**当前瓶颈**: `execute_warp_instruction()` (warp_context.cpp:146-196) 对每个活跃线程执行 `sync_from_warp_state()` → 执行 → `sync_to_warp_state()`。32 线程 × 2 次同步 = 每指令 64 次函数调用，即使只有 1 个线程活跃。

**优化方向**:
1. 只在 PC 或状态实际变化时才同步（当前全量同步）
2. 非分歧执行时批量设置 PC 而非逐线程 sync

```cpp
// 优化思路: 非分歧路径避免逐线程 sync
if (!warp_state.is_divergent()) {
    // 所有线程相同 PC → 批量设置，避免 sync_from/sync_to 开销
    for (auto& thread : active_threads) {
        thread->simple_execute(stmt);  // 跳过 warp_state 同步
    }
    advance_all_threads(pc + 1);
} else {
    // 分歧路径: 保持现有逐线程 sync 逻辑
    // ...
}
```

#### 4.3.2 ThreadState 缓存行优化 (低优先级)

`ThreadState` 数组 32×~20B = 640B，可以放入 L1 缓存。真正的开销在 `execute_warp_instruction()` 中的函数调用开销，而非数据结构布局。此优化在 SIMT 栈和 exec_mask 修复完成后再评估。

---

## 实施里程碑

### Milestone 1: Bug 修复完成 (Week 1)

- [ ] 1.1 exec_mask 收敛恢复修复
- [ ] 1.2 退出线程收敛检查修复
- [ ] 1.3 WarpState 未使用字段删除
- [ ] 所有现有测试通过 (无回归)

**验收标准**: `ctest` 全部通过，新增 2 个 bug 修复测试通过

---

### Milestone 2: 代码清理完成 (Week 2)

- [ ] 2.1 双重 PC 管理统一
  - [ ] 调用点审计 (4 个调用点)
  - [ ] 标记废弃方法
  - [ ] 更新调用点
  - [ ] 删除 pc_stacks
  - [ ] 删除废弃方法
- [ ] 2.2 SIMT 栈深度限制
- [ ] 2.3 统一 active_mask 数据源 (ISSUE-004)
- [ ] 2.4 废弃 WarpContext::pc 字段 (ISSUE-006)
- [ ] 2.5 统一 is_lane_active / is_lane_schedulable (ISSUE-005)
- [ ] 测试覆盖率报告生成

**验收标准**: `clang-tidy` 无警告，代码行数减少 10%

---

### Milestone 3: 核心测试完成 (Week 3-4)

- [ ] Phase 1 测试 (A-D 组，26 用例)
- [ ] Phase 2 测试 (E-H 组，21 用例)
- [ ] 测试覆盖率 ≥ 85%

**验收标准**: `ctest -L simt` 全部通过

---

### Milestone 4: 集成测试与文档 (Week 5)

- [ ] Phase 3 测试 (I 组，6 用例)
- [ ] 集成测试全部通过
- [ ] 更新架构文档

**验收标准**: 
- 测试覆盖率 ≥ 90%
- `docs/architecture/SIMT-ARCHITECTURE-V2.md` 更新
- 新增测试覆盖率报告

---

## 风险控制

### 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Bug 修复引入回归 | 中 | 高 | 每步运行全量测试 |
| 删除 pc_stacks 影响未知调用 | 低 | 高 | 全局搜索 + 编译验证 |
| 测试用例设计错误 | 中 | 中 | 代码审查 + 手动验证 |
| SIMT 栈深度限制影响现有内核 | 低 | 中 | 先跑 benchmark 检测 |

### 回滚策略

- 每个 Phase 创建独立 git 分支
- Phase 1-2 改动可合并，Phase 3-4 可选
- 保留旧测试文件 30 天

---

## 验收标准

### 功能验收

- [ ] 所有 7 个架构改进实施完成
- [ ] 53 个新测试用例全部通过
- [ ] 现有测试 100% 通过 (无回归)
- [ ] clang-format + clang-tidy 无警告

### 质量验收

- [ ] 测试覆盖率 ≥ 90% (SIMT 相关代码)
- [ ] 代码重复率 < 5%
- [ ] 圈复杂度 < 10 (新增函数)
- [ ] 文档更新完整

### 性能验收

- [ ] 无性能回归 (benchmark 对比 < 5% 差异)
- [ ] SIMT 栈操作开销 < 1 cycle/instruction
- [ ] 内存使用无显著增长

---

## 附录

### A. 文件变更清单

| 文件 | 变更类型 | Phase |
|------|---------|-------|
| `include/ptxsim/warp_state.h` | 删除字段 | 1 |
| `include/ptxsim/warp_context.h` | 删除方法 + 废弃标记 | 2 |
| `include/ptxsim/simt_stack.h` | 添加常量 | 2 |
| `src/ptxsim/core/warp_context.cpp` | 修复逻辑 + 统一数据源 | 1, 2 |
| `src/ptxsim/core/simt_stack.cpp` | 修复逻辑 | 1 |
| `src/ptxsim/core/sm_context.cpp` | 扩展检查 | 4 |
| `src/ptxsim/instructions/barrier.cpp` | 可选修改 | 4 |
| `tests/CMakeLists.txt` | 添加测试 | 3 |
| `tests/test_simt_stack_catch2.cpp` | 新增 | 3 |
| `tests/test_simt_stack_entry.cpp` | 新增 | 3 |
| `tests/test_warp_state.cpp` | 新增 | 3 |
| `tests/test_handle_branch_integration.cpp` | 新增 | 3 |
| `tests/test_barrier_simt_integration.cpp` | 新增 | 3 |
| `tests/test_exec_mask.cpp` | 新增 | 3 |
| `tests/test_pc_management_advanced.cpp` | 新增 | 3 |
| `tests/test_sync_mechanism.cpp` | 新增 | 3 |
| `tests/test_simt_integration.cpp` | 新增 | 3 |
| `tests/test_active_mask_consistency.cpp` | **新增** (Phase 2) | 3 |

### B. 测试用例索引

| ID | 测试名称 | 文件 | 检测 Bug |
|----|---------|------|---------|
| A1-A8 | SIMT Stack 基础 | test_simt_stack_catch2.cpp | 栈操作错误 |
| B1-B6 | SIMTStackEntry 收敛 | test_simt_stack_entry.cpp | **退出线程阻塞** |
| C1-C7 | WarpState 状态 | test_warp_state.cpp | 状态初始化 |
| D1-D5 | handle_branch 集成 | test_handle_branch_integration.cpp | 分支逻辑 |
| E1-E4 | Barrier + SIMT | test_barrier_simt_integration.cpp | **栈清理** |
| F1-F6 | exec_mask 管理 | test_exec_mask.cpp | **收敛后未恢复** |
| G1-G5 | PC 管理高级 | test_pc_management_advanced.cpp | PC 不一致 |
| H1-H6 | sync 机制 | test_sync_mechanism.cpp | **覆盖分支 PC** |
| I1-I6 | 集成场景 | test_simt_integration.cpp | 集成 bug |

### C. 参考文档

- [SIMT Architecture v2.0](../architecture/SIMT-ARCHITECTURE-V2.md)
- [SIMT Divergence Review](simt-architecture-divergence-review.md)
- [SIMT Architecture Review](simt-architecture-review-and-test-plan.md)
- [PTX ISA 9.1 - Control Flow](https://docs.nvidia.com/cuda/ptx-isa-9.1/)

---

**计划制定时间**: 2026-05-04
**最后更新**: 2026-05-05
**计划状态**: ✅ 已完成
**完成内容**:
- Phase 1: Bug 修复 (exec_mask 恢复、退出线程收敛、pc_stack 字段删除)
- Phase 2: 代码清理 (统一数据源、添加 MAX_DEPTH、废弃旧 API)
- Phase 3: 35 个新测试用例
- Phase 4: 屏障集成 (check_reconvergence 扩展到 S_BAR)

