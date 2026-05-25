# BUG-SIMT-001: Divergent Warp 单 Cycle 执行多条不同 PC 指令

## 严重程度
**高** — 功能正确性违背 SIMT 硬件基本约束

---

## 问题现象

同一 warp 在同一 cycle 内发射了两条不同 PC 的指令：

```
Cycle 9: SM 0 Warp 0 PC=8   mov.b32%r3,100;      ← not-taken 路径
Cycle 9: SM 0 Warp 0 PC=11  $L__BB0_2:           ← taken 路径
```

---

## 根因定位

**文件**: `src/ptxsim/core/sm_context.cpp`
**行号**: 219–257
**函数**: `SMContext::exe_once()`

```cpp
} else if (!lanes_by_pc.empty()) {
    // Divergent path: execute each PC group separately
    for (const auto& [pc, lanes] : lanes_by_pc) {  // ← BUG: 循环内执行所有 PC 组
        int sample_lane = lanes[0];
        ThreadContext* sample_thread = next_warp->get_thread(sample_lane);

        if (sample_thread && pc >= 0 && pc < sample_thread->statements_size()) {
            StatementContext* stmt = sample_thread->get_statement_at(pc);
            if (stmt) {
                next_warp->execute_warp_instruction(*stmt, pc);  // 每个 PC 组执行一次
            }
        }
    }
    // 所有组执行完后统一做 reconvergence 检查
    if (next_warp && !next_warp->get_simt_stack().empty()) {
        while (next_warp->check_reconvergence()) { }
    }
}
```

---

## 调用链

```
SMContext::exe_once()                          [sm_context.cpp:126]
  └─ next_warp->get_lanes_by_pc()             [warp_context.cpp:327]
       返回 std::map<int, std::vector<int>>
       key = PC, value = 在该 PC 的 lane 列表
  ├─ lanes_by_pc.size() == 1 → Fast path    [sm_context.cpp:189-218]
  │    只执行一个 PC 组，然后检查 reconvergence
  └─ lanes_by_pc.size() > 1 → Divergent path [sm_context.cpp:219-257]
       for 循环执行所有 PC 组 ← BUG
```

---

## `get_lanes_by_pc()` 实现

```cpp
// warp_context.cpp:327-340
std::map<int, std::vector<int>> WarpContext::get_lanes_by_pc() const {
    std::map<int, std::vector<int>> pc_to_lanes;
    for (int lane = 0; lane < WARP_SIZE; lane++) {
        if (lane < (int)threads.size() && threads[lane] != nullptr &&
            warp_state.threads[lane].is_active &&
            !warp_state.threads[lane].is_exited) {
            int pc = warp_state.threads[lane].pc;
            pc_to_lanes[pc].push_back(lane);
        }
    }
    return pc_to_lanes;
}
```

---

## 正确行为 vs 当前行为

| 方面 | 正确行为 (NVIDIA) | 当前行为 (BUG) |
|------|------------------|---------------|
| 每 cycle 指令数 | 1 条指令 | N 条指令（N = PC 组数） |
| 时间模型 | divergent 串行消耗 cycle | divergence "免费" |
| 性能影响 | warp divergence 导致时间倍增 | 完全隐藏 |
| reconvergence | 每步检查 | 延迟到所有组执行完 |

---

## 修复方案

**核心**: 每 cycle 只选一个 PC 组执行

```cpp
} else if (!lanes_by_pc.empty()) {
    // Divergent path: execute ONLY ONE PC group per cycle
    auto it = lanes_by_pc.begin();  // Lowest PC first
    int pc = it->first;
    const auto& lanes = it->second;

    int sample_lane = lanes[0];
    ThreadContext* sample_thread = next_warp->get_thread(sample_lane);

    if (sample_thread && pc >= 0 && pc < sample_thread->statements_size()) {
        StatementContext* stmt = sample_thread->get_statement_at(pc);
        if (stmt) {
            next_warp->execute_warp_instruction(*stmt, pc);

            // 执行后立即检查 reconvergence
            if (stmt->type == S_BRA || stmt->type == S_BAR ||
                stmt->type == S_BAR_WARP_SYNC) {
                while (next_warp->check_reconvergence()) { }
            }
        }
    }
}
```

---

## 影响评估

| 测试 | 当前行为 | 修复后 |
|------|---------|--------|
| `test_divergence` | 功能正确但 cycle 数压缩 | 功能正确，cycle 数反映真实 divergence |
| `test_barrier_*` | 可能因 cycle 数变化受影响 | 需要调整断言 |

---

## 参考文件

| 文件 | 行号 | 说明 |
|------|------|------|
| `src/ptxsim/core/sm_context.cpp` | 219-257 | Bug 所在 |
| `src/ptxsim/core/warp_context.cpp` | 327-340 | `get_lanes_by_pc()` |
| `src/ptxsim/core/warp_context.cpp` | 98-127 | `check_reconvergence()` |
| `src/ptxsim/core/simt_stack.cpp` | 75-88 | 栈弹出逻辑 |
| `include/ptxsim/warp_context.h` | 104 | 声明 |

---

## 相关文档

- [ADR-0014: Independent Thread Scheduling (ITS) 支持](./adr/0014-independent-thread-scheduling.md)

---

## 修复状态

| 项目 | 状态 |
|------|------|
| **Bug ID** | BUG-SIMT-001 |
| **Status** | ✅ 已修复 |
| **Commit** | `src/ptxsim/core/sm_context.cpp` |
| **Date** | 2026-05-13 |
| **Fix Description** | 将 divergent path 从 `for` 循环（所有 PC 组在一个 cycle 内执行）改为单 PC 组执行（Lowest PC first） |

### 修复详情

**修改文件**: `src/ptxsim/core/sm_context.cpp`

**修改位置**: 第 219-262 行

**修改内容**:
- 移除了 `for (const auto& [pc, lanes] : lanes_by_pc)` 循环
- 改为 `auto it = lanes_by_pc.begin()` 选择最低 PC 的 PC 组
- 将 reconvergence 检查从"所有组执行后统一检查"改为"当前 PC 组执行后立即检查"

**修复后行为**:
- 每个 cycle 只执行一个 PC 组
- Divergent warp 需要多个 cycle 来完成所有路径
- Reconvergence 检测更及时

### 验证结果

- [x] `test_divergence` - cycle 数增加（符合预期）
- [x] `test_barrier_*` - 断言通过
- [x] `./tests/ptx/test_all_ptx.sh` - 31/31 PTX 语法测试通过
- [x] `./scripts/sanity.sh` - 完整 sanity 检查通过（0 failures）