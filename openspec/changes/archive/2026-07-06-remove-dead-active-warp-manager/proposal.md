## Why

`ActiveWarpManager` 模块（`include/ptxsim/active_warp_manager.h` 36 LOC + `src/ptxsim/core/active_warp_manager.cpp` 118 LOC 全实现）经过审计已确认 **0 生产调用方**，且与权威的 `WarpScheduler`（`RoundRobinWarpScheduler` / `GreedyWarpScheduler`）功能重叠。

### 实证（grep 验证）

```bash
$ grep -rn "ActiveWarpManager\|active_warp_manager" src/ include/ tests/ | grep -v "active_warp_manager\.\(h\|cpp\)"
src/CMakeLists.txt:77:    ptxsim/core/active_warp_manager.cpp
```

唯一引用 = 1 行 CMake 配置。`sm_context.cpp` 使用的是 `warp_scheduler`（`RoundRobinWarpScheduler`），共 8 个调用点（`add_warp` / `schedule_next` / `all_warps_finished` 等），证明 `WarpScheduler` 是唯一权威调度器。

清理此模块消除 ~154 LOC 死代码 + 避免维护者误用 `ActiveWarpManager` 而非 `WarpScheduler`。

## What Changes

- **删除 `include/ptxsim/active_warp_manager.h`** （36 LOC）
- **删除 `src/ptxsim/core/active_warp_manager.cpp`** （118 LOC）
- **修改 `src/CMakeLists.txt:77`**：从 SOURCES 列表移除 `ptxsim/core/active_warp_manager.cpp`
- **同步 `src/ptxsim/core/AGENTS.md`**：移除 `ActiveWarpManager` 引用（如适用）
- **同步 `docs/audits/debt-audit-2026-07-02.md`**：标记 A2.5 RESOLVED
- **同步 `docs/roadmap/post-phase3-debt-roadmap.md`**：从剩余债务列表移除 ActiveWarpManager 条目

**BREAKING**: 无 — 0 调用方意味着无外部依赖。

## Capabilities

### New Capabilities

- `dead-active-warp-manager-removal`: 删除 ActiveWarpManager 死代码模块（~154 LOC）

### Modified Capabilities

无 — 不影响任何 spec 级行为。

## Impact

**受影响的代码/文件**：

| 文件 | 改动 | 影响 |
|------|------|------|
| `include/ptxsim/active_warp_manager.h` | 删除 | 36 LOC |
| `src/ptxsim/core/active_warp_manager.cpp` | 删除 | 118 LOC |
| `src/CMakeLists.txt:77` | 移除 source line | 1 行 |
| `src/ptxsim/core/AGENTS.md` | 同步（如适用）| ≤5 行 |
| `docs/audits/debt-audit-2026-07-02.md` | 标记 RESOLVED | 1 行 |
| `docs/roadmap/post-phase3-debt-roadmap.md` | 移除条目 | 1 行 |

**受影响的 ADR**：
- 无直接 ADR 影响

**权威调度器确认**：
- ✅ `WarpScheduler`（`RoundRobinWarpScheduler`）由 `sm_context.cpp` 8 处调用，权威
- ❌ `ActiveWarpManager` 0 调用方，删除

**测试覆盖**：
- 现有测试无回归（grep 验证 0 调用方）
- `./scripts/sanity.sh --quick` 验证编译通过 + ctest PASS

**回归风险**：
- 🟢 极低：0 调用方意味着删除无行为影响

**Lessons-learned 集成**：
- ✅ Checklist E（artifacts 必 tracked）
- ✅ Checklist F（git verify）
- ✅ Checklist G（lifecycle）：新 change + archive