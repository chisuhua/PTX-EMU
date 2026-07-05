## Context

`ActiveWarpManager` 是早期 PTX-EMU warp 调度器的替代实现，但从未替代 `WarpScheduler`。经过审计（2026-07-06）确认 `WarpScheduler`（具体为 `RoundRobinWarpScheduler`）是权威调度器。

### 当前状态（grep 实证）

| 文件:行 | 内容 | 状态 |
|---------|------|------|
| `include/ptxsim/active_warp_manager.h:1-36` | 完整头文件 | 死代码 |
| `src/ptxsim/core/active_warp_manager.cpp:1-118` | 完整实现（add_active_warp/get_next_warp/all_warps_finished 等）| 死代码 |
| `src/CMakeLists.txt:77` | `ptxsim/core/active_warp_manager.cpp` source line | 待删除 |

### 权威调度器对比

| 调度器 | 调用点 | 类型 | 状态 |
|--------|--------|------|------|
| `WarpScheduler`（基类）| 8 处 in `sm_context.cpp` | 抽象基类 | 权威 |
| `RoundRobinWarpScheduler` | 通过 `std::make_unique<RoundRobinWarpScheduler>()` line 23 | 具体实现 | 权威 |
| `GreedyWarpScheduler` | 通过 `set_warp_scheduler()` 注入 | 具体实现 | 备用 |
| `ActiveWarpManager` | **0 调用方** | 替代实现 | **删除** |

### Metis Review

本 change 无需 Metis pre-impl review：
- 范围明确（纯删除）
- 0 决策点（grep 实证无调用方）
- 决策记录：WarpScheduler 权威性已经在 `sm_context.cpp:23` 选择

## Goals / Non-Goals

### Goals

1. **删除 ActiveWarpManager 头文件** `include/ptxsim/active_warp_manager.h`（36 LOC）
2. **删除 ActiveWarpManager 实现** `src/ptxsim/core/active_warp_manager.cpp`（118 LOC）
3. **从 `src/CMakeLists.txt:77` 移除 source line**
4. **同步 `src/ptxsim/core/AGENTS.md`**：移除 ActiveWarpManager 引用（如适用）
5. **同步 2 个文档**：debt audit + roadmap

### Non-Goals（明确排除）

1. ❌ **保留 ActiveWarpManager 作为未来调度器替代**：违反"0 调用方 = 删除"原则；如未来需要轮询/RR 之外的策略，扩展 `WarpScheduler` 派生类
2. ❌ **合并 ActiveWarpManager 到 WarpScheduler**：两个 API 设计不同（`add_active_warp` vs `add_warp`），强行合并会增加复杂度

## Decisions

### Decision 1: 删除 vs 保留作为备用

**Choice**: 删除

**Rationale**：
- 0 调用方 = 无未来使用场景（调度需求已通过 `WarpScheduler` 抽象满足）
- 保留会增加维护负担（双重实现 = 双重 bug 风险）
- 如未来需要新策略，按 `WarpScheduler` 模式添加派生类（如 `PriorityWarpScheduler`）即可

### Decision 2: 同步 AGENTS.md

**Choice**: 同步（如适用）

**Rationale**：
- `src/ptxsim/core/AGENTS.md` 可能引用 ActiveWarpManager（需 re-verify）
- lessons-learned §21 强制：重大变更必须同步文档

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **R1: 隐藏依赖未发现** | 🟢 极低（grep 验证 0 引用）| (1) 删除前二次 grep (2) ctest 全 PASS (3) 编译通过 |
| **R2: 未来需要 ActiveWarpManager API** | 🟢 低 | (1) git log 可恢复（154 LOC 历史）(2) lessons-learned 记录决策 |
| **R3: AGENTS.md 遗漏更新** | 🟢 低 | Phase 2 文档同步 task |

## Migration Plan

### Phase 0: Artifacts Git-Tracking

```bash
git checkout -b refactor/remove-dead-active-warp-manager
git add openspec/changes/remove-dead-active-warp-manager/
git commit -m "docs(openspec): add remove-dead-active-warp-manager artifacts"
```

### Phase 1: 删除 ActiveWarpManager（Fix #1）

```bash
git worktree add .worktrees/active-warp-manager-impl refactor/remove-dead-active-warp-manager
cd .worktrees/active-warp-manager-impl

# 二次验证 0 调用方
grep -rn "ActiveWarpManager\|active_warp_manager" src/ include/ tests/ \
  | grep -v "active_warp_manager\.\(h\|cpp\)"
# 期望: 1 行 (CMakeLists.txt:77)

# 删除文件
rm include/ptxsim/active_warp_manager.h
rm src/ptxsim/core/active_warp_manager.cpp

# 修改 CMakeLists.txt
# 删除 line 77: ptxsim/core/active_warp_manager.cpp

# 验证
cmake --build build
cd build && ctest --output-on-failure

# Commit
git commit -am "refactor(ptxsim): delete dead ActiveWarpManager module (Fix #1)

Removed:
- include/ptxsim/active_warp_manager.h (36 LOC)
- src/ptxsim/core/active_warp_manager.cpp (118 LOC)
- src/CMakeLists.txt:77 source entry

Verified 0 production call sites:
- CMakeLists.txt:77 (only config-time reference)
- All sm_context.cpp scheduling goes through WarpScheduler (8 call sites)

Authoritative scheduler confirmed: RoundRobinWarpScheduler (sm_context.cpp:23).
ActiveWarpManager was an unused alternative implementation.

Per lessons-learned Checklists E/F."
```

### Phase 2: 文档同步（Fix #2）

```bash
# 检查 + 更新 AGENTS.md
grep -n "ActiveWarpManager\|active_warp_manager" src/ptxsim/core/AGENTS.md
# 如有引用 → 删除

# 更新 docs/audits/debt-audit-2026-07-02.md
# 标记 ActiveWarpManager RESOLVED（引用 commit hash）

# 更新 docs/roadmap/post-phase3-debt-roadmap.md
# 从剩余债务列表移除 ActiveWarpManager

git commit -am "docs(cleanup): sync AGENTS.md + audit + roadmap post-Fix #1 (Fix #2)

Per lessons-learned Checklists I:"
```

### Phase 3: Archive

```bash
openspec archive remove-dead-active-warp-manager --yes
git checkout main
git merge --no-ff refactor/remove-dead-active-warp-manager
```

### Rollback Strategy

```bash
git revert HEAD
cmake --build build
ctest --output-on-failure
```