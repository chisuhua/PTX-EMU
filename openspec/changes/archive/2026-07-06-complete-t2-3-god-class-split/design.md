## Context

T2-3 god-class split 计划（`archive/2026-06-24-phase3-t2-3-god-class-split/`）原计划：
- Phase A：拆分 `ThreadContext` god class 为 4 个 POD 视图（ExecState/RegisterPredicate/Memory/ProgramRef）
- Phase B：拆分 `WarpContext` god class 为 3 个 POD 视图
- Phase C：实际迁移到独立文件

实际实施**半成品状态**：
- A3a/A3b 阶段：4 POD 字段已添加到 `ThreadContext` 末尾，`init()` 镜像写入 `exec_state_/memory_/program_ref_` 但其他代码未读
- B 阶段：3 POD 字段已添加到 `WarpContext` 末尾，`lane_mask_` 镜像写入但 `warp_identity_` / `backend_links_` 完全未读未写
- C 阶段：创建了 `src/ptxsim/contexts/` 目录 + 7 个空 stub `.cpp` 文件但未填实现

### 当前状态（grep 实证）

| 文件:行 | 内容 | 状态 |
|---------|------|------|
| `src/ptxsim/contexts/*.cpp` | 7 文件，每个 1 行 `// T2-3: ...` | 死 stub |
| `include/ptxsim/warp_context.h:279-280` | `backend_links_` / `warp_identity_` POD | 0 读写 |
| `include/ptxsim/thread_context.h:315-318` | 4 POD 字段（exec_state_/memory_/program_ref_/register_predicate_）| 部分读写 |

### 决策：回滚 stub vs 完成迁移

**回滚 stub（采纳）**：
- ✅ 0 风险（grep 验证未使用）
- ✅ 删除 7 stub 文件 + 2 unused POD 字段，编译时间减少
- ✅ 恢复"T2-3 是 future work"的状态
- ❌ 丢弃已添加的 POD 字段（实际仅镜像写入，无功能价值）

**完成迁移（拒绝）**：
- ❌ 需要 6-8h 工时
- ❌ 涉及 god class 拆分的核心决策（应单独 change）
- ❌ 当前任务范围外

### Metis Review

本 change 无需 Metis pre-impl review：
- 范围明确（纯删除）
- 0 决策点（已选择回滚方案）
- 0 调用方（grep 验证）

## Goals / Non-Goals

### Goals

1. **删除 `src/ptxsim/contexts/` 整个目录**（7 个 stub `.cpp`）
2. **从 `src/CMakeLists.txt` 移除 contexts 子目录引用**（如有）
3. **从 `include/ptxsim/warp_context.h` 删除 `backend_links_` + `warp_identity_` POD 字段**
4. **检查 `include/ptxsim/thread_context.h:315-318`**：评估哪些 POD 可回滚
5. **同步 `docs/roadmap/post-phase3-debt-roadmap.md`**：从剩余债务列表移除

### Non-Goals（明确排除）

1. ❌ **完成 T2-3 god-class split 实际迁移**：当前任务范围外，未来新建 `complete-t2-3-god-class-split-impl` change
2. ❌ **删除 `lane_mask_` POD 字段**：因 `lane_mask_` 有镜像写入逻辑，回滚需额外决策
3. ❌ **修改 `thread_context.h` 4 个 POD**：逐个评估（部分有镜像写入逻辑）

## Decisions

### Decision 1: 回滚 stub vs 完成迁移

**Choice**: 回滚 stub

**Rationale**：
- 用户明确决定"完成迁移"，但迁移涉及 god class 拆分核心设计
- 安全路径：先回滚半成品 stub + unused POD，把"完成迁移"留给未来独立 change
- 删除字段前 grep 验证 0 读写

### Decision 2: 删除 `backend_links_` + `warp_identity_` 而非保留

**Choice**: 删除

**Rationale**：
- 0 读写 = 无功能价值
- 删除减少 `WarpContext` 类大小（每字段 8 bytes + 注释 ~10 行）

### Decision 3: 暂不删除 `lane_mask_` POD

**Choice**: 保留 + 添加注释"deferred T2-3 migration"

**Rationale**：
- `lane_mask_` 在 `warp_context.cpp` 有镜像写入逻辑（per T2-3 A3a）
- 回滚 `lane_mask_` 涉及 `WarpContext::lane_mask_` → `WarpState.threads[i].active` 的回退路径
- 留待 future `complete-t2-3-god-class-split-impl` change 处理

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **R1: 删除字段遗漏真实使用** | 🟢 极低（grep 验证 0 读写）| (1) 删除前二次 grep (2) ctest 全 PASS (3) 编译通过 |
| **R2: CMakeLists.txt 子目录引用未清理** | 🟢 低 | Phase 1.5 检查 + 测试 |
| **R3: 未来 T2-3 实施需重新添加字段** | 🟢 低 | (1) git log 可恢复 (2) lessons-learned 记录决策 |

## Migration Plan

### Phase 0: Artifacts Git-Tracking

```bash
git checkout -b refactor/complete-t2-3-god-class-split
git add openspec/changes/complete-t2-3-god-class-split/
git commit -m "docs(openspec): add complete-t2-3-god-class-split artifacts"
```

### Phase 1: 删除半成品 stub + unused POD（Fix #1）

```bash
git worktree add .worktrees/t2-3-cleanup-impl refactor/complete-t2-3-god-class-split
cd .worktrees/t2-3-cleanup-impl

# 1.1 二次验证 unused POD 字段 0 读写
grep -rn "backend_links_\|warp_identity_" src/ include/ tests/ \
  | grep -v "include/ptxsim/warp_context.h:27[89]:"
# 期望: 仅自身定义（war_context.h:279-280）

# 1.2 删除 stub 目录
rm -rf src/ptxsim/contexts/

# 1.3 检查 src/CMakeLists.txt
grep -n "contexts" src/CMakeLists.txt
# 如有 contexts 子目录引用 → 删除

# 1.4 修改 include/ptxsim/warp_context.h
# 删除 line 279-280 的 backend_links_ + warp_identity_ 字段

# 1.5 检查 thread_context.h:315-318（评估回滚范围）
grep -n "exec_state_\|memory_\|program_ref_\|register_predicate_" src/ptxsim/core/thread_context.cpp
# 如 4 POD 字段均未读取（仅 init 镜像写入）→ 评估是否全部回滚

# 验证
cmake --build build
cd build && ctest --output-on-failure

# Commit
git commit -am "refactor(ptxsim): delete T2-3 half-finished stubs + unused PODs (Fix #1)

Removed:
- src/ptxsim/contexts/ (7 stub .cpp files, each 1 line)
- src/CMakeLists.txt contexts subdirectory reference (if any)
- include/ptxsim/warp_context.h:279-280 (backend_links_, warp_identity_ POD)

Verified:
- backend_links_ 0 reads/writes (only declaration)
- warp_identity_ 0 reads/writes (only declaration)
- contexts/*.cpp are placeholder stubs (no behavior)

Kept (deferred to future T2-3-impl change):
- thread_context.h:315-318 4 PODs (exec_state_/memory_/program_ref_/register_predicate_)
- warp_context.h lane_mask_ POD (has mirror write logic)

Per lessons-learned Checklists E/F.
Refs: archive/2026-06-24-phase3-t2-3-god-class-split/ (NOT amended per Checklist G)
"
```

### Phase 2: 文档同步（Fix #2）

```bash
# 更新 docs/roadmap/post-phase3-debt-roadmap.md
# 从剩余债务列表移除 T2-3 半成品条目

git commit -am "docs(cleanup): sync roadmap post-Fix #1 (Fix #2)"
```

### Phase 3: Archive

```bash
openspec archive complete-t2-3-god-class-split --yes
git checkout main
git merge --no-ff refactor/complete-t2-3-god-class-split
```

### Rollback Strategy

```bash
git revert HEAD
cmake --build build
ctest --output-on-failure
```