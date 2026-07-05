## Why

T2-3 god-class split（`openspec/changes/archive/2026-06-24-phase3-t2-3-god-class-split/`）已归档但实际迁移**半成品状态**：

- `src/ptxsim/contexts/` 目录创建了 7 个 `.cpp` 文件，每个仅含 1 行 `// T2-3: POD is header-only data struct (no out-of-line definitions yet).` 占位注释，编译目标无功能
- `include/ptxsim/warp_context.h:279-280` 添加了 `backend_links_` + `warp_identity_` POD 成员，但**从未被读/写**
- `include/ptxsim/thread_context.h:315-318` 4 个 POD 成员添加，只有 `exec_state_` / `memory_` / `program_ref_` 在 `init()` 中被镜像写入

半成品状态造成：
1. 编译时间浪费（7 个空 `.cpp` 文件 + 7 次空编译）
2. 认知负担（读者会以为 `WarpContext::backend_links_` 是活跃字段）
3. "T2-3 已归档"的认知与实际状态不符

清理半成品状态：要么完成迁移（major refactor），要么回滚 stub（清理）。本次选择**回滚 stub**（更安全，不引入新设计决策）。

## What Changes

- **删除 `src/ptxsim/contexts/` 整个目录**（7 个 1 行 stub `.cpp` 文件）
- **从 `src/CMakeLists.txt` 移除 contexts 子目录引用**（如有）
- **从 `include/ptxsim/warp_context.h` 删除 `backend_links_` + `warp_identity_` POD 字段**（line 279-280）
- **检查 `include/ptxsim/thread_context.h:315-318`**：4 个 POD 字段如有未读/未写则评估是否回滚
- **同步 `docs/roadmap/post-phase3-debt-roadmap.md`**：从 §3.3 T2-3 半成品条目移除

**BREAKING**: 无 — 删除的字段从未被读/写（grep 验证）

## Capabilities

### New Capabilities

- `t2-3-half-finished-cleanup`: 清理 T2-3 god-class split 半成品状态（删除 7 stub .cpp + 2 unused POD 字段）

### Modified Capabilities

无 — 不影响任何 spec 级行为。

## Impact

**受影响的代码/文件**：

| 文件 | 改动 | 影响 |
|------|------|------|
| `src/ptxsim/contexts/*.cpp` | 删除 7 文件 | ~7 LOC（1 行 × 7） |
| `src/ptxsim/contexts/` | 删除整个目录 | （仅含 stub） |
| `src/CMakeLists.txt` | 移除 contexts 子目录引用 | ≤3 行 |
| `include/ptxsim/warp_context.h:279-280` | 删除 2 POD 字段 | 2 行 |
| `include/ptxsim/thread_context.h:315-318` | 评估回滚（如适用）| 0-4 行 |
| `docs/roadmap/post-phase3-debt-roadmap.md` | 移除条目 | 1 行 |

**受影响的 ADR**：
- 无直接 ADR 影响

**测试覆盖**：
- 现有测试无回归（grep 验证 0 引用）
- `./scripts/sanity.sh --quick` 验证编译通过 + ctest PASS

**回归风险**：
- 🟢 极低：删除的字段从未被读/写（grep 验证）

**Lessons-learned 集成**：
- ✅ Checklist G（lifecycle）：本 change 是新 change，**不 amend 已归档的 T2-3 change**
- ✅ Checklist E（artifacts 必 tracked）
- ✅ Checklist F（git verify）