## Context

`debt-audit-2026-07-02.md` §2.3 P1-D 与 §3.3 P2-D 标识了 6 条文档债务（D-1~D-6）。其中 D-1 与 D-3 在 parser-completeness 系列 change 后计数已自动对齐（commit `aed66e9` / `918891d` — `docs/README.md` 重建为 16 子目录索引），但需实证 re-verify。D-4~D-6 仍为 active debt。

**约束**：
- Checklist G: 已归档 change 不可 amend，任何修补必须通过新建 `Ref:` 链接
- Checklist F: 审计修正必须引用 git commit hash，区分 active vs stale debt
- D-4 retroactive design.md 应**简要**（1-2 页以内），描述 what was done（非 how），以 `git log` commit body 为原始材料
- `docs/skills/` 的 `README.md` 明确声明"可加载技能已迁移至 `.opencode/skills/`"，但过期副本未清理 → D-5

## Goals / Non-Goals

**Goals:**
- **D-1**: 实证验证 `docs/README.md` 索引覆盖全部 `docs/` 子目录（`ls -d docs/*/ | wc -l` vs 标题声称）
- **D-4**: 为 5 个缺 `design.md` 的已归档 change 合成 retroactive design.md，从 `git log` commit body 重建
- **D-5**: 删除 `docs/skills/` 下 3 个过期技能副本（ptx-debug/, ptxir-serialization/, ptx-grammar-modification.md）+ `three-mode-testing/`
- **D-6**: 将 ERRATA 8 项修正内联到主审计，ERRATA 文件标记"已合并"

**Non-Goals:**
- 不修改任何 `.cpp/.h` 源代码
- 不 amend 已归档 OpenSpec change（per Checklist G）
- 不处理 `docs/skills/` 内 `three-mode-testing/` 的 `__pycache__/` 垃圾（已在 `.opencode/skills.disable/` 维护）
- 不创建新的 ADR
- 不修改 `AGENTS.md` 或根 `README.md`（D-1 scope 仅为 `docs/README.md` 索引验证）

## Decisions

### Decision 1: Retroactive design.md 通过 `git log` 合成

**问题**: 5 个已归档 OpenSpec change 缺失 `design.md`，如何补充而不违反存档完整性？

**方案分析**:
- **A**: 直接 amend 已归档 change → ❌ 违反 Checklist G（Archived = 终态）
- **B**: 在归档目录外新建 `design.md` + `README.md` 中加 `Ref:` 链接 → ✅ 不触及归档内容
- **C**: 忽略不管 → ❌ 债务不消，新开发者无法理解设计决策

**选择**: **方案 B**。在 `openspec/changes/archive/<name>/` 同级创建 `design.md`，归档目录内添加 `README.md` 段落引用 retroactive design.md 的存在。这样归档内容完全未修改，历史完整。

**证据来源**: `git log --all --oneline -- <change-dir>/` 的 commit body 提供原始设计意图。

### Decision 2: ERRATA 合并策略

**问题**: 8 项事实错误应如何合并到主审计？

**方案分析**:
- **A**: 全文替换审计中的错误数字 → 破坏审计作为历史快照的完整性
- **B**: 在每个受影响段落旁添加 `**[ERRATA E1-E8]**` 标记 + 正确数字 → 保留审计原貌 + 提供正确信息
- **C**: 删除 ERRATA.md → 丢失勘误历史

**选择**: **方案 B**。受影响的 §0.2/§1.2/§2.2.1/§3.5/§8/§9.1 各段添加 inline `**[勘误: ...]**` 标记。ERRATA.md 保留并在顶部添加 `**[2026-07-06 已合并到主审计]**` 标记。

### Decision 3: D-5 删除策略

**问题**: `docs/skills/` 中 3 个过期技能副本应直接删除还是添加 deprecation 标记？

**方案分析**:
- **A**: 直接 `git rm -r` → 简单，但丢失迁移历史
- **B**: 添加 `DEPRECATED.md` stub → 保留迁移线索
- **C**: 移至 `docs/archive/` → 重量级，这些是副本非原创

**选择**: **方案 A**（直接删除）。理由：
1. `docs/skills/README.md` 已声明"可加载技能已迁移至 `.opencode/skills/`"
2. 这些是 `2026-05-26` 时间戳的旧 SKILL.md 副本，权威版本在 `.opencode/skills/`
3. 迁移已在 commit `14c8eeb` 中完成，删除是清理遗留

同时检查 `docs/skills/three-mode-testing/` 目录（已 disabled，含 `__pycache__/` 垃圾）— 该目录在 `docs/skills/README.md` 中标记为 `[disabled]`，但实际文件仍存在；因 `.opencode/skills.disable/three-mode-testing/` 已有完整副本，`docs/skills/three-mode-testing/` 一并删除。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| Retroactive design.md 可能不完整（git log commit body 有限） | 标注 `Retroactive synthesis from git log` header + "最佳努力"声明；不声称与原设计完全一致 |
| 5 个 design.md 合成可能遗漏关键决策 | 每个 design.md 包含 `git log --oneline` 引用的 commit 列表，供读者自行追溯 |
| 误删 docs/skills/ 中的非技能技术文档 | D-5 删除范围严格限定为 3 个已知过期技能目录 + 1 个已禁用目录；`docs/skills/post-dominator-algorithm.md` 和 `simt-reconvergence.md` 保留（它们是 docs/skills/README.md 明确列出的技术参考） |
| ERRATA 合并可能破坏审计的 git 历史可读性 | 采用 inline 标记而非全文替换；审计原貌完全保留，ERRATA 标记仅补充正确信息 |
| D-1 若已验证 16=16，实际无变更 → Phase 1 工作量为 0 | Phase 1 明确标注"实证优先 — 若一致则仅记录结论"，不预设必须修改 |

## Design-Time Checklist (per ptx-lessons-learned & openspec-propose)

- [x] Checklist F（Debt audit 撰写）：审计前已 `git log --all --oneline -- <path>` 验证 6 个归档 change 的文件存在状态
- [x] Checklist G（OpenSpec lifecycle）：确认 5 个孤儿 change 已归档（Archived），不 amend，通过 `Ref:` 链接
- [x] Checklist H（Pre-implementation Review 强制项）：已通过 `ls -d docs/*/` 验证 D-1 子目录数（16=16），已通过 `find ... -type f` 验证 6 个孤儿 change 的文件列表，已通过 `ls .opencode/skills/` 验证 D-5 技能计数（18）
- [x] 实证验证：所有 proposal 假设均以 `bash` 命令输出为证（见 Change Goal background context）
- [x] 不修改任何 `.cpp` / `.h` 文件 — 此 change 纯文档操作