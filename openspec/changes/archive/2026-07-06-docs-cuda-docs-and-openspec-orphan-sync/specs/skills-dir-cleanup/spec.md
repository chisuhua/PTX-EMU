## ADDED Requirements

### Requirement: docs/skills/ 不包含过期技能副本

`docs/skills/` 目录 SHALL 仅包含：
1. 人类可读导航 README（`docs/skills/README.md`）
2. 非技能类技术参考文档（`post-dominator-algorithm.md`、`simt-reconvergence.md`）

任何在 `.opencode/skills/` 已有完整副本的技能 SHALL NOT 在 `docs/skills/` 中保留过期副本。

#### Scenario: 迁移完成的技能副本被删除

- **WHEN** 技能已在 `.opencode/skills/<name>/SKILL.md` 中维护
- **THEN** `docs/skills/<name>/` 或 `docs/skills/<name>.md` SHALL NOT 存在

#### Scenario: 技术参考文档被保留

- **WHEN** 检查 `docs/skills/` 目录
- **THEN** `post-dominator-algorithm.md` 和 `simt-reconvergence.md` SHALL 保留

### Requirement: 已禁用技能在 docs/skills/ 中不保留副本

已禁用并移至 `.opencode/skills.disable/` 的技能 SHALL NOT 在 `docs/skills/` 中保留文件副本。`docs/skills/README.md` 中 `[disabled]` 标记 SHALL 链接至 `.opencode/skills.disable/<name>/`。

#### Scenario: 已禁用技能文件被清理

- **WHEN** 技能已移至 `.opencode/skills.disable/<name>/`
- **THEN** `docs/skills/<name>/` SHALL NOT 存在