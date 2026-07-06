## ADDED Requirements

### Requirement: 已归档 change 缺失 design.md 应 retroactive 合成

对于已归档但缺少 `design.md` 的 OpenSpec change，SHALL 在归档目录同级创建 `design.md`，标注为 retroactive synthesis，不修改已归档目录内任何文件。

#### Scenario: Retroactive design.md 不修改归档内容

- **WHEN** 在 `openspec/changes/archive/<name>/` 同级创建 `design.md`
- **THEN** 归档目录内所有文件的 git hash SHALL 保持不变

#### Scenario: Retroactive design.md 包含原始 commit 引用

- **WHEN** 创建 retroactive `design.md`
- **THEN** 文件 SHALL 包含 `git log --oneline` 列表引用原始实施 commits

#### Scenario: Retroactive design.md 标注 synthesis 性质

- **WHEN** 创建 retroactive `design.md`
- **THEN** 文件头 SHALL 标注 "Retroactive synthesis from git log — not an original design document" + 创建日期

### Requirement: 已归档 change 不可被 amend

已归档 OpenSpec change（路径包含 `archive/`）的目录内容 SHALL NOT 被修改；任何修补 SHALL 通过新建 change + `Ref:` 链接实现。

#### Scenario: Archive amend 被禁止

- **WHEN** 检测到对 `openspec/changes/archive/` 下文件的修改提案
- **THEN** 提案 SHALL 被拒绝，推荐新建 `fix-*` change + `Ref: archive/<date>-<name>/`