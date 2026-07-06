## ADDED Requirements

### Requirement: docs/README.md 索引覆盖所有子目录

`docs/README.md` 的 §文档导航 表格标题 SHALL 列出的子目录数与 `ls -d docs/*/ | wc -l` 的实际子目录数一致。

#### Scenario: 索引数字与文件系统一致

- **WHEN** 运行 `ls -d docs/*/ | wc -l` 得到 N
- **THEN** `docs/README.md` §文档导航 标题中的数字 SHALL 等于 N

### Requirement: 索引表格包含所有 docs/ 子目录

`docs/README.md` 的 §文档导航 表格 SHALL 为 `docs/` 下的每个一级子目录包含一个条目。

#### Scenario: 所有子目录出现在索引中

- **WHEN** 检查 `docs/` 下的一级子目录列表
- **THEN** 每个子目录 MUST 在索引中有一个对应的表格行