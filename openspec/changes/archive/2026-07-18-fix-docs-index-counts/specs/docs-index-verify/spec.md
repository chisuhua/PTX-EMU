# docs-index-verify

## ADDED Requirements

### Requirement: docs/README.md 导航表文件计数准确
docs/README.md §文档导航 表格的 "文档数" 列应与 `find docs/<dir> -type f | wc -l` 结果一致。

### Requirement: docs/adr/README.md ADR 索引完整
docs/adr/README.md 的 "ADR 总数" 与索引表条目数应与实际 `docs/adr/NNNN-*.md` 文件数一致。缺失的 ADR-0013 必须被列入索引表。

### Requirement: docs/archive/README.md 子目录计数准确
docs/archive/README.md 的子目录表格的 "文档数" 列应与 `find docs/archive/<subdir> -type f | wc -l` 结果一致。所有存在的子目录必须被列入表格。
