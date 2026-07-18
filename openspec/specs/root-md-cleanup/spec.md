# root-md-cleanup

## ADDED Requirements

### Requirement: 根目录无冗余 AI 平台配置文件
项目根目录应仅保留 AGENTS.md 作为唯一 AI agent 入口文件。内容完全相同的冗余文件（GEMINI.md、QODER.md）必须被删除。

### Requirement: AGENTS.md 无重复章节
AGENTS.md 中每个逻辑章节应仅出现一次。OpenSpec 流程章节不应在两处重复出现。

### Requirement: 过期完成声明不留在活跃文档路径下
声称 "100% 完成" 的历史文档必须归档至 docs/archive/，避免误导新开发者。
