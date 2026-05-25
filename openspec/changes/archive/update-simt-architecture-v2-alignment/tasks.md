# Tasks: update-simt-architecture-v2-alignment

## 任务清单

### Task 1: 代码审查 - 关键函数识别 ✅
**时间**: 45 分钟
**验证**: 输出代码 vs 文档对照表

MUST:
- [x] 读取 `src/ptxsim/core/warp_context.cpp` 找到：
  - [x] `advance_thread_pc()` / `advance_all_threads()`
  - [x] `get_statement_at(target_pc)`
  - [x] `sync_from_warp_state()` / `sync_to_warp_state()`
- [x] 读取 `src/ptxsim/core/thread_context.cpp` 找到同步函数
- [x] 读取 `src/ptxsim/core/sm_context.cpp` 找到调度逻辑

**结果**: §0.2 已包含所有关键函数文档

### Task 2: 文档更新 - 添加关键 API 列表 ✅
**时间**: 30 分钟
**验证**: 文档审查通过

MUST:
- [x] 在 SIMT-ARCHITECTURE-V2.md 添加 "关键 API 列表" 章节
  - [x] 附录 D 已包含所有关键函数
- [x] 记录每个函数的功能、参数、返回值
- [x] 包含代码示例

**结果**: 附录 D.1-D.6 完整记录所有关键 API

### Task 3: 验证设计决策状态 ✅
**时间**: 30 分钟
**验证**: 所有决策标记为 ✅/⚠️/❌

NOTE:
- [x] 检查 §7 的每个设计决策
- [x] 确认实现状态（已完成/部分完成/未实现）
- [x] 更新状态徽章

**结果**: §0.3 表格显示所有决策 ✅

### Task 4: 与 ADR-0014 对齐 ✅
**时间**: 15 分钟
**验证**: 文档内 ADR 引用一致

- [x] 检查文档中 ADR-0014 的引用 - 当前文档无 ADR-0014 引用
- [x] 更新任何过时引用 - 无
- [x] 验证 ADR 中描述与文档一致
- [x] 在 §11 添加 ADR-0014 引用表格
- [x] 在 docs/adr/README.md 添加 ADR-0014 到 Proposed 列表

**结果**: 已添加 ADR-0014 相关引用

### Task 5: Peer Review
**时间**: 30 分钟
**验证**: 至少一个开发者 approve

NOTE:
- [ ] 请求其他开发者 review 文档
- [ ] 根据反馈更新文档
- [ ] 最终合并到 main

**状态**: 待完成