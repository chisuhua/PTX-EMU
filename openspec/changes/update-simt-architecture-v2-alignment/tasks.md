# Tasks: update-simt-architecture-v2-alignment

## 任务清单

### Task 1: 代码审查 - 关键函数识别
**时间**: 45 分钟
**验证**: 输出代码 vs 文档对照表

MUST:
- 读取 `src/ptxsim/core/warp_context.cpp` 找到：
  - `advance_thread_pc()` / `advance_all_threads()`
  - `get_statement_at(target_pc)`
  - `sync_from_warp_state()` / `sync_to_warp_state()`
- 读取 `src/ptxsim/core/thread_context.cpp` 找到同步函数
- 读取 `src/ptxsim/core/sm_context.cpp` 找到调度逻辑

### Task 2: 文档更新 - 添加关键 API 列表
**时间**: 30 分钟
**验证**: 文档审查通过

MUST:
- 在 SIMT-ARCHITECTURE-V2.md 添加 "关键 API 列表" 章节
- 记录每个函数的功能、参数、返回值
- 包含代码示例

### Task 3: 验证设计决策状态
**时间**: 30 分钟
**验证**: 所有决策标记为 ✅/⚠️/❌

NOTE:
- 检查 §7 的每个设计决策
- 确认实现状态（已完成/部分完成/未实现）
- 更新状态徽章

### Task 4: 与 ADR-0014 对齐
**时间**: 15 分钟
**验证**: 文档内 ADR 引用一致

- 检查文档中 ADR-0014 的引用是否准确
- 更新任何过时引用
- 验证 ADR 中描述与文档一致

### Task 5: Peer Review
**时间**: 30 分钟
**验证**: 至少一个开发者 approve

NOTE:
- 请求其他开发者 review 文档
- 根据反馈更新文档
- 最终合并到 main