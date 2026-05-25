# Design: update-simt-architecture-v2-alignment

## 现状问题

`SIMT-ARCHITECTURE-V2.md` 与实际实现存在差异：

### 0.1 关键差异速查（从文档 §0.1）

| 文档描述 | 实际实现 | 状态 |
|---------|---------|------|
| `std::vector<SIMTStackEntry> simt_stack` | `ptxsim::SIMTStack simt_stack` (封装类) | ✅ 已同步 |
| `int simt_stack_depth` (ThreadState) | **不存在** — 已移除 | ✅ 已记录 |
| `pc_stack` / `pc_stack_depth` (WarpState) | **不存在** — 已移除，改用 `warp_state.threads[i].pc` | ✅ 已记录 |
| `std::atomic<bool> memory_fence_complete` (Wbar) | `bool is_initialized` + `bool memory_fence_verification_enabled` | ✅ 已同步 |
| `ThreadState::BLOCKED_BARRIER` | `BAR_SYNC` (+ `ThreadStatus::Blocked`) | ✅ 已同步 |
| `execute_branch_instruction()` | `handle_branch()` | ✅ 已同步 |
| 从 `ThreadContext::pc` 读取 PC | 从 `warp_state.threads[lane].pc` 读取 | ✅ 已同步 |

### 0.2 未文档化的关键实现

文档 §0.2 提到了以下实现，但需要进一步记录：

1. **`advance_thread_pc()` / `advance_all_threads()`** — 统一 PC 更新接口
2. **`get_statement_at(target_pc)`** — 指令按 PC 精确查找
3. **`sync_from_warp_state()` / `sync_to_warp_state()`** — ThreadContext ↔ WarpState 双同步机制
4. **`is_blocked` 过滤移除** — 防止屏障线程被调度器遗漏
5. **屏障释放使用 `arrived_mask`** — 只更新已到达线程的 PC
6. **屏障完成后重置 `status = Active`** — 防止线程状态泄漏

## 目标状态

1. **完整记录所有关键函数**: 在文档中添加 "关键 API 列表" 章节
2. **验证所有设计决策**: 确认 §7 的设计决策都已实现
3. **添加实现状态徽章**: 每个决策旁标注 ✅/⚠️/❌
4. **同步 ADR 引用**: 确保文档与 ADR-0014 等决策保持一致

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| SIMT-ARCHITECTURE-V2.md | 修改 | 主文档更新 |
| 其他架构文档 | 可能影响 | 检查是否需要同步更新 |

## 实现步骤

### Phase 1: 代码审查

1. 读取 `warp_context.cpp` 找到所有关键函数
2. 读取 `thread_context.cpp` 找到同步函数
3. 读取 `sm_context.cpp` 找到调度逻辑
4. 输出 "代码 vs 文档" 对照表

### Phase 2: 文档更新

1. 添加 "关键 API 列表" 章节
2. 更新 §0.1 差异速查表
3. 验证 §7 设计决策状态
4. 添加实现示例代码片段

### Phase 3: 验证

1. 使用 `state-modification-audit` skill 验证关键状态变量
2. 与 ADR-0014 对齐
3. 让其他开发者 review 文档准确性

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解 |
|------|--------|------|------|
| 文档更新后代码再次变化 | 高 | 低 | 添加 "最后验证日期" 和警告声明 |
| 遗漏重要实现细节 | 中 | 中 | 多轮 peer review |