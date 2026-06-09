# Tasks: add-instruction-latency-model

## 任务清单

### Task 1: 定义指令 Latency 表 ⏳
**时间**: 30 分钟
**验证**: 编译通过

MUST:
- [ ] 在 `include/ptx_ir/` 创建 `instruction_latency.h`
- [ ] 定义常见指令的 latency 值（ld.global=100, mul=4, etc.）
- [ ] 在 `ptx_op.def` 中添加 latency 属性

**状态**: 待开始 - 需要新增 latency 表定义

### Task 2: 修改 ThreadContext 添加 is_blocked ✅
**时间**: 30 分钟
**验证**: 编译通过 + 测试通过

MUST:
- [x] `ThreadState::is_blocked` 已存在于 `thread_state.h:42`
- [x] `is_schedulable()` 已正确使用 `!is_blocked` 检查
- [x] barrier 释放时清除 blocked 状态

**结果**: `is_blocked` 状态已存在，无需额外实现

### Task 3: 修改 exe_once() 调度器 ⏳
**时间**: 45 分钟
**验证**: 单元测试通过

MUST:
- [ ] 在 `exe_once()` 中检测 blocked 状态
- [ ] 实现 "选择最低 PC 的非 blocked 组" 逻辑
- [ ] 如果所有组 blocked，选择 Lowest PC（被动等待）

**状态**: 待开始 - 需要实现 blocked 检测逻辑

### Task 4: 实现 ld.global 长延迟处理 ⏳
**时间**: 30 分钟
**验证**: 相关测试通过

NOTE:
- [ ] `ld.global` 执行后标记 lane 为 blocked
- [ ] 实现 cycle 级别的 blocked 状态递减
- [ ] 测试内存加载指令的 blocking 行为

**状态**: 待开始 - 需要找到 ld.global handler 实现

### Task 5: 运行完整性检查 ⏳
**时间**: 15 分钟
**验证**: `./scripts/sanity.sh --quick`

NOTE:
- [ ] 执行 sanity.sh 确认无回归
- [ ] 如有失败，分析并修复

**状态**: 待执行