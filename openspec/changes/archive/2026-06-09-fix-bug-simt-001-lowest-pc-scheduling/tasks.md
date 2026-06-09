# Tasks: fix-bug-simt-001-lowest-pc-scheduling

## 任务清单

### Task 1: 修改 exe_once() divergent path ✅
**时间**: 已完成
**验证**: 代码检查通过

MUST:
- [x] 找到 `sm_context.cpp:219-257` 的 exe_once() divergent path 代码
- [x] 将 `for` 循环改为只选择第一个元素 (Lowest PC first)
- [x] 保持 `execute_pc_group()` 调用逻辑不变

**结果**: 代码已在 `sm_context.cpp:221-267` 实现修复
- 使用 `lanes_by_pc.begin()` 选择最低 PC
- 每 cycle 只执行一个 PC 组
- 注释明确说明 "execute ONLY ONE PC group per cycle (Lowest PC first)"

### Task 2: 验证 cycle 计数准确性 ✅
**时间**: 30 分钟
**验证**: test_barrier_module, test_barrier_module_integrated 通过

NOTE:
- [x] 运行 `ctest -R simt` 检查是否有测试验证 cycle 计数
- [x] 确认单 cycle 只执行一个 PC 组

**结果**:
- `test_barrier_module`: 45 assertions, 4 test cases - All passed
- `test_barrier_module_integrated`: Barrier 释放正确，lane 状态正确更新
- 代码已在 sm_context.cpp:221-267 实现 Lowest PC first

### Task 3: 更新 SIMT divergence 测试 ✅
**时间**: 30 分钟
**验证**: ctest 通过

- [x] 检查 `tests/` 目录下 SIMT divergence 相关测试
- [x] 更新预期行为 - 无需修改（现有测试已验证正确行为）
- [x] 运行完整测试确保无回归

**结果**: Barrier 模块测试全部通过

### Task 4: 运行完整 sanity 检查 ✅
**时间**: 15 分钟
**验证**: `./scripts/sanity.sh --quick`

NOTE:
- [x] 执行 `./scripts/sanity.sh --quick` 确认无回归
- [x] 如有失败，分析原因并修复

**结果**: All 6 critical bug tests PASS:
- BUG-001: exec_mask restore ✓
- BUG-002: SIMT stack exit handling ✓
- ISSUE-004: active_mask consistency ✓
- test_specific_bugs_unit ✓
- test_barrier_scenarios ✓
- test_barrier_verification ✓