# Tasks: fix-bug-simt-001-lowest-pc-scheduling

## 任务清单

### Task 1: 修改 exe_once() divergent path
**时间**: 30 分钟
**验证**: `clang-format -i` + 编译通过

MUST:
- 找到 `sm_context.cpp:219-257` 的 exe_once() divergent path 代码
- 将 `for (const auto& [pc, lanes] : lanes_by_pc)` 循环改为只选择第一个元素
- 保持 `execute_pc_group()` 调用逻辑不变

### Task 2: 验证 cycle 计数准确性
**时间**: 30 分钟
**验证**: 运行相关测试

NOTE:
- 运行 `ctest -R simt` 检查是否有测试验证 cycle 计数
- 确认单 cycle 只执行一个 PC 组

### Task 3: 更新 SIMT divergence 测试
**时间**: 30 分钟
**验证**: ctest 通过

- 检查 `tests/` 目录下 SIMT divergence 相关测试
- 更新预期行为（如有需要）
- 运行完整测试确保无回归

### Task 4: 运行完整 sanity 检查
**时间**: 15 分钟
**验证**: `./scripts/sanity.sh --quick`

NOTE:
- 执行 `./scripts/sanity.sh --quick` 确认无回归
- 如有失败，分析原因并修复