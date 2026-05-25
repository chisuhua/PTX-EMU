# Tasks: fix-barrier-dynamic-participation-mask

## 任务清单

### Task 1: 分析当前 barrier 实现
**时间**: 30 分钟
**验证**: 代码理解文档

MUST:
- 读取 `src/ptxsim/core/barrier.cpp` 完整实现
- 读取 `include/ptxsim/core/wbar.h` Wbar 结构
- 理解 `exe_once()` 中 barrier 调用的位置
- 输出分析文档（现状问题清单）

### Task 2: 修改动态 mask 计算逻辑
**时间**: 45 分钟
**验证**: 编译通过

MUST:
- 在 `barrier.cpp` 中修改 mask 计算逻辑
- 使用实际到达 barrier PC 的线程动态计算 mask
- 不再使用静态的 operand mask
- 保持与现有 Wbar 接口兼容

### Task 3: 修正 Wbar::is_complete() 判断
**时间**: 30 分钟
**验证**: 相关测试通过

NOTE:
- 修改 `wbar.h` 中的 `is_complete()` 方法
- 使用动态计算的 `participation_mask`
- 验证 `(arrived_mask & participation_mask) == participation_mask` 逻辑

### Task 4: 添加分叉场景测试
**时间**: 30 分钟
**验证**: 新测试通过

MUST:
- 创建 `test_barrier_divergence.cpp` 测试文件
- 测试 warp 分叉后各自到达不同 barrier 的场景
- 验证只有真正参与的线程被释放

### Task 5: 运行完整性检查
**时间**: 15 分钟
**验证**: `./scripts/sanity.sh --quick`

NOTE:
- 执行 sanity.sh 确认无回归
- 特别关注 barrier 相关测试