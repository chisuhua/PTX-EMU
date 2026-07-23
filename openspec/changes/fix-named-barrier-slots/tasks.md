## 1. 数据结构扩展

- [x] 1.1 `warp_context.cpp`: 将 `wbars[1]` 改为 `wbars[16]`，更新构造函数中默认槽 0 的初始化逻辑
  - **NOTE**: `WarpBarrier` 默认构造即可用，无需对所有 16 个槽调用 `init()`
  - **MUST**: 验证 `warp_context.h` 中 `wbars` 数组声明同步更新

## 2. Barrier 分发修改

- [x] 2.1 `barrier.cpp` `BarWarpSyncHandler::handle()`: 将 `wbars[0]` 改为按 `bar_id` 索引
  - **MUST**: 验证 `bar_id >= 16` 时抛 `std::out_of_range`
- [x] 2.2 `barrier.cpp`: 更新 `current_wbar_id` 语义为"最近访问的槽索引"，移除对 `wbars[0]` 的硬编码假设
  - **MUST**: 确认所有 `current_wbar_id` 读取处能正确处理多槽场景

## 3. 初始化/重置路径检查

- [x] 3.1 `warp_context.cpp`: 检查 `reset()` 或 `clear()` 方法是否需要对所有 16 个槽重置
- [x] 3.2 搜索全项目中 `wbars[0]` 和 `get_wbar()` 的引用，确认无遗漏的硬编码单一槽假设

## 4. 测试

- [x] 4.1 集成测试: 编写多槽并发 barrier 场景（2+ warp 各使用不同槽）
  - **MUST**: 使用 `ptxsim::testing::step_warp` 驱动
- [x] 4.2 验证所有现有 barrier 测试无回归
  - `ctest -L "barrier"` 全量通过

## 5. 验证

- [x] 5.1 运行 `./scripts/sanity.sh --quick` 确认无回归
- [x] 5.2 运行 `./scripts/sanity.sh` 全量测试通过