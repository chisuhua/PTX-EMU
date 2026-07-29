# expand-e2e-divergence-coverage - Tasks

## Task List

### Phase 1: 编写新 kernel + TEST_CASE（60 min）

- [ ] 1.1 MUST 在 `tests/e2e/divergence/test_divergence.cu` 添加 `divergence_immediate_reconvergence` kernel：
  ```cuda
  __global__ void divergence_immediate_reconvergence(int* buf) {
      int tid = threadIdx.x;
      if (tid % 2 == 0) { buf[tid] = 1; }
      buf[tid] += 100;  // even: 101, odd: 100
  }
  ```
- [ ] 1.2 MUST 添加 `divergence_deep_nesting` kernel（3+ 层嵌套 if-else，5 种分支结果）
- [ ] 1.3 MUST 添加 `divergence_memory_interaction` kernel（divergence + shared memory + `__syncthreads()`）
- [ ] 1.4 MUST 添加 `divergence_non32_threads` kernel（启动配置 `<<<1, 20>>>`，测试 partial warp）
- [ ] 1.5 MUST 为每个新 kernel 添加 `TEST_CASE("...", "[e2e][divergence]")` 测试块
- [ ] 1.6 MUST 每个 TEST_CASE 包含 host 端 buffer 验证（预期值断言）

### Phase 2: barrier_sync 结论（15 min）

- [ ] 2.1 MUST 运行 `ctest -R e2e_divergence` 确认 `divergence_barrier_sync` TEST_CASE 状态
- [ ] 2.2 如通过: MUST 移除"已知限制"注释/标记
- [ ] 2.3 如失败: MUST 添加 `INFO("Known limitation: <具体描述>")` 文档化限制
- [ ] 2.4 MUST 记录 `barrier_sync` 最终结论（修复 or 文档化）到 design.md

### Phase 3: 编译 + 验证（30 min）

- [ ] 3.1 MUST 验证编译：`. env.sh && cmake --build build`
- [ ] 3.2 MUST 验证新测试通过：`cd build && ctest -R e2e_divergence --output-on-failure`
- [ ] 3.3 MUST 验证全量 E2E：`cd build && ctest -L e2e --output-on-failure`
- [ ] 3.4 MUST 验证不影响现有测试：`cd build && ctest --output-on-failure`

### Phase 4: 提交

- [ ] 4.1 git commit -m "test(e2e): expand divergence coverage with 4 new boundary kernels"
- [ ] 4.2 MUST 运行 `openspec validate expand-e2e-divergence-coverage --strict`
- [ ] 4.3 MUST 通过所有验证后 archive 此 change

## 验收

- 新增 ≥ 3 个 divergence E2E 测试（实际 4 个）
- 所有 E2E 测试通过（`ctest -L e2e`）
- `barrier_sync` 的"已知限制"状态有明确结论（修复或文档化）
- 现有 8 个 kernel 行为不变

## 关键约束（MUST/MUST NOT）

- MUST 使用 CUDA C++ 编写（.cu 文件）
- MUST 通过 nvcc 编译为 PTX
- MUST 通过 fake libcudart.so 拦截执行
- MUST 遵循现有 `test_divergence.cu` 的测试模式（内联 kernel + 32-int buffer）
- MUST NOT 修改 SIMT stack 实现
- MUST NOT 改变现有 8 个 kernel 的行为
- MUST NOT 添加 performance benchmark
