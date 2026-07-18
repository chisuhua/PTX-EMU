# 实施任务

## Phase 1: 建立测试基础设施

- [x] Task 1.1: 创建 tests/unit/cudart/ 目录
- [x] Task 1.2: 创建 tests/unit/cudart/CMakeLists.txt（cudart 目标链接）
- [x] Task 1.3: 创建 tests/unit/cudart/cudart_test_helpers.h（GPUContext 初始化 + 设备选择辅助）
- [x] Task 1.4: 在 tests/unit/CMakeLists.txt 添加 add_subdirectory(cudart)
- [x] Task 1.5: cmake --build build 确认编译通过

## Phase 2: Memory API 测试

- [x] Task 2.1: test_cudaMalloc_basic — 有效 size→非空 ptr, cudaSuccess
- [x] Task 2.2: test_cudaMalloc_zero_size — size=0 行为
- [x] Task 2.3: test_cudaFree_valid — 分配后释放→cudaSuccess
- [x] Task 2.4: test_cudaFree_nullptr — free(nullptr) 行为
- [x] Task 2.5: test_cudaMemcpy_H2D_D2H — 写 pattern→回读验证
- [x] Task 2.6: test_cudaMemcpy_D2D — Device→Device 复制验证
- [x] Task 2.7: test_cudaMemset — 设值 0x42→回读验证

## Phase 3: Stream API 测试

- [x] Task 3.1: test_cudaStreamCreate — 默认 stream 创建→非空 handle, cudaSuccess
- [x] Task 3.2: test_cudaStreamSynchronize — 空 stream 同步→cudaSuccess

## Phase 4: 验证

- [x] Task 4.1: cd build && ctest -R unit_cudart -V（全部通过）
- [x] Task 4.2: cd build && ctest -L "e2e" -V（E2E 无回归）
