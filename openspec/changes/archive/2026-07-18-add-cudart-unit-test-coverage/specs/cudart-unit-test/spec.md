# cudart-unit-test

## ADDED Requirements

### Requirement: cudaMalloc 基础分配与边界
测试 cudaMalloc 的正确分配（有效 size→非空 ptr, cudaSuccess）、零 size 行为、大分配行为、nullptr 输出指针处理。

### Requirement: cudaFree 分配后释放与边界
测试 cudaFree 正确释放已分配内存（随后 cudaMalloc 可复用地址）、nullptr 释放行为、double-free 行为。

### Requirement: cudaMemcpy 方向路径
测试 cudaMemcpy 的 Host→Device、Device→Host、Device→Device 方向正确性（写已知 pattern→回读验证）。

### Requirement: cudaMemset 设值
测试 cudaMemset 将 Device 内存区域全部设为目标值。

### Requirement: cudaStreamCreate / cudaStreamSynchronize
测试默认 stream 创建（非空 handle、cudaSuccess）和空 stream 同步行为。
