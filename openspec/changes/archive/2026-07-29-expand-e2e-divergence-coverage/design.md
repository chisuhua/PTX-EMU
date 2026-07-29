# expand-e2e-divergence-coverage - Design

## Overview

在 `tests/e2e/divergence/test_divergence.cu` 中新增 3-5 个 divergence E2E kernel，覆盖当前缺失的边界场景：non-uniform branch + reconvergence、深层嵌套、memory 交叉、非 32 线程边界。同时对 `barrier_sync` 的"已知限制"给出明确结论。

当前状态：`test_divergence.cu` 含 8 个 kernel（`divergence_if_else`、`divergence_multi_path`、`divergence_nested_if`、`divergence_loop_if`、`divergence_uneven_loop`、`divergence_mixed`、`divergence_reduction`、`divergence_barrier_sync`），采用 1 block × 32 threads = 1 warp 模式，各 lane 将结果写到 32-int buffer，host 端验证。

## Design Decisions

### 决策 1: 新 kernel 场景选择

**选择**: 添加以下 4 个新 kernel：

1. `divergence_immediate_reconvergence` - warp 内 non-uniform branch + 立即 reconvergence
2. `divergence_deep_nesting` - 3+ 层嵌套 if-else
3. `divergence_memory_interaction` - divergence 分支内做 shared memory 写入
4. `divergence_non32_threads` - CTA 含 < 32 线程的 divergence 边界

**理由**:
- 这 4 个场景覆盖了 improvement 中明确列出的所有 In Scope 项
- 每个 kernel 针对 SIMT stack 的不同边界行为
- 与现有 8 个 kernel 无行为重叠

**替代方案**:
- A. 仅添加 3 个（minimum）-> 满足验收但覆盖不足
- B. 添加 5+ 个 -> 增加 work 量但边际收益递减
- C. **采用**: 4 个 kernel，平衡覆盖与工作量

### 决策 2: Kernel 实现模式

**选择**: 遵循现有 `test_divergence.cu` 的测试模式（内联 kernel + 32-int buffer + host 验证）

**理由**:
- 保持与现有 8 个 kernel 一致的验证风格
- 内联 kernel 避免 `__cudaRegisterFatBinary` multi-instance FATAL（见文件头注释）
- 32-int buffer 模式简单且可扩展

**实现伪码**:
```cuda
// 1. non-uniform branch + immediate reconvergence
__global__ void divergence_immediate_reconvergence(int* buf) {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        buf[tid] = 1;  // even lanes
    }
    // immediate reconvergence - all lanes execute this
    buf[tid] += 100;  // odd lanes: 100, even lanes: 101
}

// 2. deep nesting (3+ layers)
__global__ void divergence_deep_nesting(int* buf) {
    int tid = threadIdx.x;
    if (tid < 16) {           // layer 1
        if (tid < 8) {        // layer 2
            if (tid < 4) {    // layer 3
                buf[tid] = 1;
            } else {
                buf[tid] = 2;
            }
        } else {
            buf[tid] = 3;
        }
    } else {
        if (tid < 24) {       // layer 2
            buf[tid] = 4;
        } else {
            buf[tid] = 5;
        }
    }
}

// 3. divergence + memory interaction
__global__ void divergence_memory_interaction(int* buf) {
    __shared__ int smem[32];
    int tid = threadIdx.x;
    if (tid < 16) {
        smem[tid] = tid * 2;
    } else {
        smem[tid] = tid * 3;
    }
    __syncthreads();
    buf[tid] = smem[(tid + 1) % 32];
}

// 4. non-32 threads CTA divergence
__global__ void divergence_non32_threads(int* buf) {
    int tid = threadIdx.x;
    if (tid < 8) {
        buf[tid] = 10;
    } else if (tid < 16) {
        buf[tid] = 20;
    } else {
        buf[tid] = 30;
    }
    // launched with <<<1, 20>>> - tests partial warp divergence
}
```

### 决策 3: barrier_sync "已知限制" 处理

**选择**: 评估 `divergence_barrier_sync` kernel 是否仍存在限制，如有则文档化（不阻塞验收）

**理由**:
- improvement 要求给出明确结论（修复或文档化）
- 如限制已随 SIMT v2.0 修复，移除标记即可
- 如限制仍存在，在 TEST_CASE 中添加 `INFO("Known limitation: ...")` 注释

**验证步骤**:
1. 运行 `ctest -R e2e_divergence` 确认 `barrier_sync` 是否通过
2. 如通过 -> 移除"已知限制"标记
3. 如失败 -> 保留标记并记录限制详情

### 决策 4: 非 32 线程 kernel 的启动配置

**选择**: `divergence_non32_threads` 使用 `<<<1, 20>>>` 启动（20 线程 CTA）

**理由**:
- 20 不是 32 的倍数，测试 partial warp divergence 边界
- SIMT v2.0 的 WarpState 应正确处理 inactive lanes
- buffer 大小仍为 32（避免越界），但仅前 20 个 lane 写入

## Implementation Plan

### Phase 1: 编写新 kernel + TEST_CASE
1. 在 `test_divergence.cu` 末尾添加 4 个新 kernel 函数定义
2. 添加对应 TEST_CASE（遵循现有 `TEST_CASE("...", "[e2e][divergence]")` 格式）
3. 为 `divergence_non32_threads` 的 TEST_CASE 使用 `<<<1, 20>>>` 启动配置

### Phase 2: 验证编译 + 执行
1. `. env.sh && cmake --build build`
2. `cd build && ctest -R e2e_divergence --output-on-failure`
3. 确认所有新 TEST_CASE 通过

### Phase 3: barrier_sync 结论
1. 运行 `barrier_sync` TEST_CASE 确认状态
2. 如通过 -> 移除"已知限制"注释
3. 如失败 -> 添加 `INFO("Known limitation: ...")` 文档化

## Testing Strategy

| 测试场景 | 命令 | 预期 |
|---------|------|------|
| 新 kernel 编译 | `cmake --build build` | 编译通过（nvcc → PTX） |
| immediate reconvergence | `ctest -R e2e_divergence` | even lanes: 101, odd lanes: 100 |
| deep nesting | `ctest -R e2e_divergence` | 5 种分支结果正确 |
| memory interaction | `ctest -R e2e_divergence` | shared memory 交叉访问正确 |
| non-32 threads | `ctest -R e2e_divergence` | 20 lane partial warp 正确 |
| 全量 E2E | `ctest -L e2e` | 所有 E2E 测试通过 |
| barrier_sync 结论 | `ctest -R e2e_divergence` | 明确通过或文档化限制 |

### 预期输出验证

```
divergence_immediate_reconvergence:
  lane 0: 101, lane 1: 100, lane 2: 101, ... (even=101, odd=100)

divergence_deep_nesting:
  lane 0-3: 1, lane 4-7: 2, lane 8-15: 3, lane 16-23: 4, lane 24-31: 5

divergence_memory_interaction:
  lane i: smem[(i+1) % 32], smem[i] = i*2 (if i<16) or i*3 (if i>=16)

divergence_non32_threads:
  lane 0-7: 10, lane 8-15: 20, lane 16-19: 30
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| 新 kernel 触发 SIMT stack bug | 测试失败 | 这是期望行为（暴露 bug）；记录 bug 并报告 |
| `barrier_sync` 限制无法修复 | "已知限制"保留 | 文档化限制详情，不阻塞验收 |
| non-32 threads CTA 边界行为未定义 | 测试结果不确定 | 参照 SIMT v2.0 spec 确认预期行为 |
| shared memory 在 divergence 中行为异常 | memory_interaction 失败 | 验证 `__syncthreads()` 后全 warp 可见性 |

## Open Questions

1. **`barrier_sync` 是否已随 SIMT v2.0 修复？**
   - 需运行测试确认。如通过则移除标记，否则文档化

2. **non-32 threads CTA 的 divergence 语义是否与 NVIDIA 硬件一致？**
   - PTX-EMU 的 SIMT v2.0 WarpState 应正确处理 inactive lanes
   - 如行为不一致，记录为已知差异

## 关联文档

- `improvements/expand-e2e-divergence-coverage.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-24`：原债务条目
- `tests/e2e/divergence/test_divergence.cu`：当前测试文件（333 行，8 kernel）
- `docs/architecture/SIMT-ARCHITECTURE-V2.md`：SIMT v2.0 架构文档
