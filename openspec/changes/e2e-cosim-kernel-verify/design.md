## Context

### 现状

cpptlm-p1-ptxemu-shim 已归档（commit `63f10703`），shim + bridge dual-enqueue + `cpptlm_set_driver` ABI 已就绪，co-sim smoke test 通过。但缺少用**真实 CUDA kernel** 验证全链路的 E2E 测试。

### 目标

添加一个 CUDA vectorAdd kernel E2E 测试，通过 `BUILD_LIB_CPPTLM_CUDART=ON` 构建，验证：
1. `cudaLaunchKernel` bridge 路径 → `prepareKernelLaunchRequest()` → IR 正确
2. `GPUContext::exe_once()` 真实执行 PTX 指令
3. `on_complete` → `mark_complete` 回调链
4. GPU 内存输出与 CPU golden value 一致

## Decisions

### D1: test 位置

**决定**: `tests/e2e/cosim/` — 独立子目录，与现有 e2e 测试隔离。

**理由**: 这是 CppTLM 协同仿真专用测试，不应与普通 E2E 测试混在一起。`BUILD_LIB_CPPTLM_CUDART=OFF` 时此目录跳过编译。

### D2: kernel 选择

**决定**: `vectorAdd` — 最简单的有输入/输出数据的 kernel。

**理由**:
- 最小化 kernel 复杂度（~30 LOC CUDA）
- 有明确的 golden value（CPU `A[i] + B[i]`）
- 足够验证 LD/ST 指令 + 寄存器运算 + barrier sync
- 不依赖 tcgen05 或特殊指令

### D3: 构建条件

**决定**: `BUILD_LIB_CPPTLM_CUDART=ON` 时编译并运行；`OFF` 时 `SKIP`。

```cmake
if(BUILD_LIB_CPPTLM_CUDART)
    add_catch_test(e2e_cosim_vector_add
        cosim/test_cosim_vector_add.cpp
        cosim/kernel_vector_add.cu
    )
    set_tests_properties(e2e_cosim_vector_add PROPERTIES LABELS "e2e;cosim;cpptlm")
endif()
```

### D4: 测试流程

```
1. cudaMalloc host/device 内存
2. 初始化输入数据（CPU golden value 同步计算）
3. cudaMemcpy H→D
4. cudaLaunchKernel (bridge 路径: g_cpptlm_bridge != nullptr)
5. cudaDeviceSynchronize (poll_kernel + GPUContext exe_once 循环)
6. cudaMemcpy D→H
7. REQUIRE(host_output[i] == golden[i]) for all i
8. REQUIRE(is_kernel_complete(kernel_id))
```

### D5: 验收标准

- `BUILD_LIB_CPPTLM_CUDART=ON` 构建通过
- ctests 全量 PASS（含新增 `e2e_cosim_vector_add`）
- `BUILD_LIB_CPPTLM_CUDART=OFF` 时新增测试 SKIP 而不 FAIL

## Non-Goals

- 不修改任何生产代码（纯测试新增）
- 不涉及 CppTLM 侧变更
- 不处理 multi-kernel / multi-stream 场景