# ADR-0010: Fake CUDA Runtime 拦截机制

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-05 |
| **关联任务** | Phase 0 (初始架构) |
| **作者** | PTX-EMU Team |

## 上下文

PTX-EMU 的目标是拦截现有的 CUDA 程序，使其在模拟器上运行而非真实 GPU 上运行。核心问题是：

**如何在不修改 CUDA 程序源码的情况下，将其重定向到模拟器？**

## 决策驱动因素

1. **零源码修改**：用户程序无需任何修改
2. **透明拦截**：用户程序不知道自己在模拟器上运行
3. **渐进实现**：可以先拦截部分 API，逐步完善

## 考虑的替代方案

### 方案 A: 源码级插桩

**描述**: 修改用户 CUDA 源码，将 cudaMalloc 等替换为模拟器 API

**优点**:
- 完全控制

**缺点**:
- 需要修改用户源码
- 无法用于闭源程序
- 维护成本高

### 方案 B: PTX 级拦截

**描述**: 在 PTX 解析后直接执行，不拦截 CUDA Runtime API

**优点**:
- 绕过 CUDA Runtime

**缺点**:
- 需要用户直接提供 PTX 代码
- 无法模拟完整的 CUDA 执行流程

### 方案 C: Fake libcudart.so LD_PRELOAD (✅ 选中)

**描述**: 编译一个 fake libcudart.so，通过 `LD_LIBRARY_PATH` 优先加载，拦截所有 CUDA Runtime API 调用

**优点**:
- 零源码修改
- 对用户程序完全透明
- 可以渐进实现（未拦截的 API 转发到真实 libcudart）
- 支持任意 CUDA 程序

**缺点**:
- 仅 Linux 支持（LD_LIBRARY_PATH 机制）
- macOS 需要不同的机制（DYLD_INSERT_LIBRARIES）
- 需要维护与 CUDA Toolkit 版本的兼容性

**选择理由**: LD_LIBRARY_PATH 拦截是成熟的动态库劫持技术，对用户体验最友好。

## 决策内容

### 设计原则

1. **API 覆盖优先**：先覆盖常用 API（malloc、memcpy、launchKernel）
2. **未实现 API 返回成功**：避免阻塞用户程序（但记录日志）
3. **PTX 提取自动化**：使用 cuobjdump 从 CUDA binary 提取 PTX

### 实现要点

```cpp
// cudart_sim.cpp - Fake CUDA Runtime 实现

// 全局状态（需要迁移到 ExecutionContext）
std::unique_ptr<GPUContext> g_gpu_context;
std::unique_ptr<PtxInterpreter> g_ptx_interpreter;
std::map<uint64_t, std::string> func2name;
std::map<uint64_t, cudaKernel_t> func2kernel;

// PTX 提取
std::string extract_ptx_with_cuobjdump(const std::string& binary_path) {
    // 使用 cuobjdump 提取 PTX
    std::string cmd = "cuobjdump -ptx " + binary_path;
    // 执行命令，读取输出
    // 注意：使用临时文件和 system() 调用
}

// 拦截 cudaMalloc
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    // 分配模拟器内存
    auto* memory = SimpleMemory::allocate(size);
    *devPtr = memory->base_address();
    return cudaSuccess;
}

// 拦截 cudaMemcpy
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    // 根据 kind 处理不同方向的拷贝
    switch (kind) {
        case cudaMemcpyHostToDevice:
            copy_host_to_device(dst, src, count);
            break;
        case cudaMemcpyDeviceToHost:
            copy_device_to_host(dst, src, count);
            break;
        // ...
    }
    return cudaSuccess;
}

// 拦截 __cudaRegisterFatBinary
void** __cudaRegisterFatBinary(void* fatCubin) {
    // 提取 PTX
    std::string ptx = extract_ptx_with_cuobjdump(...);
    
    // 解析 PTX
    auto statements = parse_ptx(ptx);
    
    // 注册 kernel
    func2name[handle] = kernel_name;
    func2kernel[handle] = kernel;
}

// 拦截 cudaLaunchKernel
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, 
                             void** args, size_t sharedMem, cudaStream_t stream) {
    // 查找 kernel
    auto* kernel = func2kernel[func];
    
    // 执行 SIMT 模拟
    try {
        g_ptx_interpreter->launchPtxInterpreter(...);
        g_gpu_context->wait_for_completion();
    } catch (const PtxEmuException& e) {
        std::cerr << "PTX execution error: " << e.what() << std::endl;
        return (cudaError_t)999;
    }
    return cudaSuccess;
}

// 未完全实现的 API（返回成功但不执行）
cudaError_t cudaEventCreate(cudaEvent_t* event) {
    // TODO: 实现 event
    *event = fake_event;
    return cudaSuccess;
}
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `src/cudart/cudart_sim.cpp` | 新增 | Fake CUDA Runtime 实现 |
| `src/cudart/ptx_interpreter.cpp` | 新增 | PTX 解释器 |
| `CMakeLists.txt` | 修改 | 编译为共享库 |

## 后果

### 正面影响

- 用户程序零修改
- 透明的模拟体验
- 支持任意 CUDA 程序

### 负面影响

- 全局状态非线程安全
- cuobjdump 使用 system() 调用，存在安全隐患
- Event/Stream API 未真正同步
- 多 PTX cubin 不支持（只提取第一个）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| LD_LIBRARY_PATH 与其他库冲突 | 低 | 中 | 使用完整路径而非通用名称 |
| cuobjdump 失败导致无法提取 PTX | 中 | 高 | 检查返回值，提供友好错误信息 |
| system() 调用被禁用 | 低 | 高 | 改用 popen() 或直接调用 |
| 并发 kernel 启动导致数据竞争 | 中 | 高 | 迁移到 ExecutionContext（T11.3.x） |

### 未来演进方向

- 迁移全局状态到 ExecutionContext（ADR 待补充）
- 修复多 PTX cubin 提取（T12.2.1）
- 实现真正的 Event/Stream 同步
- 支持 macOS（DYLD_INSERT_LIBRARIES）

## 合规检查

后续相关开发应检查：

- [ ] 新增 CUDA API 拦截时返回合理的默认值
- [ ] 未实现的 API 记录日志
- [ ] 不使用 system() 调用外部工具
- [ ] 全局状态迁移到 ExecutionContext 后删除全局变量

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-05 | 初始版本 | PTX-EMU Team |

## 参考

- [架构评审报告 - 4.4 Fake CUDA Runtime](../reports/architecture-review-report.md#44-fake-cuda-runtime)
- [任务计划 - T11.3.x ExecutionContext 引入](../reports/task-plan.md#sprint-113-execution-context-引入day-9-10)

## 2026-07-18 Postmortem：add-cudart-unit-test-coverage 实施回顾

### 实施回顾

| Phase | 内容 | 状态 |
|-------|------|------|
| 1 | Memory API 测试（cudaMalloc/Free/Memcpy/Memset）— 14 用例 | ✅ |
| 2 | Stream API 补充测试（cudaStreamCreate/Destroy/Synchronize）— 5 用例 | ✅ |

### 关键发现

#### 发现 1: Proposal 事实性错误 — "零直接单元测试"不实
- **错误**: proposal 声称 `tests/unit/cudart/` "零直接单元测试"
- **实际**: 已有 3 个测试文件（248 行）覆盖 Stream API
- **影响**: C4 原本的 Stream API 范围需调整为"互补"而非"新建"
- **教训**: 任何 proposal 的"当前状态"声明必须用 `find`/`grep` 等工具验证，不可依赖记忆或审计文档的间接描述

#### 发现 2: `.cpp` 文件的 cudart 单测不能使用 `<cuda_runtime.h>`
- **问题**: g++ 编译的 `.cpp` 文件无法引用 CUDA Toolkit 头文件
- **解决**: 使用项目自身的 `cudart/cudart_intrinsics.h`（定义了所有 CUDA 类型），函数通过 `extern "C"` 直接链接
- **对比**: 已有 Stream 测试用 `.cu` 扩展名（nvcc 编译，可访问 `<cuda_runtime.h>`）

#### 发现 3: CudaDriver 需要显式初始化 SimpleMemory
- **问题**: `cudaMalloc` 等函数依赖 `CudaDriver::instance().get_global_pool()` 返回非空
- **解决**: 测试夹具使用 RAII 模式，每个测试前初始化 `SimpleMemory(1<<30)` 并 `set_simple_memory()`

### 验证结果

| 测试 | 结果 |
|------|------|
| `unit_cudart_memory` (14 用例, 1522 assertions) | ✅ PASS |
| `unit_cudart_stream` (5 用例, 17 assertions) | ✅ PASS |
| 已有 cudart 测试 (3 个) | ✅ 无回归 |
| barrier/simt/e2e (14 tests) | ✅ 0 回归 |

### 相关链接

- [openspec/changes/add-cudart-unit-test-coverage/](../../openspec/changes/add-cudart-unit-test-coverage/) — change artifacts
- [openspec/specs/cudart-unit-test/](../../openspec/specs/cudart-unit-test/) — synced spec
