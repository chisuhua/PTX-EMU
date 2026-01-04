# NVIDIA GPU 硬件架构模拟方案

## 概述

本文档详细描述了 PTX-EMU 项目中用于精确模拟 NVIDIA GPU 硬件架构的方案。该方案旨在实现更精确的硬件行为模拟，包括 warp 级执行、SM（Streaming Multiprocessor）抽象、内存层次结构模拟等功能。

## 现状分析

PTX-EMU 目前主要实现了线程级别的模拟，每个 ThreadContext 独立执行指令，缺乏 warp 和 SM 级别的硬件抽象。这限制了模拟器的精度，无法准确反映真实 GPU 硬件的执行特性。

## 硬件架构模拟设计

### 1. Warp 执行模型

#### 1.1 Warp Context 设计

```cpp
// warp_context.h
class WarpContext {
public:
    static const int WARP_SIZE = 32;
    
    // warp 内的线程 ID 映射
    int warp_thread_ids[WARP_SIZE];
    bool active_mask[WARP_SIZE];  // 活跃线程掩码
    int active_count;             // 活跃线程数
    
    // warp 状态
    uint32_t pc;                  // warp 的程序计数器
    bool single_step_mode;        // 单步执行模式
    
    // 执行控制
    std::vector<ThreadContext*> threads;
    
    // 构造函数
    WarpContext();
    
    // 执行单个 warp 指令
    void execute_warp_instruction(StatementContext &stmt);
    
    // 更新活跃掩码
    void update_active_mask();
    
    // 检查 warp 是否全部完成
    bool is_complete() const;
    
    // 同步 warp 内所有线程
    void sync_warp();
};
```

#### 1.2 Warp 指令执行

WarpContext 实现了 SIMD 执行模型，通过活跃掩码控制执行流程：

```cpp
void WarpContext::execute_warp_instruction(StatementContext &stmt) {
    // 根据活跃掩码执行指令
    for (int i = 0; i < WARP_SIZE; i++) {
        if (active_mask[i]) {
            ThreadContext* thread = threads[warp_thread_ids[i]];
            // 执行指令
            thread->execute_single_instruction(stmt);
        }
    }
    
    // 更新活跃掩码（例如，遇到分支指令时）
    update_active_mask();
}
```

### 2. SM（Streaming Multiprocessor）抽象

#### 2.1 SM Context 设计

```cpp
// sm_context.h
class SMContext {
public:
    // SM 配置参数
    int max_warps_per_sm;
    int max_threads_per_sm;
    int shared_mem_size_per_sm;
    int registers_per_sm;
    
    // SM 资源管理
    std::vector<WarpContext*> warps;
    int allocated_shared_mem;
    int allocated_registers;
    
    // 硬件单元
    WarpScheduler scheduler;
    int execution_cycles;  // 执行周期计数
    
    // 执行状态
    std::vector<ThreadContext*> threads;
    
    // 构造函数
    SMContext(int sm_id, int max_warps, int shared_mem_size);
    
    // 执行一个时钟周期
    void execute_cycle();
    
    // 分配资源给新的 thread block
    bool allocate_resources(int thread_count, int shared_mem_needed, int reg_count);
    
    // 释放资源
    void release_resources();
    
    // 检查是否可以接受新的 thread block
    bool can_accept_block(int thread_count, int shared_mem_needed) const;
};
```

#### 2.2 资源管理

```cpp
bool SMContext::allocate_resources(int thread_count, int shared_mem_needed, int reg_count) {
    int warps_needed = (thread_count + WarpContext::WARP_SIZE - 1) / WarpContext::WARP_SIZE;
    
    // 检查资源是否足够
    if (warps.size() + warps_needed > max_warps_per_sm ||
        allocated_shared_mem + shared_mem_needed > shared_mem_size_per_sm ||
        allocated_registers + reg_count > registers_per_sm) {
        return false;
    }
    
    // 分配资源
    allocated_shared_mem += shared_mem_needed;
    allocated_registers += reg_count;
    
    return true;
}
```

### 3. Warp 调度器

#### 3.1 调度策略

```cpp
// warp_scheduler.h
class WarpScheduler {
public:
    enum class SchedulingPolicy {
        ROUND_ROBIN,    // 轮询调度
        GREEDY,         // 贪心调度
        TWO_LEVEL       // 两级调度
    };
    
    WarpScheduler(SchedulingPolicy policy = SchedulingPolicy::ROUND_ROBIN);
    
    // 选择下一个要执行的 warp
    WarpContext* select_next_warp();
    
    // 更新 warp 状态
    void update_warp_state(WarpContext* warp);
    
    // 记录 warp 执行统计
    void record_warp_stats(WarpContext* warp);
};
```

### 4. 内存层次结构模拟

#### 4.1 内存层次设计

```cpp
// memory_hierarchy.h
class MemoryHierarchy {
public:
    // L1 缓存模拟
    struct L1Cache {
        int size;
        int line_size;
        int associativity;
        int access_latency;
    };
    
    // L2 缓存模拟
    struct L2Cache {
        int size;
        int line_size;
        int associativity;
        int access_latency;
    };
    
    // 缓存访问延迟模拟
    int simulate_cache_access(uint64_t addr, bool is_write);
    
    // 内存访问延迟模拟
    int simulate_global_memory_access(uint64_t addr, int size);
    
    // 记录内存访问统计
    void record_memory_stats();
};
```

### 5. 精确的执行模型

#### 5.1 CTA Context 修改

```cpp
// 修改 cta_context.h
class CTAContext {
public:
    // 新增 warp 相关成员
    std::vector<WarpContext*> warps;
    std::vector<SMContext*> parent_sms;
    
    // 执行一个时钟周期
    EXE_STATE execute_cycle();
    
    // 模拟 warp 调度
    void schedule_warps();
    
    // 同步所有 warp
    void sync_all_warps();
    
    // 检查 warp 分歧
    void handle_warp_divergence();
};
```

## 实施计划

### 第一阶段：架构基础（1-2个月）

1. **实现 WarpContext 类**
   - 创建 warp_context.h/.cpp 文件
   - 实现 warp 活跃掩码管理
   - 实现 warp 级 PC 管理
   - 实现 warp 内线程同步机制

2. **实现 Warp 调度器**
   - 创建 warp_scheduler.h/.cpp 文件
   - 实现基本调度策略（轮询、贪婪等）
   - 集成 warp 状态跟踪

3. **重构 ThreadContext**
   - 添加对 warp 的引用
   - 修改执行流程以适应 warp 级执行
   - 保持向后兼容性

### 第二阶段：SM 抽象（2-3个月）

4. **实现 SMContext 类**
   - 创建 sm_context.h/.cpp 文件
   - 实现资源管理机制
   - 集成 warp 调度器
   - 添加时钟周期模拟

5. **修改 CTAContext**
   - 集成 SMContext
   - 实现 thread block 到 SM 的映射
   - 添加资源分配逻辑

6. **实现内存层次结构模拟**
   - 创建 memory_hierarchy.h/.cpp
   - 实现缓存模拟器
   - 集成内存访问延迟模型

### 第三阶段：精确模拟（3-4个月）

7. **实现分支分歧处理**
   - 检测 warp 内的分支分歧
   - 模拟串行化执行路径
   - 记录分歧相关的性能损失

8. **实现内存访问模式模拟**
   - 模拟 coalesced memory access
   - 检测 bank conflicts（共享内存）
   - 实现内存延迟隐藏机制

9. **实现性能计数器**
   - 集成指令级性能统计
   - 实现吞吐量和延迟测量
   - 添加硬件利用率跟踪

### 第四阶段：高级特性（4-6个月）

10. **实现多 SM 模拟**
    - 扩展到多个 SM 的模拟
    - 实现跨 SM 的负载均衡
    - 模拟全局内存带宽限制

11. **实现动态并行性**
    - 支持 CUDA 动态并行特性
    - 实现 child kernel 启动机制
    - 添加 nested launch 支持

12. **实现高级调试功能**
    - 添加 warp 级调试视图
    - 实现可视化 warp 状态
    - 集成性能分析工具

## 验证与测试

1. **功能验证**：确保修改后的模拟器仍能正确执行现有 CUDA 程序
2. **性能验证**：与真实硬件性能数据对比验证模拟精度
3. **回归测试**：维护现有测试用例确保兼容性

## 性能优化考虑

1. **减少模拟开销**：通过批处理和缓存优化减少模拟器本身的时间开销
2. **内存效率**：使用紧凑的数据结构表示 warp 状态
3. **并行化**：利用多核 CPU 并行模拟多个 SM

## 总结

该硬件架构模拟方案将使 PTX-EMU 从简单的线程级模拟器转变为更精确的硬件行为模拟器，能够更准确地反映 NVIDIA GPU 架构的执行特性，为性能分析和优化提供更可靠的基础。通过分阶段实施，可以在保持兼容性的同时逐步提升模拟精度。

该方案遵循了 GPU 架构模拟与执行模型综合规范，确保了模拟器的扩展性和维护性。

下一步工作计划
1. 完善执行流程集成
• 修改CTAContext: 重构CTAContext以使用WarpContext，不再直接管理ThreadContext
• 更新exe_once实现: 使CTAContext::exe_once调用WarpContext执行，而不是逐线程执行
• 指令获取机制: 实现从CTAContext到WarpContext的指令流传递机制
2. 实现完整的SMContext功能
• 指令调度: 实现SMContext中的指令获取和分发逻辑
• 资源管理: 完善共享内存管理和线程块调度逻辑
• 状态同步: 实现warp间和线程间的同步机制
3. 增强调试和跟踪功能
• Warp级跟踪: 添加warp级执行跟踪，便于分析SIMT行为
• 性能计数器: 添加warp执行效率、分支分歧等性能指标
• 调试接口: 提供warp状态查看和调试接口
4. 优化Warp调度策略
• 实现更多调度算法: 例如基于优先级的调度、基于负载均衡的调度• 性能评估: 对比不同调度策略的性能表现• 动态调度: 根据运行时状态动态调整调度策略
5. 测试和验证
• 单元测试: 为新增的WarpContext、SMContext、WarpScheduler编写单元测试
• 集成测试: 验证整个执行流程的正确性
• 性能测试: 评估新架构对执行性能的影响
6. 处理分支分歧• 分支预测: 实现分支处理逻辑，正确处理warp内的分歧分支• 重新合并: 实现分支后的路径重新合并机制• 活跃掩码更新: 根据分支指令动态更新活跃掩码
7. 完善CMakeLists.txt• 添加新源文件: 将新创建的.cpp文件添加到构建系统• 更新依赖: 确保所有依赖关系正确配置这个框架已经建立了完整的四层架构（SMContext -> WarpScheduler -> WarpContext -> ThreadContext），下一步将逐步完善各层之间的交互和具体功能实现。


done:
1. 执行流程架构
• SMContext层：管理流式多处理器资源（共享内存、warp集合等），协调多个warp的执行
• WarpScheduler层：实现warp级调度策略（如轮询、贪心），选择下一个执行的warp
• WarpContext层：以warp为单位统一获取指令，根据活跃掩码分发给线程，管理warp级PC和同步
• ThreadContext层：负责单个线程的具体指令执行、寄存器状态和条件码管理

2. 已完成的改进
-. WarpContext：◦ 添加了分支分歧处理功能◦ 实现了活跃掩码管理◦ 添加了PC栈支持，用于处理分歧后的重新合并
-. ThreadContext：◦ 添加了分支处理支持◦ 正确更新PC值 
- CTAContext：◦ 重构为使用WarpContext管理线程◦ 正确更新线程状态计数

3. 执行流程验证
• CTAContext::exe_once()现在调用WarpContext执行，而不是逐线程执行
• 实现了从CTAContext到WarpContext的指令流传递机制
• 正确处理了线程状态（退出、屏障同步等）