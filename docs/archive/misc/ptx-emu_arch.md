#PTX-EMU 代码架构概览（基于 include/ 与 src/）

本文件总结当前实现的真实代码结构与数据流，避免与设计草案混淆。

## 1. 总体流程（从 CUDA 程序到 PTX 仿真）
- 入口为 fake libcudart：`src/cudart/cudart_sim.cpp` 实现 `__cudaRegisterFatBinary`、`cudaLaunchKernel` 等 API。
- 通过 `cuobjdump` 提取 PTX（`utils/cubin_utils.h`），使用 ANTLR4 解析生成 AST 与 IR（`include/ptx_parser/ptx_parser.h` 中 `PtxListener`）。
- `PtxListener` 填充 `PtxContext` → `KernelContext` → `StatementContext`/`OperandContext`（见 `include/ptx_ir/`）。
- `PtxInterpreter`（`src/cudart/ptx_interpreter.*`）构建符号表与内存映射，提交 `KernelLaunchRequest` 给 `GPUContext`。

## 2. 执行层次结构（执行核心）
- 顶层：`GPUContext`（`include/ptxsim/gpu_context.h`）维护 SM 列表、任务队列、全局内存 `SimpleMemory`。
- SM：`SMContext`（`include/ptxsim/sm_context.h`）负责资源预留、block 管理、warp 调度与 barrier 同步。
- CTA：`CTAContext`（`include/ptxsim/cta_context.h`）持有 warp、构建共享/本地内存符号表并转移所有权给 SM。
- Warp：`WarpContext`（`include/ptxsim/warp_context.h`）维护 active mask、分歧与 warp 级 PC，并驱动线程执行。
- Thread：`ThreadContext`（`include/ptxsim/thread_context.h`）实现操作数获取/提交、条件码寄存器、调用栈和 barrier 状态。

## 3. PTX IR 与解析层
- `PtxContext` 聚合 `KernelContext` 与全局 `StatementContext`（`include/ptx_ir/ptx_context.h`）。
- `KernelContext` 记录 `.entry/.func`、参数、`.shared/.const/.global` 资源、ABI Preserve 等（`include/ptx_ir/kernel_context.h`）。
- `StatementContext` 使用 `InstrVariant` 表示所有指令形态（`include/ptx_ir/statement_context.h`）。
- `OperandContext` 表示寄存器/立即数/地址/向量/谓词（`include/ptx_ir/operand_context.h`）。

## 4. 指令执行框架
- 指令分发由 `InstructionFactory::initialize()` 注册处理器（`include/ptxsim/instruction_factory.h`）。
- 所有处理器继承 `InstructionHandler`/`INSTR_BASE`（`include/ptxsim/instruction_base.h`），具体实现位于 `src/ptxsim/instructions/`。
- `InstructionHandlers` 的声明通过 `ptx_ir/ptx_op.def` 自动生成（`include/ptxsim/instruction_handlers.h`）。

## 5. 内存与寄存器模型
- 全局内存：`SimpleMemory`（`include/memory/simple_memory.h`）作为设备内存池。
- 访问接口：`HardwareMemoryManager`（`include/memory/hardware_memory_manager.h`）用于指令执行时的读写。
- 共享内存：`SharedMemoryManager`（`include/memory/shared_memory_manager.h`）按 SM 管理，由 `ResourceManager` 管理生命周期。
- 寄存器：`RegisterBankManager` 以 CTA 为单位统一分配（`include/register/register_bank_manager.h`）。
- 条件码：`ConditionCodeRegister` 存于 `ThreadContext`（`include/register/condition_code_register.h`）。

## 6. 配置与调试
- 日志与断点：`ptxsim::DebugConfig`、`LoggerConfig`（`include/ptxsim/ptx_config.h`），配置来自 `configs/config.ini` 与 `configs/debug_config.ini`。
- GPU 架构参数由 JSON 驱动（`configs/*.json`），在 `GPUContext::load_json_config()` 中加载。

## 7. 实际执行路径（简化时序）
1) `__cudaRegisterFatBinary` 解析 PTX → `PtxContext`。
2) `cudaLaunchKernel` 调用 `PtxInterpreter::launchPtxInterpreter()`。
3) `PtxInterpreter` 建立参数/常量/全局/本地内存符号表并提交请求。
4) `GPUContext` 分发到 `SMContext`，SM 构建 `CTAContext` 与 `WarpContext`。
5) `WarpContext` 迭代执行 `ThreadContext::execute_thread_instruction()`，由 `InstructionFactory` 分发到具体指令处理器。

> 以上内容以当前实现为准，后续如有新增模块，请补充对应 include/ 与 src/ 的入口与数据流。

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