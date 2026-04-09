# SIMT 架构集成实施计划

**Created**: 2026-04-03  
**Goal**: 完成 SIMT 架构完整集成，使 test_warp_divergence 测试通过  
**Status**: 📋 Planning → ⏳ Ready to Execute

---

## 🎯 目标

1. **完成 SIMT 架构集成** - 将已实现的数据结构 (ThreadState, WarpState, Wbar, ExecMask) 完全集成到执行流程中
2. **通过 test_warp_divergence 测试** - 证明新架构可以正确处理 warp 级发散和同步
3. **保持向后兼容** - 所有现有测试保持通过

---

## 📊 当前状态

### ✅ 已完成 (Stages 1-2)

| 组件 | 文件 | 状态 | 测试 |
|------|------|------|------|
| **Per-Thread PC** | `include/ptxsim/thread_state.h` | ✅ 完成 | ✅ PASS (11 tests, 294 assertions) |
| **WarpState** | `include/ptxsim/warp_state.h` | ✅ 完成 | ✅ PASS (integrated in test_simt) |
| **ExecMask** | `include/ptxsim/exec_mask.h` | ✅ 完成 | ✅ PASS (integrated in test_simt) |
| **Wbar Mechanism** | `include/ptxsim/wbar.h` | ✅ 完成 | ✅ PASS (12 tests, 265 assertions) |
| **Scheduler Config** | `include/ptxsim/scheduler_config.h` | ✅ 完成 | ✅ PASS (9 tests, 60 assertions) |
| **Grammar Rules** | `src/grammar/ptxInstructions.g4` | ✅ 完成 | ⏳ 需要 visitor |
| **Opcodes** | `include/ptx_ir/ptx_op.def` | ✅ 添加 S_BAR_WARP_SYNC, S_ACTIVEMASK | - |

### ⏳ 待完成 (Stages 2b-4)

| 阶段 | 任务 | 阻塞测试 | 预计工作量 |
|------|------|---------|-----------|
| **Stage 2b** | Parser Visitor 实现 | test_warp_divergence | 2-3 小时 |
| **Stage 3** | Instruction Dispatch 更新 | test_warp_divergence | 2-3 小时 |
| **Stage 4a** | Warp Scheduler 集成 | test_warp_divergence | 3-4 小时 |
| **Stage 4b** | __syncthreads() 翻译层 | test_warp_divergence | 2-3 小时 |
| **Stage 5** | End-to-End 测试验证 | - | 1-2 小时 |

---

## 📋 详细实施清单

### Stage 1: 数据结构 (✅ 已完成)

- [x] 创建 ThreadState 结构体 (per-thread PC)
- [x] 创建 WarpState 结构体 (32-thread array)
- [x] 创建 ExecMask 工具类
- [x] 创建 Wbar 收敛屏障
- [x] 创建 SchedulerConfig INI 解析器
- [x] 集成到 WarpContext (API 已添加)
- [x] 编写单元测试 (3 test files, 32 tests, 619 assertions)
- [x] 所有单元测试通过 ✅

### Stage 2: 语法解析 (⏳ In Progress)

#### 2a: 语法扩展 (✅ 已完成)
- [x] 添加 `bar.warp.sync` 语法到 `ptxInstructions.g4`
- [x] 添加 `activemask.b32` 语法到 `ptxInstructions.g4`
- [x] 添加 `S_BAR_WARP_SYNC` opcode 到 `ptx_op.def`
- [x] 添加 `S_ACTIVEMASK` opcode 到 `ptx_op.def`

#### 2b: Parser Visitor (⏳ 待实施)
- [ ] 实现 `PtxVisitor::visitBarWarpSyncInst()`
  - [ ] 解析参与掩码 (mask 操作数)
  - [ ] 解析汇合点 PC (labelOperand)
  - [ ] 创建 Wbar 初始化指令
  - [ ] 设置 warp_state.wbars[index]
- [ ] 实现 `PtxVisitor::visitActivemaskInst()`
  - [ ] 读取当前执行掩码
  - [ ] 返回到目标寄存器
- [ ] 更新 `ptx_visitor.h` 添加方法声明
- [ ] 测试 parser 输出 (使用简单 PTX 示例)

### Stage 3: 指令执行 (⏳ 待实施)

#### 3a: Instruction Handler (⏳ 待实施)
- [ ] 创建 `BarWarpSyncHandler` 类
  - [ ] 继承自 `InstructionHandler` 基类
  - [ ] 实现 `execute()` 方法
  - [ ] 调用 `Wbar::arrive()` 标记到达
  - [ ] 检查 `Wbar::is_complete()`
  - [ ] 更新 PC 到 reconvergence point
- [ ] 创建 `ActivemaskHandler` 类
  - [ ] 读取 `warp_state.exec_mask`
  - [ ] 写入目标寄存器
- [ ] 注册 handler 到 instruction factory

#### 3b: Instruction Dispatch (⏳ 待实施)
- [ ] 更新 `instruction_handlers.cpp`
  - [ ] 添加 `S_BAR_WARP_SYNC` case
  - [ ] 添加 `S_ACTIVEMASK` case
  - [ ] 删除或弃用旧的 `S_BAR_SYNC` handler (向后兼容保留)
- [ ] 删除或重构 `BarHandler::executeBarrier()`
  - [ ] 保留用于旧指令的兼容模式
  - [ ] 新指令使用 Wbar 机制

### Stage 4: 调度器集成 (⏳ 待实施)

#### 4a: Warp Scheduler (⏳ 待实施)
- [ ] 修改 `WarpContext::execute_warp_instruction()`
  - [ ] 从 per-warp PC 切换到 per-thread PC
  - [ ] 使用 `warp_state.threads[i].pc` 代替 `pc`
  - [ ] 支持 divergent PC 值
- [ ] 更新 warp scheduler 逻辑
  - [ ] 实现 based on priority scheduling
  - [ ] 从 scheduler_config.ini 读取权重
  - [ ] anti-starvation 机制
- [ ] 更新 ThreadContext 状态同步
  - [ ] ThreadContext ↔ WarpState 同步
  - [ ] is_blocked ↔ barrier waiting 映射

#### 4b: 翻译层 (⏳ 待实施)
- [ ] __syncthreads() → bar.warp.sync 翻译
  - [ ] 在 PTX 生成阶段或 parser 阶段翻译
  - [ ] `bar.sync %bar0, ALL;` → `bar.warp.sync 0xFFFFFFFF, reconvergence_pc;`
- [ ] 确保 CTA-level barrier 仍使用 mbarrier 机制
  - [ ] 不要破坏现有的 __syncthreads_array() 等
- [ ] 测试翻译正确性
  - [ ] 使用 nvcc --ptxas-options=-v 验证 PTX 输出

### Stage 5: 测试验证 (⏳ 待实施)

- [ ] 编译 test_warp_divergence.cu
- [ ] 运行测试，观察输出
- [ ] 如果失败，使用 debugger 定位
  - [ ] 检查 PC 值是否正确 diverge
  - [ ] 检查 Wbar 是否正确初始化
  - [ ] 检查 is_complete() 逻辑
- [ ] 记录测试结果
- [ ] 性能分析 (可选)
  - [ ] 对比新旧架构性能差异
  - [ ] 确保 overhead 可接受

---

## 🔍 关键依赖

### 文件依赖

| 文件 | 需要修改 | 依赖 |
|------|---------|------|
| `src/ptx_parser/ptx_visitor.cpp` | ✅ Yes | thread_state.h, wbar.h |
| `src/ptx_parser/ptx_visitor.h` | ✅ Yes | - |
| `src/ptxsim/instructions/barrier.cpp` | ✅ Yes | wbar.h, warp_state.h |
| `src/ptxsim/core/warp_context.cpp` | ✅ Yes | warp_state.h |
| `src/ptxsim/instruction_handlers.cpp` | ✅ Yes | ptx_op.def |

### 测试依赖

| 测试 | 依赖 | 预期结果 |
|------|------|---------|
| `test_simt_thread_pc` | 无 | ✅ 已通过 |
| `test_warp_barrier` | 无 | ✅ 已通过 |
| `test_scheduler_config` | 无 | ✅ 已通过 |
| `test_warp_divergence` | Stage 2b-4 | ⏳ 等待集成 |
| `test_divergence_sync_standalone` | Stage 2b-4 | ⏳ 等待集成 |

---

## 📅 预计时间线

| 阶段 | 预计时间 | 累计时间 |
|------|---------|---------|
| Stage 2b: Parser Visitor | 2-3 小时 | 2-3 小时 |
| Stage 3: Instruction Dispatch | 2-3 小时 | 4-6 小时 |
| Stage 4: Scheduler Integration | 4-6 小时 | 8-12 小时 |
| Stage 5: Testing & Debugging | 2-3 小时 | 10-15 小时 |

**总预计**: 10-15 小时有效工作时间  
**实际排期**: 1-2 个工作日 (包含编译、调试、验证)

---

## 🎯 成功标准

### 必须满足 (MVP)
- [ ] test_warp_divergence 测试通过 (无 timeout)
- [ ] 所有现有单元测试保持通过
- [ ] 无编译错误或警告
- [ ] LSP 诊断 clean

### 期望满足 (Stretch Goals)
- [ ] test_divergence_sync_standalone 通过
- [ ] 性能 overhead < 5%
- [ ] 文档更新完成
- [ ] 代码审查通过 (评分 > 90/100)

---

## 🔧 开发环境

### 构建命令
```bash
# 环境设置
. env.sh

# 配置 (Debug 模式便于调试)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

# 构建特定目标
cmake --build build --target test_simt_thread_pc
cmake --build build --target test_warp_barrier
cmake --build build --target test_scheduler_config

# 运行测试
cd build && ctest -R test_simt -V
cd build && ctest -R test_warp_barrier -V
cd build && ctest -R test_scheduler_config -V
```

### 调试工具
- GDB: `gdb -batch -ex "bt" ./test_binary`
- ASan: 编译时加 `-fsanitize=address`
- ncu: `ncu ./bin/test_warp_divergence`

---

## 📝 会议记录

### 用户决策
1. ✅ 使用 "wbar" (或 warpbar/convergebar) 命名 warp barrier，区别于 CTA barrier
2. ✅ Anti-Starvation 调度通过配置文件实现，不硬编码
3. ✅ Phase 和 tx_count 对 mbarrier 是必要的
4. ✅ 支持 bar.warp.sync 和 activemask.b32 作为 warp-level 同步原语

### 技术决策
1. **Per-Thread PC** (vs Per-Warp PC) - 必要，用于解决自旋锁死锁
2. **Convergence Barrier** (vs Branch Stack) - PTX 兼容，支持无限嵌套
3. **ExecMask 独立管理** - 快速路径，避免每次都扫描 32 个线程
4. **4 Wbar Registers** - 典型硬件配置，足够支持大多数场景

---

## 📄 参考文献

- PTX ISA v6.0+: https://docs.nvidia.com/cuda/parallel-thread-execution/
- PTX ISA v7.0+ (MBarrier): https://docs.nvidia.com/cuda/parallel-thread-execution/
- GPGPU-Sim 实现：https://github.com/accel-sim/gpgpu-sim_distribution
- 现有文档：`docs/architecture-upgrade/ARCHITECTURE-INTEGRATION-STATUS.md`

---

## 🚀 下一步行动

**立即开始**: Stage 2b - Parser Visitor 实现

1. 打开 `src/ptx_parser/ptx_visitor.cpp`
2. 找到 `visitBarWarpSyncInst` stub
3. 实现完整的解析逻辑
4. 编译测试

**完成后继续**: Stage 3 - Instruction Dispatch

---

**Last Updated**: 2026-04-03  
**Next Review**: After Stage 2b completion
