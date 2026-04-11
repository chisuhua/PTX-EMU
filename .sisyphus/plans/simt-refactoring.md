# PTX-EMU SIMT 架构中期改进计划

> **状态**: 🔴 Wave 1.4 完成，Test 3 仍需修复  
> **目标**: 修复 divergent execution、完善 barrier 同步、实现正确的 SIMT 语义  
> **基于**: GPGPU-Sim 架构分析 + NVIDIA SIMT 论文 (Lindholm 2008, Fung 2007)  
> **开始日期**: 2026-04-10  

---

## 📊 当前状态 (2026-04-11 06:00)

| 测试 | 状态 | 说明 |
|------|------|------|
| Test 1 (basic barrier) | ✅ PASS | - |
| Test 2 (multi-block) | ✅ PASS | - |
| Test 3 (nested sync) | ❌ FAIL | expected 3, got 496 (需要更深入的寄存器追踪) |

---

## 🎯 已完成工作

### Wave 1.1: 修复 Barrier PC 计算 (commit: 84508ec)
- 恢复使用 reconvergence_pc (由编译器/visitor 设置)
- Test 1, 2 通过

### Wave 1.2: 实现 warp-level barrier 语义 (commit: 84508ec)
- 当 participation_mask == 0xFFFFFFFF 时，标记所有 lanes 为到达
- 遵循 NVIDIA PTX ISA 规范
- Test 3 不再 hang (barrier 完成)

### Wave 1.3: 正确的活跃线程管理 (commit: 176ddad)
- 构造函数初始化为非活跃
- add_thread 时设置活跃
- 防止 inactive lanes 污染执行

### Wave 1.4: 共享内存符号解析修复 (commit: 546e167)
- mov.u32 %reg, symbol 现在正确复制偏移值
- acquire_operand 始终返回 `&(share_it->second->val)`
- get_memory_addr 负责添加 shared_mem_space

---

## 🔍 剩余问题: Test 3 数据不正确

**症状**: `expected 3, got 496` at index 1

**496 = 0x1F0 分析**:
- 非常奇怪的值，不是正常的执行结果
- 可能来源: 未初始化内存、寄存器泄漏、cudaMemcpy 问题

**建议调试方法**:
1. 在 HardwareMemoryManager::access 中添加写入日志
2. 直接打印寄存器值 (使用 printf/PTX_INFO_EMU)
3. 验证 PTX 指令翻译是否正确
4. 对比 CUDA 实际硬件的输出

---

## 🏗️ 后续计划 (Wave 2)

### 2.1 完成 Test 3 修复
- 添加详细寄存器追踪
- 对比预期和实际寄存器值

### 2.2 SIMT Stack 完善
- 实现正确的 push/pop 语义
- 支持 nested divergence

### 2.3 指令缓冲区 (IBUFFER)
- 参考 GPGPU-Sim 实现

---

*最后更新: 2026-04-11 06:00*
