# PTX-EMU SIMT 架构改进 - 问题记录

## 当前状态 (2026-04-10 22:21)

### 已完成修改 (commit f91e414 后)
1. ✅ 添加 `get_lanes_by_pc()` 和 `get_unique_pcs()` 方法
2. ✅ 修改 `execute_warp_instruction()` 支持 `target_pc` 参数
3. ✅ 修改 sm_context.cpp 按 PC 组调度
4. ✅ 修改 barrier.cpp 使用 `barrier_pc + 1` 代替静态 reconvergencePC

### 当前测试结果
```
Test 1 (basic barrier): FAIL at index 1: expected 30, got 0
Test 2 (multi-block): FAIL at block 0: expected 496, got 9796
Test 3 (nested sync):  仍在调试中
```

### 引入的问题
1. **Barrier PC 释放逻辑错误**: 修改 barrier.cpp 后，barrier 完成后线程被释放到错误 PC
2. **调度器问题**: 按 PC 组调度打破了原有的执行顺序

### 根本原因分析 (基于论文和 GPGPU-Sim)
1. **NVIDIA 硬件行为**: barrier 指令对整个 warp 生效，即使只有部分线程执行到 barrier
2. **GPGPU-Sim 做法**: 使用 per-warp PC (`m_next_pc`)，而不是 per-thread PC
3. **我们的设计**: 使用 per-thread PC，但调度器仍按 warp 调度

### 架构矛盾
- Per-thread PC 设计支持 Volta+ 独立线程调度
- 但当前实现只支持 per-warp 调度
- Result: threads with different PCs get mixed up

## 待解决的问题
1. 如何让 per-thread PC 设计与 per-warp 调度协同工作
2. 如何正确处理 divergent 路径中的 barrier
3. 如何实现正确的 SIMT Stack 语义

## 参考资料
- GPGPU-Sim: `src/gpgpu-sim/shader.cc` - per-warp PC, IBUFFER
- NVIDIA 论文: Tesla 架构 (Lindholm et al.)
- PTX ISA: barrier 语义

---

*Note: 等待架构深度梳理完成以更新此文档*
