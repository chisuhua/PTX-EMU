# PTX-EMU SIMT 中期架构改进 - 详细实施计划

> **状态**: 🔄 初步制定中 (等待架构深度梳理完成以细化)  
> **目标**: 修复 divergent execution, 完善 barrier 同步, 实现正确的 SIMT 语义

---

## 📋 Wave 1: 修复核心问题 (1-2 天)

### 1.1 回退破坏性修改，恢复测试 1, 2 通过
**问题**: 之前的 barrier PC 修改导致测试 1, 2 失败
**行动**: 
- 回退 barrier.cpp 中的 `barrier_pc + 1` 逻辑
- 恢复原有的 reconvergence_pc 计算
**验证**: test_syncthreads Test 1, 2 通过

### 1.2 修复 divergent 执行 (不破坏 Test 1, 2)
**问题**: 不同 PC 组线程无法执行不同指令
**根因**: `execute_warp_instruction` 对所有 lanes 执行 SAME statement
**方案**:
1. 保留 `get_lanes_by_pc()` 方法
2. 修改 scheduler 按 PC 组调度
3. 修改 `execute_warp_instruction` 检查 lanes 的 PC
**验证**: test_syncthreads Test 3 通过，Test 1, 2 仍通过

---

## 📋 Wave 2: 架构重构 (3-5 天)

### 2.1 统一 Barrier 模型
**问题**: warp-level barrier 和 CTA-level barrier 不协调
**方案**:
- 实现统一的 BarrierManager
- 支持 warp-level 和 CTA-level 两种 scope
- 正确处理 divergent 路径中的 barrier
**文件**: 
- 新建: `include/ptxsim/barrier_manager.h`
- 新建: `src/ptxsim/core/barrier_manager.cpp`

### 2.2 完善 SIMT Stack
**问题**: SIMT Stack 存在但未正确使用
**方案**:
- 实现正确的 push/pop 语义
- 支持 nested divergence
- 与 barrier 模型协同工作
**文件**: `simt_stack.h`, `warp_context.cpp`

### 2.3 实现指令缓冲区 (IBUFFER)
**参考**: GPGPU-Sim `shd_warp_t::m_ibuffer[IBUFFER_SIZE]`
**方案**:
- 为每个 warp 维护指令缓冲区
- 缓冲区包含来自不同 PC 组的指令
- 调度器从缓冲区取出指令分发
**文件**:
- 新建: `include/ptxsim/instruction_buffer.h`
- 修改: `warp_context.h`, `warp_scheduler.h`

---

## 📋 Wave 3: 增强与测试 (2-3 天)

### 3.1 添加单元测试
- Divergent execution 测试
- Nested barrier 测试
- SIMT stack 测试

### 3.2 回归测试
- 运行所有现有测试
- 确保无破坏性更改

### 3.3 性能分析
- 测量重构前后的性能
- 分析瓶颈

---

## 🎯 成功标准

| Wave | 标准 | 验证方法 |
|------|------|---------|
| 1 | Test 1, 2, 3 全部通过 | `test_syncthreads` |
| 2 | 支持 nested divergence | 添加测试用例 |
| 2 | Barrier 在 divergent 路径中正确工作 | 添加测试用例 |
| 3 | 所有现有测试通过 | `ctest` |
| 3 | 无性能回退 | benchmark 对比 |

---

## ⚠️ 风险和缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| Wave 1 修改引入回退 | 破坏现有功能 | 小步提交，每步验证 |
| SIMT Stack 语义复杂 | 实现错误 | 参考 GPGPU-Sim 实现 |
| IBUFFER 增加复杂度 | 维护成本增加 | 保持简单，充分测试 |

---

*最后更新: 2026-04-10 22:21*
