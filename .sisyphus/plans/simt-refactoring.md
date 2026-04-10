# PTX-EMU SIMT 架构中期改进计划

> **状态**: 🚧 Wave 1 进行中 (Test 3 数据修复)  
> **目标**: 修复 divergent execution、完善 barrier 同步、实现正确的 SIMT 语义  
> **基于**: GPGPU-Sim 架构分析 + NVIDIA SIMT 论文 (Lindholm 2008, Fung 2007)  
> **开始日期**: 2026-04-10  

---

## 📊 当前状态 (2026-04-10 23:15)

| 测试 | 状态 | 说明 |
|------|------|------|
| Test 1 (basic barrier) | ✅ PASS | - |
| Test 2 (multi-block) | ✅ PASS | - |
| Test 3 (nested sync) | ⚠️ 部分通过 | Barrier 完成 (不再 hang)，但数据结果不对 (expected 3, got 496) |

---

## 🎯 已完成工作

### Wave 1.1: 修复 Barrier PC 计算  
- **问题**: barrier.cpp 使用 `barrier_pc + 1` 代替 reconvergence_pc
- **修复**: 恢复使用 reconvergence_pc (由编译器/visitor 设置)
- **影响**: Test 1, 2 通过

### Wave 1.2: 实现 warp-level barrier 语义
- **问题**: Divergent 路径中只有部分线程执行 barrier，barrier 永远不完成
- **根因**: NVIDIA PTX ISA 规定 `bar` 指令对整个 warp 生效
- **修复**: 当 participation_mask == 0xFFFFFFFF 时，标记所有 lanes 为到达
- **影响**: Test 3 不再 hang (barrier 完成)

---

## 🔍 当前问题: Test 3 数据结果不正确

**症状**: `expected 3, got 496`

**可能原因**:
1. Barrier 完成后 PC 更新导致执行顺序错误
2. Shared memory 访问在 divergent 路径中不同步
3. Divergent threads 的数据被覆盖

**分析**:
- Test 3 使用 `@%p1 bra $L__BB2_2;` 创建 divergent 路径
- 两个路径都访问 shared memory
- Barrier 用于同步两个路径
- 期望：每个 thread 写入 `tid` 到 `data_a[tid]`，从 `data_b[31-tid]` 读取
- 实际：得到 496 (可能是累积值或地址偏移)

---

## 🏗️ 后续计划

### Wave 1.3: 修复 Test 3 数据 (1-2 小时)
**调查**:
1. 检查 divergent 路径中的 shared memory 访问顺序
2. 验证 barrier 后 PC 更新是否正确
3. 追踪 shared memory 的读写值

### Wave 2: 架构改进 (3-5 天)
#### 2.1 完善 SIMT Stack
- 实现正确的 push/pop 语义
- 支持 nested divergence
- 与 barrier 协同工作

#### 2.2 实现指令缓冲区 (IBUFFER)
- 参考 GPGPU-Sim `shd_warp_t::m_ibuffer[IBUFFER_SIZE]`

#### 2.3 统一 Barrier 模型
- 整合 SIMT Stack 和 Barrier 状态

---

## 📚 参考资料

1. GPGPU-Sim SIMT 分析: `docs/architecture/GPGPU-SIM-SIMT-ANALYSIS.md`
2. SIMT 论文: Lindholm 2008, Fung 2007, Collange 2011
3. 架构深度梳理: 2026-04-10 完成

---

*最后更新: 2026-04-10 23:15*
