# Barrier 编程参考指南

**项目**: PTX-EMU  
**状态**: 初稿  
**日期**: 2026-05-27  
**作者**: Sisyphus

---

## 1. 概述

本文档总结 PTX-EMU 中 barrier（屏障）同步机制的实现原理、调试经验和方法论，为后续 barrier 相关开发提供参考。

### 1.1 文档目的

- 记录 barrier 同步的核心语义和实现机制
- 沉淀调试方法论，加速问题定位
- 提供常见陷阱和解决方案

---

## 2. Barrier 核心语义

### 2.1 什么是 Barrier

```
bar.warp.sync 的本质是"强制汇合点"，而非"等待点"
```

**Barrier 的行为**：
1. **检测分歧**：当 warp 执行到 barrier 指令时，检查是否有多个不同 PC 的线程
2. **强制汇合**：将分散在各处的线程强制汇聚到 barrier 所在 PC
3. **初始化 wbar**：记录所有参与的 lanes，设置期望参与数
4. **等待完成**：所有 lanes 到达后，一起离开 barrier，继续执行

### 2.2 Barrier vs 普通分支

| 特性 | 普通分支 (if/else) | Barrier 同步 |
|------|-------------------|-------------|
| 分歧处理 | 允许 lanes 在不同路径执行 | 强制所有 lanes 汇合到一点 |
| 执行模式 | SIMT 并行 | SIMT 串行（汇合阶段） |
| PC 状态 | 多 PC 共存 | 强制单 PC |
| 典型用途 | 条件计算 | 共享数据同步 |

---

## 3. 实现架构

### 3.1 关键组件

```
bar.warp.sync 处理流程：
┌─────────────────────────────────────────────────────────────┐
│ BarWarpSyncHandler::processOperation()     (barrier.cpp)   │
│  ├─ 检测分歧：unique_pcs = warp_ctx->get_unique_pcs()     │
│  ├─ 触发强制汇合：warp_ctx->force_reconvergence_at_barrier()│
│  ├─ 初始化 wbar：init_wbar.init(0xFFFFFFFF, reconv_pc)    │
│  └─ 设置状态：current_wbar_id = 0                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ WarpContext::force_reconvergence_at_barrier() (warp_context.cpp) │
│  └─ 只推进 pc < barrier_pc 的 lanes                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 关键状态

```cpp
// WarpState 中的 barrier 状态
struct WarpState {
    std::array<ThreadState, 32> threads;  // 每个 lane 的状态
    uint32_t exec_mask = 0xFFFFFFFF;      // 当前活跃的 lanes
    std::array<Wbar, 4> wbars;            // 4 个 warp-level barriers
    int current_wbar_id = -1;             // -1 = 无 barrier，0-3 = barrier ID
};

// Wbar 结构
struct Wbar {
    uint32_t participation_mask = 0;      // 期望参与的线程掩码
    uint32_t arrived_mask = 0;             // 实际到达的线程掩码
    int reconvergence_pc = -1;             // 汇合点 PC
    uint32_t barrier_pc = 0;              // barrier 所在 PC
    bool is_initialized = false;
    int expected_count = 0;                // 期望参与线程数
};
```

---

## 4. 调试方法论

### 4.1 观察数据流

**基本原则**：先确认数据，再看代码

```
观察点：h_output[tid] = value
        ↓
含义：每个 thread 在 barrier 后的执行结果
        ↓
判断信号：
  - 输出全 0 → barrier 根本没工作，所有 lane 都在 idle
  - 分支值正确但 barrier 后值错误 → reconvergence 有问题
  - 部分值正确 → 可能只有部分 lane 到达了 barrier
```

### 4.2 调试检查清单

```
□ 1. barrier 指令执行前：检查 unique_pcs 是否 > 1（分歧检测）
□ 2. barrier 处理时：检查 current_wbar_id 是否从 -1 变为 0
□ 3. wbar 初始化：检查 participation_mask 是否为 0xFFFFFFFF
□ 4. force_reconvergence：检查 pc < barrier_pc 的 lanes 是否被正确推进
□ 5. barrier 完成：检查 arrived_mask 是否等于 participation_mask
□ 6. barrier 后：检查 exec_mask 是否恢复为原始值
```

### 4.3 典型信号与根因

| 观测现象 | 可能根因 |
|---------|---------|
| `h_output[tid]` 全为 0 | barrier 未工作，所有 lane 未执行 |
| lanes 0-15 值正确，16-31 全 0 | lanes 16-31 未到达 barrier 或 barrier 后未执行 |
| 分支值正确，barrier 后值错误 | reconvergence 逻辑有问题 |
| `arrived=0/32` 立即完成 | wbar 未正确初始化 |

---

## 5. 常见陷阱与解决方案

### 5.1 陷阱列表

| 陷阱 | 后果 | 解决方案 |
|------|------|---------|
| `current_wbar_id < 0` 时未设置 | wbar 状态混乱，死锁 | 在检测到分歧后立即设置 `current_wbar_id = 0` |
| 初始化 wbar 时未指定正确 mask | 部分 lane 被忽略 | 使用 `0xFFFFFFFF` 让所有 32 个 lanes 都参与 |
| 强制汇合时把在 barrier_pc 的 lane 也推进 | 它们会跳过 barrier 指令 | `force_reconvergence` 只推进 `pc < barrier_pc` 的 lanes |
| 未调用 `force_reconvergence_at_barrier` | 分歧的 lanes 不会汇合 | 检测到 `unique_pcs > 1` 时必须调用 |
| barrier 处理后未设置 `set_pc_overridden(true)` | PC 可能被错误推进 | 设置标志防止重复推进 |

### 5.2 修复位置选择原则

| 位置 | 作用 |
|------|------|
| `barrier.cpp` (BarWarpSyncHandler) | 检测分歧 + 触发强制汇合 |
| `warp_context.cpp` (force_reconvergence) | 执行汇合逻辑 |

**原则**：处理程序在 barrier.cpp，汇合逻辑在 warp_context.cpp

---

## 6. 验证方法

### 6.1 测试用例设计

**三模式测试**：
- **模式 1 (PTX)**：编写 PTX 代码直接测试 barrier 语法
- **模式 2 (IR)**：验证 IR 序列化/反序列化正确
- **模式 3 (E2E)**：编译 CUDA kernel，提取 PTX，完整执行验证

**推荐测试场景**：
```
1. 简单 barrier：所有 32 lanes 同步执行
2. 分歧 + barrier：lanes 0-15 和 16-31 分别执行不同路径后在 barrier 汇合
3. 嵌套 barrier：barrier 后再分歧，再 barrier
```

### 6.2 验证命令

```bash
# PTX 语法测试
./tests/ptx/test_all_ptx.sh

# 单元测试
cd build && ctest -L barrier -V

# E2E 测试
./build/bin/test_divergence_sync_standalone
```

---

## 7. 经验总结

### 7.1 核心经验

1. **先确认数据，再看代码** - 输出值能直接告诉你 barrier 是否工作
2. **barrier 语义是"汇合点"不是"等待点"** - 理解这点就不会在错误的地方修
3. **wbar 状态机要初始化完整** - active_mask、current_wbar_id、wbar itself 都要设置
4. **force_reconvergence 只推进落后的，不动已经在目标 PC 的**
5. **分歧检测是 barrier 处理的第一步** - 没有分歧就不需要强制汇合

### 7.2 代码审查要点

```
bar.warp.sync 实现审查清单：
□ 1. 是否正确检测分歧（unique_pcs > 1）？
□ 2. 检测到分歧后是否调用 force_reconvergence_at_barrier？
□ 3. current_wbar_id 是否从 -1 设置为有效值？
□ 4. wbar 是否用正确的 mask（0xFFFFFFFF）初始化？
□ 5. force_reconvergence 是否只推进 pc < barrier_pc 的 lanes？
□ 6. 是否正确设置 pc_overridden 标志？
□ 7. barrier 处理完成后状态是否正确恢复？
```

---

## 8. 相关文档

| 文档 | 说明 |
|------|------|
| `docs/technical_design/barrier_module_design.md` | Barrier 模块技术设计 |
| `docs/skills/ptx-barrier-mechanism/SKILL.md` | Barrier 机制技能 |
| `docs/skills/ptx-instruction-pipeline/SKILL.md` | 指令执行流水线技能 |

---

## 9. 变更历史

| 日期 | 版本 | 变更内容 |
|------|------|---------|
| 2026-05-27 | v0.1 | 初稿创建，记录 bar.warp.sync 修复经验 |