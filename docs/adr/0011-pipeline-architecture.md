# ADR-0011: PTX→PTXIR 多阶段 Pipeline 架构

| 属性 | 值 |
|------|-----|
| **状态** | Proposed |
| **日期** | 2026-05-05 |
| **关联任务** | Phase 12.1 (Sprint 12.1) |
| **作者** | PTX-EMU Team |

## 上下文

当前 PTX-EMU 的执行流程是**全链路即时执行**：

```
PTX 源码 → ANTLR4 解析 → PtxVisitor → StatementContext[] → CFG 分析 → SIMT 执行
```

这种设计存在以下问题：

1. **无法缓存**：每次运行都要重新解析 PTX，即使 PTX 代码完全相同
2. **无法独立测试**：难以单独验证 Parser、CFG Builder、或序列化组件
3. **无法增量优化**：想替换某个阶段需要重构整个流程
4. **PTXIR 未被充分利用**：已有的 PTXIR 序列化能力仅作为独立功能

## 决策驱动因素

1. **编译/运行时分离**：PTX 解析是一次性的，结果应可缓存
2. **可测试性**：每个阶段应能独立测试
3. **可扩展性**：阶段间应有标准接口，支持替换实现
4. **运行时性能**：通过 PTXIR 缓存避免重复解析

## 考虑的替代方案

### 方案 A: 保持现有全链路执行

**描述**: 不引入 Pipeline，保持现有架构

**优点**:
- 无架构变更

**缺点**:
- 无法缓存解析结果
- 无法独立测试各阶段
- 无法从 PTXIR 文件直接执行

### 方案 B: 简单的两阶段（编译 + 运行）

**描述**: 分为编译阶段（PTX→PTXIR）和运行阶段（PTXIR→执行）

**优点**:
- 简单
- 支持缓存

**缺点**:
- 粒度粗，无法独立测试中间阶段
- 无法配置起点/终点

### 方案 C: 可配置的多阶段 Pipeline (✅ 选中)

**描述**: 定义 6 个标准阶段，支持配置起点和终点，阶段间通过标准接口通信

**优点**:
- 每个阶段可独立测试
- 支持缓存（PTXIR 文件）
- 可配置执行路径
- 阶段可替换

**缺点**:
- 架构复杂度增加
- 需要迁移现有逻辑

**选择理由**: 多阶段 Pipeline 提供了最大的灵活性和可测试性，是项目走向生产就绪的必要架构改进。

## 决策内容

### 设计原则

1. **阶段枚举**：定义 6 个标准阶段（PTX_EXTRACT → SIMT_EXECUTE）
2. **可配置起点/终点**：支持从任意阶段开始，在任意阶段结束
3. **标准接口**：阶段间通过 StatementContext 或 PTXIR binary 通信
4. **中间产物可导出**：支持调试时导出各阶段的中间结果

### Pipeline 阶段定义

```cpp
enum class PipelineStage : uint8_t {
    PTX_EXTRACT = 0,         // Stage 0: cuobjdump 提取 PTX
    PTX_PARSE = 1,           // Stage 1: ANTLR4 解析
    CFG_ANALYZE = 2,         // Stage 2: CFG 后支配点分析
    PTXIR_SERIALIZE = 3,     // Stage 3: 序列化为 PTXIR
    PTXIR_DESERIALIZE = 4,   // Stage 4: 从 PTXIR 反序列化
    SIMT_EXECUTE = 5         // Stage 5: SIMT 执行
};

struct PipelineConfig {
    PipelineStage start_stage = PipelineStage::PTX_EXTRACT;
    PipelineStage end_stage = PipelineStage::SIMT_EXECUTE;
    
    std::string input_binary_path;      // Stage 0 输入
    std::string input_ptx_path;          // Stage 1 输入
    std::string input_ptxir_path;        // Stage 4 输入
    std::string output_ptxir_path;       // Stage 3 输出
    
    bool dump_ptx_after_parse = false;
    bool dump_cfg_after_analysis = false;
    bool dump_ptxir_after_serialize = true;
    bool verify_ptxir_roundtrip = false;
};
```

### 典型执行路径

| 场景 | 配置 | 用途 |
|------|------|------|
| 完整执行 | PTX_EXTRACT → SIMT_EXECUTE | 默认行为 |
| 仅生成 PTXIR | PTX_EXTRACT → PTXIR_SERIALIZE | 编译时预处理 |
| 从 PTXIR 执行 | PTXIR_DESERIALIZE → SIMT_EXECUTE | 运行时加速 |
| 从 PTX 执行 | PTX_PARSE → SIMT_EXECUTE | 调试模式 |
| 仅 CFG 分析 | PTX_PARSE → CFG_ANALYZE | 验证控制流 |
| Roundtrip 验证 | PTX_PARSE → PTXIR_DESERIALIZE | 序列化验证 |

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/pipeline_config.h` | 新增 | PipelineConfig 定义 |
| `include/ptxsim/pipeline.h` | 新增 | Pipeline 类 |
| `src/ptxsim/pipeline.cpp` | 新增 | Pipeline 实现 |
| `src/cudart/cudart_sim.cpp` | 修改 | 迁移执行逻辑到 Pipeline |

## 后果

### 正面影响

- 每个阶段可独立测试
- PTXIR 缓存避免重复解析
- 支持多种执行路径

### 负面影响

- 架构复杂度增加
- 迁移现有逻辑需要工作量

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Pipeline 与现有 cudart_sim 集成冲突 | 低 | 高 | 设计接口层，解耦依赖 |
| 阶段间数据传递不一致 | 低 | 高 | Roundtrip 验证测试 |
| 迁移引入回归 | 中 | 高 | 充分集成测试 |

## 合规检查

后续相关开发应检查：

- [ ] 每个阶段可独立执行
- [ ] 阶段间通过标准接口通信
- [ ] 支持配置起点和终点
- [ ] PTXIR Roundtrip 验证通过

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-05 | 初始版本 | PTX-EMU Team |

## 参考

- [架构评审报告 - 第八节 Pipeline 方案](../reports/architecture-review-report.md#八ptx--ptxir-多阶段-pipeline-执行方案)
- [任务计划 - Sprint 12.1](../reports/task-plan.md#sprint-121-ptxir-pipeline-核心day-11-17)
