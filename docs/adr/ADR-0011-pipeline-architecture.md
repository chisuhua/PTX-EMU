# ADR-0011: PTX→PTXIR 多阶段 Pipeline 架构

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
| **日期** | 2026-05-05（初始 Proposed）→ 2026-07-30（升级 Accepted）|
| **关联任务** | Phase 12.1 (Sprint 12.1) |
| **关联 ADR** | [ADR-0023](./ADR-0023-ptxir-binary-format.md)（PTXIR 二进制格式与 7 项决策，提供本 ADR Stage 3-4 的格式依据）|
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

### Pipeline 结构

- [ ] 每个阶段可独立执行
- [ ] 阶段间通过标准接口通信
- [ ] 支持配置起点和终点
- [ ] PTXIR Roundtrip 验证通过

### 与 ADR-0023 的对齐

- [ ] **Stage 3 (PTXIR_SERIALIZE)**: 必须使用 `ptxir_serialization` API（[`include/ptxir/ptxir_serialization.h`](../../include/ptxir/ptxir_serialization.h)）— 不允许绕过 API 直接调用 `PtxirWriter`/`PtxirReader`
- [ ] **Stage 4 (PTXIR_DESERIALIZE)**: 反序列化产物必须为 `std::vector<StatementContext>`，格式契约遵循 [ADR-0023 §Decision 1](./ADR-0023-ptxir-binary-format.md#decision-1-文件格式--扁平二进制--section-toc非-bitstream)
- [ ] **Stage 3/4 的 PTXIR 格式**: 遵循 [ADR-0023 §7 项决策](./ADR-0023-ptxir-binary-format.md#决策内容) — 扁平二进制 + TOC + 值枚举 + 字符串表末尾 + Extend-Only 版本 + `include/ptx_ir/` 放置 + CFG 反序列化后应用
- [ ] **配置可执行路径**（详见 [ADR-0023 §合规检查 - API 契约](./ADR-0023-ptxir-binary-format.md#合规检查)）:
  - "仅生成 PTXIR" 路径（PTX_EXTRACT → PTXIR_SERIALIZE）必须输出可被 Stage 4 正确解析的 `.ptxir`
  - "从 PTXIR 执行" 路径（PTXIR_DESERIALIZE → SIMT_EXECUTE）必须通过 `deserialize_statements()` 反序列化
  - "Roundtrip 验证" 路径（PTX_PARSE → PTXIR_DESERIALIZE）必须 roundtrip 一致（详见 [ADR-0023 §合规检查 - 指令覆盖](./ADR-0023-ptxir-binary-format.md#指令覆盖)）

### 当前实现差距

以下项**已识别但未修复**（参考 [差距分析](../architecture/ptxir-serialization-gaps-gap-analysis.md)），Pipeline 实施时需在 Phase 1 修复：

- [ ] **G9**: Reader 补充 12 种缺失指令类型（Writer 24/Reader 12 不对称）
- [ ] **G8 + D1-D5**: Writer/Reader 格式对齐（TOC 写入、字符串表偏移回填、Reader 按 TOC 定位 section）
- [ ] **G1**: Pipeline 集成需要 roundtrip 测试覆盖（当前无任何 PTXIR 测试）
- [ ] **G3**: Stage 3 实际使用 `serialize_statements()`（✅ 已有），`generate_ptxir()` 工具缺失
- [ ] **G4**: Stage 4 实际使用 `deserialize_statements()`（✅ 已有），`load_ptxir(apply_cfg)` 中 `apply_cfg=true` 路径需 `CFGBuilder::build()` 集成（未实现）

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-05 | 初始版本（Proposed）| PTX-EMU Team |
| 2026-07-30 | 升级为 Accepted：引用 [ADR-0023](./ADR-0023-ptxir-binary-format.md) 作为 PTXIR 格式决策依据；补充对齐 checklist；标记当前实现差距（G1/G3/G4/G8/G9/D1-D5）；待 Phase 1 修复 | PTX-EMU Architecture Team |

## 参考

### 关联 ADR

- [ADR-0023](./ADR-0023-ptxir-binary-format.md) — PTXIR 二进制格式与 7 项决策（本 ADR Stage 3-4 的格式依据）
- [ADR-0009](./ADR-0009-xmacro-instruction-dispatch.md) — X-Macro 指令分发（Stage 5 SIMT_EXECUTE 通过此机制）
- [ADR-0010](./ADR-0010-fake-cuda-runtime.md) — Fake CUDA Runtime 拦截（Stage 0-1 的入口点 `__cudaRegisterFatBinary`）
- [ADR-0012](./ADR-0012-per-thread-pc.md) — Per-Thread PC（Stage 5 SIMT_EXECUTE 的核心数据模型）

### 关联 OpenSpec change

- [`openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/`](../../openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/) — PTXIR 完整设计文档
- [`openspec/changes/archive/2026-07-29-refactor-ptxir-writer/`](../../openspec/changes/archive/2026-07-29-refactor-ptxir-writer/) — Writer 长函数拆分（C-4 债务修复）

### 关联文档

- [差距分析](../architecture/ptxir-serialization-gaps-gap-analysis.md) — 当前实现与 ADR-0023 决策的差距清单
- [技能文档](../../.opencode/skills/ptxir-serialization/SKILL.md) — PTXIR 格式规范 + API 参考

### 历史参考

- [架构评审报告 - 第八节 Pipeline 方案](../reports/architecture-review-report.md#八ptx--ptxir-多阶段-pipeline-执行方案)
- [任务计划 - Sprint 12.1](../reports/task-plan.md#sprint-121-ptxir-pipeline-核心day-11-17)
