## Why

PTXIR 二进制序列化的实现与 [ADR-0023](https://github.com/chisuhua/PTX-EMU/blob/main/docs/adr/ADR-0023-ptxir-binary-format.md) 定义的 7 项架构决策存在显著偏差。差距分析（[docs/architecture/ptxir-serialization-gaps-gap-analysis.md](../../../docs/architecture/ptxir-serialization-gaps-gap-analysis.md)）识别出 9 项能力差距（G1-G9）和 5 项格式偏差（D1-D5），其中最关键的是：
- Reader variant 覆盖仅 9/24（15 种 `InstrVariant` 类型走 `default` 分支静默跳过，G9；tasks.md §2.1-2.15 列出 15 个待补 case）
- Writer/Reader 格式实现未对齐 TOC 契约（D1-D5：TOC 未写入、字符串表偏移未回填、Reader 硬编码偏移）
- 完整 roundtrip 测试缺失（G1），无任何 PTXIR 单元测试

这些问题导致 PTXIR 在生产路径中的可靠性和前向兼容性无法保证，ADR-0023 升级的格式契约形同虚设。

## What Changes

- **修复 Writer 格式契约**（D1-D4）: 按 ADR-0023 Decision 1 写入 header → TOC entries → REGDECL section → KERNEL section → STRING_TABLE section 顺序；回填 `string_table_offset` / `string_table_size`
- **修复 Reader 格式契约**（D5）: 从 TOC 条目读取 section 偏移，移除硬编码 `sizeof(PtxirHeader)` 偏移
- **补全 Reader variant 覆盖**（G9）: 为缺失的 15 种 `InstrVariant` variant 类型（MembarInstr, FenceInstr, ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix, VoteInstr, ShflInstr, AtomInstr, TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, CpAsyncInstr, AbiDirective）添加 `case` 分支；移除 `default` 静默跳过
- **创建完整 roundtrip 测试套件**（G1）: `tests/unit/test_ptxir_serialization.cpp` 覆盖所有支持的指令类型
- **实现 `generate_ptxir()` 工具**（G3）: ANTLR 解析 + 序列化的离线工具
- **实现 `load_ptxir(apply_cfg)` 中的 `apply_cfg` 路径**（G4）: 集成 `CFGBuilder::build()` 到反序列化后流程
- **添加 ADR-0023 引用到现有 `StatementContext` 修改指引**（G2）: 同步 AGENTS.md，禁止静默改 StatementContext 而不更新 writer/reader

## Capabilities

### New Capabilities
- `ptxir-roundtrip-testing`: PTXIR 序列化 roundtrip 测试覆盖（所有 V1 支持的 24 种指令类型 + 错误路径）
- `ptxir-format-compliance`: 强制 Writer/Reader 实现符合 ADR-0023 §Decision 1（TOC 写入 + 字符串表偏移回填）
- `ptxir-coverage-parity`: Reader 指令覆盖与 Writer 对齐（消除 default 静默跳过）
- `ptxir-tooling-completion`: `generate_ptxir()` 离线工具 + `load_ptxir(apply_cfg=true)` 完整路径
- `ptxir-statement-context-change-protocol`: 修改 `StatementContext` / `OperandContext` 结构体时强制同步更新 PTXIR writer/reader 的协议（AGENTS.md 同步）

### Modified Capabilities
无现有 capability 改动。本次为纯增量修复，不修改现有 spec-level 行为。

## Impact

### 代码改动
- `include/ptx_ir/ptxir_format.h`: 移除未使用的 `header_size` 字段或补全文档；确保 TOC struct 正确
- `src/ptx_ir/ptxir_writer.cpp`: 重构 `write_header()` / `write_sections()` 以符合 ADR-0023 顺序；回填 offset
- `src/ptx_ir/ptxir_reader.cpp`: 重构 `read_header()` / `read_sections()` 以从 TOC 读取偏移；删除硬编码 `sizeof(PtxirHeader)`
- `src/ptx_ir/ptxir_reader.cpp`: 补全 15 种缺失 `InstrVariant` variant 类型的 `case` 分支；移除 `default` 静默跳过
- `src/ptxir/ptxir_serialization.cpp`: 新增 `generate_ptxir()` 和 `apply_cfg=true` 支持
- `include/ptxir/ptxir_serialization.h`: 添加 `generate_ptxir()` / `load_ptxir(apply_cfg)` 签名

### 测试改动
- **新增** `tests/unit/test_ptxir_serialization.cpp`: roundtrip 测试（每种指令类型 + 错误路径 + 跨架构字节序）
- **新增** `tests/integration/test_ptxir_pipeline_mode4.cpp`: Mode 4 端到端（PTX 文本 → ANTLR → 序列化 → 反序列化 → 执行结果对比 Mode 2）

### 文档改动
- `src/ptx_ir/AGENTS.md`: 新增「修改 StatementContext 时同步更新 PTXIR writer/reader」checklist
- `include/ptxir/AGENTS.md` (如不存在则新建): 公共头文件修改协议
- `docs/developer-guide/THREE-MODE-TESTING-GUIDE.md`: 升级为四模式（参考 [PTXIR 技能](../../../.opencode/skills/ptxir-serialization/SKILL.md) Mode 4 部分）
- `openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/tasks.md`: 标记 Phase 1 修复完成（task 10.1-10.4 解锁）

### 关联 ADR
- [ADR-0011](https://github.com/chisuhua/PTX-EMU/blob/main/docs/adr/ADR-0011-pipeline-architecture.md): PTX→PTXIR 多阶段 Pipeline 架构（Accepted 2026-07-30，引用本 change 作为 Stage 3-4 实施依据）
- [ADR-0023](https://github.com/chisuhua/PTX-EMU/blob/main/docs/adr/ADR-0023-ptxir-binary-format.md): PTXIR 二进制格式与 7 项架构决策（Accepted 2026-07-30，本 change 是其实施）

### 关联 skill
- [.opencode/skills/ptxir-serialization/SKILL.md](../../../.opencode/skills/ptxir-serialization/SKILL.md): 格式规范 + API 参考 + 工作流

### 关联 OpenSpec changes
- [archive/2026-06-09-ptxir-serialization-architecture](https://github.com/chisuhua/PTX-EMU/tree/main/openspec/changes/archive/2026-06-09-ptxir-serialization-architecture): 原始设计文档（7 项决策来源）
- [archive/2026-07-29-refactor-ptxir-writer](https://github.com/chisuhua/PTX-EMU/tree/main/openspec/changes/archive/2026-07-29-refactor-ptxir-writer): Writer 长函数拆分（C-4 债务修复，已完成）
- [archive/2026-06-09-ptxir-test-refactor](https://github.com/chisuhua/PTX-EMU/tree/main/openspec/changes/archive/2026-06-09-ptxir-test-refactor): Mode 4 roundtrip 测试设计

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- [x] Baseline 函数清单已列出（`write_header`, `write_sections`, `read_header`, `read_kernel_section`）
- [x] 跨模块状态翻译表（PTXIR 格式契约 → reader 行为对照表）已写入 design.md
- [x] 回退策略：每个 Phase 独立 commit、独立可 revert（详见 tasks.md Phase 1-4）

### 多 Phase 推进
- [x] Phase 拆分方案（4 个 Phase: P1 Reader 指令覆盖 / P2 格式契约修复 / P3 测试与工具链 / P4 文档同步）已写入 tasks.md
- [x] 基线 worktree 计划：`.worktrees/ptxir-baseline` 从 commit `5592886d`（当前 main HEAD）建立
- [x] 失败处理策略：任何测试回归 → 立即 revert 该 Phase commit，不混入后续 commit

### 文档同步
- [x] `src/ptx_ir/AGENTS.md` 同步项：新增 StatementContext 修改 checklist
- [x] `include/ptxir/AGENTS.md` 同步项：公共头文件修改协议（如不存在则新建）
- [x] ADR 追加：ADR-0011 已引用本 change；ADR-0023 不变（本 change 是其实施）
- [x] tasks.md Phase 状态变更：参考 [archive/2026-06-09-ptxir-serialization-architecture/tasks.md](https://github.com/chisuhua/PTX-EMU/blob/main/openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/tasks.md) 第 10 节「Verification」10.1-10.4 任务解锁

### 引用经验沉淀
- [ptx-lessons-learned](https://github.com/chisuhua/PTX-EMU/tree/main/.opencode/skills/ptx-lessons-learned): 跨模块状态翻译、递归锁、分 Phase commit、基线 worktree 等 14 核心经验 + 12 checklist
