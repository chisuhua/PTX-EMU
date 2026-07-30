## Context

### 现状问题

PTXIR 二进制序列化的当前实现（`src/ptx_ir/ptxir_writer.cpp` / `ptxir_reader.cpp` / `src/ptxir/ptxir_serialization.cpp`）与 [ADR-0023](https://github.com/chisuhua/PTX-EMU/blob/main/docs/adr/ADR-0023-ptxir-binary-format.md) 定义的 7 项架构决策存在显著偏差。差距分析（[docs/architecture/ptxir-serialization-gaps-gap-analysis.md](../../../docs/architecture/ptxir-serialization-gaps-gap-analysis.md)）识别出 9 项能力差距（G1-G9）和 5 项格式偏差（D1-D5）。

#### 关键偏差

**1. 格式契约破坏（D1-D5）**
- Writer 写顺序与设计文档不一致（设计：header → TOC → REGDECL → KERNEL → STRING_TABLE；实际：header → string_table → kernel_section）
- Writer 未实际写入 TOC 条目（`section_count=2` 但 0 个 TOC entries）
- Writer 未回填 `string_table_offset` / `string_table_size`（始终为 0）
- Reader 硬编码 `sizeof(PtxirHeader)` 作为字符串表偏移，未使用 TOC 解析

**2. Reader 指令覆盖不足（G9）**
- Writer 支持 24 种指令类型（BranchInstr, LabelInstr, VoidInstr, BarrierInstr, GenericInstr, DeclarationInstr, BarWarpSyncInstr, PragmaInstr, DollarNameInstr, MembarInstr, FenceInstr, ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix, VoteInstr, ShflInstr, AtomInstr, TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, CpAsyncInstr, AbiDirective）
- Reader 仅显式支持 9 种 `InstrVariant` variant（22 个 opcode），其余 15 种 variant 走 `default` 分支**静默跳过数据**
- 后果：roundtrip 测试不通过；指令数据被无声丢弃

**3. 工具链不完整（G1, G3, G4）**
- 无 roundtrip 单元测试
- 无 `generate_ptxir()` 离线工具
- 无 `load_ptxir(apply_cfg=true)` 完整路径（参数存在但实现缺失）

#### 触发事件

- 2026-07-30: 差距分析文档完成
- 2026-07-30: [ADR-0023](https://github.com/chisuhua/PTX-EMU/blob/main/docs/adr/ADR-0023-ptxir-binary-format.md) Accepted（格式决策固化）
- 2026-07-30: [ADR-0011](https://github.com/chisuhua/PTX-EMU/blob/main/docs/adr/ADR-0011-pipeline-architecture.md) 从 Proposed 升级为 Accepted（Pipeline 架构生效）
- ADR-0011 升级时附带要求：Stage 3-4 的实施必须遵循 ADR-0023 格式契约

#### 技术约束

- **向后兼容**：现有 `serialize_statements` / `deserialize_statements` 签名不变
- **构建资源**：2 核系统 OOM 阻断 ANTLR 运行时编译（参考 [archive/2026-06-09-ptxir-serialization-architecture/tasks.md §10.5-10.6](https://github.com/chisuhua/PTX-EMU/blob/main/openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/tasks.md)）；CMake 应允许独立 build ptxir 单元测试
- **格式稳定性**：V1 头版固定，V2 增量添加；不得修改 `PtxirHeader` 字段布局
- **X-Macro 一致性**：任何新增 PTX 指令必须同步更新 `ptx_op.def` + `statement_context.h`（InstrVariant）+ writer + reader

### 目标状态

#### 1. Writer 完全符合 ADR-0023 Decision 1

```cpp
void PtxirWriter::write(const std::vector<StatementContext>& statements) {
    stmts_ = statements;
    pre_pass(statements);  // 值枚举
    write_header();         // 24B header + 预留 TOC 空间
    write_toc_entries();    // 写入 section_count 个 TOC 条目
    write_regdecl_section();// 操作数表
    write_kernel_section(); // 语句流
    write_string_table();   // 末尾
    backfill_header_offsets(); // 回填 string_table_offset/size + header_size
}
```

#### 2. Reader 严格按 TOC 解析

```cpp
std::vector<StatementContext> PtxirReader::read() {
    read_header();
    std::vector<PtxirSectionTOC> toc = read_toc_entries();
    for (const auto& entry : toc) {
        seek_to(entry.offset);
        switch (entry.type) {
            case REGDECL: read_regdecl_section(); break;
            case KERNEL: read_kernel_section(); break;
            case STRING_TABLE: read_string_table(); break;
            default: throw std::runtime_error("Unknown section type");
        }
    }
    return stmts_;
}
```

#### 3. Reader 指令覆盖与 Writer 对齐（24/24）

补全 15 种缺失 `InstrVariant` variant 类型的 `case` 分支（MembarInstr, FenceInstr, ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix, VoteInstr, ShflInstr, AtomInstr, TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, CpAsyncInstr, AbiDirective）；移除 `default` 静默跳过（改为 `throw`）。

#### 4. 完整 roundtrip 测试

`tests/unit/test_ptxir_serialization.cpp` 覆盖：
- 每种 V1 指令类型
- 错误路径（magic 不匹配、version 不匹配、未知 section type、字符串表 ID 越界）
- 跨架构字节序（至少在 x86_64 little-endian 上验证；如可在大端系统测试则更好）

## Goals / Non-Goals

**Goals:**
- Writer/Reader 实现完全符合 [ADR-0023](https://github.com/chisuhua/PTX-EMU/blob/main/docs/adr/ADR-0023-ptxir-binary-format.md) 7 项决策
- Reader 指令覆盖 24/24，消除 `default` 静默跳过
- `tests/unit/test_ptxir_serialization.cpp` 完整覆盖所有指令类型的 roundtrip
- 实现 `generate_ptxir()` 离线工具（ANTLR 解析 + 序列化）
- 实现 `load_ptxir(apply_cfg=true)` 完整路径（集成 `CFGBuilder::build()`）
- 修改 `StatementContext` / `OperandContext` 时强制同步 PTXIR writer/reader 的协议（AGENTS.md 同步）
- 4 个 Phase 独立 commit、独立可 revert、构建不破坏其他模块

**Non-Goals:**
- 不修改 `PtxirHeader` 字段布局（V1 头版固定）
- 不修改 `StatementContext` / `OperandContext` 结构体本身（除非发现必须修复的设计缺陷）
- 不修改指令执行逻辑（X-Macro 分发 + per-instruction handler）
- 不修改 cudart 生产路径（`__cudaRegisterFatBinary` 仍走 ANTLR 解析）
- 不实施 V2 增量指令（wmma/mma、Hopper cluster 等）
- 不实现 VBR 编码或压缩（V1 固定宽度）
- 不实现 CI/CD Action（属 Phase 3 工具链，本 change 只到「可在 CI 运行」级别）
- 不重新设计 PTXIR 格式（仅实现 ADR-0023 已定义的契约）
- **G2 (预生成 `.ptxir` 测试数据)** — `tests/ptxir/` 目录已存在但为空。**显式推迟到后续 change**（如 `2026-08-ptxir-test-data-generation`）。本 change 只确保 `generate_ptxir()` API 可用，但不批量预生成现有 PTX 文件
- **G5 (`generate_tests.py` 集成)** — `--mode mode4` / `--ptxir` 选项不在本 change 范围。**显式推迟到后续 change**（如 `2026-08-three-mode-to-four-mode-migration`）。本 change 完成 Mode 4 手动测试，但自动化生成仍由 `three-mode-testing` 技能所有者推动
- **PR 模板 + pre-commit hook**（spec §ptxir-statement-context-change-protocol 部分场景）— 显式推迟到后续 change。AGENTS.md 协议已建立，但 PR 模板 hook 涉及 `.github/PULL_REQUEST_TEMPLATE.md` + `.git/hooks/`，属项目级流程改进

## Decisions

### Decision 1: Writer 写顺序重构（无备选 — 唯一符合 ADR-0023 的方案）

**选择**: 按 `header → TOC entries → REGDECL section → KERNEL section → STRING_TABLE section` 顺序写入

**理由**:
- 符合 [ADR-0023 §Decision 1 + §Decision 4](https://github.com/chisuhua/PTX-EMU/blob/main/docs/adr/ADR-0023-ptxir-binary-format.md#decision-1-文件格式--扁平二进制--section-toc非-bitstream)：Section TOC 提供 O(1) 随机访问，字符串表末尾布局避免 offset 不确定
- 现有实现已通过硬编码偏移偶然工作，但任何新 section 添加都会破坏该不变量
- 写顺序重构是单向的（writer 改变，reader 也必须改变），可在单 Phase 完成

**替代方案考虑**:
- **保持现有写顺序 + 修复 reader 配合**: 可行但会与设计文档永久偏离，违反 ADR-0023
- **改用 LLVM Bitstream**: 拒绝（与 ADR-0023 §方案 A 评估一致）

### Decision 2: Reader 错误处理策略 — Throw 替代静默跳过

**选择**: 未知指令类型 / 未知 section type / 字符串表 ID 越界 → 抛 `std::runtime_error`

**理由**:
- 当前 `default` 分支静默跳过会无声丢失数据（roundtrip 测试会失败但不易发现根因）
- 抛异常让错误立即可见，便于调试
- 符合 PTX-EMU 错误处理约定（参考 [ADR-0001](./ADR-0001-exception-hierarchy.md) 异常层次体系）

**替代方案考虑**:
- **保留静默跳过 + 写警告日志**: 不符合 PTX-EMU 项目约定
- **返回 `std::optional`**: 增加 API 复杂度，调用方需处理空值

### Decision 3: 测试独立构建（不受 ANTLR 编译阻塞）

**选择**: PTXIR 单元测试仅依赖 `ptxir_writer` / `ptxir_reader` / `ptxir` 静态库，不依赖 ANTLR 运行时

**理由**:
- ANTLR 运行时在 2 核系统 OOM，无法完整构建（参考 [archive/2026-06-09-ptxir-serialization-architecture/tasks.md §10.5-10.6](https://github.com/chisuhua/PTX-EMU/blob/main/openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/tasks.md)）
- 单元测试可手工构造 `StatementContext`（通过 `statement_factory`），无需 ANTLR
- 集成测试（ANTLR + PTXIR）可标记为「expected fail in 2-core system」

**替代方案考虑**:
- **等 ANTLR 编译问题解决后再写测试**: 阻塞本 change 进度
- **Docker CI 编译 ANTLR**: 属基础设施改进，本 change 不涉及

### Decision 4: `generate_ptxir()` 放置位置 — `src/ptxir/ptxir_serialization.cpp`

**选择**: 与 `serialize_statements()` / `deserialize_statements()` 同一翻译单元

**理由**:
- 三个函数都是「便捷 API 层」，逻辑类似（thin wrapper over writer/reader）
- `generate_ptxir()` 调用 `load_ptx_statements()` (ANTLR) → `serialize_statements()`，可在头文件 include ANTLR 依赖

**替代方案考虑**:
- **单独文件 `src/ptxir/ptxir_generator.cpp`**: 过度拆分，逻辑上仍属便捷 API 层
- **放在 `tests/common/`**: 测试代码不应反过来被 src/ 依赖

### Decision 5: `load_ptxir(apply_cfg=true)` 集成方式

**选择**: 直接调用 `CFGBuilder::build()` 在反序列化后

**理由**:
- `CFGBuilder` 已存在于 `src/ptx_parser/`
- 反序列化结果 `vector<StatementContext>` 直接传入 `CFGBuilder::build()`
- 现有 `load_ptx_statements(ptx_path, "", true)` 已是相同模式（参考 [PTXIR 技能 §Workflow](../../.opencode/skills/ptxir-serialization/SKILL.md)）

**替代方案考虑**:
- **在 `.ptxir` 中存储 `reconvergence_pc`**: 增加格式复杂度，违反 [ADR-0023 §Decision 7](./ADR-0023-ptxir-binary-format.md#decision-7-cfg-处理时机--反序列化后可选应用)（CFG 解耦）

## Risks / Trade-offs

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **Writer 写顺序重构破坏现有 `.ptxir` 文件读取** | 中 | 高 | **bump `PTXIR_VERSION` 从 1 → 2**（在 `ptxir_format.h`）作为前置任务。旧 V1 文件 magic 不变（`"PTXIR"`）但 `version=1` → 新 reader 检测到 `version<2` 走旧读取路径（`read_string_table` 硬编码偏移 + 不读 TOC）；新写入的 V2 文件 `version=2` → 新 reader 走 TOC 解析路径。**不依赖 section_count 区分**（旧 writer 写 2 但无 TOC 条目，新 writer 写 3+ 含 TOC 条目，靠 version 字段区分更可靠）。**实施细节**: tasks.md Phase 2 §3.1 新增 "bump version" 子任务为 §3.2-§3.3 的前置条件 |
| **Reader 错误处理 throw 改变语义，导致现有测试失败** | 中 | 中 | 在 throw 前记录警告日志（`std::cerr`）；CI 完整跑一次后调整；如需保留向后兼容，提供 `Reader::set_strict(bool)` 开关（Phase 1 不实现，Phase 2+ 视需要添加） |
| **AGENTS.md checklist 不被遵守，StatementContext 修改再次破坏 PTXIR** | 中 | 高 | 在 PR 模板中添加「是否修改 `StatementContext` / `OperandContext`」checkbox；如选 Yes，必须列出同步的 writer/reader 改动 |
| **ANTLR 编译阻塞使 generate_ptxir() / load_ptxir(apply_cfg) 集成测试无法运行** | 高 | 中 | 单元测试不依赖 ANTLR（手工构造 StatementContext）；集成测试添加「expected fail in 2-core system」标签；CI 完整系统上运行 |
| **15 种新 case 分支的解析错误，导致更多静默数据丢失** | 低 | 中 | 每个 case 分支单独单元测试；roundtrip 测试对比所有字段（不仅语句数量） |
| **格式版本兼容：未来 V2 增加 section type 时 V1 reader 无法解析** | 低 | 中 | V1 reader 遇到未知 section type → throw 明确错误（已有 catch 路径）；V2 设计时考虑 block-skip 机制 |
| **4 Phase commit 累积冲突** | 中 | 中 | 每个 Phase 独立 worktree；Phase 1 完成后立即合并到 main；Phase 2/3/4 各自 rebased on main |

## Migration Plan

### 阶段化实施（4 Phase 独立 commit、独立可 revert）

**Phase 1: Reader 指令覆盖补全（G9 修复）**
- Commit: `feat(ptxir): complete reader instruction coverage (24/24)`
- 范围: 仅 `src/ptx_ir/ptxir_reader.cpp`
- 不依赖：可独立实施
- 回退策略: `git revert <commit>` → reader 回到 9/24 variant 状态（不会破坏现有硬编码偏移 reader）

**Phase 2: 格式契约对齐（D1-D5 修复）**
- **前置任务 (§3.0)**: bump `PTXIR_VERSION` 1 → 2 in `ptxir_format.h`（必做，消除旧 V1 文件解析歧义）
- **子阶段 2A (commit 1)**: writer 输出 V2 格式（写入 TOC 条目 + 回填 string_table_offset/size）
- **子阶段 2B (commit 2)**: reader 支持 V1 旧格式（version=1 走硬编码偏移路径，保持向后兼容）
- **子阶段 2C (commit 3)**: reader 支持 V2 新格式（version=2 走 TOC 解析路径）
- 范围: `src/ptx_ir/ptxir_format.h` + `src/ptx_ir/ptxir_writer.cpp` + `src/ptx_ir/ptxir_reader.cpp`
- 依赖: Phase 1（确保 reader case 完整后再改 reader 主体）
- 回退策略: 单 commit revert；writer/reader 各一次 revert
- **破损中间态消除**: 3A/3B/3C 三 commit 顺序保证任意中间态可用 — (a) 仅 writer 写 V2 但 reader 暂不读 V2 → 老 V1 文件照常可读；(b) writer 写 V2 + reader 支持 V1 → 双向兼容；(c) 全切 V2 路径 → 老 V1 仍可读

**Phase 3: 测试 + 工具链（G1, G3, G4 修复）**
- Commit 1: `test(ptxir): add roundtrip unit tests for all 24 instruction types`
- Commit 2: `feat(ptxir): add generate_ptxir() offline tool`
- Commit 3: `feat(ptxir): complete load_ptxir(apply_cfg=true) path with CFGBuilder integration`
- 范围: `tests/unit/test_ptxir_serialization.cpp` + `src/ptxir/ptxir_serialization.cpp` + `include/ptxir/ptxir_serialization.h`
- 依赖: Phase 2（格式契约稳定后才有意义测试）
- 回退策略: 单 commit revert；如某 commit 引入回归，revert 该 commit 而非整体

**Phase 4: 文档同步 + 协议建立（G6 修复，AGENTS.md 协议；G2/G5 推迟到后续 change）**
- Commit 1: `docs(ptxir): add StatementContext modification protocol to AGENTS.md`
- Commit 2: `docs(ptxir): update THREE-MODE-TESTING-GUIDE.md to four-mode framework`
- 范围: `src/ptx_ir/AGENTS.md` + `include/ptxir/AGENTS.md` (如新建) + `docs/developer-guide/THREE-MODE-TESTING-GUIDE.md`
- 依赖: Phase 3（工具有了才能更新文档说「如何用」）
- 回退策略: 单 commit revert

### 基线 worktree

```bash
# 在 Phase 1 开始前建立基线
git worktree add .worktrees/ptxir-baseline main

# 验证基线可构建（如果 2 核 OOM，至少验证 ptxir 子模块可构建）
cmake --build .worktrees/ptxir-baseline/build --target ptxir
```

### 失败处理策略

- **任何 Phase 完成后测试回归** → 立即 revert 该 Phase commit，不尝试 fix
- **2 核 OOM 导致 ANTLR 编译失败** → 标记集成测试为「expected fail in 2-core system」，继续推进单元测试
- **跨 Phase 冲突** → 在 worktree 内 rebase on main，逐 Phase 验证

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptx_ir/ptxir_format.h` | 修改 | 移除未使用 `header_size` 字段或补全文档 |
| `src/ptx_ir/ptxir_writer.cpp` | 重构 | 写顺序 + TOC 写入 + offset 回填 |
| `src/ptx_ir/ptxir_reader.cpp` | 重构 | 从 TOC 解析 + 15 种 case 补全 + 移除 default |
| `src/ptxir/ptxir_serialization.cpp` | 扩展 | 新增 `generate_ptxir()` + 完善 `load_ptxir(apply_cfg=true)` |
| `include/ptxir/ptxir_serialization.h` | 扩展 | 2 个新 API 签名 |
| `tests/unit/test_ptxir_serialization.cpp` | 新增 | roundtrip 单元测试 |
| `tests/integration/test_ptxir_pipeline_mode4.cpp` | 新增 | Mode 4 端到端（可能标 expected fail） |
| `src/ptx_ir/AGENTS.md` | 扩展 | 新增 StatementContext 修改 checklist |
| `include/ptxir/AGENTS.md` | 新增 | 公共头文件修改协议 |
| `docs/developer-guide/THREE-MODE-TESTING-GUIDE.md` | 重写 | 三模式 → 四模式 |
| `tests/CMakeLists.txt` | 修改 | 注册新测试 target |
| `archive/2026-06-09-ptxir-serialization-architecture/tasks.md` | 更新 | 标记 §10.1-10.4 解锁（待 Phase 3 完成后） |

## Open Questions

1. **`PtxirHeader::header_size` 字段** — 当前是冗余的（`sizeof(PtxirHeader)` 已固定为 24），但保留它有助于未来 V2 兼容。**决策**: Phase 2 保留，不删除
2. **错误日志格式** — `std::cerr` vs `printf` vs `nlohmann/json` logger？**决策**: Phase 1 使用 `std::cerr`（最小依赖），未来如需统一日志再重构
3. **`generate_ptxir()` 的 ANTLR 失败处理** — 当前 `load_ptx_statements` 失败时返回空 vector。**决策**: 沿用现有行为，`generate_ptxir()` 返回 `false`，不抛异常
4. **V2 增量时的 block-skip 机制** — 当前未实现，未来 V2 reader 遇到未知 block 时如何跳过？**决策**: 暂不实现，留给 V2 设计阶段。本 change 仅在 unknown section type 时 throw
5. **`include/ptxir/AGENTS.md` 是否已存在**？**待 Phase 4 确认**: 如不存在则新建
