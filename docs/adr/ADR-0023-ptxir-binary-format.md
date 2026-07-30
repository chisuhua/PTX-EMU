# ADR-0023: PTXIR 二进制序列化格式与 7 项架构决策

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
| **日期** | 2026-07-30 |
| **关联任务** | Phase 12.1 (Sprint 12.1), 2026-06-09-ptxir-serialization-architecture |
| **关联 OpenSpec change** | [openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/](../../openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/) |
| **关联差距分析** | [docs/architecture/ptxir-serialization-gaps-gap-analysis.md](../architecture/ptxir-serialization-gaps-gap-analysis.md) |
| **关联技能** | [.opencode/skills/ptxir-serialization/SKILL.md](../../.opencode/skills/ptxir-serialization/SKILL.md) |
| **关联 ADR** | [ADR-0011](./ADR-0011-pipeline-architecture.md)（Pipeline 架构，2026-07-30 升级 Accepted，引用本 ADR 决策）|
| **作者** | PTX-EMU Architecture Team |
| **审核人** | Oracle (architecture review), Metis (decision completeness) |

---

## 上下文

### 问题背景

PTX-EMU 的测试管道在每次运行时都通过 ANTLR 重新解析 PTX 文本：

```
PTX 文本 → ANTLR 词法/语法分析 → PTX AST → PtxVisitor → StatementContext[] → CFGBuilder → 执行
```

对于大型 kernel（光线追踪等上千条指令），ANTLR 解析耗时 ~200ms，而实际执行仅需 ~10ms。**重复解析是测试框架最大的性能瓶颈**，且无法独立测试各阶段。

### 触发事件

1. **2026-06-09** — 提交 OpenSpec change `2026-06-09-ptxir-serialization-architecture`，定义 `.ptxir` 二进制格式与 Mode 4 快速加载模式
2. **2026-06-19** — `tests/three_mode_testing/` 目录被清理（P1-4 清理 commit `7c583c3`），`test_helpers.hpp` 和 Mode 4 测试文件被删除
3. **2026-07-29** — 完成 `refactor-ptxir-writer` change，将 232 行巨型 `write_instruction()` 函数拆分为 22 个 per-type helper 方法
4. **2026-07-30** — `gap-analysis` 文档（`docs/architecture/ptxir-serialization-gaps-gap-analysis.md`）填充完成，识别 9 项差距（G1-G9）+ 5 项格式偏差（D1-D5）

### 技术约束

- **PTX-EMU 模拟器场景**：需在无 NVIDIA GPU 环境仿真执行 CUDA 程序
- **三模式测试框架**（已存在的 Mode 1/2/3/3C）必须**完全保留**（向后兼容）
- **不修改 ANTLR 解析器本身**（语法规则、词法分析不变）
- **不修改 `StatementContext` 结构体**（data variant 不变）
- **不修改指令执行逻辑**（X-Macro 分发 + per-instruction handler）
- **不修改 cudart 生产路径**（`__cudaRegisterFatBinary` 仍走 ANTLR 解析）

### 目标架构

```
PTX 文本 (.ptx)                                    .ptxir (二进制)
    ↓                                                  ↓
ANTLR 解析（首次,~200ms）                        反序列化（~5ms）
    ↓                                                  ↓
StatementContext[] ←───── PTXIR 序列化 ───── StatementContext[]
    ↓                                                  ↓
CFGBuilder::build()  （可选,反序列化后）        CFGBuilder::build()  （可选）
    ↓                                                  ↓
GPUContext 执行                                  GPUContext 执行
```

**关键原则**（来自 ADR-0011 Pipeline 架构，本 ADR 落地实施细节）：
- 编译/运行时分离 — PTX 解析结果应可缓存
- 阶段可独立测试 — 每阶段接口清晰
- 阶段可独立替换 — 阶段间标准接口
- 性能优化 — 避免重复 ANTLR 解析

---

## 决策驱动因素

1. **测试性能瓶颈**：ANTLR 解析占测试时间 ~95%，需绕过以加速
2. **测试可重现性**：Mode 2/3 依赖 PTX 文本和 ANTLR，binary 缓存更稳定
3. **实现复杂度控制**：序列化层 ~500 行（writer + reader + format），相比 ANTLR ~2000 行不算大
4. **格式可读性 vs 紧凑性**：V1 数据量小（~1000 条指令），不需要复杂压缩
5. **调试便利性**：十六进制编辑器应能直接读取关键结构
6. **版本兼容性**：V1 稳定后很少变更，V2 增量添加
7. **CFG 解耦**：CFG builder 结果不存 `.ptxir`，避免运行时状态污染
8. **基础设施复用**：目录结构与现有 `ptx_ir/` 头文件布局对齐

---

## 考虑的替代方案

### 方案 A: LLVM Bitstream 编码（❌ 未采用）

**描述**: 采用 LLVM bitcode 的 VBR（Variable Bit Rate）+ 缩略符编码 + 多层架构

```
Layer 3: ModuleBitcodeWriter — 领域语义（类型/常量/函数）
Layer 2: BitcodeWriterBase   — 字符串表、值枚举
Layer 1: BitstreamWriter    — 原始二进制编码（块/记录）
```

**优点**:
- 体积最小（VBR 编码对重复整数优化）
- 前向兼容设计完善（块跳过机制）
- 与 LLVM 生态工具链兼容

**缺点**:
- 实现复杂度高（~2000 行 vs 我们 500 行）
- 调试困难（VBR 解码不直观）
- V1 数据量（~1000 条指令）不需要此优化
- 学习曲线陡

**未采用理由**: 实现成本与数据量不匹配。

### 方案 B: JSON 文本格式（❌ 未采用）

**描述**: 每条 StatementContext 序列化为 JSON 对象

**优点**:
- 完全可读
- 调试极方便
- 现成 JSON 库支持

**缺点**:
- 体积膨胀 ~10x
- 反序列化 CPU 开销高（字符串解析 vs 二进制 memcpy）
- 与"快速加载"目标直接矛盾

**未采用理由**: 性能无法满足 Mode 4 ~5ms 目标。

### 方案 C: Protocol Buffers（❌ 未采用）

**描述**: 定义 `.proto` schema，用 `protoc` 生成序列化代码

**优点**:
- Schema 即文档
- 跨语言兼容
- 类型安全

**缺点**:
- 新增依赖（`.proto` 编译 + 运行时）
- 构建系统复杂度增加
- Schema 变更需要重新生成代码
- 调试二进制不直观

**未采用理由**: 依赖增加但收益有限，V1 自定义格式足够。

### 方案 D: 自描述 TLV（Type-Length-Value）指令（❌ 未采用）

**描述**: 每条指令前写类型 + 长度字段，未知指令可跳过

**优点**:
- 前向兼容（未知指令可跳过）
- 解析容错强

**缺点**:
- PTX 指令长度固定（Opcode 决定操作数格式），TLV 冗余
- 体积增加 ~30%

**未采用理由**: PTX 指令结构紧凑，TLV 过度设计。

### 方案 E: 扁平二进制 + Section TOC + 尾部字符串表（✅ 选中）

**描述**: 见下方 7 项决策

**优点**:
- 实现简单（~500 行）
- TOC 提供 O(1) 随机访问
- 十六进制编辑器可读
- 字符串表尾部布局与 LLVM STRTAB_BLOCK、DWARF debug_str 模式一致

**缺点**:
- 缺乏 VBR 压缩
- V2 增量需手动管理

**选择理由**: 实现成本与功能匹配，V1 阶段 VBR 收益不抵成本。

---

## 决策内容

### 设计原则

1. **最小实现**：V1 仅测试路径，不进生产 `cudart_sim.cpp`
2. **Extend-Only 版本管理**：Opcode ID 永不改变，新指令往后追加
3. **LLVM 模式借鉴**：值枚举（pre-pass）+ 字符串表尾部 + 头部版本字段
4. **CFG 解耦**：CFG builder 结果不存 `.ptxir`，反序列化后可选应用
5. **可丢弃实现成本**：序列化层维护成本需要 `StatementContext` 冻结

### 7 项决策（逐项展开）

#### Decision 1: 文件格式 — 扁平二进制 + Section TOC（非 Bitstream）

**选择**: 固定宽度整数字节序 + Section TOC 头部 + 尾部字符串表

```
┌─────────────────────────────────┐
│         PTXIRHeader (24B)       │
├─────────────────────────────────┤
│      Section TOC (6B * N)       │
├─────────────────────────────────┤
│         REGDECL Section         │
├─────────────────────────────────┤
│        KERNEL Section           │
├─────────────────────────────────┤
│      STRING_TABLE Section       │
└─────────────────────────────────┘
```

**理由**:
- V1 数据量小（~1000 条指令），VBR 编码无必要
- Section TOC 提供 O(1) 随机访问
- 实现/维护成本低，调试容易（十六进制编辑器可读）

**Header 字段**（24 字节，小端序）:
| 字段 | 类型 | 偏移 | 说明 |
|------|------|------|------|
| `magic[4]` | char[4] | 0-3 | `"PTXIR"` |
| `version` | u16 | 4-5 | 当前 `PTXIR_VERSION = 1` |
| `flags` | u16 | 6-7 | 保留，必须为 0 |
| `section_count` | u16 | 8-9 | TOC 条目数 |
| `reserved` | u16 | 10-11 | 保留 |
| `string_table_offset` | u32 | 12-15 | 字符串表绝对偏移 |
| `string_table_size` | u32 | 16-19 | 字符串表字节数 |
| `header_size` | u32 | 20-23 | `sizeof(PtxirHeader) = 24` |

**TOC 条目**（6 字节）:
| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | u8 | `PtxirSectionType` 枚举值 |
| `reserved` | u8 | 保留，必须为 0 |
| `offset` | u32 | section 起始绝对偏移 |

**Section 类型枚举**:
| 名称 | 值 | 描述 |
|------|---|------|
| `REGDECL` | 1 | 寄存器声明（操作数表） |
| `TYPE` | 2 | 类型信息（预留） |
| `KERNEL` | 3 | kernel 语句 |
| `CONSTANT` | 4 | 常量（预留） |
| `STRING_TABLE` | 5 | 标签/标识符字符串池 |

#### Decision 2: 指令编码 — Opcode ID + 操作数扁平紧凑

**选择**: 每条指令编码为 `[opcode:u16, ...per-type fields]`

**理由**:
- Opcode 枚举值（`StatementType`）已经是紧凑整数 ID
- 操作数通过预遍历赋值（Decision 3）变成 `u32` 索引
- `std::variant<BranchInstr, GenericInstr, ...>` 的 22 种指令类型用 Opcode ID 区分
- Reader 按 `switch(opcode)` 分发到 per-type 反序列化

**指令编码大小常量**（在 `include/ptx_ir/ptxir_format.h`）:
| 指令 | 编码字段 | 大小 |
|------|---------|------|
| BranchInstr | opcode + pred_str_id + target_str_id + pred_negated + reconvergence_pc | 2+4+4+1+4 = 15B |
| LabelInstr | opcode + label_str_id | 2+4 = 6B |
| VoidInstr | opcode only | 2B |
| BarrierInstr | opcode + bar_id | 2+4 = 6B |
| DeclarationInstr | opcode + kind + type + name_str_id + array_size | 2+1+1+4+4 = 12B |
| GenericInstr | opcode + qualifiers[] + dst_id + src_ids[] | 变长 |

#### Decision 3: 值枚举（预遍历赋值）— 与 LLVM ValueEnumerator 一致

**选择**: 序列化前遍历一次 `kernelStatements`，为所有 `RegOperand` 分配紧凑 `u32` ID

```cpp
// 预遍历：建立 operand_table
std::vector<OperandContext> operand_table;
std::unordered_map<std::string, uint32_t> reg2id;
for (auto& stmt : statements) {
    for (auto& op : stmt.operands) {
        if (op.kind == RegOperand && reg2id.count(op.name) == 0) {
            reg2id[op.name] = operand_table.size();
            operand_table.push_back(op);
        }
    }
}
// 写指令时：op → reg2id[op.name]
```

**理由**:
- PTX 寄存器名 `%r5`、`%p1` 等字符串若在指令中重复存储，`.ptxir` 体积膨胀 ~5x
- 值枚举后，指令中只存储 `u32` ID，`operand_table[ID]` 提供原始字符串
- 与 LLVM `ValueEnumerator` 模式完全一致

#### Decision 4: 字符串表位置 — 最后写

**选择**: 所有 Section 写完后，最后写 STRING_TABLE，offset 记录在 TOC 头部

**理由**:
- 与 LLVM STRTAB_BLOCK、DWARF debug_str 相同模式
- 避免序列化过程中字符串 offset 不确定
- 读取时先读 TOC，再读取字符串表供其他 Section 引用

**字符串编码**: `[length:u16, bytes:u8[N]]`，每个字符串独立带长度前缀

#### Decision 5: 版本管理 — Extend-Only（只增不改）

**选择**: 版本字段在头部；现有 Opcode ID 永不改变；新增 Opcode 只能往后追加

**理由**:
- `.ptxir` 是测试基础设施，不需要生产级前向兼容
- V1 只支持 sm_50-sm_80 的基本 PTX ISA
- 未来 V2 增加 wmma/mma、Hopper cluster 时，V1 reader 可跳过未知 block（根据 TOC 里的 block type 字段）

**当前 V1 支持指令**（22 个 opcode / 9 种指令 variant 显式处理，详见 [差距分析 §2.4](../architecture/ptxir-serialization-gaps-gap-analysis.md)）:
- 控制流: S_BRA, S_LABEL, S_EXIT, S_RET, S_PRAGMA, S_DOLLOR
- 通用算术: S_MOV, S_ADD, S_SUB, S_MUL, S_SETP, S_CVT
- 访存: S_LD, S_ST
- 屏障: S_BAR, S_BAR_WARP_SYNC
- 声明: S_REG, S_CONST, S_SHARED, S_LOCAL, S_GLOBAL, S_PARAM

**注**: 24 种 `InstrVariant` 类型中（BranchInstr, LabelInstr, VoidInstr, BarrierInstr, GenericInstr, DeclarationInstr, BarWarpSyncInstr, PragmaInstr, DollarNameInstr, MembarInstr, FenceInstr, ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix, VoteInstr, ShflInstr, AtomInstr, TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, CpAsyncInstr, AbiDirective），Reader 显式 case 覆盖 9 种 variant，对应 22 个 opcode（多个 opcode 共享同一 variant，如 S_REG/S_CONST/S_SHARED/S_LOCAL/S_GLOBAL/S_PARAM 共享 DeclarationInstr）。**15 种 variant 缺失 reader case**（MembarInstr, FenceInstr, ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix, VoteInstr, ShflInstr, AtomInstr, TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, CpAsyncInstr, AbiDirective），走 `default` 静默跳过。详见差距分析 G9 + tasks.md §2.1-2.15。

#### Decision 6: 放置位置 — `include/ptx_ir/` + `src/ptx_ir/` + `src/ptxir/`

**选择**: 三层目录结构

```
include/ptx_ir/
  ├── ptxir_format.h       # 格式定义（24 字节 Header + 6 字节 TOC）
  ├── ptxir_writer.h        # PtxirWriter 类声明
  └── ptxir_reader.h        # PtxirReader 类声明

src/ptx_ir/
  ├── ptxir_writer.cpp     # 序列化实现
  └── ptxir_reader.cpp     # 反序列化实现

src/ptxir/                  # 顶层便捷 API
  └── ptxir_serialization.cpp  # 4 个自由函数封装
```

**理由**:
- `include/ptx_ir/` 已有 `StatementContext`/`OperandContext`，同目录便于包含
- 核心 writer/reader 实现与 IR 类型同模块（`ptx_ir`）
- 顶层便捷 API（`serialize_statements` / `deserialize_statements`）单独静态库（`ptxir`）
- 区分底层机制（writer/reader）和高层便利 API

#### Decision 7: CFG 处理时机 — 反序列化后可选应用

**选择**: `load_ptxir(path, apply_cfg=false)` 的 `apply_cfg` 参数控制

**理由**:
- Mode 3a（CFG 前）对应 `apply_cfg=false` — 原始解析状态
- Mode 3b（CFG 后）对应 `apply_cfg=true` — 已填充 `reconvergence_pc`
- `.ptxir` 文件中 `reconvergence_pc` 可以不存储（由反序列化调用方决定）
- 保持文件内容与运行时状态解耦
- 简化 `.ptxir` 格式：不需要存储 label2pc 等中间状态

---

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptx_ir/ptxir_format.h` | 新增 | 格式定义头文件（73 行） |
| `include/ptx_ir/ptxir_writer.h` | 新增 | PtxirWriter 类（80 行） |
| `include/ptx_ir/ptxir_reader.h` | 新增 | PtxirReader 类（30 行） |
| `src/ptx_ir/ptxir_writer.cpp` | 新增 | 序列化实现（296 行，refactored 至 22 per-type helper） |
| `src/ptx_ir/ptxir_reader.cpp` | 新增 | 反序列化实现（229 行） |
| `src/ptxir/ptxir_serialization.cpp` | 新增 | 4 个自由函数封装（35 行） |
| `include/ptxir/ptxir_serialization.h` | 新增 | 公共 API 头文件 |
| `src/ptx_ir/CMakeLists.txt` | 新增 | `ptxir_writer` + `ptxir_reader` 静态库 |
| `src/ptxir/CMakeLists.txt` | 新增 | `ptxir` 静态库（链接 writer + reader） |
| `tests/ptxir/` | 新增 | 预生成 `.ptxir` 文件目录（当前空，待 Phase 2 填充） |
| `.opencode/skills/ptxir-serialization/SKILL.md` | 新增 | 操作参考技能文档 |
| `tests/CMakeLists.txt` | 修改 | 链接 `ptxir_writer` |

---

## 后果

### 正面影响

- **测试加速**：ANTLR 解析 ~200ms → `.ptxir` 反序列化 ~5ms（40x 加速）
- **独立测试**：每阶段可独立验证（parse / serialize / deserialize / execute）
- **缓存友好**：CI/CD 可缓存 `.ptxir` 文件，避免每次 ANTLR 编译
- **格式稳定**：V1 头版固定，V2 增量添加，前向兼容
- **调试便利**：二进制格式 + 字符串表末尾 + TOC 头部 = 十六进制编辑器友好

### 负面影响

- **维护成本**：每次 `StatementContext` 结构体变更需同步更新 writer/reader
- **C-4 债务**：`write_instruction()` 巨型函数（已被 2026-07-29-refactor-ptxir-writer 拆分为 22 per-type helper）
- **格式实现偏差**（详见差距分析 D1-D5）:
  - TOC 条目未实际写入（writer 跳过）
  - header 中 `string_table_offset` / `string_table_size` 未回填
  - Reader 硬编码偏移（`sizeof(PtxirHeader)`），破坏格式契约
  - Writer 实际写顺序与设计文档不一致
- **Reader 指令覆盖不足**（15/24 variant 缺失，详见差距分析 G9）: 15 种 `InstrVariant` 类型（MembarInstr, FenceInstr, ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix, VoteInstr, ShflInstr, AtomInstr, TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, CpAsyncInstr, AbiDirective）走 `default` 分支静默跳过；Writer 写入这 15 种 variant 的字段后，Reader 无法重建为正确的 variant 类型
- **测试缺失**（G1）: 无 roundtrip 测试，无 Mode 4 测试
- **工具链缺失**（G3, G4, G5, G6, G7）: `generate_ptxir()` / `load_ptxir(apply_cfg)` / `generate_tests.py` 集成 / 文档更新 / CI Action

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| `StatementContext` 变更破坏序列化兼容性 | 中 | 高 | 冻结 V1 期间 `StatementContext` 结构体；变更需同步更新 writer/reader + bump version |
| Reader 静默跳过未知指令（default case） | 中 | 中 | Phase 1 修复：移除 default 静默跳过，改为 `throw` 或 `assert` |
| Reader/Writer 格式实现偏差 | 中 | 中 | Phase 1 修复：对齐实现与设计文档（详见差距分析 D1-D5） |
| 2 核系统 OOM 阻断 build verification | 已发生 | 中 | 迁移到 ≥4 核 ≥16GB RAM 系统；CI/CD 编译可解决 |
| V2 增量时前向兼容失败 | 低 | 中 | 实施前先在 V1 验证 block-skip 机制 |
| `.ptxir` 文件意外被 git 追踪 | 低 | 低 | `tests/ptxir/.gitignore` 已在；CI Action 缓存 |

---

## 合规检查

后续相关开发应检查：

### 格式契约

- [ ] Writer 写入顺序必须为：header → TOC entries → REGDECL section → KERNEL section → STRING_TABLE section
- [ ] Reader 解析顺序必须从 TOC 条目读取 section 偏移，不允许硬编码
- [ ] header 中 `string_table_offset` / `string_table_size` / `header_size` 字段必须正确回填
- [ ] 任何 `.ptxir` 写入/读取必须经过 `ptxir_format.h` 常量（`BRANCH_ENCODED_SIZE` 等），禁止硬编码尺寸

### 指令覆盖

- [ ] 新增 PTX 指令时必须同步更新 `ptx_op.def` + `statement_context.h`（InstrVariant）+ writer + reader
- [ ] Reader 不允许有 `default` 静默跳过分支；未知 Opcode 应 throw
- [ ] 完整 roundtrip 测试覆盖所有支持的指令类型

### API 契约

- [ ] 4 个核心 API（`serialize_statements` / `deserialize_statements` / `generate_ptxir` / `load_ptxir`）签名必须保持稳定
- [ ] `load_ptxir(apply_cfg=true)` 必须正确应用 `CFGBuilder::build()`
- [ ] `generate_ptxir` 必须经过 ANTLR 解析 + 序列化两步

### 构建集成

- [ ] 任何修改 `StatementContext` / `OperandContext` 结构体的 commit 必须同步更新 ptxir_writer 和 ptxir_reader
- [ ] `tests/ptxir/*.ptxir` 默认 git 忽略，仅在 CI/CD Action 中生成缓存
- [ ] ctest Mode 4 测试通过率必须 100%（roundtrip + 跨 V1 指令类型）

### 与其他 ADR 的关系

- [ ] 引用本 ADR 时同时引用 [ADR-0011](./ADR-0011-pipeline-architecture.md)（Pipeline 架构）
- [ ] `StatementContext` 变更需同时检查 [ADR-0012](./ADR-0012-per-thread-pc.md)（Per-Thread PC）和 [ADR-0019](./ADR-0019-pc-management-extraction.md)（PC management extraction）

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-07-30 | 初始版本（从 OpenSpec change `2026-06-09-ptxir-serialization-architecture/design.md` 7 项决策提取 + 补充差距分析 G1-G9 / D1-D5）| PTX-EMU Architecture Team |

---

## 参考

### 关联 ADR

- [ADR-0009](./ADR-0009-xmacro-instruction-dispatch.md) — X-Macro 指令分发（PTXIR 通过 `StatementType` 枚举复用相同机制）
- [ADR-0010](./ADR-0010-fake-cuda-runtime.md) — Fake CUDA Runtime 拦截（PTXIR 不影响生产路径）
- [ADR-0011](./ADR-0011-pipeline-architecture.md) — PTX→PTXIR 多阶段 Pipeline 架构（Proposed，待升级为 Accepted 引用本 ADR）
- [ADR-0019](./ADR-0019-pc-management-extraction.md) — ThreadContext 持续瘦身（`OperandContext::operand_phy_addr` 不序列化原则）

### 关联 OpenSpec changes

- [`openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/`](../../openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/) — 完整设计文档（proposal + design + tasks + 4 specs）
- [`openspec/changes/archive/2026-07-29-refactor-ptxir-writer/`](../../openspec/changes/archive/2026-07-29-refactor-ptxir-writer/) — Writer 长函数拆分（C-4 债务修复）
- [`openspec/changes/archive/2026-06-09-ptxir-test-refactor/`](../../openspec/changes/archive/2026-06-09-ptxir-test-refactor/) — Mode 4 roundtrip 测试设计

### 关联文档

- [差距分析](../architecture/ptxir-serialization-gaps-gap-analysis.md) — 当前实现与本 ADR 7 项决策的差距清单
- [技能文档](../../.opencode/skills/ptxir-serialization/SKILL.md) — 格式规范 + API 参考 + 工作流
- [PTX-EMU 文档索引](../README.md)

### 外部参考

- [LLVM Bitcode 格式](https://llvm.org/docs/BitCodeFormat.html) — Bitstream 编码参考（已评估未采用）
- [DWARF Debugging Information Format](https://dwarfstd.org/) — 字符串表末尾布局参考
- [PTX ISA 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) — 指令集定义来源
