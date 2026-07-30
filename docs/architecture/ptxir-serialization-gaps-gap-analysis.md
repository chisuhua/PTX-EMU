# 架构差距分析: ptxir-serialization-gaps

> **生成日期**: 2026-07-30
> **状态**: 正式
> **关联 ADR**: ADR-0016 (Blackwell tcgen05), 关联 change: `2026-06-09-ptxir-serialization-architecture`
> **分析者**: Sisyphus (guide-arch Phase 3)

## 1. 目标架构

[ADR-0016](docs/adr/ADR-0016-blackwell-only-tcgen05.md) 和 [2026-06-09-ptxir-serialization-architecture](openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/) 定义的 PTXIR 序列化目标架构：

### 1.1 格式定义

- **二进制格式**：`.ptxir`，扁平二进制 + Section TOC，头部 24 字节
- **Magic**: `"PTXIR"`，Version 1
- **Section 类型**: REGDECL=1, TYPE=2, KERNEL=3, CONSTANT=4, STRING_TABLE=5
- **指令编码**: Opcode(u16) + 操作数 ID 索引
- **值枚举**: 预遍历所有 `RegOperand` → 分配紧凑 u32 ID（类似 LLVM ValueEnumerator）
- **字符串表**: 尾部存储，offset 记录在 TOC 头部

### 1.2 测试模式

- **Mode 4 (快速加载)**: 反序列化 `.ptxir` → `vector<StatementContext>`，绕过 ANTLR，目标 ~5ms vs ~200ms
- 保留 Mode 1/2/3/3C 全部向后兼容

### 1.3 API 层

- `serialize_statements(stmts, path)` — 序列化
- `deserialize_statements(path)` — 反序列化
- `generate_ptxir(ptx_path, ptxir_path, kernel_name)` — PTX 文本 → ANTLR 解析 → 序列化
- `load_ptxir(ptxir_path, apply_cfg)` — 加载 + 可选 CFG Builder

### 1.4 工具链

- `generate_tests.py --mode mode4` 自动生成 Mode 4 测试
- `generate_tests.py --ptxir` 预生成 `.ptxir` 文件
- CI/CD Action 自动缓存 `.ptxir` 文件
- `tests/ptxir/` 目录存放预生成文件

## 2. 当前架构

### 2.1 已实现（✅）

| 组件 | 文件 | 状态 |
|------|------|------|
| 格式定义 | `include/ptx_ir/ptxir_format.h` | ✅ 完整 (header, TOC, section types, encoding constants) |
| Writer | `src/ptx_ir/ptxir_writer.cpp` + `.h` | ✅ 支持 20+ 指令类型 |
| Reader | `src/ptx_ir/ptxir_reader.cpp` + `.h` | ✅ 支持 12 种指令类型回读 |
| 序列化 API | `src/ptxir/ptxir_serialization.cpp` + `.h` | ✅ `serialize_statements()` / `deserialize_statements()` / `serialize_to_string()` / `deserialize_from_string()` |
| 构建集成 | `src/ptxir/CMakeLists.txt` + `src/ptx_ir/CMakeLists.txt` | ✅ 静态库 `ptxir` + `ptxir_writer` + `ptxir_reader` |
| 技能文档 | `.opencode/skills/ptxir-serialization/SKILL.md` | ✅ 完整格式说明 + API 参考 |

### 2.2 未实现/缺失（❌）

| # | 组件 | 预期 | 实际 | 影响 |
|---|------|------|------|------|
| G1 | **测试文件** | `test_ptxir_serialization.cpp` (Mode 4) | 不存在 | 无法验证序列化/反序列化 roundtrip |
| G2 | **预生成 `.ptxir` 文件** | `tests/ptxir/*.ptxir` | 目录存在但为空 | 无现成 fast-load 测试数据 |
| G3 | **`generate_ptxir()`** | ANTLR 解析 + 序列化一站式函数 | 不存在 | 无法从 PTX 源码生成 `.ptxir` |
| G4 | **`load_ptxir(apply_cfg)`** | 加载 + 可选 CFG Builder | 不存在 | 缺少 `apply_cfg` 参数支持 |
| G5 | **`generate_tests.py` 集成** | `--mode mode4` + `--ptxir` 选项 | 不存在 | 无法自动化生成 Mode 4 测试 |
| G6 | **四模式测试文档** | `THREE-MODE-TESTING-GUIDE.md` 升级 | 未更新 | 文档仍为三模式框架 |
| G7 | **CI/CD Action** | 自动生成+缓存 `.ptxir` | 不存在 | 每次 CI 需重新 ANTLR 解析 |
| G8 | **Reader 格式兼容性** | 严格按 TOC 解析 | `read_string_table()` 硬编码偏移 | 格式不匹配时静默错误 |

### 2.3 格式实现偏差（⚠️）

| # | 问题 | 现象 | 根因 |
|---|------|------|------|
| D1 | **TOC 未写入** | `write_header()` 设置 `section_count=2` 但未写入 TOC 条目 | 实现简化——Writer 跳过 TOC 直接写数据 |
| D2 | **字符串表偏移未更新** | header 中 `string_table_offset` 和 `string_table_size` 始终为 0 | 未在 `write_string_table()` 后回填 |
| D3 | **Reader 硬编码偏移** | `read_string_table()` 从 `sizeof(PtxirHeader)` 读取 | 碰巧正确（Writer 在 header 后直接写字符串表），但破坏格式契约 |
| D4 | **Writer 写顺序偏差** | 设计文档：header → TOC → REGDECL → KERNEL → STRING_TABLE；实际：header → string_table → kernel_section | REGDECL 和 TYPE section 完全未实现 |
| D5 | **Writer 未写 TOC 条目** | `write_header()` 后应写 `section_count` 个 `PtxirSectionTOC` 条目 | 当前为 0 个 TOC 条目 |

### 2.4 Writer vs Reader 指令覆盖差异

| 指令类型 | Writer | Reader | 差距 |
|----------|--------|--------|------|
| BranchInstr | ✅ | ✅ | 一致 |
| LabelInstr | ✅ | ✅ | 一致 |
| VoidInstr | ✅ | ✅ | 一致 |
| BarrierInstr | ✅ | ✅ | 一致 |
| GenericInstr | ✅ | ✅ (S_MOV/S_ADD/S_SUB/S_MUL/S_LD/S_ST/S_SETP/S_CVT) | Reader 仅覆盖 8/106 种 GenericInstr |
| DeclarationInstr | ✅ | ✅ | 一致 |
| BarWarpSyncInstr | ✅ | ✅ | 一致 |
| PragmaInstr | ✅ | ✅ | 一致 |
| DollarNameInstr | ✅ | ✅ | 一致 |
| MembarInstr | ✅ | ❌ | Reader 缺 |
| FenceInstr | ✅ | ❌ | Reader 缺 |
| ReduxSyncInstr | ✅ | ❌ | Reader 缺 |
| MbarrierInstr | ✅ | ❌ | Reader 缺 |
| CallInstr | ✅ | ❌ | Reader 缺 |
| PredicatePrefix | ✅ | ❌ | Reader 缺 |
| VoteInstr | ✅ | ❌ | Reader 缺 |
| ShflInstr | ✅ | ❌ | Reader 缺 |
| AtomInstr | ✅ | ❌ | Reader 缺 |
| TextureInstr | ✅ | ❌ | Reader 缺 |
| SurfaceInstr | ✅ | ❌ | Reader 缺 |
| ReductionInstr | ✅ | ❌ | Reader 缺 |
| PrefetchInstr | ✅ | ❌ | Reader 缺 |
| CpAsyncInstr | ✅ | ❌ | Reader 缺 |
| AbiDirective | ✅ | ❌ | Reader 缺 |

**总计**: Writer 支持 24 种指令类型，Reader 仅完整支持 12 种，12 种缺失（Reader 对未知类型走 `default` 分支静默跳过）。

## 3. 差距清单

| # | 差距项 | 严重程度 | 优先级 | 关联 change |
|---|--------|---------|--------|------------|
| G1 | 无 PTXIR 序列化测试文件 | 高 | P1 | 需新建 `test_ptxir_serialization.cpp` |
| G2 | 无预生成 `.ptxir` 测试数据 | 中 | P2 | 需 `generate_ptxir()` 工具 |
| G3 | `generate_ptxir()` 未实现 | 中 | P2 | 2026-06-09-ptxir-serialization-architecture |
| G4 | `load_ptxir(apply_cfg)` 未实现 | 低 | P3 | 同上 |
| G5 | `generate_tests.py` 未集成 Mode 4 | 中 | P2 | 同上 |
| G6 | 测试文档未升级为四模式 | 低 | P3 | 文档更新 |
| G7 | 无 CI/CD `.ptxir` 缓存 Action | 低 | P3 | 基础设施 |
| G8 | Reader 格式兼容性问题（硬编码偏移） | 高 | P1 | 需修复 reader |
| D1-D5 | 格式实现偏差（TOC 未写入、顺序错误） | 中 | P2 | 需对齐 writer 实现与设计文档 |
| G9 | Reader 指令覆盖不足（12/24 种，`default` 静默跳过） | 高 | P1 | 需补全 12 种缺失指令类型 |

## 4. 补齐路径

### Phase 1: 核心修复（P1 项）

**依赖**: 无外部依赖，可在当前编译条件下独立完成

1. **修复 Reader 指令覆盖**（G9）
   - 为 MembarInstr, FenceInstr, ReduxSyncInstr, MbarrierInstr, CallInstr, PredicatePrefix, VoteInstr, ShflInstr, AtomInstr, TextureInstr, SurfaceInstr, ReductionInstr, PrefetchInstr, CpAsyncInstr, AbiDirective 添加 `case` 分支
   - 每个分支的解析逻辑：读取 qualifiers + operands，按 Writer 写出的格式反序列化
   - 移除 `default` 分支的静默跳过行为，改为 `throw` 或至少 `assert`（未知 Opcode 属异常情况）

2. **修复 Reader 格式兼容性**（G8, D1-D5）
   - 更新 Writer：按设计文档顺序写入 header → TOC entries → REGDECL section → KERNEL section → STRING_TABLE section
   - 更新 Reader：按 TOC 索引定位各个 section，而非硬编码偏移
   - 回填 header 中的 `string_table_offset` 和 `string_table_size`

3. **创建 Mode 4 测试文件**（G1）
   - 创建 `tests/unit/test_ptxir_serialization.cpp`
   - roundtrip 测试：serialize → deserialize → 比较 statement 数量和类型
   - 覆盖：BranchInstr, GenericInstr, BarrierInstr, DeclarationInstr, VoidInstr, LabelInstr 等核心类型

### Phase 2: 工具链（P2 项）

**依赖**: Phase 1 完成

4. **实现 `generate_ptxir()`**（G3）
   - 调用 `load_ptx_statements()` (ANTLR) → `serialize_statements()` (PTXIR)
   - 放置在 `src/ptxir/ptxir_serialization.cpp` 或 `tests/common/` 中

5. **实现 `load_ptxir(apply_cfg)`**（G4）
   - `deserialize_statements()` + 可选 `CFGBuilder::build()`
   - 放置在 `src/ptxir/ptxir_serialization.cpp`

6. **预生成 `.ptxir` 文件**（G2）
   - 扫描 `tests/` 下所有 `.ptx` 文件，调用 `generate_ptxir()` 生成对应 `.ptxir`
   - 存入 `tests/ptxir/`

### Phase 3: 完善（P3 项）

**依赖**: Phase 1-2 完成

7. **更新测试文档**（G6）
   - `docs/developer-guide/THREE-MODE-TESTING-GUIDE.md` → 四模式框架
   - 更新 `docs/developer-guide/README.md` 索引

8. **CI/CD Action**（G7）
   - `.github/workflows/generate-ptxir.yml`
   - 检测 `.ptx` 变更 → 重新生成 `.ptxir` → 缓存到 CI artifact

### 执行顺序

```
Phase 1 (G9, G8, D1-D5, G1)
  ├── G9: Reader 补充 12 种指令类型
  ├── G8+D1-D5: Writer/Reader 格式对齐 + TOC 实现
  └── G1: Mode 4 测试文件（roundtrip 验证）
      ↓
Phase 2 (G3, G4, G2)
  ├── G3: generate_ptxir() API
  ├── G4: load_ptxir(apply_cfg) API
  └── G2: 预生成 .ptxir 文件
      ↓
Phase 3 (G5, G6, G7)
  ├── G5: generate_tests.py 集成
  ├── G6: 文档更新
  └── G7: CI/CD Action
```

## 5. 参考资料

- **ADR 相关**:
  - ADR-0016: Blackwell-only tcgen05 → PTXIR 序列化架构
  - ADR-0003: 三阶段架构（arch → plan → ship）

- **Change artifacts**:
  - `openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/` — 完整设计文档（proposal.md, design.md, tasks.md, specs/）

- **当前实现文件**:
  - `include/ptx_ir/ptxir_format.h` — 格式定义
  - `src/ptx_ir/ptxir_writer.cpp` — 序列化实现
  - `src/ptx_ir/ptxir_reader.cpp` — 反序列化实现
  - `src/ptxir/ptxir_serialization.cpp` — 便捷 API 层
  - `include/ptxir/ptxir_serialization.h` — API 头文件

- **技能文档**:
  - `.opencode/skills/ptxir-serialization/SKILL.md` — PTXIR 格式技能文档
  - `.opencode/skills/ptxir-serialization/SKILL.md` 中"Limitations"章节

- **测试文件**:
  - `tests/ptxir/` — 预生成 `.ptxir` 目录（当前为空）
  - 无现有 PTXIR 测试文件