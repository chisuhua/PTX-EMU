## Context

### 当前 PTX 执行管道

```
CUDA Binary (.cu)
    ↓ nvcc 编译
PTX 文本 (AT&T 语法格式)
    ↓ cuobjdump -xptx (Mode 1) 或直接读取 .ptx 文件 (Mode 2)
PTX 文本字符串 (std::string)
    ↓ ANTLR 词法/语法分析
PTX AST (ParseTree)
    ↓ PtxVisitor::visit()
PtxContext { ptxKernels[], ptxStatements[] }
    ↓ 提取 kernelStatements
std::vector<StatementContext>
    ↓ CFGBuilder::build()
reconvergence_pc 填充
    ↓ PtxInterpreter::submit_kernel_request()
GPUContext 执行
```

**问题**：ANTLR 解析在每次测试时重复执行。对于大型 kernel（如光线追踪，上千条指令），解析耗时 ~200ms，而实际执行只需 ~10ms。重复解析是测试框架最大的性能瓶颈。

### 现有三模式测试框架

| Mode | 加载方式 | ANTLR | 适用场景 |
|------|----------|-------|----------|
| Mode 1 | cuobjdump 动态提取 | ✅ 每次 | CI/CD、端到端集成 |
| Mode 2 | 读取预存 .ptx 文件 | ✅ 每次 | 稳定复现、版本控制 |
| Mode 3a/3b | 直接构造 StatementContext | ❌ | 单元测试、精确定位 |
| Mode 3C | popen() 外部二进制 | N/A | FAIL 复现 |

### 参考：LLVM-IR 序列化

LLVM 的 bitcode 格式采用三层架构：

```
Layer 3: ModuleBitcodeWriter   — 领域语义（类型/常量/函数）
Layer 2: BitcodeWriterBase     — 字符串表、值枚举
Layer 1: BitstreamWriter       — 原始二进制编码（块/记录）
```

**关键模式**：
1. **预遍历赋值**：写入前对所有 Value 做一次编号，指令通过紧凑的数字 ID 而非字符串引用操作数
2. **Section TOC + 尾部字符串表**：头部索引各 Section 偏移/大小，字符串表最后写（所有引用已解析）
3. **自描述指令格式**：每条指令自己记录长度，未知指令可跳过（前向兼容）
4. **版本字段置顶**：读取第一件事就是检查版本

## Goals / Non-Goals

**Goals:**

1. 定义 `.ptxir` 二进制格式，持久化已解析的 `StatementContext` 序列
2. 实现序列化（PTX 文本 → `.ptxir`）和反序列化（`.ptxir` → `vector<StatementContext>`）
3. 增加 Mode 4 测试：从 `.ptxir` 快速加载，绕过 ANTLR 解析
4. 性能目标：`.ptxir` 加载比完整 ANTLR 解析快 **40x+**（目标 ~5ms vs ~200ms）
5. 向后兼容：现有 Mode 1/2/3/3C 全部保留

**Non-Goals:**

- 不修改 ANTLR 解析器本身（语法规则、词法分析不变）
- 不修改 `StatementContext` 结构体（data variant 不变）
- 不修改 `InstructionFactory` 或指令执行逻辑
- 不支持 `.ptxir` 在生产路径（`cudart_sim.cpp`）中使用——仅用于测试加速
- 不实现压缩或 VBR 编码（V1 阶段用固定宽度整数足够）

## Decisions

### Decision 1: 文件格式 — 扁平二进制 + Section TOC（非 Bitstream）

**选择**：固定宽度整数字节序 + Section TOC 头部 + 尾部字符串表

**理由**：
- 相比 LLVM Bitstream 的 VBR/缩略符编码，PTX-EMU V1 数据量不需要复杂压缩
- Section TOC 头部提供 O(1) 随机访问能力
- 实现和维护成本低，调试容易（十六进制编辑器直接可读）

**替代方案考虑**：
- **LLVM Bitstream**：太复杂，VBR 编码对 ~1000 条指令的 kernel 无必要
- **JSON 格式**：可读但体积大（~10x），反序列化 CPU 开销高于二进制
- **Protocol Buffers**：需要 `.proto` 定义和 `protoc` 编译，依赖增加

### Decision 2: 指令编码 — Opcode ID + 操作数扁平紧凑二进制

**选择**：每条指令编码为 `[opcode:u16, pred:u8, type:u8, dst:u32, src_count:u8, srcs[src_count]:u32]`

**理由**：
- Opcode 枚举值（来自 `ptx_op.def` 的 `StatementType`）已经是紧凑整数 ID
- 操作数通过预遍历赋值（见下）变成 `u32` 索引
- `std::variant<BranchInstr, GenericInstr, ...>` 的 22 种指令类型在序列化时用 Opcode ID 区分

**替代方案考虑**：
- **自描述 TLV**：每条指令先写类型长度，但 PTX 指令长度是固定的（Opcode 决定操作数格式），TLV 过于冗余
- **文本格式 (.ptxir.txt)**：可读但解析慢，与目标"快速加载"矛盾

### Decision 3: 值枚举（预遍历赋值）— 与 LLVM 相同的 ID 分配策略

**选择**：序列化前遍历一次 `kernelStatements`，为所有 `RegOperand` 分配紧凑 `u32` ID，构建 `operand_table`

**理由**：
- PTX 寄存器名 `%r5`、`%p1` 等字符串如果在指令中重复存储，`.ptxir` 体积膨胀 ~5x
- 值枚举后，指令中只存储 `u32` ID，`operand_table[ID]` 提供原始字符串
- 与 LLVM 的 `ValueEnumerator` 模式完全一致

**实现**：
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

### Decision 4: 字符串表位置 — 最后写

**选择**：所有 Section 写完后，最后写 STRING_TABLE_BLOCK，offset 记录在 TOC 头部

**理由**：
- 与 LLVM STRTAB_BLOCK、DWARF debug_str 相同模式
- 避免序列化过程中字符串 offset 不确定的问题
- 读取时先读 TOC，再读取字符串表供其他 Section 引用

### Decision 5: 版本管理 — Extend-Only（只增不改）

**选择**：版本字段在头部；现有 Opcode ID 永不改变；新增 Opcode 只能往后追加

**理由**：
- .ptxir 是测试基础设施，不需要生产级的前向兼容
- V1 只支持 sm_50-sm_80 的基本 PTX ISA
- 未来 V2 增加 wmma/mma、Hopper cluster 时，V1 reader 可以跳过未知 block（根据 TOC 里的 block type 字段）

### Decision 6: 放置位置 — `include/ptx_ir/ptxir_format.h` + `src/ptx_ir/`

**选择**：格式定义头文件在 `include/ptx_ir/`，序列化实现在 `src/ptx_ir/`（新增目录）

**理由**：
- `StatementContext`、`OperandContext` 在 `include/ptx_ir/`，`.ptxir` 格式与它们同目录便于包含
- `src/ptx_ir/` 中已有的 IR 相关实现在 `src/ptx_parser/` 和 `src/ptxsim/`，新增 `src/ptx_ir/` 用于"IR 层"序列化，与解析器分离
- `test_helpers.hpp` 引用 `include/ptx_ir/` 下的头文件，序列化函数可以无缝加入

### Decision 7: CFG 处理时机 — 反序列化后可选应用

**选择**：`load_ptxir(path, apply_cfg=false)` 的 `apply_cfg` 参数控制是否在反序列化后运行 `CFGBuilder`

**理由**：
- Mode 3a（CFG 前）对应 `apply_cfg=false` — 原始解析状态
- Mode 3b（CFG 后）对应 `apply_cfg=true` — 已填充 `reconvergence_pc`
- `.ptxir` 文件中 `reconvergence_pc` 可以不存储（由反序列化调用方决定是否需要 CFG），保持文件内容与运行时状态解耦
- 简化 `.ptxir` 格式：不需要存储 label2pc 等中间状态

## Risks / Trade-offs

**[Risk] StatementContext 包含 std::variant（InstrVariant）无法直接二进制序列化**

→ **Mitigation**：`std::variant` 本身是 POD 兼容的（只包含union + discriminant），但 C++ variant 的二进制表示未标准化。用Opcode ID 区分类型后，每种指令类型单独序列化其字段（如 `BranchInstr { target_id, predicate_id, reconvergence_pc }`），反序列化时按 Opcode 重建 variant。

**[Risk] OperandContext 有 shared_ptr 和 string 等动态对象**

→ **Mitigation**：`shared_ptr<Predicate>` 在 PTX 中罕见（Predicate 通常是内联字面值），序列化时展开为直接量。字符串 `ImmOperand::value` 和 `RegOperand::name` 通过字符串表去重存储。

**[Risk] X-Macro 冲突导致头文件无法同时包含**

→ **Mitigation**：Mode 4 测试只 include `statement_context.h` + `operand_context.h` + `ptx_types.h`，不 include `ptx_parser.h`。test_helpers.hpp 已经做到这点——它只引用 IR 层头文件，不引用 ANTLR 解析器。

**[Risk] 序列化格式变更导致历史 `.ptxir` 文件无法加载**

→ **Mitigation**：头部有 `version:u16` 字段，reader 检查版本不匹配时抛出异常而非静默错误。V1 格式稳定后很少变更。

**[Trade-off] 实现复杂度 vs 收益**

→ 序列化层本身 ~500 行代码（writer + reader + format.h），相比 ANTLR 解析 ~2000 行不算大。但维护成本是真实的——每次 `StatementContext` 结构体变更需要同步更新序列化器。建议 V1 阶段 freeze `StatementContext` 结构体。

## Migration Plan

**阶段 1（V1）**：仅测试路径
1. 新增 `include/ptx_ir/ptxir_format.h`，定义 `.ptxir` 头部结构和 Section 枚举
2. 实现 `src/ptx_ir/ptxir_writer.cpp` 和 `ptxir_reader.cpp`（约 500 行）
3. 在 `test_helpers.hpp` 中增加 `serialize_statements()` / `deserialize_statements()` / `generate_ptxir()` / `load_ptxir()`
4. 增加 Mode 4 测试文件 `test_ptxir_mode4.cpp`，验证序列化和执行
5. 使用 `generate_tests.py --mode mode4` 自动生成 Mode 4 测试

**阶段 2（V1）**：工具链集成
1. 在 `docs/skills/ptxir-serialization/` 增加技能文档
2. 更新 `THREE-MODE-TESTING-GUIDE.md` 为四模式文档
3. 预生成现有 PTX 文件的 `.ptxir` 版本，存入 `tests/ptxir/`
4. 将 `generate_ptxir()` 集成到 `generate_tests.py`

**回滚策略**：删除 `src/ptx_ir/` 和相关代码，`test_helpers.hpp` 中移除 4 个函数，`CMakeLists.txt` 移除 Mode 4 构建目标即可。无破坏性变更。

## Open Questions

1. **`.ptxir` 文件应该 git 追踪吗？**
   - 正方：版本化的 PTX 内容可以被构建产物缓存
   - 反方：二进制文件在 git diff 中无意义，增加仓库大小
   - 建议：`.gitignore` 默认忽略 `tests/ptxir/*.ptxir`，只在 CI/CD 中通过 Action 自动生成缓存

2. **是否需要支持反序列化后直接通过 `run_statement_sequence()` 执行？**
   - `run_statement_sequence()` 需要 `label2pc` 映射（从 StatementContext 中的 `S_LABEL` 构建）
   - `.ptxir` 本身不存储 label2pc（由序列化器从 StatementContext 重建）
   - 这意味着 Mode 4 等价于 `load_ptxir → apply_cfg → run_statement_sequence`，技术上可行

3. **`OperandContext::operand_phy_addr`（运行时物理地址缓存）是否序列化？**
   - 该字段是运行时填充的（`ThreadContext` 执行时设置），不应序列化
   - 反序列化时应为 `nullptr`，由执行时重新计算
