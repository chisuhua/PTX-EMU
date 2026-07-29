# refactor-ptxir-writer - Design

## Overview

`PtxirWriter::write_instruction()` 是 PTXIR 二进制序列化的核心函数。当前实现在单个 `stmt.visit()` lambda 内用 31 个 `if constexpr (std::is_same_v<T, ...>)` 分支处理所有 `InstrVariant` 子类型，导致 232 行巨型函数。

本 change 将每个 variant 分支提取为独立的 private helper 方法，按指令类别分组，保持二进制输出 byte-identical。

## Design Decisions

### 决策 1: 拆分策略 - per-type private helper 方法

**选择**: 在 `PtxirWriter` 类中新增 private helper 方法，每个 `InstrVariant` 子类型一个方法

**理由**:
- 最小化变更范围：不需要新建文件，仅扩展类接口
- 保持 `write_instruction()` 作为唯一公开入口，签名不变
- helper 方法可直接访问 `out_`、`str2id_`、`reg2id_` 等 private 成员

**替代方案**:
- A. 提取为自由函数 + 传 `out_` 参数 -> 参数传递冗余，丢失封装
- B. 按类别拆为多文件 -> 过度工程，当前单文件 360 行不算大
- C. **采用**: 类内 private helper 方法

### 决策 2: 分组方式 - 按指令类别注释分组

**选择**: 在头文件和实现文件中用注释分组，不引入额外类/命名空间

**分组**:
1. **Control flow**: `write_branch()`, `write_label()`, `write_void()`, `write_abi_directive()`
2. **Barrier/Sync**: `write_barrier()`, `write_bar_warp_sync()`, `write_membar()`, `write_fence()`, `write_redux_sync()`, `write_mbarrier()`
3. **Generic/Declaration**: `write_generic()`, `write_declaration()`, `write_predicate_prefix()`
4. **Warp collective**: `write_vote()`, `write_shfl()`
5. **Memory/Atomic**: `write_atom()`, `write_texture()`, `write_surface()`, `write_reduction()`, `write_prefetch()`, `write_cp_async()`
6. **Misc**: `write_pragma()`, `write_dollar_name()`, `write_call()`

**理由**: 注释分组提供视觉结构，同时保持单类单文件的简洁性

### 决策 3: 分发方式 - 保持 std::visit + if constexpr

**选择**: `write_instruction()` 保留 `stmt.visit()` + `if constexpr` 链，但每个分支仅调用对应 helper

**理由**:
- `std::visit` + `if constexpr` 是 `InstrVariant` (std::variant) 的标准分发方式
- 不引入 visitor 类或函数指针表（过度工程）
- 分支体从多行序列化逻辑缩减为单行 `write_xxx(instr)` 调用

**write_instruction() 重构后伪码**:
```cpp
void PtxirWriter::write_instruction(const StatementContext& stmt) {
    write_u16(out_, static_cast<uint16_t>(stmt.type));
    stmt.visit([this](const auto& instr) {
        using T = std::decay_t<decltype(instr)>;
        if constexpr (std::is_same_v<T, BranchInstr>)     write_branch(instr);
        else if constexpr (std::is_same_v<T, LabelInstr>) write_label(instr);
        // ... 31 branches, each one-liner
    });
}
```

### 决策 4: 提取公共 operand 序列化逻辑

**选择**: 提取重复的 operand 遍历模式为 `write_operands()` helper

**理由**:
- `GenericInstr`、`AtomInstr`、`CallInstr` 等分支有几乎相同的 operand 遍历逻辑
- 差异仅在是否处理 `ImmOperand`（部分类型不支持 IMM）
- 提取为带 `bool write_imm` 参数的 helper 可消除约 100 行重复代码

**helper 签名**:
```cpp
void write_operands(const std::vector<OperandContext>& operands, bool write_imm);
void write_qualifiers(const std::vector<Qualifier>& qualifiers);
```

## Implementation Plan

### Phase 1: 提取公共 helpers（qualifiers + operands）
1. 新增 `write_qualifiers()` 和 `write_operands()` private 方法
2. 在头文件添加声明
3. 替换 31 个分支中的 qualifier/operand 遍历循环
4. 验证: `cmake --build build && ctest` 全绿

### Phase 2: 提取 per-type helper 方法
1. 为每个 variant 分支创建独立 helper 方法
2. 在头文件添加声明（按类别注释分组）
3. 将 `write_instruction()` 各分支体替换为单行调用
4. 验证: `cmake --build build && ctest` 全绿

### Phase 3: Round-trip 验证
1. 运行所有 PTXIR 相关测试（`ctest -R ptxir`）
2. 运行 `./tests/ptx/test_all_ptx.sh` 确认 PTX 语法测试全绿
3. 确认 `write_instruction()` < 50 行

### Phase 4: 提交
1. 验证 LSP 诊断无错误
2. git commit

## Testing Strategy

### 验证维度

| 测试类型 | 命令 | 预期 |
|---------|------|------|
| PTXIR round-trip | `ctest -R unit_ptxir` | 全绿（write->read 无损） |
| PTX 语法 | `./tests/ptx/test_all_ptx.sh` | 全绿 |
| 全量 ctest | `cd build && ctest` | 全绿 |
| 行数检查 | `wc -l` on write_instruction | < 50 行 |
| 二进制一致性 | 对比拆分前后 .ptxir 文件 | byte-identical |

### 二进制一致性验证方法

```bash
# 拆分前生成基线
cmake --build build && ./build/bin/test-ptx --serialize-sample > before.ptxir

# 拆分后重新生成
cmake --build build && ./build/bin/test-ptx --serialize-sample > after.ptxir

# 逐字节比较
diff before.ptxir after.ptxir  # 应无输出
# 或
cmp before.ptxir after.ptxir   # 应无输出
```

### 每个指令类别的序列化函数可独立测试

- 拆分后可通过 mock ostream 或直接调用 helper 方法验证单个类型的序列化输出
- 现有 round-trip 测试已覆盖所有类型的正确性验证

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| helper 方法签名错误 | 编译失败 | MUST 保持 `const T&` 参数类型一致 |
| 二进制输出偏移 | round-trip 失败 | MUST 对比拆分前后 .ptxir 文件；Phase 2 每步验证 |
| 头文件膨胀 | 编译时间微增 | 可接受（helper 方法声明约 30 行） |
| 公共 operand helper 行为偏移 | 多个类型受影响 | MUST `write_imm` 参数正确区分 IMM 支持类型 |

## Open Questions

1. **是否将 helper 方法设为 `protected` 以支持子类化？**
   - 推荐：NO（当前无子类化需求，private 即可）
   - 决定：private

2. **是否同时重构 `pre_pass()` 中的类似 variant 分支？**
   - 推荐：NO（`pre_pass` 仅 35 行，未达拆分阈值）
   - 决定：明确划入 Out Scope

## 关联文档

- `improvements/refactor-ptxir-writer.md`：完整 5 段提案
- `src/ptx_ir/AGENTS.md`：PTXIR 模块结构
- `.opencode/skills/ptxir-serialization/SKILL.md`：PTXIR 二进制格式规范
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-4`：原债务条目
