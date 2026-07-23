# ADR-0013: StatementContext 测试统一模式 — statement_factory + execute_warp_instruction

| 属性 | 值 |
|------|-----|
| **状态** | Proposed |
| **日期** | 2026-05-09 |
| **作者** | Sisyphus |
| **审核人** | — |

## 上下文

### 问题描述

当前项目的单元测试在构造 `StatementContext` 序列时存在大量重复模式：

1. **`make_barrier_stmt` 重复定义**：12 个测试文件各自实现了签名相同的本地 `make_barrier_stmt(uint32_t mask, int reconv_pc)` 函数
2. **工厂函数缺失**：虽然 `statement_factory.h` 提供了 `makeBarWarpSyncInstr`，但缺少接受 `mask`（uint32_t）+ `reconv_pc`（int）参数的便捷重载
3. **测试模式碎片化**：有的测试直接用 `statement_factory.h`，有的用本地辅助函数，没有统一约定

### 根因

`statement_factory.h` 的 `makeBarWarpSyncInstr` 最初只提供基于 `qualifiers` + `operands` 的低阶 API：

```cpp
inline StatementContext makeBarWarpSyncInstr(
    const std::vector<Qualifier> &qualifiers,
    const std::vector<OperandContext> &operands,
    const std::string &text = "")
```

测试代码需要手动构建 operands list（包含 mask 的十六进制字符串 + reconv_pc 的整数字符串），这导致了每个测试文件都自行封装 `make_barrier_stmt` 包装层。

## 决策驱动因素

1. **DRY 原则**：消除 12 个测试文件中 `make_barrier_stmt` 的重复实现
2. **可发现性**：新的测试作者应能通过 `statement_factory.h` 直接找到所需工厂函数，无需复制粘贴
3. **一致性**：所有通过 `execute_warp_instruction` 驱动的单元测试应遵循相同构造模式
4. **可维护性**：当 `BarWarpSyncInstr` 的 operand 格式变更时，只需修改一处工厂函数

## 考虑的替代方案

### 方案 A：不修改 statement_factory.h，仅统一测试中的 make_barrier_stmt 实现（✅ 已实施部分）

**描述**: 在 `statement_factory.h` 中添加 `makeBarWarpSyncInstr(mask uint32_t, reconv_pc int)` 重载，消除重复的本地包装函数。

**优点**:
- 改动范围小，仅影响 `statement_factory.h`
- 现有测试文件的本地 `make_barrier_stmt` 可逐步替换

**缺点**:
- 需要逐个修改 12 个测试文件
- 不解决测试构造模式碎片化问题

### 方案 B：建立 TestStatementFactory 头文件（不推荐）

**描述**: 创建一个 `test_utils.h`，包含所有测试专用的 `StatementContext` 工厂函数。

**优点**:
- 测试代码集中管理

**缺点**:
- 增加间接层，测试作者需知道使用 `test_utils.h` 而非 `statement_factory.h`
- 与项目架构中 `ptxir::factory` 命名空间分离
- 本质上是复制 `statement_factory.h` 到测试目录，没有消除重复

### 方案 C：仅通过 ADR 建立约定，不修改现有代码（不推荐）

**描述**: 通过 ADR 约定"测试中应使用 `statement_factory.h`，不应自行实现 `make_barrier_stmt`"，但不实际修改现有代码。

**优点**:
- 无代码改动

**缺点**:
- 无法解决当前 12 个测试文件已有的重复代码
- 没有强制性，约定会被遗忘

## 决策内容

### 1. 扩展 statement_factory.h 的工厂函数

在 `statement_factory.h` 中添加两个便捷重载：

```cpp
// 重载 1：接受 mask (string) + reconv_pc
inline StatementContext makeBarWarpSyncInstr(
    const std::string &mask, int reconv_pc, const std::string &text = "");

// 重载 2：接受 mask (uint32_t) + reconv_pc，自动转十六进制
inline StatementContext makeBarWarpSyncInstr(
    uint32_t mask, int reconv_pc, const std::string &text = "");
```

### 2. 建立测试构造模式规范

所有通过 `execute_warp_instruction` 驱动的单元测试应遵循：

```cpp
// ✅ 正确：使用 statement_factory.h
#include "ptx_ir/statement_factory.h"
stmts.push_back(ptxir::factory::makeBarWarpSyncInstr(0xFF, 9));

// ❌ 错误：本地重复实现 make_barrier_stmt
static StatementContext make_barrier_stmt(uint32_t mask, int reconv_pc) { ... }
```

### 3. 统一测试文件本地 make_barrier_stmt 的替换

12 个测试文件按以下规则替换：

| 文件 | 当前 `make_barrier_stmt` 签名 | 替换为 |
|------|------------------------------|--------|
| `test_warp_barrier_integrated.cpp` | `make_barrier_stmt(uint32_t, int)` | `makeBarWarpSyncInstr(mask, reconv_pc)` |
| `test_barrier_reconvergence.cpp` | `make_barrier_stmt()` | `makeBarWarpSyncInstr(0xFFFFFFFF, 0)` |
| `test_barrier_scenarios_integrated.cpp` | `make_barrier_stmt(uint32_t, int)` | `makeBarWarpSyncInstr(mask, reconv_pc)` |
| `test_barrier_verification_integrated.cpp` | `make_barrier_stmt(uint32_t, int)` | `makeBarWarpSyncInstr(mask, reconv_pc)` |
| `test_divergence_sync_isolated.cpp` | `make_barrier_stmt(uint32_t, int)` | **已删除（合并到 standalone）** |
| `test_pc_management_integrated.cpp` | `make_barrier_stmt(uint32_t, int)` | `makeBarWarpSyncInstr(mask, reconv_pc)` |
| `test_post_barrier_divergence.cpp` | `make_barrier_stmt(uint32_t, int)` | **已合并（5→2 TEST_CASE）** |
| `test_simt_stack_entry_integrated.cpp` | `make_barrier_stmt(uint32_t, int)` | `makeBarWarpSyncInstr(mask, reconv_pc)` |
| `test_simt_thread_pc_integrated.cpp` | `make_barrier_stmt(uint32_t, int)` | `makeBarWarpSyncInstr(mask, reconv_pc)` |
| `test_sync_mechanism_integrated.cpp` | `make_barrier_stmt(uint32_t, int)` | `makeBarWarpSyncInstr(mask, reconv_pc)` |
| `test_syncthreads_direction.cpp` | `make_barrier_stmt()` | `makeBarWarpSyncInstr(0xFFFFFFFF, 0)` |
| `test_warp_state_integrated.cpp` | `make_barrier_stmt(uint32_t, int)` | `makeBarWarpSyncInstr(mask, reconv_pc)` |
| `test_shortest_path_first.cpp` | — | **已删除（无实际行为覆盖）** |

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptx_ir/statement_factory.h` | 修改 | 添加 `makeBarWarpSyncInstr` 重载 |
| `tests/test_warp_barrier_integrated.cpp` | 重构 | 移除本地 `make_barrier_stmt`，使用 `ptxir::factory::makeBarWarpSyncInstr` |
| `tests/test_barrier_reconvergence.cpp` | 重构 | 同上 |
| `tests/test_barrier_scenarios_integrated.cpp` | 重构 | 同上 |
| `tests/test_barrier_verification_integrated.cpp` | 重构 | 同上 |
| `tests/test_divergence_sync_isolated.cpp` | **已删除** | 合并到 `test_divergence_sync_standalone_integrated.cpp` |
| `tests/test_pc_management_integrated.cpp` | 重构 | 同上 |
| `tests/test_post_barrier_divergence.cpp` | **已合并** | 5→2 TEST_CASE，保留已知问题文档 |
| `tests/test_simt_stack_entry_integrated.cpp` | 重构 | 同上 |
| `tests/test_simt_thread_pc_integrated.cpp` | 重构 | 同上 |
| `tests/test_sync_mechanism_integrated.cpp` | 重构 | 同上 |
| `tests/test_syncthreads_direction.cpp` | 重构 | 同上 |
| `tests/test_warp_state_integrated.cpp` | 重构 | 同上 |
| `tests/test_shortest_path_first.cpp` | **已删除** | 无实际行为覆盖，已移除 |

## 后果

### 正面影响

- 消除 12 个测试文件中 `make_barrier_stmt` 的重复实现（约 200 行重复代码）
- `statement_factory.h` 成为测试中 `StatementContext` 构造的唯一权威来源
- 新增测试可直接使用 `ptxir::factory::makeBarWarpSyncInstr`，无需复制粘贴

### 负面影响

- 需要修改 12 个测试文件，验证所有测试仍然通过
- 部分使用 `to_hex()` 辅助函数的测试（如 `test_divergence_sync_isolated.cpp`）需要调整

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 重构过程中引入测试回归 | 中 | 高 | 每修改一个文件立即运行测试验证 |
| 部分测试依赖本地 `make_barrier_stmt` 的隐式行为（如 `to_hex` 格式） | 低 | 中 | 审查每个文件的使用点，确保替换后语义一致 |

## 合规检查

后续相关开发应检查：

- [ ] 新增测试使用 `ptxir::factory::makeBarWarpSyncInstr` 而非本地 `make_barrier_stmt`
- [ ] `statement_factory.h` 保持完整，所有便捷工厂函数已实现
- [ ] 所有重构后的测试通过原有测试用例

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-09 | 初始版本 | Sisyphus |
| 2026-06-04 | 标记 `test_divergence_sync_isolated.cpp` / `test_shortest_path_first.cpp` 已删除，`test_post_barrier_divergence.cpp` 已合并（5→2 TEST_CASE） | Sisyphus |

## 参考

- [statement_factory.h](file:///workspace/project/PTX-EMU/include/ptx_ir/statement_factory.h)
- [ADR-0009: X-Macro + Weak Symbol 指令分发模式](./ADR-0009-xmacro-instruction-dispatch.md)