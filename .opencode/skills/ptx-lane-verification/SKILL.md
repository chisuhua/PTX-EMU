# PTX Lane Verification Skill

## 用途

基于 lane report 的分析结论，通过 `statement_factory.h` 构建与原始 PTX 等价的 `StatementContext` 序列，使用 `execute_warp_instruction()` 批量驱动 32 lane 执行，验证每个 lane 的分支决策符合 report 结论。

## 核心思想

PTX 分支验证分为两个阶段：
1. **LLM 分析阶段** → 理解 register 赋值链、predicate 逻辑、分支条件语义（由 `ptx-lane-tracer` 完成）
2. **Warp 验证阶段** → 基于分析结果，使用 `statement_factory` 构建语句序列，通过 `execute_warp_instruction()` 验证每个 lane 的分支决策

**混合架构**：
1. **Lane Tracer (LLM)** → 生成 lane report，包含唯一路径、tid.x 范围、关键分支点
2. **Lane Verification (C++)** → 构建 StatementContext 序列，批量执行，验证分支决策

## 工作流程

### 步骤 1: 使用 Lane Tracer 生成 Report

```bash
python3 ptx_lane_tracer.py test.ptx _Z16kernelIiEvPT 25 --generate-analysis > analysis.json
```

将 PTX 文件和模板发给 LLM 分析，填充 predicates、branches、loop 信息。

```bash
python3 ptx_lane_tracer.py test.ptx _Z16kernelIiEvPT 25 -a analysis.json -o lane_report.md
```

### 步骤 2: 配置验证参数

根据 lane report 配置 `PathConfig`：

```cpp
using namespace ptxsim::verification;

std::vector<PathConfig> paths = {
    {
        .name = "Path 1",
        .lane_ids = {0},
        .statements = build_statements_from_ptx(),  // 使用 statement_factory
        .expected_decisions = {
            {.pc = 47, .predicate = "%p3", .expect_taken = false, .fallback_pc = 50, .target_pc = 52}
        }
    },
    {
        .name = "Path 2",
        .lane_ids = {1, 2, /* ... */ 15},
        .statements = build_statements_from_ptx(),
        .expected_decisions = {
            {.pc = 47, .predicate = "%p3", .expect_taken = true, .fallback_pc = 50, .target_pc = 52}
        }
    }
};
```

### 步骤 3: 执行验证

```cpp
ExecutionEngineConfig config;
config.label2pc["$L__BB0_7"] = 7;
config.label2pc["$L__BB0_2"] = 2;
config.name2RegIndex["tid.x"] = 1;  // %r1 存储 tid.x
config.enable_pc_trace = false;

auto results = verify_warp_branch_decisions(warp, paths, config);

// 输出结果
for (const auto& result : results) {
    fmt::print("Path: {} - Result: {}\n", result.path_name, result.passed ? "PASS" : "FAIL");
    if (!result.passed) {
        fmt::print("  Error: {}\n", result.error_msg);
    }
}
```

---

## API 参考

### 数据结构

#### `ptxsim::verification::PCTraceEntry`

| 字段 | 类型 | 说明 |
|------|------|------|
| `pc` | `int` | 程序计数器 |
| `line` | `int` | 源码行号 |
| `instruction` | `std::string` | 指令文本 |

#### `ptxsim::verification::PathVerificationResult`

| 字段 | 类型 | 说明 |
|------|------|------|
| `path_name` | `std::string` | 路径名称 |
| `passed` | `bool` | 验证是否通过 |
| `error_msg` | `std::string` | 错误信息（如果失败） |
| `expected_lanes` | `std::vector<int>` | 期望的 lane ID 列表 |
| `actual_lanes` | `std::vector<int>` | 实际的 lane ID 列表 |

#### `ptxsim::verification::PathConfig`

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `std::string` | 路径名称 |
| `lane_ids` | `std::vector<int>` | 该路径包含的 lane ID |
| `statements` | `std::vector<StatementContext>` | 语句序列（由 statement_factory 构建） |
| `expected_decisions` | `std::vector<BranchDecision>` | 期望的分支决策 |

#### `ptxsim::verification::BranchDecision`

| 字段 | 类型 | 说明 |
|------|------|------|
| `pc` | `int` | 分支指令的 PC |
| `predicate` | `std::string` | predicate 名称，如 `"%p1"`, `"%p3"` |
| `expect_taken` | `bool` | true=分支跳转, false=不跳转(fallthrough) |
| `fallback_pc` | `int` | 不跳转时的下一 PC |
| `target_pc` | `int` | 跳转时的目标 PC |

#### `ptxsim::verification::ExecutionEngineConfig`

| 字段 | 类型 | 说明 |
|------|------|------|
| `label2pc` | `std::map<std::string, int>` | 标签 → PC 映射 |
| `name2RegIndex` | `std::map<std::string, int>` | 特殊寄存器名 → 寄存器索引 |
| `enable_pc_trace` | `bool` | 是否启用 PC trace（预留接口） |

---

### 核心函数

#### `verify_warp_branch_decisions`

```cpp
std::vector<PathVerificationResult> verify_warp_branch_decisions(
    WarpContext* warp,
    const std::vector<PathConfig>& paths,
    const ExecutionEngineConfig& config
);
```

**功能**: 批量验证 warp 分支决策

**参数**:
- `warp` - WarpContext 指针
- `paths` - 路径配置列表
- `config` - 执行引擎配置

**返回**: 每个 PathConfig 对应一个 PathVerificationResult

**执行流程**:
```
FOR each path in paths:
  1. reset_path_state(warp)  // 重置 simt_stack, exec_mask, PC=0
  2. warp = create_execution_warp(sm, path.statements, config)
  3. Execute path:
     WHILE any lane not finished:
       FOR each stmt in path.statements:
         warp->execute_warp_instruction(stmt, stmt_pc)
       END WHILE
  4. Collect branch decisions for all active lanes
  5. Compare with path.expected_decisions
  6. Record result
END FOR
```

---

#### `create_execution_warp`

```cpp
WarpContext* create_execution_warp(
    SMContext* sm,
    const std::vector<StatementContext>& statements,
    const ExecutionEngineConfig& config
);
```

**功能**: 辅助函数 - 构建完整执行环境

**参数**:
- `sm` - SMContext 指针
- `statements` - 语句序列
- `config` - 执行引擎配置

**返回**: WarpContext 指针

---

#### `set_predicate_value`

```cpp
void set_predicate_value(
    ThreadContext* thread,
    const std::string& pred_name,
    bool value
);
```

**功能**: 辅助函数 - 设置 predicate 寄存器值

**参数**:
- `thread` - ThreadContext 指针
- `pred_name` - predicate 名称（如 `"%p1"`）
- `value` - predicate 值（true/false）

---

#### `reset_path_state`

```cpp
void reset_path_state(WarpContext* warp);
```

**功能**: 辅助函数 - 重置路径间状态

**参数**:
- `warp` - WarpContext 指针

**重置内容**:
- SIMT stack
- Execution mask
- PC 复位到 0

---

#### `collect_pc_traces`

```cpp
std::map<int, std::vector<PCTraceEntry>> collect_pc_traces(
    WarpContext* warp,
    bool enable
);
```

**功能**: 可选 - 收集 PC trace

**参数**:
- `warp` - WarpContext 指针
- `enable` - 是否启用收集

**返回**: lane_id → PC trace 序列的映射

---

## 输出格式

```
=== PTX Lane Verification ===
Source: test_divergence_sync_standalone.ptx
Kernel: test_divergence_sync_standalone
Start Line: 25

--- Path 1: tid.x = [0] (1 lane) ---
Expected: tid.x=0, @%p3 bra NOT taken
Actual: tid.x=0, @%p3 bra NOT taken
Result: PASS

--- Path 2: tid.x = [1-15] (15 lanes) ---
Expected: tid.x=1-15, @%p3 bra taken
Actual: tid.x=1-15, @%p3 bra taken
Result: PASS

--- Path 3: tid.x = [16-31] (16 lanes) ---
Expected: tid.x=16-31, @%p3 bra taken + loop
Actual: tid.x=16-31, @%p3 bra taken + loop
Result: PASS

=== OVERALL: PASS (3/3 paths verified) ===
```

### 失败输出格式

```
--- Path 2: tid.x = [1-15] (15 lanes) ---
Expected: @%p3 bra taken (target_pc=52)
Actual: @%p3 bra NOT taken (fallback_pc=50)
Result: FAIL
Error: Branch decision mismatch at PC=47: expected taken=true, actual taken=false
```

---

## 使用示例

### 示例 1: 基本路径验证

```cpp
#include "utils/ptx_lane_verification.h"

void verify_branch_decisions() {
    auto* gpu = GPUContext::create();
    auto* sm = gpu->get_sm(0);
    auto* cta = sm->create_cta(1, 1);
    auto* warp = cta->get_warp(0);

    ExecutionEngineConfig config;
    config.label2pc["$L__BB0_7"] = 7;
    config.label2pc["$L__BB0_2"] = 2;
    config.name2RegIndex["tid.x"] = 1;

    std::vector<PathConfig> paths = {
        {
            .name = "Path 1",
            .lane_ids = {0},
            .statements = build_path_statements(),
            .expected_decisions = {
                {.pc = 47, .predicate = "%p3", .expect_taken = false, .fallback_pc = 50, .target_pc = 52}
            }
        }
    };

    auto results = verify_warp_branch_decisions(warp, paths, config);

    for (const auto& result : results) {
        fmt::print("{}: {}\n", result.path_name, result.passed ? "PASS" : "FAIL");
    }
}
```

### 示例 2: 带循环的路径验证

```cpp
// Path 3 (tid.x 16-31) 包含循环，需要多次迭代
PathConfig path3;
path3.name = "Path 3";
path3.lane_ids = {16, 17, /* ... */ 31};
path3.statements = build_loop_statements();
path3.expected_decisions = {
    {.pc = 47, .predicate = "%p3", .expect_taken = true, .fallback_pc = 50, .target_pc = 52},
    // 循环分支决策
    {.pc = 2, .predicate = "%p2", .expect_taken = true, .fallback_pc = 5, .target_pc = 2}
};

// verify_warp_branch_decisions 自动处理循环迭代
auto results = verify_warp_branch_decisions(warp, {path3}, config);
```

---

## 验证检查清单

实现时必须处理的依赖项：

| 依赖项 | 说明 | 处理方式 |
|--------|------|----------|
| `InstructionFactory::initialize()` | 指令处理函数映射 | 在测试前调用一次 |
| `label2pc` 映射 | 分支目标解析 | 通过 config 传入 |
| RegisterBankManager | 寄存器预分配 | CTAContext.init() 时设置 |
| Predicate 寄存器值 | 分支条件判断 | 通过 set_predicate_value() 设置 |
| simt_stack 重置 | 路径间清理 | 通过 reset_path_state() |
| name2Sym 符号表 | 变量解析 | 通过 create_execution_warp() 传入 |
| ExecutionTracer | PC trace 收集 | 通过 collect_pc_traces() 启用 |

---

## 故障排除

### 问题 1: 分支决策不匹配

**症状**: `Branch decision mismatch at PC=X`

**可能原因**:
1. Predicate 值设置错误 → 检查 `set_predicate_value()` 调用
2. label2pc 映射不完整 → 确认所有分支目标标签都在 config 中
3. 循环未正确处理 → 验证执行循环逻辑

**排查步骤**:
```cpp
// 启用 PC trace 辅助调试
config.enable_pc_trace = true;
auto results = verify_warp_branch_decisions(warp, paths, config);
auto traces = collect_pc_traces(warp, true);

// 检查每个 lane 的执行轨迹
for (const auto& [lane_id, entries] : traces) {
    fmt::print("Lane {}:\n", lane_id);
    for (const auto& e : entries) {
        fmt::print("  PC={} {}\n", e.pc, e.instruction);
    }
}
```

### 问题 2: 执行循环未终止

**症状**: `while (any_lane_active)` 无限循环

**可能原因**:
1. 循环条件判断错误 → 检查 `is_lane_active()` 实现
2. FINISHED_PC 未正确设置 → 检查 ret/exit 指令处理
3. 循环计数寄存器未更新 → 检查循环增量逻辑

**排查步骤**:
```cpp
// 添加调试输出
while (any_lane_active) {
    for (size_t pc = 0; pc < statements.size(); ++pc) {
        warp->execute_warp_instruction(statements[pc], static_cast<int>(pc));
    }

    any_lane_active = false;
    for (int i = 0; i < 32; ++i) {
        if (warp->is_lane_active(i)) {
            auto* thread = warp->get_thread(i);
            int thread_pc = thread->get_pc();
            if (thread_pc != FINISHED_PC) {
                any_lane_active = true;
                fmt::print("Lane {} still active at PC={}\n", i, thread_pc);
            }
        }
    }
}
```

### 问题 3: 路径状态残留

**症状**: 路径 2 的结果受路径 1 影响

**原因**: 路径间状态未完全重置

**解决方案**: 确保每次验证前调用 `reset_path_state(warp)`

```cpp
for (const auto& path : paths) {
    reset_path_state(warp);  // 重置 simt_stack, exec_mask, PC
    // ... 执行验证
}
```

---

## 已知限制

1. **不支持函数调用**：当前设计只处理单 kernel，不支持跨函数调用
2. **不支持动态分支**：循环次数必须在编译时确定
3. **不支持异步 barrier**：假设所有 barrier 都是同步的

---

## 参考文档

- 设计文档: `docs/superpowers/specs/2026-05-11-ptx-lane-verification-design.md`
- Lane Tracer: `.opencode/skills/ptx-lane-tracer/SKILL.md`
- 参考实现:
  - `tests/test_divergence_sync_isolated.cpp` - 使用 statement_factory 构建指令序列
  - `tests/test_barrier_verification_integrated.cpp` - 使用 execute_warp_instruction 驱动
  - `tests/test_nested_divergence.cpp` - 分歧验证示例
- 相关 API:
  - `statement_factory.h` - 工厂函数
  - `warp_context.cpp:execute_warp_instruction()` - 执行入口
  - `thread_context.cpp:_execute_once()` - 单条指令执行