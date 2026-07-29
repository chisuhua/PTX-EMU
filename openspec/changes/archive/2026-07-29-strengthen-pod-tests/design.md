# strengthen-pod-tests - Design

## Overview

深化 4 个现有 context 单元测试文件，从"仅验证构造"扩展为行为验证（状态转换、边界条件、错误路径）。同时确保每个新增 unit 测试有对应的 `execute_warp_instruction` 集成测试（test-coverage-enforcer 要求）。

当前状态：4 个测试文件共 873 行，但有效行为断言稀疏。以 `test_warp_context.cpp`（98 行）为例，仅测试 `WarpContext` 的构造、active_mask setter、thread addition、completion —— 缺少状态转换语义、边界值、错误路径。

## Design Decisions

### 决策 1: 测试深化策略 - 行为维度

**选择**: 按 3 个行为维度深化每个测试文件：

1. **状态转换验证** - 验证状态变更的正确性（非仅构造后初始值）
2. **边界条件** - 0 线程、全活跃、最大值/最小值边界
3. **错误路径** - 非法参数、空指针、越界访问

**理由**:
- improvement 明确要求"状态转换、边界条件、错误路径"
- 3 维度覆盖确保测试深度而非仅广度
- 与 test-coverage-enforcer 的 unit -> integration 桥接要求一致

**替代方案**:
- A. 仅补充断言数量（凑行数）-> 不满足"行为验证"要求
- B. 重写整个测试文件 -> 风险高，可能破坏现有通过的测试
- C. **采用**: 增量深化，保留现有测试 + 新增行为测试 SECTION

### 决策 2: 每个文件的深化内容

**选择**: 按文件特性定制深化内容：

#### `test_warp_context.cpp` (98 -> ≥150 行)
- **状态转换**: `set_active_mask()` 的 overwrite 语义（非 OR 合并，AGENTS.md anti-pattern）
- **边界条件**: `active_mask = 0x0`（全 inactive）、`0xFFFFFFFF`（全 active）、`0x1`（仅 lane 0）
- **错误路径**: `is_lane_active(32)` 越界、`get_warp_thread_id(32)` 越界

#### `test_sm_context.cpp` (126 -> ≥180 行)
- **状态转换**: SM warp 调度状态转换（idle -> running -> done）
- **边界条件**: 0 warp SM、最大 warp 数 SM
- **错误路径**: 越界 warp index 访问

#### `test_cvt_context.cpp` (274 -> ≥330 行)
- **边界条件**: f16 最小/最大值、denormalized float、NaN/Inf 转换
- **错误路径**: 不支持的类型组合（如 f16 -> s64 with .sat）

#### `test_smcontext_injection.cpp` (375 -> ≥420 行)
- **错误路径**: null bridge 注入、detach 后操作、重复 attach
- **状态转换**: attach -> detach -> reattach 生命周期

### 决策 3: test-coverage-enforcer 合规

**选择**: 每个新增 unit 测试场景检查是否有对应 `execute_warp_instruction` 集成测试

**理由**:
- test-coverage-enforcer 技能要求：直接测试 barrier/warp/gpu 场景的 unit 测试必须有对应集成测试
- 避免 unit 测试通过但集成层面失败的情况

**实现**:
- `test_warp_context.cpp` 的新增状态转换测试 -> 检查 `tests/integration/exec/` 是否有对应 `execute_warp_instruction` 测试
- 如缺失 -> 在 `tests/integration/warp/` 或 `tests/integration/exec/` 新增对应集成测试
- 集成测试使用 `step_warp()` + `make_*` helper 驱动指令序列

### 决策 4: 测试框架和标签

**选择**: 使用 Catch2 框架 + `<type>;<subject>` 标签格式

**理由**:
- 项目约定（AGENTS.md: `ctest 命名必须前缀 unit_`，`测试标签格式 <type>;<subject>`）
- Catch2 是项目唯一测试框架

**标签分配**:
- `test_warp_context.cpp`: `[unit;warp]`
- `test_sm_context.cpp`: `[unit;sm]`
- `test_cvt_context.cpp`: `[unit;cvt]`
- `test_smcontext_injection.cpp`: `[unit;cpptlm]`
- 新增集成测试: `[integration;warp]` / `[integration;exec]`

### 决策 5: 不修改被测源码

**选择**: 仅修改测试文件，不修改 `src/ptxsim/` 源码

**理由**:
- improvement 明确 Out Scope: "不修改被测源码"
- 如测试暴露 bug，记录为单独的 fix change，不在此 change 中修复

## Implementation Plan

### Phase 1: 深化 test_warp_context.cpp
1. 新增 SECTION: active_mask overwrite 语义验证（非 OR 合并）
2. 新增 SECTION: 边界条件（0x0, 0xFFFFFFFF, 0x1）
3. 新增 SECTION: 越界 lane index 错误路径
4. 检查/新增对应 integration 测试（`execute_warp_instruction` 驱动）

### Phase 2: 深化 test_sm_context.cpp
1. 新增 SECTION: SM warp 调度状态转换
2. 新增 SECTION: 边界条件（0 warp, max warp）
3. 新增 SECTION: 越界 warp index 错误路径
4. 检查/新增对应 integration 测试

### Phase 3: 深化 test_cvt_context.cpp
1. 新增 SECTION: f16 边界值转换（min/max/denorm）
2. 新增 SECTION: NaN/Inf 转换行为
3. 新增 SECTION: 不支持的类型组合错误路径
4. 检查/新增对应 integration 测试

### Phase 4: 深化 test_smcontext_injection.cpp
1. 新增 SECTION: null bridge 注入错误路径
2. 新增 SECTION: detach 后操作错误路径
3. 新增 SECTION: attach -> detach -> reattach 生命周期
4. 检查/新增对应 integration 测试

## Testing Strategy

| 测试场景 | 文件 | 预期 |
|---------|------|------|
| active_mask overwrite | `test_warp_context.cpp` | setter 覆盖旧值，非 OR 合并 |
| 全 inactive warp | `test_warp_context.cpp` | `is_active() == false`, `get_active_count() == 0` |
| 越界 lane index | `test_warp_context.cpp` | 返回 false 或抛异常（按现有行为） |
| SM 状态转换 | `test_sm_context.cpp` | idle -> running -> done 正确转换 |
| f16 边界值 | `test_cvt_context.cpp` | min/max/denorm 转换正确 |
| null bridge 注入 | `test_smcontext_injection.cpp` | 返回错误或抛异常 |
| 全量 unit | `ctest -L unit` | 全绿 |
| 对应 integration | `ctest -L integration` | 全绿 |

### 验证流程

```bash
# 1. 编译
. env.sh && cmake --build build

# 2. 运行 unit 测试
cd build && ctest -L unit --output-on-failure

# 3. 运行对应 integration 测试
cd build && ctest -L integration --output-on-failure

# 4. 确认行数
wc -l tests/unit/warp/test_warp_context.cpp  # >= 150
wc -l tests/unit/warp/test_sm_context.cpp    # >= 180
wc -l tests/unit/ptx/test_cvt_context.cpp     # >= 330
wc -l tests/unit/cpptlm/test_smcontext_injection.cpp  # >= 420
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| 新增测试暴露已有 bug | unit 测试失败 | 记录 bug，创建单独 fix change；此 change 不修复源码 |
| 越界行为未定义 | 测试结果不确定 | 参照现有源码行为，测试实际行为而非期望行为 |
| integration 测试缺失需新建 | 增加 work 量 | 优先检查现有 integration 测试是否已覆盖；仅补充缺失部分 |
| 测试文件行数膨胀 | 可维护性下降 | 用 SECTION 组织，保持逻辑清晰；每文件不超过 500 行 |

## Open Questions

1. **越界访问的行为是返回 false 还是抛异常？**
   - 需检查源码实际行为，测试实际行为而非期望行为
   - 如行为不安全（如 segfault），测试应使用防御性检查

2. **test-coverage-enforcer 的对应 integration 测试是否全部缺失？**
   - Phase 1-4 中逐个检查；如已有覆盖则无需新增

## 关联文档

- `improvements/strengthen-pod-tests.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-6`：原债务条目
- `.opencode/skills/test-coverage-enforcer/SKILL.md`：test-coverage-enforcer 技能
- `tests/AGENTS.md`：3-tier 测试套件约定
