# Phase 3 CVT Precision Bugfix

## Why

T2-6 Step 1 (commit `d3c77b5`) 暴露了 `arithmetic_conversion.cpp` 中 4 个 helper 的 **2 个预先存在 bug**。这些 bug 在 refactor 过程中**保留**（zero behavior change 约束），但需要独立 change 修复以恢复 PTX CVT 正确性。

## Bug 1: `half_to_float` denormal 路径错（**严重**）

- **位置**: `src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float`（原 `arithmetic_conversion.cpp:25-58`）
- **症状**: `half_to_float(0x0001)` 返回 `5.07e30`（IEEE 754 float 表示 `+2^102`）
- **期望**: `half_to_float(0x0001) = 2^-24 ≈ 5.96e-8`（half 最小 denormal 值）
- **根因**: denormal 路径的指数/尾数计算错误
  - 当前代码：`exp = 127 - 15 = 112`，然后循环左移 `mantissa` 直到 `bit 10` 设置
  - 对 `h=0x0001`：`mantissa=1`，左移 10 次后 `mantissa=0x400`（bit 10），exp 减到 102
  - 输出：`f = (0 << 31) | ((102+127) << 23) | (0 << 13) = 229 << 23 = +2^102`
  - **正确算法**：denormal 应该 = `0.xxx × 2^-14`（half bias 15，-bias-23 = -14）
- **影响**: 任何 PTX 含 `cvt.f32.f16` + 输入是 denormal half 值时静默返回错误结果

## Bug 2: `should_saturate_uint32` 边界判断错（**中等**）

- **位置**: `src/ptxsim/instructions/cvt/cvt_helpers.cpp::should_saturate_uint32`（原 `arithmetic_conversion.cpp:67-69`）
- **症状**: 当 `sat_high == 4294967295.0f` 时**永远返回 false**
- **期望**: `should_saturate_uint32(4294967295.0f, 4294967295.0f) = true`（值已在边界）
- **根因 1**: 实现用严格 `<` 而非 `<=`：
  ```cpp
  return temp >= 4294967295.0f && temp < sat_high;  // 应改为 <=
  ```
- **根因 2**: `4294967295.0f` 在 float32 中被舍入为 `4294967296.0f`（IEEE 754 精度损失）
- **影响**: PTX `cvt.u32.f32.sat` 在边界值处静默不饱和

## What Changes

- **修复 `half_to_float` denormal 路径**：重写 denormal 处理逻辑，对齐 `half_utils.h::f16_to_f32` 的正确实现
- **修复 `should_saturate_uint32` 边界**：`<` 改为 `<=`
- **更新 unit tests** `tests/unit/ptx/test_cvt_helpers.cpp`：解除 Step 1 中为保留 buggy 行为而**故意宽松**的断言，恢复正确 IEEE 754 行为验证
- **新增 integration tests** `tests/integration/ptx/test_cvt_edge_cases.cpp`：覆盖 PTX `cvt.f32.f16` denormal + `cvt.u32.f32.sat` 边界两种端到端场景

## Out of Scope

- T2-6 Step 2-6 推迟到此 change 完成后继续
- 其他 CVT 指令族的精度/边界修复（需后续独立审计）
- `half_utils.h::f16_to_f32` 自身的精度问题（已存在但需单独审计）

## Pre-conditions

- ✅ T2-6 Step 1（commit `d3c77b5`）— helpers 已抽离到 `cvt_helpers.cpp`
- ✅ `half_utils.h::f16_to_f32/f32_to_f16` 已存在（**注意**：T2-6 Step 2 计划复用它，但本 change 修复本地实现，不强制复用）

## Capabilities

### Modified Capabilities
- `ptx-cvt-instruction-execution`: 修复 `cvt.f32.f16` denormal + `cvt.u32.f32.sat` 边界正确性

## Reference

- **T2-6 Step 1 报告**: 见 master plan 上下文（commit `d3c77b5` 的 agent 报告，5 TEST_CASEs / 30 assertions）
- **Master plan**: `docs/superpowers/plans/2026-06-23-phase3-critical-debt.md`（未来需更新：T2-6 流程插入本 bugfix 步骤）
- **T2-6 tasks**: `openspec/changes/phase3-t2-6-cvt-strategy-pattern/tasks.md` Sub-task 1 Step 2 注释含详细 bug 分析
- **`half_utils.h` 参考**: `include/ptxsim/utils/half_utils.h`（修复 `half_to_float` denormal 时的正确实现参考）
- **IEEE 754 half precision 规范**: denormal = `0.sig × 2^-14`（half bias 15, denormal exponent = 1-15 = -14）

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|:---:|------|
| 修复 `half_to_float` 改变 denormal 之外的 behavior | 🟢 低 | 🟡 中 | 完整 5 TEST_CASEs 回归 + 16/16 CVT/PTX 集成测试 |
| `should_saturate_uint32` 改 `<=` 后破坏其他路径 | 🟢 低 | 🟡 中 | 改后跑全量 `ctest -L "ptx;cvt"` + sanity.sh --quick |
| PTX 用户已经依赖 buggy 行为 | 🟢 极低 | 🟢 低 | 行为本身违反 IEEE 754，依赖视为错误；dev 角度 0 用户 |
| `half_utils.h::f16_to_f32` 本身也有 bug | 🟡 中 | 🟡 中 | 修复前对比测试 `half_utils.h` vs `half_to_float` denormal 输出 |
