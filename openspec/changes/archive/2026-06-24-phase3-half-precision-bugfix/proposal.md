# Phase 3 Half Precision Bugfix

## Why

T2-6 Step 1 (commit `d3c77b5`) 抽离 CVT helpers 时 agent 通过对比 `include/ptxsim/utils/half_utils.h::f16_to_f32` 发现 **同源 bug**：

`half_utils.h::f16_to_f32` 的 denormal 路径错误（与 `arithmetic_conversion.cpp::half_to_float` 历史 bug 同源）。`phase3-cvt-precision-bugfix` (commit `32ce8a0`) 修复了**本地** `cvt_helpers.cpp::half_to_float` 但未触碰 `half_utils.h` 源文件。

**影响范围**：`half_utils.h` 是项目内任何 f16 ↔ f32 转换的共用工具。`grep -rn "f16_to_f32\|f32_to_f16" src/ include/ tests/` 找到所有调用方（包括 T2-6 之外的其他模块）。

## Bug 1: `f16_to_f32` denormal 路径错（**严重**）

- **位置**: `include/ptxsim/utils/half_utils.h::f16_to_f32` + 实现（在 `src/ptxsim/utils/half_utils.cpp`）
- **症状**: `f16_to_f32(0x0001)` 返回 `~4.6e-5`（IEEE 754 不正确）
- **期望**: `f16_to_f32(0x0001) = 2^-24 ≈ 5.96e-8`（half 最小 denormal）
- **根因**: 同一历史 bug（同 `arithmetic_conversion.cpp::half_to_float` 的原始实现）
  - denormal (exp=0) 走 `else` 分支，按 normal 处理：`f32 = (sign << 31) | ((0 + 112) << 23) | (mant << 13)`
  - 对 `h=0x0001` (mant=1): 指数 = 112 → 2^(112-127) = 2^-15 ≈ 3e-5，尾数 1<<13=8192 加进去 → 实际 ~4.6e-5
  - **正确**: denormal = `mantissa × 2^-24`；exp_f = 103 + p（p 是 mantissa 高 bit 位置）
- **已在 `cvt_helpers.cpp::half_to_float` 修复**（commit `32ce8a0`），但 `half_utils.h` 源未同步

## Bug 2: `f32_to_f16` denormal 路径错（**推测**，待确认）

- **位置**: `include/ptxsim/utils/half_utils.h::f32_to_f16`
- **症状**: 推测类似 — 极小 float 转 half 时返回错误结果
- **根因**: 同源（denormal 路径未正确处理）
- **需 agent 在 Task 1 调研时确认**：如 f32_to_f16 已正确处理 denormal，标记 N/A；如也有 bug，Task 1 一并修

## What Changes

- **修复 `half_utils.h::f16_to_f32` denormal 路径**：参考 `cvt_helpers.cpp::half_to_float`（commit `32ce8a0`）的正确算法
- **修复 `half_utils.h::f32_to_f16` denormal 路径**（如 Task 1 调研发现 bug）
- **新增 unit tests** `tests/unit/utils/test_half_utils.cpp`：覆盖 denormal + Inf + NaN + 边界 case
- **新增 integration tests** `tests/integration/utils/test_half_utils_ptx.cpp`（可选）：PTX `cvt.f32.f16` / `cvt.f16.f32` 端到端验证（f16_to_f32 / f32_to_f16 已通过 CVT 路径间接覆盖，本测试为冗余，按需）

## Out of Scope

- `cvt_helpers.cpp::half_to_float`（已在 `phase3-cvt-precision-bugfix` 修）
- 改 API 签名（保持 `inline uint16_t f32_to_f16(float)` 等）
- 性能优化（不引入 SIMD/查找表）

## Pre-conditions

- ✅ T2-6 Step 1（commit `d3c77b5`）
- ✅ `phase3-cvt-precision-bugfix`（commit `32ce8a0`）— 提供 `cvt_helpers.cpp::half_to_float` 正确算法参考
- 测试基础设施 `tests/unit/utils/` 存在

## Capabilities

### Modified Capabilities
- `half-precision-conversion`: `f16_to_f32` / `f32_to_f16` denormal 路径正确性恢复

## Reference

- **正本算法参考**: `src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float`（commit `32ce8a0`，已修）
- **CVT bugfix 报告**: `openspec/changes/phase3-cvt-precision-bugfix/proposal.md` Bug 1 节
- **T2-6 探索报告**: bg_ec7aaee3 8m50s 报告 §E.4 确认 `half_utils.h` 重复实现
- **IEEE 754 half precision**: denormal = `mantissa × 2^-14`（half bias 15, denormal exp = 1-15 = -14）

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|:---:|------|
| 修复破坏 `f32_to_f16` 的正常范围（如最大 half normal） | 🟢 低 | 🔴 高 | Task 1 调研后跑全量 `ctest -L "ptx;cvt"` + sanity.sh --quick 验证 |
| 修复后 `half_utils.h` 与 `cvt_helpers.cpp::half_to_float` 行为不一致 | 🟢 低 | 🟡 中 | Task 1 末对比测试（已在 `phase3-cvt-precision-bugfix` Task 2 Step 2 提到但未执行） |
| `half_utils.h` 有调用方已依赖 buggy 行为 | 🟢 极低 | 🟢 低 | 行为违反 IEEE 754，依赖视为错误；当前 bug 状态是"silent wrong" |
| T2-6 Step 2 后续复用变可执行 | 🟢 低 | 🟡 中 | 本 change 完成即解封 Step 2 |
