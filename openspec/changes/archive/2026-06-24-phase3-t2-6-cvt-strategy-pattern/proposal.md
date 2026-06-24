# Phase 3 T2-6: CVT 策略模式重构

## Why

`src/ptxsim/instructions/arithmetic_conversion.cpp` 是 PTX-EMU 中最长的单 handler 实现（1288 行），`CvtHandler::processOperation` 单方法占 1145 行（line 141-1286），核心逻辑全部在一个嵌套 4 层深的 `switch (dst_bytes)`（line 224-1284，1063 行）中。`half_to_float` / `float_to_half` 与 `half_utils.h::f16_to_f32` / `f32_to_f16` 重复 ~70 行。int→int 4×4×4 嵌套逻辑在 4 个 case 中几乎相同模板。

维护成本：每次 CVT 相关 bug 修复需在 4 个 case 中同步修改，新指令族扩展需复制整个 case 块。审计 P2-1 估计 3 天，本计划评估 3.5 天（含测试补全 + P1-4.1 协同）。

## What Changes

- **提取共享 helpers**：4 个 inline helper（`round_half_to_even`, `half_to_float`, `float_to_half`, `should_saturate_uint32`）从 `arithmetic_conversion.cpp` 顶部抽到 `include/ptxsim/instructions/cvt/cvt_helpers.h` + `src/ptxsim/instructions/cvt/cvt_helpers.cpp`
- **复用 `half_utils.h`**：删除文件内重复的 `half_to_float`/`float_to_half`，改用 `f16_to_f32`/`f32_to_f16`（含 1-2 个边界 case 对比测试）
- **策略模式重构（Composition 而非 Inheritance）**：保留 `CvtHandler` 单类（X-Macro 约束），内部通过 `ConversionStrategy` 接口 + 5+ 策略实现拆分
  - `IntToIntStrategy`（最大复用价值）
  - `IntToFloatStrategy`
  - `FloatToFloatStrategy`
  - `FloatToIntStrategy`（含 .sat / 5 种舍入 / 默认 truncate）
  - `RoundingMode` 工具类（独立头 `cvt_rounding.h`）
- **新增 94 个 integration tests**：覆盖 P0/P1/P2 优先级指令族（int↔int 全组合、float→int 全舍入、int→float、.sat 全部、f16↔f64 等）
- **P1-4.1 bug 修复**：f32→s32 / f64→s64 路径补齐 r2 写入（与 CVT 路径同 PR，注释引用 `BUG-P1-4.1-CVTR2`）

## Out of Scope

- CvtHandler 类拆分（X-Macro 约束禁止）
- CVT 指令的精度修复（属于精度任务，与本 change 正交）
- WMMA / Tensor Core CVT 变体
- T2-4（PTX 8.7+ 占位清理）— 留待 Phase 4

## Pre-conditions

- T1-1..T1-5 ✅（Phase 2 完成）
- T2-2, T2-4 Step 1, T2-5, T2-7 ✅
- P1-4.1 bug 修复与本 change 同 PR 提交（在 Step 5 启动前完成）

## Capabilities

### Modified Capabilities
- `ptx-cvt-instruction-execution`: 重构实现为 Composition 策略模式；外部行为完全不变

## Reference

- **Master plan**：`docs/superpowers/plans/2026-06-23-phase3-critical-debt.md` §Task 1
- **详细 tasks**：`openspec/changes/phase3-t2-6-cvt-strategy-pattern/tasks.md`
- **Explore 盘点报告**（会话上下文 bg_ec7aaee3 8m50s）：8 sections A-H 含完整 file:line 引用
- **现有测试**：`tests/integration/ptx/test_cvt.cpp`（224 行，3 个 case，1 SKIP）
- **遗留 reference**：`tests/reference/ptx_builtin/test_ptx_cvt.cu`（73 个 case，**未集成 ctest**）
- **CMake 注册**：`src/CMakeLists.txt:105`、`tests/integration/CMakeLists.txt:247`、`tests/unit/CMakeLists.txt:383`（注释掉 `unit_ptx_cvt`）
