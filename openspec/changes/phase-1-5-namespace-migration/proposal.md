## Why

HSK-8 spec 决策点 1（[decision-1.md §117](../2026-08-22-hsk-8-ptxemu-public-api-ack.md)）承诺 `include/ptx_ir/statement_context.h` 等 5 个 IR 类型文件晋升至 `include/ptxemu/ir/`，并提供 `ptxemu::ir` 命名空间包裹。Phase 1 已在 commit `564174f7` 完成 scaffolding（`include/ptxemu/ir/` 5 个 header 含完整定义 + ptxemu::ir 命名空间），但实测 **218 个 src/include/tests 文件**仍包含未限定 IR 类型名或使用 `#include "ptx_ir/..."`，未实际迁移（分项：src 58、include 40、tests 120；其中 `include/ptx_ir/` 非 shim 头与 `include/ptxir/` 亦在范围内）。HSK-8 audit §Postmortem 明确标记 Phase 1.5 deferred item 触发窗口为"HSK-8 ack 2026-08-22 + 1 release cycle"（≈ 2026-09 中旬），已逼近。

本次 change 完成 `ptxemu::ir` 命名空间承诺，消除 spec/code drift，并为未来 HSK-9（PTXEMU_API_VERSION=1→2）准入提供干净的公共面基线。

## What Changes

- **`include/ptx_ir/{ptx_types,operand_context,statement_context}.h` 改造为 forwarding shim**：canonical 定义在 `include/ptxemu/ir/`，shim 通过 `using` 声明保持旧路径可用（per `statement-ir-public/spec.md:46-48` Scenario）
- **`src/ptx_ir/*.cpp` 自身迁移到 `ptxemu::ir` 命名空间**：函数实现（`Q2s/S2s/Q2bytes/extractREG` 等）必须加 `ptxemu::ir::` 前缀以避免 ODR 冲突
- **218 个 caller 文件按目录/子批次切 commit 迁移到 `ptxemu::ir::*` 限定名**：`src/ptx_ir/` 自身 → `src/ptxir/` → `src/ptx_parser/` → `src/ptxsim/instructions/` → `src/ptxsim/core+utils` → `src/cudart/` → `include/ptx_ir/` 非 shim 头 + `include/ptxir/` → 其他 `include/` → `tests/unit/` → `tests/integration/` → `tests/e2e/`
- **`include/ptxsim/gpu_context.h` 接口重签名**：3 处 `std::vector<StatementContext>` → `std::vector<ptxemu::ir::StatementContext>`（line 58/80/173）
- **drift_check workflow 新增 Invariant 8**：使用 token-aware scanner 禁止 src/include/tests 出现裸 IR 类型名（仅豁免 3 个 forwarding shim header，且忽略 `ptxemu::ir::` 限定、注释和字符串），防止后续回归
- **OpenSpec 文档同步**：`openspec/specs/statement-ir-public/spec.md` Scenario "5 文件自洽 include" + "旧路径 forwarding header 一个 release 周期" 验证通过；HSK-8 audit §Postmortem 标记 Phase 1.5 关闭

**BREAKING**: 内部 src/include/tests caller 全部由裸 IR 类型名改为 `ptxemu::ir::Qualifier` 等限定形式（218 个文件，实际替换数以每 phase 的结构化扫描为准）。`libcudart.so` / `libptxemu_device.so` / `libptxemu_core.so` 公共 ABI 不变（PTXEMU_API_VERSION=1 冻结）。

## Capabilities

### New Capabilities

- `ptxemu-ir-namespace-contract`: 显式锁定 `ptxemu::ir` 命名空间为 IR 类型公共面，定义 20+ 指令结构 + 6 operand variant + InstructionState/Qualifier/StatementType 枚举的 namespace 边界、ODR 兼容性约束、forwarding shim 契约

### Modified Capabilities

- `statement-ir-public`: 现有 spec §46-48 Scenario "旧路径 forwarding header 一个 release 周期" 由"承诺"变"已实现"；新增 Scenario "src/ptx_ir/*.cpp 函数实现在 ptxemu::ir 命名空间"以锁定 ODR 契约
- `ci-drift-check`: 现有 spec 新增 Invariant 8 需求（禁止裸名 IR 类型回归），保留 Invariant 1-7 不变

## Impact

**代码影响**：
- `include/ptx_ir/{ptx_types,operand_context,statement_context}.h` 改为 forwarding shim，包含类型和自由函数的显式 `using ::ptxemu::ir::...` 声明
- `src/ptx_ir/*.cpp` 8 个文件全部 wrap 到 `ptxemu::ir` 命名空间，函数实现加 `ptxemu::ir::` 前缀
- 218 个 src/include/tests caller 文件裸类型名 → 限定名，覆盖 `include/ptx_ir/` 非 shim 头和 `include/ptxir/`
- `include/ptxsim/gpu_context.h` 3 处 type signature 修改
- `.github/workflows/drift_check.yml` 新增 Invariant 8 步骤
- `include/ptx_ir/AGENTS.md`（已在 commit `d7890a61` 修正）状态更新

**API 影响**：PTXEMU_API_VERSION=1 不变；`include/ptxemu/device_api.h` 公共 ABI 不变；`libptxemu_core.so` / `libcudart.so` / `libptxemu_device.so` 对外符号不变。CppTLM-as-consumer 路径不受影响（`cpp 不暴露` 不变量下 CppTLM 看不到 `ptx_ir/`，更看不到 namespace 迁移）。

**测试影响**：252 ctest target 全部需要 ctest 252/252 绿（每 phase 验证）；PTXIR roundtrip 测试 (`unit_ptxir_serialization` 等) 特别敏感（直接构造 `StatementContext{type, data}`，namespace wrap 后需限定）。

**工时估算**：Medium (1-2d 累计)，按 Oracle SPLIT 建议拆为 **11-13 个原子 commit**：1.5c+d (1 commit, ~3h) → 1.5e (src/ptx_parser, ~1h) → 1.5f1/f2 (src/ptxsim/instructions + core/utils, ~3h) → 1.5g (src/cudart, ~1h) → 1.5h1/h2 (include/ptx_ir 非 shim + include/ptxir、其他 include, ~2h) → 1.5i1/i2/i3 (tests unit/integration/e2e, ~3h) → 1.5j (gpu_context, ~30min) → 1.5k (drift_check, ~30min)。

**风险**：高 — 218 files 跨 9+ 目录，AI 误改率较高。Oracle 建议"per-directory 子批次切 commit + each ≤30 files"是核心缓解；任何 phase 失败立即 revert 独立 commit 不污染后续。

**前置已完成（本次 change 起点）**：
- commit `d7890a61` Phase 1.5a: AGENTS.md 文档谎言修正 + HSK-8 audit postmortem 同步
- commit `2cd8449e` Phase 1.5b: def 文件单源化（`include/ptx_ir/ptx_{op,qualifier}.def` → 7-8 行 shim）
- baseline = `2cd8449e` (HEAD, ctest 252/252 verified)
