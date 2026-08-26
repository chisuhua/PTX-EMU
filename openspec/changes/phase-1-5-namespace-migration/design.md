## Context

**当前状态**（baseline `2cd8449e`）：
- `include/ptxemu/ir/` canonical headers 含完整 `ptxemu::ir` 命名空间定义（Phase 1 scaffolding commit `564174f7`），但**0 caller**（canonical 是合同基线，但 PTX-EMU 内部全部使用 `include/ptx_ir/...`）；实际 caller 文件集合以 Phase 0 scanner 生成并冻结的清单为准
- `include/ptx_ir/` 11 个 header 在全局命名空间，实测 **218 个 src/include/tests 文件**含未限定 IR 类型名（src 58、include 40、tests 120），其中 `include/ptx_ir/` 非 shim 头和 `include/ptxir/` 也必须迁移
- `include/ptx_ir/AGENTS.md` 此前文档谎言（声称 forwarding shim + using-directive），已在 commit `d7890a61` 修正
- `include/ptx_ir/{ptx_op,ptx_qualifier}.def` byte-equivalent 双维护，已在 commit `2cd8449e` 单源化为 11 行 shim
- `src/ptx_ir/` 实际包含 7 个 `.cpp`，其 namespace ownership 固定为：`ptx_types.cpp`、`operand_context.cpp`、`statement_context.cpp` 属于 `ptxemu::ir`；`instruction_latency_table.cpp` 属于 `ptxsim`；`ptx_syntax_utils.cpp` 属于 `ptx::syntax`；`ptxir_reader.cpp`、`ptxir_writer.cpp` 保持其头文件声明的 namespace。该 7 文件清单是本 change 的完整 `.cpp` 集合。
- `libcudart.so` / `libptxemu_device.so` / `libptxemu_core.so` 公共 ABI 冻结（`PTXEMU_API_VERSION=1`，HSK-8 spec §Decision 3）
- `cpp 不暴露` 约束（drift_check Invariant 4）保证 CppTLM 仓仅 `add_subdirectory(external/PTX-EMU)` 消费 `ptxemu_core` library，看不到 `ptx_ir/` 内部 header
- 252/252 ctest 绿

**驱动约束**：
- `openspec/specs/statement-ir-public/spec.md:33-40` 明确承诺 `ptxemu::ir::Statement` 等 5 文件晋升；spec §46-48 Scenario 描述旧路径 forwarding shim 形式（`#include <ptxemu/ir/foo.h> + namespace ptx_ir = ::ptxemu::ir;`）
- `ptxemu-core-library/spec.md:17-18` PUBLIC/PRIVATE 拆分：Phase 1.5 后 `ptx_ir/` 仍 PRIVATE（CppTLM 不可见），但 `ptxemu/ir/` 是 PUBLIC 一部分
- HSK-8 audit §Postmortem 标记 Phase 1.5 deferred item 触发窗口 = HSK-8 ack 2026-08-22 + 1 release cycle (≈ 2026-09 中旬)
- PTXEMU_API_VERSION=1 冻结 → HSK-9 准入是 PTXEMU public ABI 唯一变更触发点；本 change 不触及 public ABI
- ptx-lessons-learned §1 跨模块间接状态翻译警示：`using namespace` 全局 shim 风险（本 change 用 per-file 显式 `using ::ptxemu::ir::Type` 而非 `using namespace`）
- ptx-lessons-learned §3-4 分 Phase commit 纪律：任何 phase 失败立即 revert，不混入后续 commit

**Stakeholders**：
- PTX-EMU maintainer (chi-suhua)：负责本 change 实施
- CppTLM 仓 owner：消费方，**本 change 不影响**（`cpp 不暴露` 约束下 CppTLM 看不到 `ptx_ir/` 内部路径）
- 未来 HSK-9 触发者：需要 canonical public ABI 稳定

## Goals / Non-Goals

**Goals:**
- 完成 `statement-ir-public` spec §33-48 全部承诺（5 文件晋升 + 旧路径 forwarding shim + 1 release cycle 后清理）
- 218 个 src/include/tests caller 文件全部迁移到 `ptxemu::ir::*` 限定名，无函数行为变化
- `include/ptxsim/gpu_context.h` 3 处 type signature 重命名（`StatementContext` → `ptxemu::ir::StatementContext`）
- drift_check Invariant 8 防止后续代码回归到裸 IR 类型名
- HSK-8 audit §Postmortem Phase 1.5 状态从"deferred" → "completed"
- 252/252 ctest 全程绿（每 phase 验证）
- PTXEMU_API_VERSION=1 冻结面不受影响
- 公共 ABI（`libptxemu_core.so` / `libcudart.so` / `libptxemu_device.so`）不变

**Non-Goals:**
- 不重命名 `StatementContext` → `Statement`（虽然 spec L33 表格写了 `ptxemu::ir::Statement`，但实际 `include/ptxemu/ir/statement.h:308` 仍用 `StatementContext`；保持类名不变仅 namespace wrap 是最小风险路径；如需重命名类另开 change）
- 不触 `include/ptxemu/device_api.h` 公共面（`PTXEMU_API_VERSION=1` 冻结）
- 不删除 `include/ptx_ir/` 转发 shim（per `task 9.4` 1 release cycle 后再删）
- 不改 PTXIR 二进制格式（`include/ptxir/ptxir_format.h` 与 namespace 迁移正交）
- 不更新 CppTLM 仓（无 HSK 触发需要）
- 不引入 `using namespace ::ptxemu::ir;` 全局污染方案（违反 HSK-8 spec §Decision 6 "no namespace pollution"，ptx-lessons-learned §1 警示）
- 不动 ANTLR4-generated 头（`build/antlr4_generated_src/` 由 ANTLR runtime 管辖）

## Decisions

### D1: shim 形式 — per-file 显式 `using` 而非 `using namespace`

**Decision**：shim 兼容策略固定为“canonical 类型名兼容、unscoped enumerator 不兼容”。`include/ptx_ir/{ptx_types,operand_context,statement_context}.h` 改造为：
```cpp
#include <ptxemu/ir/foo.h>
namespace ptx_ir = ::ptxemu::ir;  // alias namespace, optional opt-in
using ::ptxemu::ir::Qualifier;
using ::ptxemu::ir::StatementContext;
// ... 每个类型一行
```

**Rationale**：
- 显式 `using` 列名避免 `using namespace` 引入的所有潜在 ODR 冲突（ptx-lessons-learned §1 警示）
- 旧 caller 用裸 scoped 类型（如 `Qualifier`）走 `using` 声明解析到 `ptxemu::ir::Qualifier`
- unscoped `StatementType`/`OperandType` 的裸 enumerator（如 `S_REG`/`O_REG`）不会随类型 using 自动导入；本 change 固定采用“类型名兼容、enumerator 不兼容”策略，避免在 shim 中重新导出大规模全局 enumerator 集合
- 旧 caller 用 `Qualifier::Q_F32` 通过 type using 等价 lookup 仍工作
- 旧 caller 用 `ptx_ir::Qualifier` 显式限定也工作（namespace alias）

**Alternatives considered**：
- (a) `using namespace ::ptxemu::ir;` 全局污染 — 拒绝（违反 HSK-8 Decision 6 + ptx-lessons-learned §1）
- (b) 重命名 `StatementContext` → `Statement` — 推迟（statement.h:308 当前名字，最小风险路径；如需重命名另开 change）
- (c) 完全删除 `ptx_ir/` 旧路径 — 拒绝（违反 spec §46-48 Scenario 承诺 + `task 9.4` 1 release cycle 清理窗口未到）

### D2: src/ptx_ir/*.cpp 命名空间 wrap 必须与 shim 同步

**Decision**：canonical IR 定义/实现文件（`ptx_types.cpp`、`operand_context.cpp`、`statement_context.cpp`，以及确实定义 canonical IR 类的方法）wrap 到 `ptxemu::ir`，函数定义与类方法保持 canonical 声明一致。`instruction_latency_table.cpp` 保留 `namespace ptxsim`，`ptx_syntax_utils.cpp` 保留 `namespace ptx::syntax`，两者仅将 IR 类型限定为 `ptxemu::ir::*`。`ptxir_reader.cpp`/`ptxir_writer.cpp`/serialization 文件保持其头文件声明的 namespace，不能盲目整体 wrap。

**Rationale**：
- 2026-08-26 实测发现：canonical `ptxemu/ir/ptx_types.h` 声明 `Q2s(Q2bytes/extractREG)` 但**无函数体**（仅声明），函数体仍在 `src/ptx_ir/ptx_types.cpp` 全局。Shim 把 `using Q2s = ptxemu::ir::Q2s` 暴露后，cpp 中 `Q2s(Qualifier q)` 重声明与 canonical 签名 ODR 冲突。
- 解决路径：cpp 内函数定义改用 `ptxemu::ir::Q2s(Qualifier q)` 限定形式，函数体包入 `namespace ptxemu::ir {}` 即可
- 验证：1.5c+d 合并 commit 后 ctest 252/252 必须绿

**Alternatives considered**：
- (a) shim 单独 commit（不变 cpp）— 失败，已在本次 session 验证（ODR 冲突立即 revert）
- (b) 删除 canonical 中的函数声明，保留 cpp 全局定义 — 拒绝（canonical 必须完整，cpp 仅实现细节）

### D3: per-directory 切 commit + 单 commit ≤30 sites

**Decision**：218 caller files 按调用拓扑和 ≤30 files/commit 约束切子批次，每批立即 ctest 252/252：`src/ptx_parser/` → `src/ptxsim/instructions/` → `src/ptxsim/core+utils/` → `src/cudart/` → `include/ptx_ir/` 非 shim + `include/ptxir/` → 其他 `include/` → `tests/unit/` → `tests/integration/` → `tests/e2e/`。

**Rationale**：
- ptx-lessons-learned §3-4 分 Phase commit 纪律
- Oracle SPLIT 建议：per-directory 子批次切 commit 才能 bisect；单 commit 218 files 出问题难以定位
- 拓扑顺序保证依赖方向：parser → sim → cudart → include（反向依赖 sim） → tests（消费所有）
- 每 commit 后 ctest 验证：失败立即 revert 不污染后续

**Alternatives considered**：
- (a) 单 commit 218 files — 拒绝（不可 bisect，review 不可读）
- (b) 随机目录顺序 — 拒绝（依赖方向会导致 forward-decl 错误）

### D4: GPUContext 接口重签名作为独立 commit (1.5j)

**Decision**：`include/ptxsim/gpu_context.h:58,80,173` 三处 `std::vector<StatementContext>` → `std::vector<ptxemu::ir::StatementContext>` 独立 commit。

**Rationale**：
- 头文件 type signature 修改会影响所有 include `gpu_context.h` 的 TU（~10 个 src/ + 3 个 include/）
- 必须独立 commit 让 review 可读，并允许在 1.5e-1.5i 之前或之后单独 verify
- 与 caller sweep 解耦：先做 caller sweep（仅类型名加限定），再做 GPUContext 重签名（实际类型变 namespace），降低 bisect 复杂度

**Alternatives considered**：
- (a) 与 1.5d 合并 — 拒绝（namespace wrap 阶段不该触动 type signature，1.5j 单独便于 bisect）

### D5: drift_check Invariant 8 — 禁止裸名 IR 类型回归

**Decision**：在 `.github/workflows/drift_check.yml` 新增 Invariant 8：
```yaml
# Invariant 8 uses a token-aware Python scanner, not bare `\bType\b` grep:
# `\bQualifier\b` also matches `ptxemu::ir::Qualifier` because `:` is non-word.
- name: Check no bare IR type names outside ptx_ir shim (Invariant 8)
  run: |
    python3 scripts/check_ptxemu_ir_names.py \
      --roots src include tests \
      --exclude include/ptx_ir/ptx_types.h \
      --exclude include/ptx_ir/operand_context.h \
      --exclude include/ptx_ir/statement_context.h
```

The scanner MUST:
- skip comments, ordinary/char literals, and C++ raw string literals;
- scan caller roots `src/`, `include/ptxsim/`, `include/ptxemu/` (excluding canonical definitions under `include/ptxemu/ir/`), `include/cudart/`, `include/ptx_parser/`, `include/register/`, `include/utils/`, `include/ptx_ir/`, `include/ptxir/`, and `tests/`;
- exclude only the three forwarding shim headers plus `include/ptxemu/ir/` canonical definition headers;
- ignore IR tokens already qualified by `ptxemu::ir::` and tokens lexically inside the canonical namespace block; and
- fail on a bare caller code token outside those explicit definition/shim exclusions.
The token set MUST include at least `StatementType`, `OperandType`, `InstructionState`, `Qualifier`, `OperandContext`, `InstrVariant`, `Tcgen05Instr`, `Tcgen05OpKind`, and `Tcgen05Dtype`; the implementation may derive the complete type/enumerator set from the canonical headers/def files.

**Rationale**：
- 防止 1.5c-1.5i 完成后，新代码又用裸 `Qualifier` 写，污染 namespace 迁移成果；扫描范围必须包含 `include/ptx_ir/` 非 shim 头和 `include/ptxir/`
- ptx-lessons-learned §1 警示：状态/类型翻译漂移是无声回归的常见来源
- 灰度期（1.5i 完成后 1 release cycle）期间 shim 仍存在，caller 可临时回退到 shim 路径，但应在 deadline 前清理

**Alternatives considered**：
- (a) 不加 invariant — 拒绝（漂移风险敞口）
- (b) 仅 grep `using namespace` — 拒绝（语义粗，且与 D1 显式 using 冲突）

## Risks / Trade-offs

- **[R1] 218 files 跨 9+ 目录，AI 误改率高** → Mitigation: per-directory 子批次 ≤30 files + 每 commit ctest 验证 + 失败立即 revert (per ptx-lessons-learned §3-4)
- **[R2] `StatementContext::toString()` out-of-line 定义（`src/ptx_ir/statement_context.cpp`）namespace wrap 后需重定位** → Mitigation: 1.5c+d 阶段明确将 `ptxemu::ir::StatementContext::toString` 定义 wrap 进 namespace
- **[R3] `std::visit` + ADL 在 namespace wrap 后行为可能微妙变化** → Mitigation: 1.5d 阶段重点验证 `src/ptx_parser/ptx_visitor*.cpp` 9 个文件 + `src/ptxsim/instruction_factory.cpp` 的 `handler_map` 分发
- **[R4] ANTLR4-generated 头（`build/antlr4_generated_src/`）含未限定 `StatementType/Qualifier`** → Mitigation: 本 change 不 sweep generated 头；这些头是 ANTLR4 runtime 管辖；如未来 ANTLR 升级触发 issue 另开 change
- **[R5] `src/ptxir/ptxir_serialization.cpp` 与 `include/ptxir/ptxir_serialization.h` 使用旧的全局 `StatementContext` 前置声明** → Mitigation: 1.5c+d 阶段改为 canonical `ptxemu::ir::StatementContext`，header 增加 canonical include 或合法 namespace forward declaration，implementation 与 API roundtrip 一并验证；不整体 wrap serialization namespace
- **[R6] `include/ptxemu/ir/statement.h` 头本身未实现类方法（与 `ptx_types.h` 同样模式）** → Mitigation: shim 后旧路径 caller 仍依赖 `src/ptx_ir/*.cpp` 实现，1.5c+d 阶段必须 wrap cpp；如未 wrap，链接失败
- **[R7] HSK-9 提前签发需要重新评估** → Mitigation: 1.5i 完成后未到 HSK-9 触发窗口前，本 change 闭合 spec/code drift 承诺；如 HSK-9 提前签发，新 change 重新评估
- **[R8] git 合并冲突与 PTX-EMU 作为 CppTLM submodule 约束** → Mitigation: 每个 phase 在独立分支提交并通过 PR 审查后合并；不直接 push `origin/main`，C++ code 冲突概率低（per-directory 切 commit 独立）

## Migration Plan

**Phase 顺序（12 implementation groups, 1.5a/1.5b 已完成）**：

1. **1.5a** ✅ (commit `d7890a61`): AGENTS.md 文档谎言修正
2. **1.5b** ✅ (commit `2cd8449e`): def 文件单源化
3. **1.5c+d** (1 commit, ~3h): shim 改造 + src/ptx_ir/*.cpp namespace wrap + 函数体加 `ptxemu::ir::` 前缀 + ctest 验证
4. **1.5e** (1 commit, ~1h): `src/ptx_parser/` caller sweep（实测 13 files）
5. **1.5f1** (1 commit, ~1.5h): `src/ptxsim/instructions/` caller sweep
6. **1.5f2** (1 commit, ~1.5h): `src/ptxsim/core+utils+debug/` caller sweep（src/ptxsim 合计实测 33 files）
7. **1.5g** (1 commit, ~1h): `src/cudart/` caller sweep（实测 4 files）
8. **1.5h1** (1 commit, ~1h): `include/ptx_ir/` 非 shim + `include/ptxir/` caller sweep
9. **1.5h2** (1 commit, ~1h): 其他 `include/{ptxsim,ptxemu,cudart,ptx_parser,register,utils}/` caller sweep（include 合计实测 40 files）
10. **1.5i1** (1 commit, ~1h): `tests/unit/` caller sweep
11. **1.5i2** (1 commit, ~1h): `tests/integration/` caller sweep
12. **1.5i3** (1 commit, ~1h): `tests/e2e/` caller sweep（tests 合计实测 120 files）
13. **1.5j** (1 commit, ~30min): `include/ptxsim/gpu_context.h` 3 处 type signature 重命名
14. **1.5k** (1 commit, ~30min): drift_check Invariant 8 + `openspec/specs/statement-ir-public/spec.md` Scenario 验证 + HSK-8 audit postmortem 关闭

**Rollback strategy**：
- 每 phase 独立 commit → 任何 phase 失败 `git revert HEAD` 不污染后续
- 1.5c 失败（已发生，已 revert 1 次）：`git checkout -- include/ptx_ir/*` + rebuild
- 1.5d-1.5i 失败：`git revert HEAD~N..HEAD --no-edit` 回滚到上一 commit
- 1.5j 失败：`git revert HEAD` 单 commit
- 1.5k 失败：drift_check workflow 自身可 disable（workflow_dispatch: false），不影响 main

**Baseline commit**：`2cd8449e`（1.5b 完成后状态）；Phase 0 必须先创建 scanner、冻结 caller 清单，并执行 `cmake --build build && cd build && ctest --output-on-failure`，以实际结果确认基线，不将历史声明视为独立证据

**Defer 触发条件**：
- 1.5c+d 失败 2 次 → 暂存，提交 `fix-phase-1-5-...` change 单独排查
- HSK-9 提前签发 → 暂停本 change，等 HSK-9 准入流程
- `cpp 不暴露` 不变量被破坏 → 立即 revert，触发新 HSK 协商

## Open Questions

1. **`StatementContext` 重命名 → `Statement` 不在本 change 范围。** 已固定保持类名不变；如未来需要重命名，另开 change。
2. **`include/ptxir/ptxir_serialization.h` 的 IR 引用已实证**：当前通过全局 `struct StatementContext` 前置声明使用；Phase 1.5c+d 必须迁移为 `ptxemu::ir::StatementContext`，并按 task 1.6 增加 canonical include 或合法 namespace forward declaration。
3. **ANTLR4-generated 头无需迁移。** 已核对 generated headers 不包含实际 IR 类型引用；不得修改 `build/antlr4_generated_src/`，若未来生成代码行为变化则另开 change。
4. **`include/ptxemu/ir/statement.h` 头里类方法实现位置？** 当前 header 不实现类方法（仅声明），如发现 1.5d 阶段某些 `std::get<I>` 调用期望 out-of-line 定义，需检查 statement.h 实际内容。
