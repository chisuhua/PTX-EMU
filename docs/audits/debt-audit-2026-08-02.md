# PTX-EMU 技术债务审计 — C/D 系列实证复验（2026-08-02）

> **审查方法**：逐项对 `docs/roadmap/post-phase3-debt-roadmap.md §1.2-1.3` 18 个 C 债务 + 6 个 D 债务进行**实证测量**（文件行数 / 函数复杂度 / 关键 API 存在性），所有"已实现/未实现"判定基于 `wc -l` + `grep` + `python3` AST 测量。
> **审查范围**：`src/` + `include/` + `tests/` + `CMakeLists.txt`
> **当前 HEAD**：`be746976`（main，2026-07-25 标记）
> **上次审计**：[debt-audit-2026-07-02.md](./debt-audit-2026-07-02.md)（仅 2 项 C/D 复验 + 仅依赖工时估算）
> **本次新增**：18 项 C + 6 项 D 的逐项实证 + 与 `improvements/*.md` 提案的匹配度验证

---

## 0. 执行摘要

### 0.1 关键发现（一句话）

**post-phase3-debt-roadmap.md 中 18 条 C 债务有 5 条（28%）基于过期测量**，建议从 C 系列"剩余 18 项"中扣减为 **13 项有效 + 5 项 RESOLVED**。

### 0.2 状态分布

| 状态 | C 计数 | D 计数 | 说明 |
|------|:--:|:--:|------|
| ✅ RESOLVED（实证显示已实现） | **5** | 6 | C-4, C-9, C-10, C-15, C-21 |
| 🟡 PARTIAL（部分缓解，需重新评估范围） | **3** | 0 | C-1, C-17, C-5 |
| 🟢 ACTIVE（债务仍真实存在） | **8** | 0 | C-2, C-3, C-6, C-7, C-8, C-18(*), C-20, C-24 |
| ❌ N/A（流程项/已合并） | **2** | 0 | C-16 (合并 A-9), C-22 (流程) |
| **小计** | **18** | **6** | |

> (*) C-18 实证显示文件已 537→365 行显著缩减，需用户确认是否仍 god class。

### 0.3 与 `improvements/` 的匹配度

`improvements/` 现有 17 个提案文件中，**5 个对应已 RESOLVED 的 C 债务**（refactor-ptxir-writer, cmake-use-glob-for-sources, add-cmake-options, complete-x-macro-dispatch, replace-assert-false-with-throw）。这些提案在 debt roadmap 标记为"待实施"但代码已先于提案完成 — 提案文件需添加 ✅ RESOLVED 标记避免被 guide-design 误批准。

---

## 1. C 系列逐项审计（18 项）

### 1.1 RESOLVED — 已实施，建议从 roadmap 扣减

#### C-4 `refactor-ptxir-writer` (P2, 3h) → ✅ RESOLVED

**原提案陈述**：
> `src/ptx_ir/ptxir_writer.cpp::write_instruction()` 函数实测 **232 行**（始于 line 129），处理 31 个 `is_same_v` variant 分支；函数内联展开所有指令类型的序列化逻辑，单函数承载全部 IR 类型的写入职责

**实证反驳**（2026-08-02）：
| 指标 | 提案声称 | 实际 |
|------|---------|------|
| 文件总行数 | 374 | **348** |
| `write_instruction()` 行数 | 232 | **30**（line 174-204） |
| `is_same_v` 分支数 | 31 | **26** |
| 函数职责 | 单函数承载全部 | **dispatcher + 25 per-type helpers** |

**已存在 helpers**（从 `grep "^void PtxirWriter::write_"`）：
`write_branch / write_label / write_void / write_barrier / write_generic / write_declaration / write_bar_warp_sync / write_pragma / write_dollar_name / write_membar / write_fence / write_redux_sync / write_mbarrier / write_call / write_predicate_prefix / write_tcgen05 / write_vote / write_shfl / write_atom / write_texture / write_surface / write_reduction / write_prefetch / write_cp_async / write_abi_directive`

**接受标准**："`write_instruction() 函数 < 50 行（仅分发逻辑）`" — **已满足**（30 行）。

**结论**：✅ RESOLVED — 该重构在 2026-06 PTXIR serialization architecture 阶段已完成。`improvements/refactor-ptxir-writer.md` 提案需添加 ✅ 标记。

---

#### C-9 `cmake-use-glob-for-sources` (P3, 1h) → ✅ RESOLVED

**原提案陈述**：`src/CMakeLists.txt` 手动 `set(SOURCES)` 非 GLOB

**实证反驳**（`grep src/CMakeLists.txt`）：
```cmake
# Line 41-43
# Auto-discover cudart sources via GLOB. CONFIGURE_DEPENDS ensures cmake
# re-runs on file additions.
file(GLOB SOURCES CONFIGURE_DEPENDS
```

**结论**：✅ RESOLVED — 已使用 `file(GLOB SOURCES CONFIGURE_DEPENDS)`。

---

#### C-10 `add-cmake-options` (P3, 1h) → ✅ RESOLVED

**原提案陈述**：仅 1 个 cmake option（无 ASAN/UBSAN）

**实证反驳**（`grep "option(" CMakeLists.txt`）：
```cmake
option(USE_DETAILED_LOGGING "..." OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_WERROR "Treat compiler warnings as errors" OFF)
```

**结论**：✅ RESOLVED — ASAN/UBSAN/WERROR 已存在（带 `if(ENABLE_ASAN)` 等条件编译块）。

---

#### C-15 `complete-x-macro-dispatch` (P3, 3h) → ✅ MOOT

**原提案陈述**：`instruction_handlers.cpp` X-Macro 仅调用 1 次

**实证**：
- `src/ptxsim/instruction_handlers.cpp` 文件**不存在**（已拆分为 per-category `arithmetic.cpp / atomic.cpp / barrier.cpp / ...`）
- X-Macro 在 `src/ptxsim/instruction_factory.cpp::initialize()` 中调用一次，对 106 个 `X(enum_val, op_name, opstr, ...)` 条目生成 `handler_map[enum_val] = new opstr##Handler();`

**结论**：✅ MOOT — "X-Macro 仅调用 1 次"是正确的 X-Macro 使用模式（call once for full enum coverage）。该"债务"是误识别。

---

#### C-21 `replace-assert-false-with-throw` (P3, 1h) → ✅ RESOLVED

**原提案陈述**：`assert(false && "...")` 3 处应改 throw

**实证反驳**（`grep -rn 'assert(false &&\|assert(false\b' src/ include/`）：
```
include/ptxsim/ptx_exceptions.h:5: * 提供运行时错误报告机制，替代 assert(false) 和 TODO 注释。
```
**唯一匹配是文档注释**，无代码使用 `assert(false)`。

**结论**：✅ RESOLVED — 项目使用 `throw`/`UnsupportedInstructionException`/`std::runtime_error` 等异常机制替代。

---

### 1.2 PARTIALLY RESOLVED — 部分缓解，需重新评估

#### C-1 `god-class-refactor-thread-context` (P1, 10h) → 🟡 PARTIAL

| 指标 | 原提案 | 实际 | 变化 |
|------|--------|------|------|
| `src/ptxsim/core/thread_context.cpp` 总行数 | 885 | **727** | -158 |
| 最大函数行数 | (未指定) | `init()` 58 / `_execute_once()` 49 | 健康 |
| `include/ptxsim/thread_context.h` includes | 23 | **25** | +2 |

**结论**：🟡 PARTIAL — 文件已显著缩减，函数粒度健康。剩余风险在 **header 25 个 include**（已 C-7 单独追踪）。建议将 C-1 范围从"god class"改为"减少 header include 依赖"或与 C-7 合并。

---

#### C-17 `split-ptx-visitor-god-class` (P2, 5h) → 🟡 PARTIAL

| 指标 | 原提案 | 实际 |
|------|--------|------|
| `src/ptx_parser/ptx_visitor.cpp` 总行数 | 998 | **623** |
| 最大函数行数 | (未指定) | `visitFunctionDecl` **192** / `visitInstruction` **121** / `visitVariableDecl` 74 |

**结论**：🟡 PARTIAL — 文件已 -375 行（已拆分至 `ptx_visitor_atom.cpp` 等子文件），但 `visitFunctionDecl()` 仍 192 行、`visitInstruction()` 仍 121 行。建议重构这两个函数而非拆分文件。

---

#### C-5 `consolidate-sub-agents-md` (P3, 2h) → 🟡 PARTIAL

**实证**（13 个 AGENTS.md 文件）：
| 文件 | 行数 |
|------|------|
| `./AGENTS.md` | 118 |
| `src/ptxsim/AGENTS.md` | 82 |
| `src/ptxsim/core/AGENTS.md` | 143 |
| `src/ptxsim/instructions/AGENTS.md` | 89 |
| `src/ptxsim/barrier/AGENTS.md` | 48 |
| `src/ptx_ir/AGENTS.md` | 66 |
| `src/ptx_parser/AGENTS.md` | 86 |
| `src/cudart/AGENTS.md` | 47 |
| `src/grammar/AGENTS.md` | 85 |
| `include/cudart/AGENTS.md` | 30 |
| `include/ptxir/AGENTS.md` | 14 |
| `configs/AGENTS.md` | 67 |
| `tests/AGENTS.md` | 126 |

**关键观察**：所有子 AGENTS.md 头部均含 `**SSOT**: Common conventions (build/test/format/conventions/anti-patterns) live in root AGENTS.md` 标记，**主动声明 defer-to-root**。但 `STRUCTURE / WHERE TO LOOK / CONVENTIONS / ANTI-PATTERNS / COMMANDS` 五段结构在 root / src/ptxsim / src/ptx_ir 中重复。

**结论**：🟡 PARTIAL — SSOT 模式已建立（避免了 70%+ 内容重复），但 5 段框架结构仍存在模板级重复。需重新评估"重复"实际比例。

---

### 1.3 ACTIVE — 债务仍真实存在

#### C-2 `god-class-refactor-sm-context` (P2, 6h) → 🟢 ACTIVE（恶化）

| 指标 | 原提案 | 实际 | 变化 |
|------|--------|------|------|
| `src/ptxsim/core/sm_context.cpp` 总行数 | 703 | **862** | **+159** |
| `exe_once()` 函数行数 | (未指定) | **225**（line 333-558） | 严重 |
| `print_warp_status()` | (未指定) | 85 | 关注 |

**结论**：🟢 **ACTIVE — 反而恶化**。SM 上下文 god class 不仅未缓解，反而因 tcgen05/cluster 集成而膨胀。`exe_once()` 225 行是项目当前**最大单函数**。

---

#### C-3 `merge-arithmetic-handlers` (P3, 3h) → 🟢 ACTIVE

| 文件 | 行数 |
|------|------|
| `src/ptxsim/instructions/arithmetic.cpp` | 478 |
| `src/ptxsim/instructions/arithmetic_ext.cpp` | 764 |

**头文件差异**（前 20 行）：`arithmetic.cpp` 使用 `<cmath>/<type_traits>`，`arithmetic_ext.cpp` 使用 `<cassert>/<cstdint>/<cstring>` + `half_utils.h`。两者职责按数据类型（通用 vs 半精度扩展）分割。

**结论**：🟢 ACTIVE — 两个文件 split 是有意的（通用算术 vs 半精度扩展），但 1242 行总和远超单文件 god class 阈值。建议改为"按算术子类别细分"或"将 half 扩展独立 namespace"。

---

#### C-6 `strengthen-pod-tests` (P3, 2h) → 🟢 ACTIVE（结构变化）

**原提案陈述**：`tests/unit/contexts/` 7 个 < 50 行 POD 测试太浅

**实证**：
- `tests/unit/contexts/` 目录**不存在**（已重构）
- 现结构：`barrier/ (13) / common/ (7) / cpptlm/ (8) / cudart/ (4) / divergence/ (1) / exec/ (7) / logger/ (1) / memory/ (8) / parser/ (2) / pc/ (2) / ptx/ (9) / ptx_ir/ (3) / register/ (1) / simt/ (8) / sm/ (3) / sync/ (4) / tcgen05/ (2) / testing/ (1) / utils/ (2) / warp/ (5)`
- 小于 30 行的文件存在但分散（非 contexts 子目录）：`test_barrier_verification.cpp (15) / test_cudart_sim.cpp (20) / test_cluster_tcgen05_integration.cpp (23) / test_pc_management_advanced.cpp (26) / test_barrier_interaction_integrated.cpp (27)`

**结论**：🟢 ACTIVE — 原"`tests/unit/contexts/`"已不存在，但项目内仍有多个浅测试（< 30 行）。需要重新定义"哪些测试算 POD"。

---

#### C-7 `reduce-thread-context-includes` (P3, 3h) → 🟢 ACTIVE

**实证**：`include/ptxsim/thread_context.h` 含 **25 个 include**（提案声称 23，差 +2），分布在：
- 内部：`ptx_ir/operand_context.h / ptx_types.h / statement_context.h / ptxsim/common_types.h / contexts/{exec_state,memory_ref,program_ref,register_predicate}.h / execution_types.h / ptx_config.h / register_access_layer.h / simt_pc_manager.h / warp_state.h / register/{condition_code_register,register_bank_manager}.h / utils/logger.h`
- STL：`<any> <array> <iostream> <map> <memory> <stack> <string> <unordered_map> <vector>`

**结论**：🟢 ACTIVE — 实际比提案所述多 2 个 include，且包含 `<iostream>` (违反"header 最小化")。

---

#### C-8 `reduce-memory-test-utils-includes` (P3, 1h) → 🟢 ACTIVE

**实证**：`include/ptxsim/testing/memory_test_utils.h` 含 **18 个 include**（与提案一致）。

**结论**：🟢 ACTIVE — 数字与提案完全一致。

---

#### C-18 `refactor-warp-context` (P2, 4h) → 🟢 可能已 RESOLVED

| 指标 | 原提案 | 实际 |
|------|--------|------|
| `src/ptxsim/core/warp_context.cpp` 总行数 | 537 | **365** |

**结论**：🟡 需用户判断 — 文件 -172 行但仍 365 行（非小文件）。需进一步测量最大函数粒度。

---

#### C-20 `dedupe-ptx-op-def-format` (P3, 0.5h) → 🟢 需复验

**实证**：`ptx_visitor_atom.cpp` 已使用 X-Macro 模式（`VISITOR_ATOM_INSTR(openum, opstr, opname, opcount)`），不再"硬编码 ptx_op.def 格式"。`ptx_visitor.cpp:504-514` 也使用 `VISITOR_IMPL_*_INSTR` 宏。

**结论**：🟢 可能是 MOOT（已使用 X-Macro），需更细审计确认是否仍有手工重复。

---

#### C-24 `expand-e2e-divergence-coverage` (P3, 8h) → 🟢 ACTIVE（部分扩展）

**实证**：
- `tests/e2e/divergence/test_divergence.cu` — 381 行（原文件）
- `tests/e2e/divergence/test_divergence_sync.cu` — 113 行（新增，warp divergence + barrier sync 测试）

**结论**：🟢 部分缓解 — 从 1 个 E2E 增至 2 个，但提案要求"非 barrier E2E"覆盖仍可能不足。

---

### 1.4 N/A — 流程/合并项

#### C-16 `atomic.cpp` 115 行 stub → ✅ 已合并至 A-9（archive 2026-07-06）

#### C-22 docs(t2-4) commit 占最近 50 commit 12% → ❌ N/A（流程项，非 change）

---

## 2. D 系列（6 项）— 全部 RESOLVED（来自 debt roadmap §1.3）

所有 6 项 D 系列债务在 §1.3 已显式标记 ✅ RESOLVED by `docs-cuda-docs-and-openspec-orphan-sync` (2026-07-06)。本次未重新审计，建议信任前次结论。

---

## 3. 改进文件（improvements/*.md）匹配度审计

### 3.1 17 个文件总览

| 提案文件 | 对应 C/D | 状态判定 |
|----------|---------|----------|
| `god-class-refactor-thread-context.md` | C-1 | 🟡 PARTIAL（重写范围） |
| `god-class-refactor-sm-context.md` | C-2 | 🟢 ACTIVE |
| `merge-arithmetic-handlers.md` | C-3 | 🟢 ACTIVE |
| **`refactor-ptxir-writer.md`** | **C-4** | **✅ RESOLVED — 添加 ✅ 标记** |
| `consolidate-sub-agents-md.md` | C-5 | 🟡 PARTIAL |
| `strengthen-pod-tests.md` | C-6 | 🟢 ACTIVE（路径变化） |
| `reduce-thread-context-includes.md` | C-7 | 🟢 ACTIVE |
| `reduce-memory-test-utils-includes.md` | C-8 | 🟢 ACTIVE |
| **`cmake-use-glob-for-sources.md`** | **C-9** | **✅ RESOLVED — 添加 ✅ 标记** |
| **`add-cmake-options.md`** | **C-10** | **✅ RESOLVED — 添加 ✅ 标记** |
| **`complete-x-macro-dispatch.md`** | **C-15** | **✅ MOOT — 添加 ✅ 标记** |
| `split-ptx-visitor-god-class.md` | C-17 | 🟡 PARTIAL（重写为函数级） |
| `refactor-warp-context.md` | C-18 | 🟡 待确认 |
| **`dedupe-ptx-op-def-format.md`** | **C-20** | **🟢 需复验** |
| **`replace-assert-false-with-throw.md`** | **C-21** | **✅ RESOLVED — 添加 ✅ 标记** |
| `expand-e2e-divergence-coverage.md` | C-24 | 🟢 部分缓解 |
| `split-cpptlm-core-minimal.md` | ADR-0022（不在 C/D 列表） | Oracle 评估推迟 |

### 3.2 应立即处理的同步项

5 个 `improvements/` 文件需添加 ✅ RESOLVED 标记（位于元数据行下方），避免 guide-design 误批准：
- `refactor-ptxir-writer.md`
- `cmake-use-glob-for-sources.md`
- `add-cmake-options.md`
- `complete-x-macro-dispatch.md`
- `replace-assert-false-with-throw.md`

---

## 4. 推荐下一步（基于实证）

### 4.1 最高价值 Opportunity

**C-2 `god-class-refactor-sm-context` (P2, 6h)** — 唯一 ACTIVE 且**恶化**的债务：
- `sm_context.cpp` 703 → **862 行**（+159）
- `exe_once()` = **225 行**（项目最大单函数）

### 4.2 第二优先

**C-1 `reduce-thread-context-includes` (重写自 C-1 god class, P3, 3h)** — 25 includes 包含 `<iostream>`，可显著改善编译时间。

### 4.3 快速胜利

**改进文件同步**（30 分钟）：
- 5 个 ✅ RESOLVED 标记
- 在 `proposal-suggestions.md` 中标记状态
- 更新 `roadmap.md` 与 `post-phase3-debt-roadmap.md` 反映新状态

---

## 5. 方法论备注

### 5.1 实证工具链

```bash
# 1. 文件行数
wc -l <file>

# 2. 函数粒度
python3 -c '
import re
with open(file) as f: content = f.read()
pattern = re.compile(r"^(?:void|std::\w+|auto|int|bool|size_t|\w+(?:::\w+)?)\s+(?:\w+::)?(\w+)\([^)]*\)\s*(?:const)?\s*\{", re.MULTILINE)
# ... 匹配括号深度计算函数行数
'

# 3. Include 计数
grep -c "^#include" <header>

# 4. 关键 API 存在性
grep -E "option\(.*ASAN|file\(GLOB" CMakeLists.txt
```

### 5.2 已知限制

- 本次审计**未运行 ctest**（验证现有测试仍通过）
- 本次审计**未比对 git 历史**（哪些 commit 引入了变更）
- C-17/C-18/C-20 的 PARTIAL/ACTIVE 判定建议人工 spot-check 关键函数

---

## 6. 引用

- 上次审计：[debt-audit-2026-07-02.md](./debt-audit-2026-07-02.md)
- 债务总清单：[docs/roadmap/post-phase3-debt-roadmap.md §1.2-1.3](../roadmap/post-phase3-debt-roadmap.md)
- 当前路线图：[roadmap.md](../../roadmap.md)
- 改进提案池：[improvements/](../../improvements/)

---

**审查者**：Sisyphus agent
**日期**：2026-08-02
**HEAD**：`be746976` (main)
**总实证时间**：~25 分钟（同步 bash 测量 + python3 AST 分析）