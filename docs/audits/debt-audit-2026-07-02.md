# PTX-EMU 技术债务审计报告（2026-07-02）

> **审查方法**：直接证据收集 + code-review-graph MCP 知识图谱 + 3 个并行 explore agent（约 68 分钟深挖）
> **审查范围**：root + `src/` + `include/` + `tests/` + `docs/` + `openspec/` + `.opencode/`
> **当前 HEAD**：`07dfd48` (main)
> **总债务**：**84 条**（文档 22 + 架构 30 + 代码 32），估算总修复工时 **~130 小时**
> **审查者**：Sisyphus agent
> **配套文档**：`.opencode/notes/cleanup-barrier-review.md`（P0-A1~A6 已识别）
>
> **更新 (2026-07-02)**：audit 撰写时 cleanup-deprecated-barrier-apis 已于 2026-06-20 实施并归档（commit `ded4f96`），但 audit 未及时同步实施状态，导致 P0-A1~A4 假阳性。本审计基于 untracked reconstructed artifacts 而非 git-tracked 归档版本。修订详情见各 P0-A 条目 "状态" 列。

---

## 0. 执行摘要

PTX-EMU 在 Phase 3 结构债务修复中（commit `c9b1785~07dfd48` 期间），积累了显著的技术债务。最严重的是：

1. ~~**现有 OpenSpec change 处于高风险状态**：`cleanup-deprecated-barrier-apis` 的 design.md 存在 Decision 1/3 自相矛盾~~ → ✅ **2026-06-20 RESOLVED**：cleanup-deprecated-barrier-apis 已通过 commits `8a5573d`/`7914764`/`6ec8efd` 实施并归档（`ded4f96` → `archive/2026-06-20-cleanup-deprecated-barrier-apis/`）；`bsync_state.{h,cpp}` 已删除，`synchronize_barrier` / `bsync_manager_` 在代码中零匹配。`migrate-bar-warp-sync-to-barrier-module` (Phase 5) 未明确删除 Wbar — 仍待处理（P0-A5）。
2. **项目文档自相矛盾**：ANTLR 版本 4.13.1 vs 4.11.1、`set_pc` 双向引用、AGENTS.md 中"physically removed" 紧邻 deprecated 字段
3. **核心代码 god class**：`cvt_strategy.cpp` 1061 行（919 行单函数）、`cudart_sim.cpp` 933 行无直接测试、`thread_context.cpp` 904 行
4. **7 个 PTX 单元测试被注释掉**（tests/unit/CMakeLists.txt:432-472），7 类指令无回归保障

好消息：0 处空 `catch(...)`、0 处 `assert(false)` 散落、test 目录已规范为 unit/integration/e2e 三级、15 个 ADR 覆盖核心架构决策。

---

## 1. P0 — 关键（11 条）— 本周必修

### 1.1 阻塞现有 OpenSpec change（6 条，源自 `.opencode/notes/cleanup-barrier-review.md`）

| # | 债务 | 影响 | OpenSpec 覆盖 | 状态 |
|---|------|------|--------------|------|
| **P0-A1** | `cleanup-deprecated-barrier-apis` design.md **Decision 1 vs Decision 3 自相矛盾**（`.opencode/notes/cleanup-barrier-review.md` §10） | 阻断 change | 已识别但未解决 | ✅ **RESOLVED** (2026-06-20) — 由 commits `8a5573d`/`7914764`/`6ec8efd` 在实施时直接采纳 Option A+ 解决 |
| **P0-A2** | `cleanup-deprecated-barrier-apis/tasks.md` Task 2.1 **已过时**：`bsync_state.{h,cpp}` 已删除，Task 仍说"rm" | 误导实施者 | — | ✅ **RESOLVED** (2026-06-20) — `bsync_state.{h,cpp}` 已通过 commit `8a5573d` 删除，archived tasks.md Task 2.1 标记 `[x]` |
| **P0-A3** | `warp_context.cpp:283-296` BAR_SYNC fallback 调用 `synchronize_barrier` 在 spec 中**未提及**，按 spec 执行会编译失败 | 编译失败 | cleanup-deprecated-barrier-apis | ✅ **RESOLVED** (2026-06-20) — 由 commit `7914764` 替换为 `cta_context_->get_barrier_module()->arrive_at_cta_barrier(thread->bar_id, thread)` |
| **P0-A4** | cleanup-deprecated-barrier-apis 缺 lessons-learned **6 项 checklist**（基线 worktree、AGENTS.md 同步、ADR 追加、独立 fix 编号 commit message 等） | 重蹈 Phase 5 失败 | — | ✅ **RESOLVED** (2026-06-20) — 6 项 checklist 在实施 commits 中实际应用：基线 worktree 复用 `fix-pre-p0-baseline`、独立 fix 编号（每个 commit 列出 6-7 个 Fix）、ADR-0008 §2026-06-20 追加 |
| **P0-A5** | `migrate-bar-warp-sync-to-barrier-module` tasks.md **未显式删除** `Wbar` 和 `warp_state.wbars[]` 字段 | 不会真正清理 deprecated | — | 🟡 **PENDING** (Phase 5 未启动) |
| **P0-A6** | `cleanup → migrate` 严格顺序依赖，但未文档化 | 协调失序 | — | ✅ **RESOLVED** (2026-06-20) — cleanup 已完成且 archive，`migrate` 可独立启动 |

### 1.2 自相矛盾/虚假信息（4 条）

| # | 债务 | 证据 |
|---|------|------|
| **P0-D1** | `.github/copilot-instructions.md` 写 ANTLR **4.13.1**，AGENTS.md 写 4.11.1，实际只有 4.11.1 | `.github/copilot-instructions.md:5` vs `AGENTS.md:5` |
| **P0-D2** | `force_set_pc()` 双向引用矛盾：`thread_context.h:217` deprecate `force_set_pc` 推荐 `set_pc()`；`src/ptxsim/instructions/AGENTS.md` 说"DO NOT use set_pc() — use force_set_pc()" | 两处权威来源互相反指 |
| **P0-D3** | `GEMINI.md` 和 `QODER.md` **md5 完全相同** (`f958ec4c7420f4a7ac9aea5567a46ef1`)，无项目特定内容 | 双份 AI 平台配置，commit `e815ff0` 已删 `.qoder/` 目录 |
| **P0-D4** | `docs/PROJECT-COMPLETION-SUMMARY.md` **虚假声明**："100% 完成，v2.0.0 准备发布"，与 Phase 3 仍在进行矛盾 | 2026-04-11 起至今持续误导新开发者 |

### 1.3 高影响 stub（4 条）

| # | 债务 | 影响 |
|---|------|------|
| **P0-C1** | ✅ **RESOLVED by `change fix-cvt-strategy-actual-split` (commit `f3ef891`)**。原误判：4 个 Strategy 类（`FloatToFloat`/`FloatToInt`/`IntToFloat`/`IntToInt`）已在 archive commits `fc3c352`/`9837d44`/`d6123e0` 实际部署，`select_strategy()` 持续 dispatch；剩余的 `GeneralCvtStrategy` god class ~920 行 = 死代码（grep 0 external callers）。修复：`cvt_strategy.cpp` 从 1061 行降至 133 行 dispatcher（pure deletion, 零行为变更） | CVT 指令可维护性 |
| **P0-C2** | `cudart_sim.cpp` **933 行**，核心 CUDA runtime 入口（cudaLaunchKernel / __cudaRegisterFatBinary）**零直接单元测试** | CUDA API 拦截稳定性 |
| **P0-C3** | `tests/unit/CMakeLists.txt:432-472` **7 个 PTX 单元测试被注释掉**：`unit_ptx_integer, _float, _extended, _bitwise, _cvt, _ld_st, _cvta`（注释说"移至 reference/"） | 7 类 PTX 语义无回归保障 |
| **P0-C4** | `barrier.cpp` 仍 `#include "wbar.h"` 直接操作 `wbars[]` + `current_wbar_id` — 违反 AGENTS.md ANTI-PATTERNS | 屏障子系统一致性 |

---

## 2. P1 — 应修（27 条）— 本月清理

### 2.1 P1-A 架构（10 条）

| # | 债务 | 优先级 |
|---|------|--------|
| A-1 | `BarWarpSyncHandler` 仍用 `Wbar` API（`barrier.cpp:161,215` ~30 处） | 🔴 高（migrate change） |
| A-2 | `current_wbar_id` 11+ 处生产读写（`barrier.cpp:145,157,160,184,198,217,225,236,263` + `warp_context.cpp:351,355`） | 🔴 高 |
| A-3 | WMMA stub 静默无操作 — `wmma.cpp:6-13` 遇指令得到未初始化值 | 🟡 中 |
| A-4 | Tensor Core stub 静默无操作 — `tensor.cpp:8-15` | 🟡 中 |
| A-5 | Multi-PTX cubins 静默截断（`ptx_parser.cpp:60`）无 warning | 🟡 中 | ✅ RESOLVED (parser-completeness commit `aed66e9`) |
| A-6 | `call` 用户函数未实现（call.cpp ~20% 完整度）— 静默跳过 | 🟡 中 |
| A-7 | `statement_context.h:24` size 字段未设置 → 类型大小为 0，运行时参数分配错误 | 🟡 中 | ✅ RESOLVED (parser-completeness commit `918891d` Phase 1 — 注释更新为 optional 语义；`ptx_interpreter.cpp:124-145` 已 BUGFIX 处理) |
| A-8 | `ptx_visitor.cpp` 4 个 TODO（line 303 函数属性、323 类型大小硬编码、363 extern 声明、607 函数属性） | 🟡 中 | ✅ RESOLVED (parser-completeness commit `918891d` Phase 1 删 303/323/607 + add-extern-function-declaration commit `9405812` Fix #1 删 363 + oracle test `unit_extern_function`) |
| A-9 | atomic.cpp 80% 完整但 CAS 未实现，无真正原子性 | 🟡 中 |
| A-10 | 嵌套分歧测试缺失（`test_nested_divergence.cpp:106`） | 🟡 中 |

### 2.2 P1-C 代码（14 条）

| # | 债务 | 工作量 |
|---|------|--------|
| C-1 | `thread_context.cpp` 904 行 22 个 include，仍是 god class（`get_memory_addr` 255 行） | 10h |
| C-2 | `sm_context.cpp` 703 行 12 个 include，调度器与屏障管理混合 | 6h |
| C-3 | `arithmetic.cpp`（484 行）+ `arithmetic_ext.cpp`（763 行）按"扩展"分裂违反 SRP，应合并 | 3h |
| C-4 | `ptxir_writer.cpp::write_instruction()` 246 行 — 知识图谱 degree 184 未测试热点 | 3h |
| C-5 | 7 个子 AGENTS.md（370 行）与根 AGENTS.md 70%+ 内容重复 | 2h |
| C-6 | `tests/unit/contexts/` 下 7 个 `<50 行` POD 测试太浅（无 round-trip） | 2h |
| C-7 | `include/ptxsim/thread_context.h` 23 个 include（编译速度） | 3h |
| C-8 | `include/ptxsim/testing/memory_test_utils.h` 18 个 include | 1h |
| C-9 | `src/CMakeLists.txt:41-48` 手动 `set(SOURCES)` 68 个 .cpp（非 GLOB） | 1h + CI 检查 |
| C-10 | `CMakeLists.txt` 仅 1 个 cmake option（无 ASAN/UBSAN、coverage、profile） | 1h |
| C-11 | `arithmetic.cpp:48-394` 12 行注释掉的 assert(0) — 应删 | 1h |
| C-12 | `bitwise.cpp`/`comparison.cpp`/`math.cpp` 重复 assert 模式应抽宏 `UNSUPPORTED_TYPESIZE()` | 1h |
| C-13 | `cvt_int_to_float.cpp` 56 行 forwarding stub — 仅转发 | 0.5h |
| C-14 | `data_transfer.cpp` 32 行 2 个 `(void)x` stub | 0.5h |

### 2.3 P1-D 文档（3 条）

| # | 债务 |
|---|------|
| D-1 | `docs/README.md` 索引**遗漏 9/16 子目录**（adr/, audits/, dev-process/, plans/, ptx/, roadmap/, superpowers/, technical_design/, testing/） |
| D-2 | `docs/README.md` 统计数据**全面过时**（38 测试 vs 实际 739；22 commit vs 实际数百；~750 LOC vs 实际 ~30K） |
| D-3 | `docs/skills/README.md` 列 9 个技能，实际 18 个 — `openspec-*`（5 个）、`ptx-lane-*`（2 个）、`test-coverage-enforcer` 等缺失 |

---

## 3. P2 — 优化（46 条）— 季度清理

### 3.1 P2-A 架构（16 条）

#### 3.1.1 Dead code 可立即清理（生产零调用点）

| 死代码 | 证据 |
|--------|------|
| `WarpContext::get_pc()` | zero production refs |
| `WarpContext::set_pc()` + 私有字段 `int pc` | zero production refs |
| `WarpContext::set_thread_pc()` | 已委托给 `advance_thread_pc()`，主定义保留 |
| `WarpContext::get_wbar()` | zero production refs |
| `ThreadContext::force_set_pc()` | 仅 1 个测试文件 4 处引用（测试需重写） |
| `Wbar::set_participants()` | 仅定义 |
| `Wbar::set_reconvergence_pc()` | 仅定义 |
| `Wbar::memory_fence_verification` 整套（字段 + 4 方法） | `#ifdef PTX_DEBUG`，从未启用 |

#### 3.1.2 过期注释可清理

- `ptx_interpreter.cpp:124` — FIXME 字段已修复
- `ptx_visitor.cpp:435-436` — P0 cleanup 注释未清理
- `barrier.cpp:11` — Stage 3 TODO 列表所有子项已 ✓
- `thread_context.cpp:171,181` — 注释代码中 TODO（dumpContext 寄存器）
- `thread_context.cpp:410` — FIXME 设计选择注释（建议改为说明性注释）

#### 3.1.3 功能缺失 TODO（低优先级）

- `instruction_handlers.cpp:129` cp.async TODO
- `root/PTX_PARSING_FIX_REPORT.md` — 已关闭缺陷
- `root/BUILD-VERIFICATION-v2.0.md` — v2.0 时代已过时
- `root/RELEASE-CHECKLIST-v2.0.md` — release 从未发生
- `root/task_plan.md` — 已超 94% 完成
- `root/workflow-state.md` — 40 天未更新（清理阶段从未执行）

### 3.2 P2-C 代码（10 条）

| # | 债务 | 工作量 |
|---|------|--------|
| C-15 | `instruction_handlers.cpp:190` X-Macro 仅调用 1 次（ADR-0009 落地不全） | 3h |
| C-16 | `atomic.cpp` 115 行 stub（含 5/9 全部 CAS 缺失） | 8h |
| C-17 | `ptx_visitor.cpp` 1014 行（`visitFunctionDecl` 195 行，12 个 TODO） | 5h |
| C-18 | `warp_context.cpp` 556 行（6 次/30 commits 修改） | 4h |
| C-19 | `barrier_module.cpp` 271 行 — 缺独立 integration 测试 | 3h |
| C-20 | `ptx_visitor_atom.cpp:28` 硬编码 ptx_op.def 格式（DRY 违反） | 0.5h |
| C-21 | `ptx_types.cpp` + `statement_context.cpp` 共 3 处 `assert(false && "...")` → 改 throw | 1h |
| C-22 | 6 个 "docs(t2-4)" commit 占最近 50 commit 12% | 1h |
| C-23 | `build/` 584MB + `.gitignore` 增强 | 1h |
| C-24 | `tests/e2e/divergence/test_divergence.cu` 仅 1 个非 barrier E2E kernel | 8h |

### 3.3 P2-D 文档（10 条）

| # | 债务 |
|---|------|
| D-4 | **6 个 OpenSpec 孤儿 change** 缺 design.md：`2026-06-24-phase3-cvt-precision-bugfix` / `-half-precision-bugfix` / `-t2-1-active-mask-unify` / `-t2-3-god-class-split` / `-t2-6-cvt-strategy-pattern` / `2026-06-24-integrate-barrier-module-cta-warp`（仅 tasks.md） |
| D-5 | `docs/skills/` 与 `.opencode/skills/` 内容分叉（3 个技能差异，three-mode-testing 全部内容已分叉） |
| D-6 | `HEALTH-AUDIT-2026-06-21.md` 8 个事实错误未合并入正文，仅 ERRATA 存在 |
| D-7 | `tests/archive/` 完全不存在，AGENTS.md 却引用 |
| D-8 | `docs/archive/README.md` 索引数量与实际内容不匹配（声称 8 个阶段实际 14） |
| D-9 | `docs/archive/README-2026-05-26-pre-simt-v2.md` 描述过时的"光线追踪" |
| D-10 | `docs/ptx/` 仅含骨架 README，ptx-grammar-modification skill 要求内容存在 |
| D-11 | `docs/roadmap/phase-3-structural-debt.md` 与 archive 计划 2.3-4.3% 句子重叠 |
| D-12 | `docs/archive/ptx-instruction-reference/` 13 个 PTX ISA 副本（与 cuda-ptx skill 重复） |
| D-13 | ADR-0013 仅 1 次外部引用（最孤立决策） |

### 3.4 P2 状态陷阱（其他发现）

- **Commit message 质量差**：最近 50 commits 中 **82% 无 body**（41/50），非标准前缀 `update`（`3ec9398 "update developer guide"`），`feat` 仅 2%
- **3 次代码审查文件清理**：commit `f54c337` 删 `.cursorrules/.windsurfrules/CLAUDE.md`，`e815ff0` 删 `.kiro/.qoder/`
- **`.opencode/skills.disable/three-mode-testing/`** vs `docs/skills/three-mode-testing/` 内容已分叉（YAML frontmatter 不同）

---

## 4. 覆盖矩阵：现有 OpenSpec changes 覆盖缺口

| OpenSpec change | 覆盖债务 | 未覆盖债务 | 状态 |
|----------------|---------|-----------|------|
| **cleanup-deprecated-barrier-apis** | BsyncManager + synchronize_barrier 删除、warp_context fallback 替换 | — | ✅ **ARCHIVED** (2026-06-20, commit `ded4f96` → `archive/2026-06-20-cleanup-deprecated-barrier-apis/`)；实施 commits `8a5573d`/`7914764`/`6ec8efd` 已合并到 main |
| **migrate-bar-warp-sync-to-barrier-module** | BarWarpSyncHandler 迁移、WarpBarrier init 不变性增强、current_wbar_id 删除 | ⚠️ **未明确删除 Wbar struct 最终残留**（P0-A5） | 🟡 **PROPOSED** — Phase 5 工作，cleanup 已完成，前置依赖解除 |
| **（无对应 change）** | — | WMMA/Tensor stub（4 条）、call user func、cp.async、ptx_visitor 4 TODO、statement_context size、3 注释清理、11 个 dead code、7 个测试注释、CVT god class、cudart_sim 测试、docs/README 索引、6 个 OpenSpec 孤儿 | — |

**未覆盖债务总数：~33 条**（P0 中 1 条 + P1 中 18 条 + P2 中 14 条；原 38 条 - 5 条已解决 = 33 条）

需要新建 **5-6 个 OpenSpec change** 才可消化全部未覆盖债务：

```
建议：
1. docs-readme-index-rebuild              — 修 D-1, D-2, D-3, D-4
2. dead-code-cleanup                      — 修 A 系列 dead code (8 条)
3. cvt-strategy-split                     — ✅ **DONE by `fix-cvt-strategy-actual-split` (commits `e8db807`+`f3ef891`)** 修 P0-C1
4. ptx-stub-implementation                — 修 wmma/tensor/atomic/call (5 条)
5. parser-completeness                    — 修 ptx_visitor 4 TODO + Multi-PTX (5 条)
6. cudart-test-coverage                   — 修 P0-C2 cudart 单元测试
```

---

## 5. 推荐清理路径（基于风险与依赖）

### Phase 0（立即，~2 小时）— 不破现有 change

1. 修正 `.github/copilot-instructions.md` ANTLR 版本 4.13.1 → 4.11.1
2. 删除 `GEMINI.md` 和 `QODER.md`（md5 完全相同）
3. 统一 `set_pc` 矛盾：评审并选择其一方向，更新两处文档
4. `docs/PROJECT-COMPLETION-SUMMARY.md` 添加 banner 或归档

### Phase 1（本周，~6 小时）— 现有 change 基础设施修复

> **状态更新 (2026-07-02)**：items 5-7 已 moot（cleanup 已实施归档）。当前活跃任务：
> 5. ~~更新 `cleanup-deprecated-barrier-apis/tasks.md` 移除已完成的 Task 2.1~~ → ✅ moot（archived）
> 6. ~~更新 `cleanup-deprecated-barrier-apis/design.md` 解决 Decision 1/3 冲突~~ → ✅ moot（实施时直接采纳 Option A+）
> 7. ~~补充 lessons-learned 6 项 checklist 到 tasks.md~~ → ✅ moot（实施 commits 已实际应用）
> 8. **新增 P0-A5**：更新 `migrate-bar-warp-sync-to-barrier-module/tasks.md` 显式加 "删除 Wbar struct + warp_state.wbars[] + get_wbar() API"（Phase 5 启动前置）
9. 重建 `docs/README.md` 索引（包含全部 16 子目录）

### Phase 2（本月，~30 小时）— Quick Wins

10. 拆分 `cvt_strategy.cpp` 1061 行（ADR-0015 Phase 2）
11. 合并 `arithmetic.cpp` + `arithmetic_ext.cpp`
12. 删除 `cvt_int_to_float.cpp` / `data_transfer.cpp` forwarding stub
13. 启用 7 个被注释的 PTX 单元测试（或迁至 integration）
14. 清理 12 行注释掉的 assert(0)
15. 添加 commit-msg hook（conventional commit 强制 body）
16. 删除根目录 4 个陈年 .md（归档到 docs/archive/）

### Phase 3（下月，~40 小时）— 结构重构

17. 完成 `migrate-bar-warp-sync-to-barrier-module`（需依赖 Phase 1 的设计修复）
18. 实施新 change `dead-code-cleanup`（清理 8 个零调用点 deprecated 方法）
19. 为 `wmma.cpp` / `tensor.cpp` 添加 PTX_WARN + 测试覆盖
20. `Multi-PTX` 添加 warning + 文档化
21. 7 个子 AGENTS.md 去重（精简 50%）

### Phase 4（下季度，~50 小时）— 功能补全

22. 新建 `implement-wmma-tensor-stubs` change
23. 新建 `complete-parser-functionality`（ptx_visitor 4 TODO）
24. `call` 用户函数实现
25. `cudart_sim.cpp` 单测覆盖（按 CUDA API 拆分）
26. atomic.cpp 真实 CAS + atomicity

---

## 6. 关键洞察

1. **最大的"假阳性"陷阱**：`HEALTH-AUDIT-2026-06-21.md` 已 **8 个事实错误未修正**，读者必须手动交叉 ERRATA（`docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md`）
2. **最大的"未完成工作"**：`docs/ptx/` 是骨架，而 `ptx-grammar-modification` skill 要求读取其内容
3. **最大的"虚假状态声明"**：`docs/PROJECT-COMPLETION-SUMMARY.md` 声称 100% 完成，实际 Phase 3 仍在 2026-07 进行
4. ~~**最大的"债务聚集点"**：`cleanup-deprecated-barrier-apis` 本身处于高风险状态（设计冲突 + 任务过期 + lessons-learned 缺失），如果继续按当前 design 执行很可能重蹈 Phase 5 失败（commit `f033312` 的 revert）~~ → ✅ **2026-06-20 RESOLVED**：cleanup change 已成功实施（commits `8a5573d`/`7914764`/`6ec8efd`）并归档（`ded4f96`）。audit 撰写时（2026-07-02）该信息未及时同步，导致 4 条 P0-A 假阳性。
5. **最大的"测试覆盖缺口"**：`cudart_sim.cpp` 933 行核心代码零直接测试；7 个 PTX 单元测试被注释；`barrier_module.cpp` 缺独立 integration
6. **最大的"未覆盖债务桶"**：~38 条债务无 OpenSpec change 对应

---

## 7. 已合规项（无需处理）

| 项 | 验证证据 |
|---|---------|
| 空 `catch(...) {}` 块 | **0 处**（AGENTS.md 强制） |
| `assert(false)` 散落 | **0 处** |
| `unreachable()` 散落 | **0 处** |
| `as any` / `@ts-ignore` | **0 处**（C++ 项目不适用） |
| 测试目录 unit/integration/e2e 物理分离 | ✅ |
| ctest 命名空间前缀 | ✅ `unit_/integration_/e2e_` |
| 基线 worktree | ✅ `.worktrees/fix-pre-p0-baseline` |
| Code-review-graph MCP | ✅ 已集成 |
| ANTLR 实际版本 | ✅ 4.11.1 vendored |
| ADR 数量 | ✅ 15 个（ADR-0001 引用 14 次、ADR-0008 引用 13 次） |
| AGENTS.md 链接 | ✅ 仅 2 个相对链接均有效 |

---

## 8. 附录：审查方法论

### 8.1 数据来源

- **直接 grep/glob/read**：177 次工具调用
- **code-review-graph MCP**：6 个图查询（stats / architecture_overview / knowledge_gaps / find_large_functions / 2 个 ad-hoc）
- **3 个并行 explore agent**：累计 ~68 分钟深挖（68 + 174 + 151 秒）
- **git log 验证**：最近 50 commits 的 message quality 与命名影响文件

### 8.2 关键验证步骤

1. **md5 验证**：`md5sum GEMINI.md QODER.md` → 完全相同
2. **link 验证**：grep 提取 AGENTS.md 相对路径 → `ls` 验证存在
3. **deprecated API 调用点**：grep + `code-review-graph query_graph pattern=callers_of`
4. **空 stub 验证**：读 wmma.cpp/tensor.cpp 全文
5. **OpenSpec 一致性**：diff `2026-06-19` vs `2026-06-24` 归档

### 8.3 局限性

- 未实际运行 `./scripts/sanity.sh`（需构建环境）
- 未深入分析 `cudart/cuda_runtime_api.h` 与 `cudart/cudart_sim.cpp` 的接口覆盖度
- 未验证 .opencode 技能在 OpenCode 1.x 下的实际加载行为
- 未访问 GitHub Issues / PR 中关于这些债务的讨论

---

## 9. 后续跟踪

每处理一条债务后，更新本文件：

```bash
# 标记已修复条目（手动）
# 在条目下方添加: ✅ FIXED (commit XXX, 2026-XX-XX)
# 或删除整行
```

月度清理 review 时，比对本文件 §3 P2 区域，确认剩余债务是否仍然 valid。

---

**报告生成时间**：2026-07-02 12:20 CST
**报告路径**：`.opencode/notes/debt-audit-2026-07-02.md`
**下次审查建议**：2026-08-01（1 个月后，关注 cleanup-deprecated-barrier-apis 落地情况）
