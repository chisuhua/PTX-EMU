# sm-context-component-split — Spec

## Purpose

`src/ptxsim/core/sm_context.cpp` 862 lines (verified `wc -l` on 2026-08-03, HEAD `041ae7d1`) mix three cohesive responsibilities — CTA block admission, warp lifecycle, and SM-level barrier glue — beyond the already-shipped Phase 1+2 helpers (`sm_context_reconvergence`, `sm_context_cpptlm_inject`). This spec defines the contract for the next split; decomposing the 226-line `exe_once()` is deferred to a follow-up change (`exe-once-decomposition`).

## ADDED Requirements

### Requirement: sm_context.cpp 主文件行数 ≤ 600

The system MUST reduce `src/ptxsim/core/sm_context.cpp` to ≤ 600 lines by extracting three cohesive helpers.

#### Scenario: 主文件行数验证

- **WHEN** Phase 6 完成后运行 `wc -l src/ptxsim/core/sm_context.cpp`
- **THEN** 命中 ≤ 600 行
- **AND** 净减少 ≥ 262 行（862 → ≤600）

#### Scenario: <250 行目标分阶段处理

- **GIVEN** 原始提案 `<250` 行目标
- **AND** `src/ptxsim/core/AGENTS.md:89` 与 commit `d9172014` 均指出 226 行 `exe_once()` 不应在单次 change 中拆分
- **WHEN** 本 change 完成并 archive
- **THEN** `<250` 行目标作为 multi-change end-state，由 follow-up `exe-once-decomposition` change 实现
- **AND** 本 change 在 tasks.md / proposal.md / spec.md 显式记录这一决策

### Requirement: 三组件 ≤ 5 helper 上限

The system MUST keep the helper-component count ≤ 5 (existing 2 + this change adds ≤ 3).

#### Scenario: helper 组件数量

- **WHEN** 运行 `ls src/ptxsim/core/sm_context_*.{h,cpp}`
- **THEN** 命中 10 文件（5 helpers × .h+.cpp，parent `sm_context.{h,cpp}` 单独计）
- **AND** `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` cap 由 ≤4 上调至 ≤5，记录本 change 引用

#### Scenario: 新增 3 个 helper 组件

- **WHEN** Phase 3+4+5 全部 commit 完成
- **THEN** `src/ptxsim/core/sm_block_dispatch.{h,cpp}` 存在
- **AND** `src/ptxsim/core/sm_warp_lifecycle.{h,cpp}` 存在
- **AND** `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}` 存在 **OR** Phase 5 走 fold-back path（仅在 SM barrier glue < 50 行时允许，并在 tasks.md 4.1 标注原因）

### Requirement: helper-namespace 模式（不引入新类）

The system MUST use the existing helper-namespace pattern (`sm_reconvergence::`, `sm_cpptlm_inject::`), not new classes.

#### Scenario: 命名空间模式

- **WHEN** 阅读新 helper 头文件
- **THEN** 每个新文件声明 namespace `sm_block_dispatch::` / `sm_warp_lifecycle::` / `sm_barrier_wrapper::`
- **AND** `SMContext` friend-declares each namespace

#### Scenario: WarpContext public API 不变

- **WHEN** 拆分后 `cmake --build build`
- **THEN** 无 `WarpContext` public API 签名不匹配错误
- **AND** `tests/integration/.../execute_warp_instruction` 路径零回归

### Requirement: SMContext::exe_once() 签名不变

The system MUST NOT change `SMContext::exe_once()` signature.

#### Scenario: 编译期验证

- **WHEN** 拆分后 `cmake --build build`
- **THEN** 无 `exe_once()` signature mismatch 错误
- **AND** `exe_once()` 主体保留在 `sm_context.cpp` 内（本 change 不分解）

### Requirement: update_active_mask() 站点行级随迁（lessons-learned §1）

The system MUST move the `w->update_active_mask()` call site (current verified location `src/ptxsim/core/sm_context.cpp:362`) and its BUG-001 active_count synchronization comment block (current verified location `src/ptxsim/core/sm_context.cpp:354-359`, key sentence at line 357) together with whichever helper ends up owning the call site.

#### Scenario: 调用点与注释同步迁移

- **WHEN** 任意 Phase commit 改变 `update_active_mask()` 所在 helper 的归属
- **THEN** 该 helper 内的 call site 仍包含完整 `BUG-001 active_count sync fix` 注释
- **AND** `grep -n 'update_active_mask' src/ptxsim/core/sm_*.cpp` 列出所有 call site，**无 orphan**（无 call 无 comment，无 comment 无 call）
- **AND** `grep -n 'BUG-001' src/ptxsim/core/sm_*.cpp` 列出所有 comment block，**无 orphan**

#### Scenario: 行号引用必须基于当前 verified 值

- **WHEN** 文档化任何行号引用
- **THEN** 使用 `362` (call) / `354-359` (comment block) / `357` (key sentence)，不是过时的 `379` / `374` / `:374`

### Requirement: step_b no-op byte-identical fallback 4 分支测试锁定（lessons-learned §14）

The system MUST keep `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` passing unchanged across all Phase commits.

#### Scenario: 4 分支测试不变

- **WHEN** 每个 Phase commit 后运行 `cd build && ctest -R test_step_b_set_blocked_cycles --output-on-failure`
- **THEN** 4 分支断言全绿
- **AND** 字节级 fallback 契约未漂移

### Requirement: 递归锁审计（lessons-learned §2）

The system MUST run a recursive-lock audit before every Phase commit.

#### Scenario: 持锁方法不调用同锁 public 方法

- **WHEN** 任意 Phase 提交前运行 `grep -n 'lock_guard\|unique_lock' src/ptxsim/core/sm_*.cpp`
- **THEN** 无新增 helper 在持锁方法内调用同锁 public 方法
- **AND** Phase commit message 引用审计结果

### Requirement: 3 Phase commit + 独立可 revert（Checklist B）

The system MUST split work into 3 independent Phase commits, each revert-safe.

#### Scenario: 独立可 revert

- **GIVEN** Phase 3/4/5 三次 commit
- **WHEN** 任意 Phase 引入回归
- **THEN** `git revert HEAD` 或 `git revert HEAD~N..HEAD` 可独立回滚
- **AND** 不影响其余 Phase

#### Scenario: Phase 顺序与基线 worktree

- **WHEN** Phase 0 完成
- **THEN** `.worktrees/baseline-god-class-p3` 创建并 `ctest` 全绿（Checklist B 基线对照）
- **AND** `.worktrees/refactor-god-class-p3` 创建在 branch `openspec/god-class-refactor-sm-context-phase3`

### Requirement: 3 个新单元测试文件

The system MUST add direct unit tests for each new helper component.

#### Scenario: test_sm_block_dispatch.cpp

- **WHEN** Phase 3 完成
- **THEN** `tests/unit/sm/test_sm_block_dispatch.cpp` 存在
- **AND** `cd build && ctest -R unit_sm_sm_block_dispatch --output-on-failure` ≥ 3 case PASS

#### Scenario: test_sm_warp_lifecycle.cpp

- **WHEN** Phase 4 完成
- **THEN** `tests/unit/sm/test_sm_warp_lifecycle.cpp` 存在
- **AND** `ctest -R unit_sm_sm_warp_lifecycle --output-on-failure` ≥ 3 case PASS

#### Scenario: test_sm_barrier_wrapper.cpp（仅当 Phase 5 走 go 路径）

- **WHEN** Phase 5 走 go 路径（非 fold-back）
- **THEN** `tests/unit/sm/test_sm_barrier_wrapper.cpp` 存在
- **AND** `ctest -R unit_sm_sm_barrier_wrapper --output-on-failure` ≥ 2 case PASS

### Requirement: 集成测试零回归

The system MUST keep `tests/integration/barrier/*` and `tests/integration/divergence/*` green throughout all Phase commits.

#### Scenario: barrier integration 测试套件

- **WHEN** 任意 Phase commit 后运行 `cd build && ctest -L barrier --output-on-failure`
- **THEN** 全部 PASS，无回归

#### Scenario: divergence integration 测试套件

- **WHEN** 任意 Phase commit 后运行 `cd build && ctest -L divergence --output-on-failure`
- **THEN** 全部 PASS，无回归

### Requirement: exe_once() 分解推迟到 follow-up

The system MUST defer decomposition of the 226-line `SMContext::exe_once()` to a follow-up change named `exe-once-decomposition`.

#### Scenario: out-of-scope 文档化

- **WHEN** 完成 archive
- **THEN** proposal.md / design.md / tasks.md / spec.md 均显式声明 `exe_once()` 分解为 follow-up
- **AND** `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` 状态记录指向 follow-up

### Requirement: OpenSpec strict validation 通过

The system MUST pass `openspec validate god-class-refactor-sm-context-phase3 --strict`.

#### Scenario: 验证阶段

- **WHEN** Phase 6 完成前运行 `openspec validate god-class-refactor-sm-context-phase3 --strict`
- **THEN** exit 0，无 delta parse error
- **AND** spec file `specs/sm-context-component-split/spec.md` 含 ADDED Requirements + Scenarios

## 关联

- `openspec/changes/archive/2026-07-27-god-class-refactor-sm-context/` — Phase 1+2 source (applied)
- `openspec/changes/archive/2026-07-27-refactor-warp-context/` — C-18 WarpContext API freeze
- `improvements/god-class-refactor-sm-context.md` — origin 5-section proposal
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2` — debt entry (cap bumped to ≤5)
- `.opencode/skills/ptx-lessons-learned/SKILL.md` — §1, §2, §14, Checklist B
- `.opencode/skills/ptx-barrier-mechanism/SKILL.md` — barrier semantics baseline
- `.opencode/skills/ptx-instruction-pipeline/SKILL.md` — Warp → SM call chain