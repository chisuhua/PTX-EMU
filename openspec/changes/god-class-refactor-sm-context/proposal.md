# god-class-refactor-sm-context — Proposal

## Why

`src/ptxsim/core/sm_context.cpp` 实测 965 行（roadmap 703 行已过期，post-phase3-debt-roadmap.md:53）。增长主因 ADR-0020 cpptlm 注入（commits `367fd6a5`/`a53508c2`/`5831623c`），增长 ~260 行。

**Oracle Round-3 实证发现**：
- §1 跨模块状态翻译实证站点：`sm_context.cpp:379` `w->update_active_mask()`（注释 :374 明示 "only updated by update_active_mask(). Without this fix, active_count…"）
- C-2/C-18 边界：sm_context.cpp:455-490 与 :580-623 存在两段**近乎逐行重复的 SIMT reconvergence 编排循环**（约 130 行）
- ADR-0020 注入点代码（step_b_set_blocked_cycles + 3 setter + 3-step 编排）未在原提案 scope

现在的问题：原 C-2 提案缺 §1 行级 diff + Checklist B + 重复循环去重 + WarpContext API 边界声明。本 change 重新定基线，明确三个 Phase 拆分目标。

## What Changes

- **新增** 共享 helper：去重 sm_context.cpp:455-490 与 :580-623 两段 reconvergence 编排循环
- **新增** ADR-0020 注入点代码归属决策（step_b_set_blocked_cycles + 3 setter + 3-step 编排）
- **拆分** sm_context.cpp 为 ≤ 4 个职责单一组件（CTA 调度 / warp 生命周期 / SM barrier 封装 / ADR-0020 注入编排）
- **不可破坏**：step_b no-op byte-identical fallback 契约的 4 分支测试（lessons-learned §14）
- **依赖 C-18**：必须 WarpContext public API 冻结后执行

## Capabilities

### New Capabilities
- `sm-context-dedup-reconvergence`: 共享 reconvergence helper 提取（去重 130 行）
- `sm-context-step-b-preservation`: step_b byte-identical fallback 4 分支测试锁定
- `sm-context-god-class-split`: sm_context 拆分为 ≤ 4 组件
- `sm-context-warp-api-consumer`: WarpContext API 消费方契约（不修改 API）

### Modified Capabilities
（无现有 spec-level 行为变更。CTA 调度/SM barrier 行为不变。）

## Impact

**受影响代码**：
- `src/ptxsim/core/sm_context.cpp`（主文件，965 → < 250 行）
- `src/ptxsim/core/sm_context.h`（如新增 helper 类）
- `src/ptxsim/core/CMakeLists.txt`（添加新源文件）

**§1 强制项**：
- `sm_context.cpp:379` `w->update_active_mask()` 必须行级随迁
- 注释 `sm_context.cpp:374` "only updated by update_active_mask(). Without this fix, active_count…" 完整保留

**重复循环去重**：
- sm_context.cpp:455-490（~35 行）
- sm_context.cpp:580-623（~35 行）
- 提取为 1 个共享 helper（约 -65 行净减少）

**WarpContext API 冻结**（C-18 已建立）：
| 消费方位置 | 调用 | 来源 |
|----------|------|------|
| sm_context.cpp:379 | `update_active_mask()` | C-18 spec |
| sm_context.cpp:461 | `get_simt_stack().depth()` | C-18 spec |
| sm_context.cpp:463 | `get_simt_stack().empty()` | C-18 spec |
| sm_context.cpp:464 | `get_simt_stack().top()` | C-18 spec |
| sm_context.cpp:468 | `check_reconvergence()` | C-18 spec |
| sm_context.cpp:583 | `get_simt_stack().depth()` | C-18 spec |
| sm_context.cpp:585 | `get_simt_stack().empty()` | C-18 spec |
| sm_context.cpp:586 | `get_simt_stack().top()` | C-18 spec |
| sm_context.cpp:590 | `check_reconvergence()` | C-18 spec |

**不变范围**：
- `warp_context.cpp`（C-18 已冻结）
- `BarrierModule` 内部实现
- `exe_once()` 主循环签名
- `CTAContext` 接口

**依赖**：
- **必须 C-18 已落地**（WarpContext API 冻结）
- C-17 split-ptx-visitor-god-class（独立，无依赖）

**Oracle 评审链**：
- Round-1: APPROVE-WITH-CHANGES
- Round-2: NEEDS-MORE-CHANGES
- Round-3: **CONDITIONAL APPROVE**（在 C-18 之后执行）

**关键约束**：
- MUST §1 行级 diff（SKILL.md:48-77，sm_context.cpp:379）
- MUST Checklist B（SKILL.md:474-483）：worktree + 3 Phase commit
- MUST §14 step_b no-op 4 分支测试锁定（SKILL.md:409-455）
- MUST NOT 改 exe_once() 签名、WarpContext public API 签名

**工时**: 10-12h（去重 + Phase 化开销）