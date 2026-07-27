# refactor-warp-context — Proposal

## Why

`src/ptxsim/core/warp_context.cpp` 实测 558 行（roadmap 537 行已过期），最近 30 commits 触碰 6 次（高 churn）。职责混合：execute_warp_instruction 分发、active mask 管理、BarrierModule 同步、SIMT 编排。simt_stack 数据结构已抽离至独立模块（`src/ptxsim/core/simt_stack.cpp`），残留 :64-143 为编排逻辑。

**Oracle Round-3 实证发现**：
- 4 处 `thread->sync_to_warp_state()` 调用（warp_context.cpp:337/:345/:370/:375）是 lessons-learned §1 跨模块状态翻译的实例，迁移时漏任一处即调度器死循环
- 消费方 `sm_context.cpp:379/:461/:468/:583/:590` 依赖 WarpContext public API（`update_active_mask`/`check_reconvergence`/`get_simt_stack`）

现在的问题：原 C-18 提案缺 §1 行级 diff + Checklist B + API 边界声明。本 change 重新定基线，明确三个提取目标 + API 冻结约束。

## What Changes

- **新增** `warp_context_dispatch.{h,cpp}`：提取指令分发为策略表
- **新增** `warp_context_active_mask.{h,cpp}`：提取 active mask 操作为 helper（含 set_active_mask overwrite 语义）
- **新增** `warp_context_simt.{h,cpp}`：提取 SIMT 编排逻辑（divergence/reconvergence orchestration）
- **API 冻结**：`WarpContext::update_active_mask` / `check_reconvergence` / `get_simt_stack` / `get_lanes_by_pc` public 签名不变（消费方 sm_context.cpp:379/:461/:468/:583/:590）
- **不可破坏**：4 处 `sync_to_warp_state()` 行级随迁（§1 强制项）

## Capabilities

### New Capabilities
- `warp-context-dispatch-extraction`: 指令分发策略表化
- `warp-context-active-mask-helper`: set_active_mask overwrite 语义锁定
- `warp-context-simt-orchestration`: SIMT 编排独立化（不重抽数据结构）
- `warp-context-api-freeze`: WarpContext public API 签名不变（消费者契约）

### Modified Capabilities
（无现有 spec-level 行为变更。本 change 为纯重构，调度语义不变。）

## Impact

**受影响代码**：
- `src/ptxsim/core/warp_context.cpp`（主文件，558 → < 300 行）
- `src/ptxsim/core/warp_context.h`（如新增组件需导出 helper 接口）
- `src/ptxsim/core/CMakeLists.txt`（添加新源文件）

**API 冻结（消费方契约）**：
| 方法 | 消费方位置 | 用途 |
|------|----------|------|
| `update_active_mask()` | sm_context.cpp:379 | active_count 同步 |
| `check_reconvergence()` | sm_context.cpp:468, :590 | 汇聚检查 |
| `get_simt_stack()` | sm_context.cpp:461, :583 | SIMT stack 访问 |
| `get_lanes_by_pc()` | sm_context.cpp:489（隐式）| 汇聚 lanes 列表 |

**§1 强制项**：
- warp_context.cpp:337 — `thread->sync_to_warp_state()`
- warp_context.cpp:345 — `thread->sync_to_warp_state()`
- warp_context.cpp:370 — `thread->sync_to_warp_state()`
- warp_context.cpp:375 — `thread->sync_to_warp_state()`

**不变范围**：
- `src/ptxsim/core/simt_stack.cpp`（数据结构已抽离）
- `src/ptxsim/core/simt_stack.h`（数据结构）
- BarrierModule API
- ThreadContext 行为

**依赖**：
- **必须在 C-2 之前执行**：本 change 冻结 WarpContext API 后，C-2 才能在稳定接口上做 SM 侧拆分
- C-17 split-ptx-visitor-god-class（独立，无依赖）

**Oracle 评审链**：
- Round-1: APPROVE-WITH-CHANGES（§2 引用错误）
- Round-2: NEEDS-MORE-CHANGES（缺 §1/Checklist B + API 边界）
- Round-3: **APPROVE**

**关键约束**：
- MUST §1 行级 diff（lessons-learned §1，SKILL.md:48-77）：4 站点列入迁移清单
- MUST Checklist B（SKILL.md:474-483）：worktree + 3 Phase commit
- MUST set_active_mask overwrite 语义（失败模式速查表 + AGENTS.md ANTI-PATTERNS）
- MUST NOT 改变 WarpContext public API 签名

**工时**: 6h（§1 迁移清单 + Phase 化开销）