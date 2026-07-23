# Dead Code Cleanup — PC API

## Why

PTX-EMU 在 commit `12390b7`（barrier 重构）后遗留了 3 个生产零调用者的
deprecated PC API + 1 个废弃私有字段，触发大量编译警告、误导新开发者、
模糊设计意图。

**核心触发点**：`docs/audits/debt-audit-2026-07-02.md` P0-D2
（`thread_context.h` 与 `src/ptxsim/core/AGENTS.md` 对 `set_pc` /
`force_set_pc` 的推荐方向**互相矛盾**）。本 change 的真实目标：

1. **消除** `[[deprecated]]` 调用产生的编译警告；
2. **统一** PC 管理语义（`commit_pc` / `set_pc` / `set_next_pc` 为唯一入口）；
3. **同步** ADR-0003 / ADR-0008 / 两处 AGENTS.md / barrier.cpp 头注释。

"删除 dead code" 是**手段**，不是目的。

**当前状态**（已验证，HEAD `5f8e5fe`）：

| 死代码 | 位置 | 验证 |
|-------|------|------|
| `WarpContext::get_pc()` | `include/ptxsim/warp_context.h:59` | `grep -r "warp.*get_pc(" src/ include/` → 0 hits |
| `WarpContext::set_pc()` | `include/ptxsim/warp_context.h:65` | `grep -r "warp.*set_pc(" src/ include/` → 0 hits |
| 私有字段 `WarpContext::pc` | `include/ptxsim/warp_context.h:247` | 仅被 `get_pc/set_pc` + 构造 + `reset` 引用 |
| `ThreadContext::force_set_pc()` | `include/ptxsim/thread_context.h:217` | `grep -r "force_set_pc" src/ include/` → 0 hits（仅 tests） |

**前置条件**（C1 + C2 已合并到 main）：

```bash
grep -rn "get_wbar\|wbar\.h\|warp_state\.wbars\|current_wbar_id" include/ src/ tests/
# 预期：空（C2 已删除）
```

如未通过则 STOP，本 change 不可在 C2 完成前实施。

## What Changes

物理删除以下 4 个废弃符号 + 1 个废弃字段，并重写 1 个测试文件中 4 处
引用，同步 4 个文档：

- **代码删除**：
  - `WarpContext::get_pc()` 声明 + 实现
  - `WarpContext::set_pc()` 声明 + 实现
  - `WarpContext::pc` 私有字段 + 构造函数初始化列表 + `reset()` 引用
  - `ThreadContext::force_set_pc()` 声明 + 实现
- **测试重写**：`tests/unit/pc/test_pc_management.cpp` 4 处引用（按行决策，
  见 design.md Decision 1）
- **文档同步**：`docs/adr/ADR-0003-commit-pc-pattern.md`、
  `docs/adr/ADR-0008-barrier-semantics.md`、`src/ptxsim/core/AGENTS.md`、
  `src/ptxsim/instructions/AGENTS.md`、`src/ptxsim/instructions/barrier.cpp`

## Non-Goals

- **不删除** `WarpContext::set_thread_pc()`：仍被 29 处测试调用（详见
  `test_sync_mechanism.cpp` / `test_simt_integration.cpp` /
  `test_exec_mask.cpp` 等），删除会引发大范围重写，**超出本 change 范围**。
  应单独建 change 评估。
- **不修改** `ThreadContext::set_pc()` / `commit_pc()` / `get_pc()` /
  `get_next_pc()` 的实现语义。
- **不修改** `WarpContext::advance_thread_pc()` 实现。
- **不修改** barrier / branch / dispatch 等 PC 设置逻辑。
- **不触碰** `WarpState::threads[i].pc`（这是权威源，不是 dead code）。
- **不删除** `Wbar` 相关任何内容（C2 已完成）。
- **不实施** `set_thread_pc` → `advance_thread_pc` 的迁移（单独 change）。

## Goals

1. 删除 4 个 deprecated 符号的物理代码（保留 `// Removed 2026-07-XX` 占位注释）
2. 同步 4 处文档以解决 P0-D2 矛盾
3. 编译警告数 -4（每删除一个 deprecated 标记）
4. `force_set_pc → set_pc` 的语义差异**逐行**评估（不是全局替换）
5. 现有 ctest 全 PASS，无新增 FAIL

## Risks

- `force_set_pc → set_pc` 不是语义等价替换：前者只写 `pc`，
  后者同时写 `pc` + `next_pc`。直接字符串替换会让"preserves next_pc"
  用例失败。
- 增量 build 可能隐藏残留引用（缓存未失效），必须 clean build 后再做"0 引用"
  断言。
- ADR-0003 / ADR-0008 仍推荐 `force_set_pc` 作为 init/sync 入口，文档同步是
  **阻断项**，不解决会留下新的 P0-D2 残留。
- `WarpContext::pc` 字段删除需同步清理构造函数初始化列表
  (`warp_context.cpp:212`) 和 `reset()` (`warp_context.cpp:461`)，否则会
  出现未初始化字段或 `[[deprecated]]` 状态污染。

## 审计依据

- `.opencode/notes/debt-audit-2026-07-02.md` §1.2 (P0-D2) + §3.1.1 (dead code)
- `.opencode/notes/cleanup-barrier-review.md` (C1 相关风险模式参考)
- `docs/dev-process/lessons-learned.md` (3-Commit 拆分 + commit message 规范)
