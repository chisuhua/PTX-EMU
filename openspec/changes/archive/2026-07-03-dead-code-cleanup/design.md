# Design — Dead Code Cleanup (PC API)

## Decision 1: `ThreadContext::force_set_pc` 逐行处理

`force_set_pc(int)` 写 `pc`，**不写** `next_pc`；`set_pc(int)` 同时写
`pc` 和 `next_pc`。**不能**做全局字符串替换。

`tests/unit/pc/test_pc_management.cpp` 中 4 处逐行决策：

| 位置 | 当前代码 | 决策 | 理由 |
|------|---------|------|------|
| `test_pc_management.cpp:81-92` | `force_set_pc(20)` 后断言 `next_pc == 15` | **删除整个用例** | "只写 pc" 语义已不存在 |
| `test_pc_management.cpp:164-182` | `force_set_pc(10); set_next_pc(10);` | 改为 `set_pc(10);` | 等价合并 |
| `test_pc_management.cpp:227-248` | `force_set_pc(10); set_next_pc(10);` | 改为 `set_pc(10);` | 等价合并 |
| `test_sync_mechanism.cpp:29-34` | 标题提到 `force_set_pc`（实际测 `set_thread_pc`） | **重命名测试标题** | 误标，需修正 |

## Decision 2: 删除 `WarpContext::pc` 字段的完整清理

`pc` 字段影响 3 个位置，必须同步删除：

1. `include/ptxsim/warp_context.h:247` — 字段声明
2. `src/ptxsim/core/warp_context.cpp:212` — 构造函数初始化列表 `pc(0)`
3. `src/ptxsim/core/warp_context.cpp:461` — `reset()` 中 `pc = 0;`

**反例**：仅删字段声明不删初始化列表会导致编译错误（member init list
引用不存在成员）。

## Decision 3: 保留 `WarpContext::set_thread_pc()` 但**不**改动

虽然 29 处测试仍调用 `set_thread_pc`，但本 change **不删除**它。理由：

- 删除需要重写 29 处测试 → scope creep
- `set_thread_pc` 委托给 `advance_thread_pc`，运行时无副作用
- 应单独建 change 评估（提议：`deprecate-set-thread-pc`）

**占位注释**：保留 `[[deprecated]]` 标记和现有 deprecation 消息。

## Decision 4: 文档同步是阻断项

按 P0-D2 修正顺序：

1. **ADR-0003** (`docs/adr/ADR-0003-commit-pc-pattern.md`)：
   在 "PC 写入入口" 章节追加 "Removed 2026-07-XX: `force_set_pc`" 段落，
   明确由 `set_pc()` 取代。
2. **ADR-0008** (`docs/adr/ADR-0008-barrier-semantics.md`)：
   在代码示例旁加 `// 历史实现，已移除` 注释（不删示例，保留历史）。
3. **`src/ptxsim/core/AGENTS.md`** + **`src/ptxsim/instructions/AGENTS.md`**：
   "DO NOT use `set_pc()` — use `commit_pc()` or `force_set_pc()`"
   → "DO NOT use `force_set_pc()` — use `set_pc()` for init/sync/reset,
   `commit_pc()` for normal advancement"。
4. **`barrier.cpp` 头注释**（P1-5）：
   移除 "Wbar 数据结构" 描述，改为 "通过 `BarrierModule` /
   `WarpBarrier` 实现"。

## Decision 5: 3-Commit 拆分（与 C1 一致）

```
Commit 1: refactor(pc): remove WarpContext dead get_pc/set_pc and pc field
  - 删除 WarpContext::get_pc / set_pc / pc 字段 + 构造 + reset
  - 验证：ctest -L "pc;warp" all pass
  - 独立可 revert

Commit 2: refactor(pc): remove ThreadContext::force_set_pc and rewrite tests
  - 删除 ThreadContext::force_set_pc 声明 + 实现
  - 重写 test_pc_management.cpp 4 处 + test_sync_mechanism.cpp 标题
  - 验证：ctest -R unit_pc_management -V pass
  - 独立可 revert（仅 force_set_pc + 测试）

Commit 3: docs(pc): sync ADR/AGENTS after removing force_set_pc
  - ADR-0003、ADR-0008、core/AGENTS.md、instructions/AGENTS.md、barrier.cpp
  - 验证：grep -r "force_set_pc" docs/ src/ 仅在 .h 注释占位
  - 独立可 revert（仅文档）
```

## Decision 6: 验证必须 clean build

不可信增量构建的"0 引用"断言。Phase 5 验证要求：

```bash
rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build -j$(nproc) 2>&1 | tee /tmp/build.log
```

然后断言 `grep -iE "deprecated|warning" /tmp/build.log` 在排除
`bench/` 和 `antlr4_generated_src/` 后**为空**。

## 失败模式预防

按 `ptx-lessons-learned` §1（行级 diff）和 §3（分 Phase commit）：

- **每个 commit 前**：在 worktree 中跑 `./scripts/sanity.sh --quick > baseline.txt`
- **每个 commit 后**：在 main worktree 跑全量测试
- **任何已有测试回归**：立即 revert 该 commit，**不**混入下一个 commit
