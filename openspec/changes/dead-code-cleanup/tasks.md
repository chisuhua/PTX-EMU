# Tasks — Dead Code Cleanup (PC API)

## 1. 审计与基线

- [ ] 1.1 **C2 前置验证**（必须通过）：
      ```bash
      grep -rn "get_wbar\|wbar\.h\|warp_state\.wbars\|current_wbar_id" \
        include/ src/ tests/
      ```
      输出必须为空。否则 STOP。
- [ ] 1.2 **全量审计**（合并为一个命令）：
      ```bash
      grep -rnE "force_set_pc|WarpContext::(get_pc|set_pc)|warp\.(get_pc|set_pc)\s*\(" \
        --include="*.cpp" --include="*.h" src/ include/ tests/ \
        | grep -v "build/" | grep -v "antlr4_generated_src/" \
        | grep -v "bench/" > /tmp/dead_code_audit.txt
      ```
      预期：4 处 `force_set_pc`（全部在 `tests/unit/pc/`）+ 0 处
      `WarpContext::get_pc/set_pc` 实际引用。
- [ ] 1.3 **构造 + reset 引用审计**：
      ```bash
      grep -n "\bpc\b" src/ptxsim/core/warp_context.cpp include/ptxsim/warp_context.h \
        | grep -v "//\|Removed 2026"
      ```
      预期：`pc(0)` 初始化 + `pc = 0;` reset + `return pc;` set_pc 实现。
- [ ] 1.4 **基线 worktree**（C1/C2 已有 `.worktrees/fix-pre-p0-baseline`，如不存在则建）：
      ```bash
      git worktree add .worktrees/baseline-dead-code HEAD 2>/dev/null \
        || git worktree list  # 复用已存在
      ```
- [ ] 1.5 **基线测试快照**：
      ```bash
      .worktrees/baseline-dead-code/scripts/sanity.sh --quick > /tmp/sanity-baseline.txt
      ```

## 2. Commit 1 — 删除 `WarpContext` 死 PC API（Fix #1）

- [x] 2.1 删除 `include/ptxsim/warp_context.h:59-68` 的 `get_pc()` / `set_pc()` 声明 + `[[deprecated]]`
- [x] 2.2 删除 `include/ptxsim/warp_context.h:247` 的 `int pc;` 字段
- [x] 2.3 删除 `src/ptxsim/core/warp_context.cpp:212` 构造函数初始化列表中的 `pc(0)`
- [x] 2.4 删除 `src/ptxsim/core/warp_context.cpp:461` `reset()` 中的 `pc = 0;`
- [x] 2.5 删除 `src/ptxsim/core/warp_context.cpp` 中 `get_pc()` / `set_pc()` 实现
- [x] 2.6 在 `warp_context.h:59` 处添加占位注释：
- [x] 2.7 验证：`grep -rn "warp.*get_pc\|warp.*set_pc\|WarpContext::pc" src/ include/` 仅命中占位注释
- [x] 2.8 `cmake --build build --target ptxsim` 编译通过
- [x] 2.9 `ctest -L "pc;warp"` 全部 PASS（与 baseline 对比无新增 FAIL）
- [x] 2.10 commit：
      ```bash
      git add -A
      git commit -m "refactor(pc): remove WarpContext dead get_pc/set_pc and pc field (Fix #1)

      - Remove WarpContext::get_pc() and WarpContext::set_pc() (zero production refs)
      - Remove WarpContext::pc private field + ctor init list + reset()
      - Replaced by: warp_state.threads[lane_id].pc (authoritative source)
      - Independent revert-safe commit
      "
      ```

## 3. Commit 2 — 删除 `ThreadContext::force_set_pc` + 重写测试（Fix #2 + #3）

- [x] 3.1 删除 `include/ptxsim/thread_context.h:217-218` 的 `force_set_pc()` 声明
- [x] 3.2 删除 `src/ptxsim/core/thread_context.cpp` 中 `force_set_pc()` 实现（按 grep 定位）
- [x] 3.3 在 `thread_context.h:217` 处添加占位注释：
      ```cpp
      // Removed 2026-07-XX — dead-code-cleanup (Fix #2)
      // Replaced by: set_pc() writes both pc and next_pc
      ```
- [x] 3.4 **`test_pc_management.cpp:81-92`**：删除整个 "force_set_pc: sets pc only,
      preserves next_pc" 用例（语义已不存在）
- [x] 3.5 **`test_pc_management.cpp:164-182`**：将 `force_set_pc(10); set_next_pc(10);`
      合并为 `set_pc(10);`
- [x] 3.6 **`test_pc_management.cpp:227-248`**：同上
- [x] 3.7 **`test_sync_mechanism.cpp:29-34`**：重命名测试标题，移除
      `force_set_pc` 误导（实际测 `set_thread_pc`）
- [x] 3.8 验证：`grep -rn "force_set_pc" tests/` 仅命中 `Removed 2026-07` 占位注释
- [x] 3.9 `cmake --build build` 编译通过
- [x] 3.10 `ctest -R unit_pc_management -V` 全部 PASS
- [x] 3.11 `ctest -L "pc;sync"` 全部 PASS
- [x] 3.12 commit：
      ```bash
      git add -A
      git commit -m "refactor(pc): remove ThreadContext::force_set_pc and rewrite tests (Fix #2)

      - Remove ThreadContext::force_set_pc() (zero production refs)
      - Delete test_pc_management.cpp 'preserves next_pc' case (semantic gone)
      - Merge 2x 'force_set_pc + set_next_pc' pairs to set_pc()
      - Rename test_sync_mechanism.cpp title to reflect actual set_thread_pc
      - Independent revert-safe commit
      "
      ```

## 4. Commit 3 — 文档同步解决 P0-D2（Fix #4）

- [ ] 4.1 **`docs/adr/0003-commit-pc-pattern.md`**：在 "PC 写入入口" 章节追加
      "Removed 2026-07-XX: `force_set_pc`" 段落，明确由 `set_pc()` 取代
- [ ] 4.2 **`docs/adr/0008-barrier-semantics.md`**：在引用 `force_set_pc` 的代码示例
      旁加 `// 历史实现，已移除` 注释（不删示例，保留历史）
- [ ] 4.3 **`src/ptxsim/core/AGENTS.md`**：将
      "DO NOT use `set_pc()` — use `commit_pc()` or `force_set_pc()`"
      改为
      "DO NOT use `force_set_pc()` — use `set_pc()` for init/sync/reset,
      `commit_pc()` for normal advancement"
- [ ] 4.4 **`src/ptxsim/instructions/AGENTS.md`**：同上修改
- [ ] 4.5 **`src/ptxsim/instructions/barrier.cpp` 头注释**（P1-5）：移除
      "Wbar 数据结构" 描述，改为 "通过 `BarrierModule` / `WarpBarrier` 实现"
- [ ] 4.6 验证：
      ```bash
      grep -rn "force_set_pc" docs/adr/0003-commit-pc-pattern.md \
        docs/adr/0008-barrier-semantics.md \
        src/ptxsim/core/AGENTS.md \
        src/ptxsim/instructions/AGENTS.md
      ```
      预期：仅命中 "Removed 2026-07" 或 "历史实现"，**无** "use `force_set_pc`"
- [ ] 4.7 验证：更新根 `AGENTS.md` 引用（如有 `force_set_pc` 推荐）
- [ ] 4.8 commit：
      ```bash
      git add -A
      git commit -m "docs(pc): sync ADR/AGENTS after removing force_set_pc (Fix #4)

      - ADR-0003: mark force_set_pc as Removed
      - ADR-0008: annotate historical code samples
      - core/AGENTS.md + instructions/AGENTS.md: invert recommendation direction
      - barrier.cpp: remove Wbar header description
      - Resolves P0-D2 doc contradiction
      - Independent revert-safe commit
      "
      ```

## 5. 全量验证（clean build + 回归）

- [ ] 5.1 **clean build**（P0-10）：
      ```bash
      rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        && cmake --build build -j$(nproc) 2>&1 | tee /tmp/build.log
      ```
- [ ] 5.2 **warning 检查**（P1-4）：
      ```bash
      grep -iE "deprecated|warning" /tmp/build.log \
        | grep -v "bench/" \
        | grep -v "antlr4_generated_src/" \
        | grep -v "Removed 2026-07"
      test ! -s <(grep -iE "deprecated|warning" /tmp/build.log \
        | grep -v "bench/" | grep -v "antlr4_generated_src/" \
        | grep -v "Removed 2026-07")
      ```
      预期：命令退出码 0
- [ ] 5.3 **ctest 完整回归**：
      ```bash
      cd build && ctest --output-on-failure
      ```
      预期：与 baseline 对比无新增 FAIL
- [ ] 5.4 **sanity quick**：
      ```bash
      ./scripts/sanity.sh --quick
      ```
- [ ] 5.5 **PTX 语法全量**：
      ```bash
      ./tests/ptx/test_all_ptx.sh
      ```
- [ ] 5.6 **zero-refs 终验**：
      ```bash
      grep -rnE "force_set_pc|WarpContext::(get_pc|set_pc)|warp\.(get_pc|set_pc)\s*\(" \
        --include="*.cpp" --include="*.h" src/ include/ \
        | grep -v "antlr4_generated_src/"
      ```
      预期：仅命中 `warp_context.h` 占位注释

## 6. 合并与归档

- [ ] 6.1 确认 3 个 commit 各自独立可 revert：
      ```bash
      for sha in $(git log --format=%H -3); do
        git revert --no-commit $sha  # 干跑
        git revert --abort
      done
      ```
- [ ] 6.2 切换到 main：
      ```bash
      git checkout main
      git merge --no-ff refactor/dead-code-cleanup \
        -m "Merge: dead-code-cleanup (Fix #1-#4)"
      ```
- [ ] 6.3 触发 OpenSpec archive 流程，**生成 postmortem**（P2-4）：
      将"set_pc / force_set_pc 文档矛盾" 模式追加到
      `docs/dev-process/lessons-learned.md` 失败模式表
- [ ] 6.4 清理 worktree：
      ```bash
      git worktree remove .worktrees/baseline-dead-code --force
      ```
- [ ] 6.5 验证根 `AGENTS.md` 状态（更新 `Last Updated: 2026-07-XX`）

## 工时估算

| Phase | 工时 |
|-------|------|
| 1. 审计 + 基线 | 1h |
| 2. Commit 1（PC 字段） | 0.5h |
| 3. Commit 2（force_set_pc） | 1h |
| 4. Commit 3（文档） | 0.5h |
| 5. 全量验证 | 1h |
| 6. 合并 + 归档 | 0.5h |
| **合计** | **4.5h** |
