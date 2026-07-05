# Tasks: Remove Dead ActiveWarpManager

> **Type**: 3-Phase 纯删除（清理 change）
> **HEAD baseline**: `c338f12`
> **Risk**: 🟢 极低（0 调用方，grep 验证）
> **Lessons-learned**: §6（artifacts 必 tracked）+ Checklist E/F/G

---

## Phase 0: Artifacts Git-Tracking + Baseline（强制）

- [ ] 0.1 创建工作分支
  ```bash
  git checkout -b refactor/remove-dead-active-warp-manager
  ```
- [ ] 0.2 git-tracked artifacts
  ```bash
  git add openspec/changes/remove-dead-active-warp-manager/
  git ls-files openspec/changes/remove-dead-active-warp-manager/
  ```
- [ ] 0.3 commit artifacts
  ```bash
  git commit -m "docs(openspec): add remove-dead-active-warp-manager artifacts"
  ```
- [ ] 0.4 建立 baseline worktree（~15-20 分钟）
  ```bash
  git worktree add .worktrees/active-warp-manager-baseline HEAD
  cd .worktrees/active-warp-manager-baseline
  . env.sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  cd build && ctest --output-on-failure 2>&1 | tee /tmp/active-warp-baseline.log
  ```
- [ ] 0.5 验证 baseline 100% PASS
  ```bash
  grep -c "FAILED\|XPASS\|FAIL!" /tmp/active-warp-baseline.log  # 期望: 0
  ```

---

## Phase 1: 删除 ActiveWarpManager（Fix #1）

> **Risk**: 🟢 极低

- [ ] 1.1 创建实施 worktree
  ```bash
  cd /workspace/project/PTX-EMU
  git worktree add .worktrees/active-warp-manager-impl refactor/remove-dead-active-warp-manager
  cd .worktrees/active-warp-manager-impl
  ```
- [ ] 1.2 **二次验证 0 调用方**
  ```bash
  grep -rn "ActiveWarpManager\|active_warp_manager" src/ include/ tests/ \
    | grep -v "active_warp_manager\.\(h\|cpp\)"
  # 期望: 1 行 (CMakeLists.txt:77)
  ```
- [ ] 1.3 **删除头文件**
  ```bash
  rm include/ptxsim/active_warp_manager.h
  ```
- [ ] 1.4 **删除实现**
  ```bash
  rm src/ptxsim/core/active_warp_manager.cpp
  ```
- [ ] 1.5 **修改 src/CMakeLists.txt:77**
  ```cmake
  # 删除该行:
  # ptxsim/core/active_warp_manager.cpp
  ```
- [ ] 1.6 **Phase 1 验证**
  ```bash
  cmake --build build  # 编译通过
  cd build && ctest --output-on-failure  # 100% PASS
  ./tests/ptx/test_all_ptx.sh
  ```
- [ ] 1.7 **Commit Fix #1**
  ```bash
  git commit -am "refactor(ptxsim): delete dead ActiveWarpManager module (Fix #1)

  Removed:
  - include/ptxsim/active_warp_manager.h (36 LOC)
  - src/ptxsim/core/active_warp_manager.cpp (118 LOC)
  - src/CMakeLists.txt:77 source entry

  Verified 0 production call sites:
  - CMakeLists.txt:77 (only config-time reference)
  - All sm_context.cpp scheduling goes through WarpScheduler (8 call sites)

  Authoritative scheduler confirmed: RoundRobinWarpScheduler (sm_context.cpp:23).
  ActiveWarpManager was an unused alternative implementation.

  Per lessons-learned Checklists E/F.
  "
  ```

---

## Phase 2: 文档同步（Fix #2）

- [ ] 2.1 **检查 src/ptxsim/core/AGENTS.md 引用**
  ```bash
  grep -n "ActiveWarpManager\|active_warp_manager" src/ptxsim/core/AGENTS.md
  ```
- [ ] 2.2 如有引用，更新 AGENTS.md（删除对应行）
- [ ] 2.3 **更新 docs/audits/debt-audit-2026-07-02.md**
  - 标记 ActiveWarpManager RESOLVED（引用 Phase 1 commit hash）
- [ ] 2.4 **更新 docs/roadmap/post-phase3-debt-roadmap.md**
  - 从剩余债务列表移除
- [ ] 2.5 **二次 ctest 验证**
  ```bash
  cd build && ctest --output-on-failure
  ./scripts/sanity.sh --quick
  ```
- [ ] 2.6 **Commit Fix #2**
  ```bash
  git commit -am "docs(cleanup): sync AGENTS.md + audit + roadmap post-Fix #1 (Fix #2)"
  ```

---

## Phase 3: Archive + Merge

- [ ] 3.1 Archive change
  ```bash
  openspec archive remove-dead-active-warp-manager --yes
  ```
- [ ] 3.2 清理 worktree
  ```bash
  git worktree remove .worktrees/active-warp-manager-impl
  ```
- [ ] 3.3 Merge to main
  ```bash
  git checkout main
  git merge --no-ff refactor/remove-dead-active-warp-manager
  ```

---

## 风险缓解矩阵

| 风险 | 缓解任务 | 验证 |
|------|---------|------|
| R1: 隐藏依赖 | 1.2 + 1.6 | grep + ctest 100% |
| R2: 未来需要 ActiveWarpManager | git history | 154 LOC 可恢复 |
| R3: 文档遗漏 | 2.1-2.4 | AGENTS.md + audit + roadmap |