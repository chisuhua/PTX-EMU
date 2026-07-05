# Tasks: Complete T2-3 God-Class Split Cleanup

> **Type**: 3-Phase 清理 change（删除 T2-3 半成品 stub + unused POD）
> **HEAD baseline**: `c338f12`
> **Risk**: 🟢 极低（0 读写验证）
> **Lessons-learned**: §6 + Checklist E/F/G（**NOT amend 已归档 T2-3 change per Checklist G**）

---

## Phase 0: Artifacts Git-Tracking + Baseline

- [ ] 0.1 创建工作分支
  ```bash
  git checkout -b refactor/complete-t2-3-god-class-split
  ```
- [ ] 0.2 git-tracked artifacts
  ```bash
  git add openspec/changes/complete-t2-3-god-class-split/
  git ls-files openspec/changes/complete-t2-3-god-class-split/
  ```
- [ ] 0.3 commit artifacts
  ```bash
  git commit -m "docs(openspec): add complete-t2-3-god-class-split artifacts"
  ```
- [ ] 0.4 建立 baseline worktree（~15-20 分钟）
  ```bash
  git worktree add .worktrees/t2-3-baseline HEAD
  cd .worktrees/t2-3-baseline
  . env.sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  cd build && ctest --output-on-failure 2>&1 | tee /tmp/t2-3-baseline.log
  ```
- [ ] 0.5 验证 baseline 100% PASS
  ```bash
  grep -c "FAILED\|XPASS\|FAIL!" /tmp/t2-3-baseline.log
  ```

---

## Phase 1: 删除半成品 stub + unused POD（Fix #1）

> **Risk**: 🟢 极低

- [ ] 1.1 创建实施 worktree
  ```bash
  cd /workspace/project/PTX-EMU
  git worktree add .worktrees/t2-3-cleanup-impl refactor/complete-t2-3-god-class-split
  cd .worktrees/t2-3-cleanup-impl
  ```
- [ ] 1.2 **二次验证 unused POD 字段 0 读写**
  ```bash
  grep -rn "backend_links_\|warp_identity_" src/ include/ tests/ \
    | grep -v "include/ptxsim/warp_context.h:27[89]:"
  # 期望: 0 行（仅自身定义）
  ```
- [ ] 1.3 **检查 src/CMakeLists.txt 是否有 contexts 子目录引用**
  ```bash
  grep -n "contexts" src/CMakeLists.txt
  ```
- [ ] 1.4 **删除 stub 目录**
  ```bash
  ls src/ptxsim/contexts/  # 验证内容
  rm -rf src/ptxsim/contexts/
  ```
- [ ] 1.5 **如有 contexts CMake 引用则删除**
- [ ] 1.6 **修改 include/ptxsim/warp_context.h:279-280**
  ```cpp
  // 删除该 2 行:
  // backend_links_type backend_links_;
  // warp_identity_type warp_identity_;
  ```
- [ ] 1.7 **检查 thread_context.h:315-318 是否需要回滚**
  ```bash
  grep -n "exec_state_\|memory_\|program_ref_\|register_predicate_" src/ptxsim/core/thread_context.cpp
  ```
  - 如 4 POD 字段均仅 init 镜像写入，无其他代码读取 → 评估是否全部回滚
  - 如有任何字段被读取 → **保留**并加注释"deferred T2-3"
- [ ] 1.8 **Phase 1 验证**
  ```bash
  cmake --build build  # 编译通过
  cd build && ctest --output-on-failure  # 100% PASS
  ```
- [ ] 1.9 **Commit Fix #1**
  ```bash
  git commit -am "refactor(ptxsim): delete T2-3 half-finished stubs + unused PODs (Fix #1)

  Removed:
  - src/ptxsim/contexts/ (7 stub .cpp files, each 1 line)
  - src/CMakeLists.txt contexts subdirectory reference (if any)
  - include/ptxsim/warp_context.h:279-280 (backend_links_, warp_identity_ POD)

  Verified:
  - backend_links_ 0 reads/writes (only declaration)
  - warp_identity_ 0 reads/writes (only declaration)
  - contexts/*.cpp are placeholder stubs (no behavior)

  Kept (deferred to future T2-3-impl change):
  - thread_context.h:315-318 4 PODs (exec_state_/memory_/program_ref_/register_predicate_)
  - warp_context.h lane_mask_ POD (has mirror write logic)

  Per lessons-learned Checklists E/F.
  Refs: archive/2026-06-24-phase3-t2-3-god-class-split/ (NOT amended per Checklist G)
  "
  ```

---

## Phase 2: 文档同步（Fix #2）

- [ ] 2.1 更新 docs/roadmap/post-phase3-debt-roadmap.md
  - 从 §3.3 剩余债务列表移除 T2-3 半成品条目
- [ ] 2.2 二次 ctest 验证
  ```bash
  cd build && ctest --output-on-failure
  ```
- [ ] 2.3 Commit Fix #2
  ```bash
  git commit -am "docs(cleanup): sync roadmap post-Fix #1 (Fix #2)"
  ```

---

## Phase 3: Archive + Merge

- [ ] 3.1 Archive change
  ```bash
  openspec archive complete-t2-3-god-class-split --yes
  ```
- [ ] 3.2 清理 worktree
  ```bash
  git worktree remove .worktrees/t2-3-cleanup-impl
  ```
- [ ] 3.3 Merge to main
  ```bash
  git checkout main
  git merge --no-ff refactor/complete-t2-3-god-class-split
  ```

---

## 风险缓解矩阵

| 风险 | 缓解任务 | 验证 |
|------|---------|------|
| R1: 删除字段遗漏使用 | 1.2 + 1.8 | grep + ctest 100% |
| R2: CMake 引用遗漏 | 1.3-1.5 | grep 检查 + 编译通过 |
| R3: 未来 T2-3 实施需重新添加 | git history | 已记录决策 |