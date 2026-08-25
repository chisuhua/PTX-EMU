# antlr4-path-hardcoding-fix — Tasks

## 1. Pre-Implementation Setup (per ptx-lessons-learned §4 + §6)

- [x] 1.1 MUST: 提交 `2026-08-24-hsk8-followup-task-path.md` 入 git (untracked file, per ptx-lessons-learned §6 "artifacts-first"): `git add 2026-08-24-hsk8-followup-task-path.md && git commit -m "docs(plan): commit HSK-8 follow-up task path"` — untracked artifacts 在 implementation 开始时破坏 audit chain
- [x] 1.2 MUST: 建立基线 worktree `.worktrees/antlr4-path-fix-baseline` (per §4 15-20min): `git worktree add .worktrees/antlr4-path-fix-baseline HEAD`
- [x] 1.3 MUST: 跑基线 ctest 验证 baseline 健康: `cd .worktrees/antlr4-path-fix-baseline && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build --output-on-failure` — expected 249/249 PASS (per device-api-delegation archive)
- [x] 1.4 验证 `CMakeLists.txt:98-99` 当前使用 `${CMAKE_SOURCE_DIR}/antlr4/` (基线确认): `grep -n "CMAKE_SOURCE_DIR.*antlr4" CMakeLists.txt`
- [x] 1.5 创建 `feat/antlr4-path-fix` 分支: `git checkout -b feat/antlr4-path-fix` (在 main 工作 tree)

## 2. CMakeLists.txt 修复 (per design Decision 1)

- [x] 2.1 修改 `CMakeLists.txt:98`: `CMAKE_SOURCE_DIR` → `PROJECT_SOURCE_DIR`
- [x] 2.2 修改 `CMakeLists.txt:99`: `CMAKE_SOURCE_DIR` → `PROJECT_SOURCE_DIR`
- [x] 2.3 验证 diff 仅限 2 行: `git diff CMakeLists.txt | grep -E "^[+-]" | grep -v "^+++\|^---"` 应只显示 2 行修改
- [x] 2.4 build 测试 standalone: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build` — expected 100% build success, 0 错误
- [x] 2.5 ctest 验证: `ctest --test-dir build --output-on-failure` — expected 249/249 PASS (零回归)

## 3. drift_check Invariant 7 (per design Decision 2)

- [x] 3.1 修改 `.github/workflows/drift_check.yml`: 在 Invariant 6 (line 102-164) 之后添加 Invariant 7
  - 实现: `! grep -nE "CMAKE_SOURCE_DIR.*antlr4|antlr4.*CMAKE_SOURCE_DIR" CMakeLists.txt`
  - 匹配规则: `${CMAKE_SOURCE_DIR}/antlr4` 任何出现 → FAIL
  - 排除: `${CMAKE_CURRENT_SOURCE_DIR}/antlr4` (acceptable for subdirectory-relative references)
  - 排除: `${PROJECT_SOURCE_DIR}/antlr4` (correct fix)
  - 路径 trigger filter **必须**添加 `CMakeLists.txt` (BOTH `pull_request.paths` L11-17 AND `push.paths` L20-24 sections)
- [x] 3.2 验证 Invariant 7 grep syntax: `grep -nE "CMAKE_SOURCE_DIR.*antlr4|antlr4.*CMAKE_SOURCE_DIR" CMakeLists.txt` 应返回 0 行
- [x] 3.3 验证 Invariant 7 reverse test: 在临时分支尝试 revert CMakeLists.txt → 应被 Invariant 7 检测到 (本地 manual check,无需 push)

## 4. 文档同步 (per ptx-lessons-learned §21 Checklist I)

- [x] 4.1 编辑 `docs/dev-process/lessons-learned.md` 追加新章节 (按现有 lessons 格式):
  - 章节标题: "## N. CMake `CMAKE_SOURCE_DIR` vs `PROJECT_SOURCE_DIR` for vendored dependencies"
  - 现象: CppTLM `add_subdirectory(external/PTX-EMU)` 时 `CMAKE_SOURCE_DIR` 指向 CppTLM 根,导致 ANTLR4 path 解析失败
  - 教训: vendored dependencies 的 CMake 路径必须用 `PROJECT_SOURCE_DIR` (project-relative),不能用 `CMAKE_SOURCE_DIR` (top-level-relative)
  - 真实案例: `device-api-delegation` archive 完成后 CppTLM-side chained builds 仍因 ANTLR4 path 失败 (Doc2 §8 item 4)
  - commit: `2148e15c`
- [x] 4.2 修改 `AGENTS.md` §HSK-8 follow-up: 在 Phase 2.2/2.3 + cpptlm bridge cleanup 之后添加 Phase 2.4 条目
  - 内容: "✅ Phase 2.4 ANTLR4 path fix (commit `<hash>`): `${CMAKE_SOURCE_DIR}` → `${PROJECT_SOURCE_DIR}` for `ANTLR_EXECUTABLE` + `ANTLR4_RUNTIME_SOURCE_DIR`. CppTLM-side chained builds now succeed. drift_check Invariant 7 added."
- [x] 4.3 修改 `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md:265` §7.2 item 4:
  - 将 `⏳ Continue ANTLR4 path hardcoding fix` 改为 `✅ ANTLR4 path hardcoding fix landed (commit `<hash>`, drift_check Invariant 7 added, see `openspec/changes/antlr4-path-hardcoding-fix/`)`
- [x] 4.4 `README.md` §已实现功能: 无需修改 (build infra fix,不影响 user-visible features)

## 5. PR + Commit (DO NOT auto-commit — per HSK-8 follow-up plan §Phase 1.3)

- [x] 5.1 验证 ctest + drift_check 验证: `ctest --test-dir build --output-on-failure` → 249/249 PASS
- [x] 5.2 验证 drift_check.yml syntax: `cat .github/workflows/drift_check.yml | python3 -c "import yaml; yaml.safe_load(open('/dev/stdin'))"` 应无 YAML error
- [x] 5.3 commit 5 文件修改: `git add CMakeLists.txt .github/workflows/drift_check.yml docs/dev-process/lessons-learned.md AGENTS.md docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md`
- [x] 5.4 commit message: `fix(build): use PROJECT_SOURCE_DIR for vendored ANTLR4 paths (drift_check Invariant 7)`
- [x] 5.5 push `feat/antlr4-path-fix` 分支: `git push origin feat/antlr4-path-fix`
- [x] 5.6 创建 PR (建议标题): "fix(build): use PROJECT_SOURCE_DIR for vendored ANTLR4 paths"
- [x] 5.7 PR merge (squash): 合并后 origin/main HEAD advance
- [x] 5.8 (可选) 通知 CppTLM owner: PTX-EMU 端 ANTLR4 path 已修复, CppTLM 可以测试 chained build via `add_subdirectory`

## 6. Archive + Cleanup (per openspec-archive-change skill)

- [x] 6.1 验证 merge 后 OpenSpec artifacts 仍 tracked: `git ls-files openspec/changes/antlr4-path-hardcoding-fix/`
- [x] 6.2 ac-verifier archive-time check: drift_check Invariant 7 PASS + `CMakeLists.txt` 无 `${CMAKE_SOURCE_DIR}/antlr4`
- [x] 6.3 `openspec archive antlr4-path-hardcoding-fix`: `mv openspec/changes/antlr4-path-hardcoding-fix openspec/changes/archive/$(date +%Y-%m-%d)-antlr4-path-hardcoding-fix`
- [x] 6.4 验证 `openspec list` 不再包含 `antlr4-path-hardcoding-fix`
- [x] 6.5 update HSK-PROTOCOL-NOTES.md §HSK-8 实践示例: 引用此 change 作为 Doc2 §8 item 4 解决案例 (可选)
- [x] 6.6 清理基线 worktree: `git worktree remove .worktrees/antlr4-path-fix-baseline`

## Reference

- **Tracking issue / Doc2 §8 item 4**: `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md:265`
- **Parent HSK-8 follow-up plan**: `2026-08-24-hsk8-followup-task-path.md` (will commit before this change's Phase 1.1)
- **HSK-2 ANTLR4 contract**: `docs/superpowers/hsk-drafts/2026-07-16/HSK-2-antlr4-version.md`
- **Affected code**: `CMakeLists.txt:98-99`
- **Affected workflow**: `.github/workflows/drift_check.yml` (Invariant 7 added)
- **Affected docs**: `docs/dev-process/lessons-learned.md`, `AGENTS.md`, `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md`
- **Skills referenced**:
  - `ptx-lessons-learned` §3 (multi-phase criterion — N/A, single commit) + §4 (baseline worktree) + §6 (artifacts-first) + §21 (README Checklist I)
  - `cmake` (best practices for subproject-consumed libraries)
  - `openspec-archive-change` (archive procedure)
  - `ac-verifier` (archive-time check)