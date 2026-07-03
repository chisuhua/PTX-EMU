## 1. Pre-flight Verification

- [ ] 1.1 Verify clean working tree: `git status` shows no uncommitted source changes (only `.opencode/notes/` from C4 is OK)
- [ ] 1.2 Verify C4 scripts exist: `ls scripts/check-docs-index.py scripts/check-docs-index.sh`
- [ ] 1.3 Verify active spec exists: `cat openspec/specs/docs-discoverability/spec.md | head -5`
- [ ] 1.4 Run baseline: `bash scripts/check-docs-index.sh` returns PASS (confirms C4 working state)
- [ ] 1.5 Verify skills sync state: `ls .opencode/skills/ | wc -l` = 18 directories, `ls docs/skills/README.md` exists
- [ ] 1.6 git add OpenSpec artifacts (Checklist E — FIRST commit): `git add openspec/changes/test-docs-readme-rebuild/` then commit

## 2. Tier 1: Unit Tests + Check 5/6 + pre-commit (Fix #1-#3)

### 2.1 Test infrastructure

- [ ] 2.1.1 Create directory `tests/unit/scripts/`
- [ ] 2.1.2 Create `tests/unit/scripts/CMakeLists.txt` with single test target using `add_test + add_custom_target` (Catch2 unit test infrastructure via bash)
- [ ] 2.1.3 Create `tests/unit/scripts/README.md` documenting Python test pattern (not Catch2)

### 2.2 Unit test fixtures (per Check)

- [ ] 2.2.1 Write `tests/unit/scripts/test_check_1_subdirs.py`: setup makes temp dir with 16 subdirs + valid README, runs validator, asserts PASS
- [ ] 2.2.2 Write `tests/unit/scripts/test_check_1_mismatch.py`: setup makes 17 subdirs but README indexes 16, asserts FAIL with `NOT_INDEXED: <name>`
- [ ] 2.2.3 Write `tests/unit/scripts/test_check_2_links_pass.py`: temp doc with valid links, asserts PASS
- [ ] 2.2.4 Write `tests/unit/scripts/test_check_2_links_fail.py`: temp doc with broken link `./nonexistent.md`, asserts FAIL
- [ ] 2.2.5 Write `tests/unit/scripts/test_check_3_stats.py`: temp doc with `38 测试` row, asserts FAIL (post-Tier-2)
- [ ] 2.2.6 Write `tests/unit/scripts/test_check_4_orphan.py`: creates 2 archive subdirs (1 with README, 1 without), asserts mixed PASS/FAIL
- [ ] 2.2.7 Write `tests/unit/scripts/test_check_5_banner.py`: creates 2 stale docs (1 with expected banner, 1 without), asserts PASS for each
- [ ] 2.2.8 Write `tests/unit/scripts/test_check_6_skills.py`: creates 17 + 1 skills dirs + matching docs/README.md, asserts PASS; then add new skill not in docs, assert FAIL

### 2.3 Test runner script

- [ ] 2.3.1 Create `tests/unit/scripts/run_all_tests.sh` that loops and runs all `test_check_*.py` files, reports aggregated result
- [ ] 2.3.2 Verify: `bash tests/unit/scripts/run_all_tests.sh` returns exit 0 with all 8 tests passing
- [ ] 2.3.3 Add to `tests/unit/CMakeLists.txt` `add_test(NAME unit_doc_validator COMMAND bash ...)` integration

### 2.4 Check 5: Banner verification

- [ ] 2.4.1 Edit `scripts/check-docs-index.py`: add `EXPECTED_BANNERED` dict + `check_5_banners()` function
  - Expected: `HEALTH-AUDIT-2026-06-21.md` contains `ERRATA` or `事实错误已修正` substring
  - Expected: `PROJECT-COMPLETION-SUMMARY.md` contains `已过期` substring
- [ ] 2.4.2 Pattern: `^>\s+\*\*⚠️\s+.+` (blockquote bold with warning emoji at first 5 lines after title)
- [ ] 2.4.3 Update Check 1-4 numbering → shift to Check 6/7 (Check 5 inserts before existing logic)
- [ ] 2.4.4 Add `sys.exit(1)` summary if Check 5 FAILs

### 2.5 Check 6: Skills sync verification

- [ ] 2.5.1 Edit `scripts/check-docs-index.py`: add `check_6_skills_sync()` function
- [ ] 2.5.2 Logic:
  - active_skills = `[d for d in os.listdir(".opencode/skills") if is_dir]`
  - disabled_skills = `[d for d in os.listdir(".opencode/skills.disable")]`
  - docs_active = parse docs/skills/README.md table, return list
  - docs_disabled = parse docs/skills/README.md, find `[disabled]` markers
  - error if active_skills != docs_active, or disabled_skills != docs_disabled
- [ ] 2.5.3 FAIL message format: `MISSING_IN_DOCS: {name}` / `STALE_IN_DOCS: {name}` / `DISABLED_MISSING_MARKER: {name}`

### 2.6 pre-commit hook

- [ ] 2.6.1 Create `.git/hooks/pre-commit`:
      ```bash
      #!/usr/bin/env bash
      if git diff --cached --name-only | grep -q '^docs/'; then
        echo "docs/ changed — running check-docs-index..."
        bash "$(git rev-parse --show-toplevel)/scripts/check-docs-index.sh" || {
          echo "❌ docs-index check failed. To skip: git commit --no-verify"
          exit 1
        }
      fi
      ```
- [ ] 2.6.2 `chmod +x .git/hooks/pre-commit`
- [ ] 2.6.3 Test: stage a deliberate docs/ change → commit → expect validator run
- [ ] 2.6.4 Document: add to root AGENTS.md "开发流程" 章节 一行说明 pre-commit hook

### 2.7 Tier 1 verification

- [ ] 2.7.1 `bash tests/unit/scripts/run_all_tests.sh` exits 0
- [ ] 2.7.2 `bash scripts/check-docs-index.sh` exits 0 (all 6 checks PASS)
- [ ] 2.7.3 Verify pre-commit hook by intentionally breaking then fixing
- [ ] 2.7.4 `git diff --stat` shows only intended changes
- [ ] 2.7.5 Commit: `test(docs): add 8 unit tests + check 5/6 + pre-commit hook (Fix #1)`

## 3. Tier 2: Check 3 FAIL + Check 4 commit hash (Fix #2)

### 3.1 Check 3 WARN → FAIL

- [ ] 3.1.1 Edit `scripts/check-docs-index.py`: in `log_warn` for stats, change to `log_fail` + increment `FAIL_COUNT`
- [ ] 3.1.2 Update summary logic: any FAIL exits 1
- [ ] 3.1.3 Verify: `bash scripts/check-docs-index.sh` still exits 0 (no hand-edited stats in current docs)
- [ ] 3.1.4 Test fixture update: `test_check_3_stats.py` was written for FAIL — ensure it still passes after behavior change

### 3.2 Check 4 commit hash verification

- [ ] 3.2.1 Extend Check 4 to parse each orphan README for `**Implementation commit**: \`xxx\`` markdown code block
- [ ] 3.2.2 For each extracted hash: run `git cat-file -t <hash>` — if not found, FAIL with `INVALID_COMMIT: <change-name> -> <hash>`
- [ ] 3.2.3 Also verify: hash must appear in `git log --all --oneline -- openspec/changes/archive/<change>/` — else FAIL

### 3.3 Tier 2 verification

- [ ] 3.3.1 All 6 checks (Check 1-6) PASS
- [ ] 3.3.2 Re-run unit tests — all still PASS
- [ ] 3.3.3 Commit: `feat(docs): upgrade check 3 to FAIL + verify orphan commit hashes (Fix #2)`

## 4. Tier 3: Check 7 + CI workflow (Fix #3)

### 4.1 Check 7: banner body byte-identical

- [ ] 4.1.1 Add `check_7_banner_body_unchanged()` function
- [ ] 4.1.2 Logic: for each file in `EXPECTED_BANNERED`, find the commit that introduced the banner by `git log --diff-filter=M -p --pickaxe-regex -S "⚠️" -- <file>`
- [ ] 4.1.3 Compare pre-banner commit body (`git show <commit>^:<file>`) vs current body — if differs, FAIL with `BODY_CHANGED: <file>`
- [ ] 4.1.4 Edge case: first commit has no parent, FAIL with `NO_PRE_BANNER_COMMIT: <file>`

### 4.2 GitHub Actions workflow

- [ ] 4.2.1 Create `.github/workflows/docs-validate.yml`:
      ```yaml
      name: docs-validate
      on:
        pull_request:
          paths: ['docs/**']
      jobs:
        validate:
          runs-on: ubuntu-latest
          steps:
            - uses: actions/checkout@v4
              with:
                fetch-depth: 0  # need git history for Check 7
            - uses: actions/setup-python@v5
              with:
                python-version: '3.11'
            - run: bash scripts/check-docs-index.sh
            - run: bash tests/unit/scripts/run_all_tests.sh
      ```
- [ ] 4.2.2 Document: add to `.github/code-review-graph.instruction.md` (if exists) 或 root README 关于此 workflow

### 4.3 Tier 3 verification

- [ ] 4.3.1 All 7 checks PASS on current main
- [ ] 4.3.2 All 8+ unit tests PASS
- [ ] 4.3.3 `bash scripts/check-docs-index.sh --verbose` shows Check 7 in output
- [ ] 4.3.4 YAML lint: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/docs-validate.yml'))"`
- [ ] 4.3.5 Commit: `ci(docs): add github actions workflow + check 7 banner-body (Fix #3)`

## 5. Post-Implementation Verification (per AGENTS.md TDD §8)

- [ ] 5.1 Run `./scripts/sanity.sh --quick` — must PASS (no regression)
- [ ] 5.2 Run all C4 tests: `bash tests/unit/scripts/run_all_tests.sh` — must PASS
- [ ] 5.3 Run validator: `bash scripts/check-docs-index.sh` — must PASS
- [ ] 5.4 Verify each Phase commit is independently revertible:
      - `git revert <hash> --no-commit` then `bash scripts/check-docs-index.sh` still passes (test 1 of 3)
- [ ] 5.5 Update `openspec/changes/test-docs-readme-rebuild/tasks.md` — check all boxes
- [ ] 5.6 Generate postmortem per lessons-learned Checklist E:
      - `.opencode/notes/test-docs-readme-rebuild-postmortem.md` with:
        - 实测时间 vs 估算时间
        - regex escaping 在 unit test 中的处理方式
        - 任何新发现的失败模式
- [ ] 5.7 Archive change: `openspec archive test-docs-readme-rebuild --yes --skip-specs`
- [ ] 5.8 Add postmortem reference to `.opencode/skills/ptx-lessons-learned/SKILL.md` if new pattern discovered (e.g., "Python unit test + subprocess validator integration")
- [ ] 5.9 Final commit: `chore(openspec): archive test-docs-readme-rebuild + postmortem`

## 6. Lessons-Learned Compliance Check

- [ ] 6.1 Checklist A (函数迁移): N/A — 文档工具，无 API 迁移
- [ ] 6.2 Checklist B (重构前): baseline 已通过 (1.4)
- [ ] 6.3 Checklist C (写注释): Phase 2 unit test fixtures 必须清晰可读
- [ ] 6.4 Checklist D (Commit 前): 每个 commit message 包含 `Fix #N` 编号 (已在每 Phase commit message 中)
- [ ] 6.5 Checklist E (OpenSpec 实施后): artifacts 在 Phase 1.6 即 git-tracked (避免 8a5573d 模式)
- [ ] 6.6 Checklist F (Debt audit): 本 change 引用 commit hash 不引用文件路径
- [ ] 6.7 Checklist G (lifecycle): 通过 delta spec 增量修改 docs-discoverability，不 amend C4 archived change
