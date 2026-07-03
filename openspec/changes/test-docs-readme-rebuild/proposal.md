## Why

C4 change `docs-readme-rebuild` 在 archived 时未创建任何测试，违反 AGENTS.md §测试覆盖率检查清单 中"**新增功能 → 必须 unit + 推荐 integration + 推荐 e2e**"的要求。`scripts/check-docs-index.py`（105 行验证器，6 个 spec requirements，12 个 scenarios）当前仅 0 个单元测试 + 0 个 CI 集成。结果：

- spec 12 个 scenarios 中仅 4 个被 validator 主动检查（覆盖率 33%）
- 5 个 scenarios 完全无覆盖（banner 检查、skills 同步、孤儿 commit hash 验证、auto-generated stats、banner 不改 body）
- 2 个 scenarios 仅 WARN 不 FAIL（手写统计、添加新子目录）
- 改动 Check 1 regex 时无回归保护（已踩坑：`_` 不在 `[a-z0-9-]` 中）

**Why now**：本 change 在 C4 上线 3 小时内审计发现，避免下一波 Phase 3 债务修复时再次因缺测试导致同类问题。

## What Changes

**新增**：
- `tests/unit/scripts/test_check_docs_index.py` — 8 个 unit tests（每个 Check 覆盖 PASS + FAIL 两个场景）
- `.git/hooks/pre-commit` — 触发 Check 1/2/4 在 docs/ 改动时自动跑
- `.github/workflows/docs-validate.yml` — CI 在 PR 跑 Check 1-6 (Tier 3)
- Check 5：验证 stale 文档顶部含预期 banner 关键词
- Check 6：验证 `docs/skills/README.md` 与 `.opencode/skills/` 同步（数量 + disabled 标记）
- Check 7：验证 banner 添加 commit 不修改 body 字节内容（可选）

**修改**：
- `scripts/check-docs-index.py`：
  - Check 3 由 WARN 提升为 FAIL（强制"统计必自动生成"）
  - Check 4 升级为验证 commit hash 可通过 `git cat-file -t` 解析
  - 新增 Check 5/6/7
- `openspec/specs/docs-discoverability/spec.md`：通过 delta spec 添加 3 个新 requirements + 修正 2 个 scenarios 为严格 SHALL

**无破坏性变更**：所有新增 Check 采用"加而不删"策略，老的 Check 1-4 行为完全保留。

## Capabilities

### New Capabilities

（无 — 不创建新 capability spec 文件）

### Modified Capabilities

- `docs-discoverability`: 通过 delta spec 添加以下 requirements + scenarios（补充 C4 未覆盖的 5 个 scenarios）：
  - Req 4（Stale docs banner）补充 2 个新 scenarios + Check 5 验证
  - Req 6（Skill mirror）补充 2 个新 scenarios + Check 6 验证
  - Req 3（Orphan README）补充 1 个 scenario（commit hash 可验证）+ Check 4 升级
  - Req 2（Auto stats）scenario 由 WARN 提升为 FAIL（Check 3）
  - 新增 Req 7（Pre-commit enforcement）：docs/ 改动时 validator 自动触发
  - 新增 Req 8（CI enforcement）：PR 时 validator 6-check 在 GitHub Actions 跑

## Impact

**Affected files**：
- `tests/unit/scripts/test_check_docs_index.py`（新建）
- `tests/unit/scripts/CMakeLists.txt`（新建 or 追加）
- `scripts/check-docs-index.py`（扩展 — 增加 Check 5/6/7，提升 Check 3 severity，升级 Check 4）
- `.git/hooks/pre-commit`（新建）
- `.github/workflows/docs-validate.yml`（新建）
- `openspec/changes/test-docs-readme-rebuild/specs/docs-discoverability/spec.md`（delta spec）

**Affected systems**：
- 仅文档/验证系统；不影响 C++ 代码、CMake 构建、PTX 解析、模拟器执行
- 影响范围：所有 review docs/ 改动的开发者 + CI 系统

**Dependencies**：
- 复用 `.opencode/notes/debt-audit-2026-07-02.md` §3.3 文档债务审计
- 复用 C4 实现：`scripts/check-docs-index.py` (105 行) + `docs/README.md` (16 子目录索引) + 6 个孤儿 README + 2 个 banner

**Risks**：
- R1：Check 5/6/7 加入后第一次跑可能 FAIL（被现有状态偏离）→ 需先 fix 偏离再启用 strict 模式
- R2：pre-commit hook 可能误报（用 `git diff --cached --name-only` 检查 docs/ 改动限定）→ 限定范围
- R3：CI workflow 需 GitHub 仓库配置 → 仅当 GitHub repo 存在时生效

**Lessons-learned 集成**：
- ✅ Checklist D（Commit 前）：每个 commit message 列 Fix #N
- ✅ Checklist E（OpenSpec 实施后）：artifacts 在 Phase 1 即 git-tracked
- ✅ Checklist G（lifecycle）：本 change 修改 archived C4 的 spec — **通过 delta spec 机制**而非 amend C4 change（避免违反 lifecycle 约束）

**Estimated effort**：7.5 小时，分 3 个 Tier：
- Tier 1（最优先，3h）：unit tests + pre-commit + Check 5/6
- Tier 2（推荐，1.5h）：Check 3 FAIL + Check 4 commit hash
- Tier 3（CI 集成，3h）：GitHub Actions + Check 7
