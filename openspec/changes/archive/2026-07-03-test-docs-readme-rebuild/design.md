## Context

C4 change `docs-readme-rebuild` (archived 2026-07-03) 引入了：
- `scripts/check-docs-index.py`（105 行验证器）
- `scripts/check-docs-index.sh`（bash wrapper）
- 6 个孤儿 OpenSpec README
- 2 个 staleness banner

但 C4 archived 时**0 个 unit test + 0 个 integration test + 0 个 CI enforcement**。

验证器 4 个 check 与 active spec `docs-discoverability`（6 个 requirements，12 个 scenarios）对照：
- ✅ 覆盖：Check 1/2/4 — 3 个 scenarios
- ⚠️ 部分覆盖：Check 3 — 2 个 scenarios 仅 WARN 不 FAIL
- ❌ 完全未覆盖：5 个 scenarios（banner 检查、skills 同步、commit hash 验证、auto-generated 强制、banner body byte-identical）

**当前状态**：spec 承诺的能力与 validator 实际实现存在巨大 gap。Check 1 regex 已踩坑（`_` 不在 `[a-z0-9-]`）— 修改 Check 1 时无回归保护，debug 时间 ~30 分钟。

**目标状态**：8 个 unit tests 保护 4 个 Check（PASS + FAIL 双向），pre-commit hook 自动触发，CI 在 PR 跑全 6-check。

## Goals / Non-Goals

**Goals**:
- 100% spec scenarios 主动验证（12/12 含新增 3 个 Tier-3 requirements）
- Check 1-4 各有 ≥1 个 PASS + 1 个 FAIL 单元测试
- docs/ 改动时自动 validator 触发（pre-commit + CI）
- 单 test 运行 < 5s（避免开发者抗拒运行）
- 任何 Check 改 regex 时有回归保护

**Non-Goals**:
- 不修改 C++ 代码、CMake 构建、PTX 解析、模拟器执行
- 不重构 `scripts/check-docs-index.py` 整体架构
- 不修改 active `docs-discoverability` spec 的 6 个 requirement 内容（仅通过 delta spec 增量）
- 不为 `docs/skills/README.md` 添加额外功能（仅同步检查）
- 不替代 `scripts/sanity.sh`（保留现有测试入口）

## Decisions

### Decision 1: 测试目录放 `tests/unit/scripts/`

**选择**: 新建 `tests/unit/scripts/` 子目录存放 unit tests

**理由**:
- 命名一致：已有 `tests/unit/contexts/`、`tests/unit/barrier/`、`tests/unit/cudart/` 等
- 单元测试本质：测试单个 Python 脚本的逻辑，不需要 PTX 解析
- 不要混进 `tests/integration/`：integration 测试已定义为"指令序列"，与本测试无关

**考虑的替代**：
- `tests/unit/utils/` — 工具类但不准确，scripts/ 已超 utility 范围
- 独立顶层 `docs-validator/tests/` — 增加认知负担，与现有结构不统一
- **采用**：新建 `tests/unit/scripts/`

### Decision 2: Check 5/6/7 采用"加而不删"策略

**选择**: 在 validator 末尾追加 3 个新 Check，原有 Check 1-4 行为完全不变

**理由**:
- C4 已 archived，archived change 不应受新 check 影响（OpenSpec lifecycle）
- 新 check 首次跑可能 FAIL（如果当前 docs/ 状态偏离）— 需先 fix 再启用
- "加而不删"允许先 force-on-green，再扩大 enforcement

**考虑的替代**：
- 4-check 拆除：分裂违反 lessons-learned（修改应 incremental）
- 强制启用 strict：现有 docs 状态已偏离多个 check，强制启用导致 change 自身失败
- **采用**：先加 Check 5/6/7（WARN 模式），下一次 change 升级为 FAIL

### Decision 3: Check 3 升级为 FAIL（Tier 2）

**选择**: Check 3 (hand-edited statistics) 从 WARN 升级为 FAIL

**理由**:
- 当前 `docs/README.md` 无手写 statistics（C4 已清空）— 升级零回归风险
- 升级防止未来 PR 重新引入
- 与 spec Req 2 "MUST" 字面一致

**考虑的替代**：
- 保留 WARN：未来易被绕过
- 完全删除 Check 3：失去 inspection value
- **采用**：直接升级为 FAIL

### Decision 4: Test framework 选 Catch2 + Python subprocess

**选择**: C++ test 文件用 Catch2（与现有 `tests/unit/*/` 一致），Python test 文件用独立 pytest-style（不引入新依赖）

**理由**:
- Catch2 是项目统一 unit test framework（`tests/catch_amalgamated.hpp`）
- 验证器是 Python，不能用 Catch2 直接测 — 需要 subprocess 调用
- 但项目目前**无 Python test framework 集成** — 加 pytest 需 4h+（cmake 集成、pip install）

**妥协方案**：
- Tier 1 用 Python standalone script（`tests/unit/scripts/test_check_docs_index.py`）
- 通过 `bash` 调用 `python3 scripts/check-docs-index.py` 在测试 fixtures 上跑
- Test 本身用纯 Python unittest，避免引入新依赖
- Tier 3 可考虑加 GitHub Actions Python setup

**考虑的替代**：
- pytest：需要 CMake 集成，~4h 工作量（超出 7.5h 预算）
- shell script 只测 exit code：无法验证具体 check 输出
- **采用**：纯 Python unittest + subprocess 调用 validator + 临时目录 fixture

### Decision 5: pre-commit hook 仅触发 docs/ 改动

**选择**: pre-commit hook 用 `git diff --cached --name-only | grep '^docs/'` 限定范围

**理由**:
- 仅 docs 改动时跑 — 避免 C++ commit 触发 5s 延迟
- 与 spec Req "any change to docs/<dir>" 语义一致
- 失败时 exit 1 阻止 commit

**考虑的替代**：
- 任何 commit 都跑：增加开发摩擦
- 完全无 pre-commit：失去了 lessons-learned §3 "phase 间 invariant" 保护
- **采用**：docs/-only pre-commit

### Decision 6: Tier 3 CI workflow 用 GitHub Actions

**选择**: `.github/workflows/docs-validate.yml` — ubuntu-latest + python3 + check-docs-index.sh

**理由**:
- 项目已有 `.github/` 目录（copilot-instructions.md 存在）
- GitHub Actions 是事实标准
- 配置 30 行 yaml 即可

**考虑的替代**：
- GitLab CI：项目在 GitHub（`github.com/chisuhua/PTX-EMU` per README）
- Jenkins：不必要 — GitHub Actions 完全够
- **采用**：GitHub Actions，1.5h 配置

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|-----------|
| R1: Check 5/6/7 首次跑可能 FAIL（当前状态偏离） | 第一次提交 change 自身可能失败 | 先 fix 现状偏离（见 Phase 3.0 prep），再 enable Check 5/6/7 |
| R2: pre-commit hook 误触发（不相关 commit） | 开发者绕过或禁用 hook | 仅 docs/ 改动触发 + 提供 hook 跳过提示 |
| R3: Python unittest 与 Catch2 风格不一致 | 团队认知摩擦 | unit test 显式标注"Python test (not Catch2)" |
| R4: Check 5 banner regex 误报 | false positive 阻塞 PR | banner 用最宽松 pattern（"⚠️" 或 "**⚠️"），人工 review |
| R5: test fixture 临时目录可能未清理 | CI 缓存膨胀 | 用 `tempfile.TemporaryDirectory()` 自动清理 |
| R6: GitHub Actions 需要 Python 3 显式安装 | ubuntu-latest 自带 python3，但版本可能变化 | `actions/setup-python@v5` 显式锁定 3.11+ |
| R7: 修改 active spec 而违反 lifecycle | Cannot amend C4 archived change | 通过 delta spec（specs/ 文件）增量 |
| R8: 8 个 unit test 时间 > 5s | 开发者抗拒 | 每个 test < 100ms（fixture 最小化）|

## 影响范围

| 组件 | 影响类型 |
|------|---------|
| `tests/unit/scripts/test_check_docs_index.py` | 新建 — 8 unit tests |
| `tests/unit/scripts/CMakeLists.txt` | 新建 — 1 target |
| `scripts/check-docs-index.py` | 修改 — 加 Check 5/6/7 + Check 3 FAIL + Check 4 commit hash |
| `scripts/check-docs-index.sh` | 不变（wrapper 已正确） |
| `.git/hooks/pre-commit` | 新建 — 6 行 bash |
| `.github/workflows/docs-validate.yml` | 新建 — 30 行 yaml |
| `openspec/changes/test-docs-readme-rebuild/specs/docs-discoverability/spec.md` | 新建 delta spec — 3 ADDED Requirements |
| `openspec/specs/docs-discoverability/spec.md` | **不变**（通过 delta 机制） |
| C++ 代码 / CMake build / PTX simulator | **无影响** |

## Migration Plan

**前置 (Phase 3 prep)**：
1. 验证 C4 当前状态：`bash scripts/check-docs-index.sh` 应 PASS（已验证）
2. 验证 docs/skills/README.md 与 .opencode/skills/ 数量一致（17 active + 1 disabled = 18）

**Phase 1 — Tier 1 (3h)**：
1. 创建 8 unit tests，验证现有 Check 1-4 behavior 不回归
2. 扩展 validator 加 Check 5/6 (banner + skills 同步)
3. 配置 pre-commit hook

**Phase 2 — Tier 2 (1.5h)**：
4. Check 3 WARN → FAIL
5. Check 4 升级：验证 commit hash 可 `git cat-file -t` 解析
6. 同步 unit tests

**Phase 3 — Tier 3 (3h)**：
7. Check 7：banner 添加 commit body byte-identical 验证
8. GitHub Actions workflow
9. README 更新：测试运行说明

**回退策略**：
- 每个 Tier 独立可 revert（unit test 是 additive，不影响现有 C4）
- 若 Tier 1 失败 → 仅 revert Check 5/6/pre-commit，不影响 Check 1-4
- 若 Tier 2 失败 → 仅 revert Check 3 升级
- 若 Tier 3 失败 → 仅 revert Check 7 + CI workflow

## Open Questions

1. **Q1**: 是否需要为本 change 单独 OpenSpec change proposal phase？
   - 当前选择：复用 `openspec change propose` 标准流程
   - resolved

2. **Q2**: Python test 是否需要 Catch2 风格的测试命名？
   - 当前选择：unittest 标准 (`test_check_*` 前缀)
   - 项目惯例：测试需 Catch2（`TEST_CASE`），Python test 是新模式
   - 决议：在 tests/unit/scripts/README.md 标注"Python tests (NOT Catch2)"

3. **Q3**: 是否将 GitHub Actions workflow 推迟到下个 Phase？
   - 当前选择：Tier 3 一并交付（30 行 yaml，1.5h 足够）
   - 替代：仅做 Tier 1+2，CI 推迟
   - **决策**：先做全部 3 个 Tier；如果时间不允许，从 Tier 3 减项
