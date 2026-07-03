## Why

`docs/README.md` 是 PTX-EMU 项目唯一文档索引入口，但当前 171 行 36 表格行仅覆盖 6/17 子目录（缺少 adr/, appendix/, archive/, audits/, dev-process/, plans/, ptx/, reports/, research/, roadmap/, skills/, technical_design/, testing/）。新人查找 ADR 决策、审计报告、PTX 指令参考时被阻挡。同时 `docs/skills/README.md` 列 9 个技能而 `.opencode/skills/` 实际有 18 个。`tests/archive/` 被 AGENTS.md 引用但目录不存在。6 个 OpenSpec archive change 缺 design.md 导致历史决策无法追溯。

**问题根因**（参考 `docs/audits/debt-audit-2026-07-02.md` §3.3）：docs/ 子目录是 2026-06 重组（commit 历史中 `DOCUMENTATION-REORGANIZATION-SUMMARY.md` 记录）后陆续新建的，但 README 索引未同步。属于"已修复架构、未修复文档"的典型债务。

**Why now**：当前 `docs/audits/` 已沉淀 4 个审计（HEALTH-AUDIT + ERRATA + 2026-07-02 debt-audit），新人需要快速定位这些审计；OpenSpec v1.4.1（commit `c20a93b`）升级后归档速度加快，孤儿 change 数量增加。

## What Changes

- **重建 `docs/README.md` 索引**：从 6 子目录扩展到 17 子目录，每个子目录添加 1-2 句功能描述
- **删除过时统计信息**：移除"38 测试、~750 LOC、22 commit"等手写数字，改为 `grep -r "TEST_CASE" tests/ | wc -l` 自动生成
- **同步 `docs/skills/README.md`**：从 9 技能扩展到 18 技能（参考 `.opencode/skills/` 实际目录），标注 `three-mode-testing` 为 disabled（commit `14c8eeb`）
- **处理 `tests/archive/` 引用冲突**：AGENTS.md 引用不存在的路径，二选一（删引用或建 `.gitkeep`）
- **6 个 OpenSpec 孤儿 change 添加 README.md**：列出实施 commit 哈希和关键决策（按 Oracle A2 推荐的"语义参考"模式）
- **添加审计勘误 banner**：`HEALTH-AUDIT-2026-06-21.md` 和 `PROJECT-COMPLETION-SUMMARY.md` 顶部添加"已过期/有勘误"banner 指向 ERRATA
- **验证所有 README 链接可达**：用脚本提取 `markdown-link-check` 验证

**BREAKING**: 无（本 change 仅修改文档，不影响代码、API、构建）

## Capabilities

### New Capabilities

- `docs-discoverability`: 项目文档索引与导航规范。定义 README 索引表格应覆盖全部子目录、每个子目录有 1-2 句功能描述、统计信息自动生成而非手写。后续任何 docs/ 子目录新增必须同步更新 README 索引。

### Modified Capabilities

（无 — 本 change 不修改任何已有 spec 的 requirements）

## Impact

- **Affected files**:
  - `docs/README.md`（重写）
  - `docs/skills/README.md`（重写）
  - `AGENTS.md`（可能修改对 `tests/archive/` 的引用）
  - `docs/audits/HEALTH-AUDIT-2026-06-21.md`（添加 banner）
  - `docs/PROJECT-COMPLETION-SUMMARY.md`（添加 banner）
  - `openspec/changes/archive/2026-06-24-phase3-*/`（6 个目录各加 README.md）
  - `tests/archive/.gitkeep`（如选择保留目录）
- **Affected systems**: 仅文档系统；不影响 C++ 代码、CMake 构建、PTX 解析、模拟器执行
- **Dependencies**: 无外部依赖
- **Migration**: 无需迁移（文档是 addition-only）
- **Risks**:
  - R1：修改 `docs/README.md` 可能断链 → 用 `markdown-link-check` 验证
  - R2：6 个 OpenSpec 孤儿 README 需准确引用 commit hash → 用 `git log --all --oneline -- <path>` 验证每个
  - R3：删除手写统计信息可能被认为"信息丢失" → 改为自动生成脚本保留同样信息
- **ADR 引用**: 本 change 符合 ADR-0013 "statement factory test unification" 隐含的"测试/文档分离"原则
- **Skill 引用**: 无直接 skill 依赖（文档维护非编程任务）
- **Lessons-learned 集成**:
  - ✅ Checklist D（Commit 前）：修改 docs/ 前 AGENTS.md 同步
  - ✅ Checklist E（OpenSpec 实施后）：artifacts (proposal/design/tasks) 必须 git-tracked（避免 8a5573d 模式重演）
  - ✅ Checklist F（Debt audit 撰写）：本 change 引用 commit `3f46a3e` HEAD 而非文件路径
  - ✅ Checklist G（lifecycle）：本 change 是 NEW change 不 amend 任何 archived change

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- [x] 不适用（本 change 无 API 迁移，仅文档修改）

### 状态修改
- [x] 不适用（本 change 无状态修改）

### 多 Phase 推进
- [x] 本 change 预计 1-2 commit 单 Phase 推进（独立 worktree 仍按 Checklist B 准备基线作为安全网）

### 文档同步
- [x] AGENTS.md 修改 `tests/archive/` 引用（如选删除路径）
- [x] docs/README.md 是 change 主目标
- [x] docs/skills/README.md 是 change 子目标
- [x] OpenSpec artifacts (proposal/design/tasks) 必须 git-tracked（Checklist E）
