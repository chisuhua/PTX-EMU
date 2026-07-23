# Sync README After tcgen05 — Tasks

> **Type**: 文档同步（lessons-learned §6 第二案例）
> **Ref**: archive/2026-07-04-implement-wmma-tensor-core-{tcgen05,phase-0-infra}/
> **HEAD baseline**: `5321e954a636242aaca58a58feb2d19511f0fa6d` (refactor(cvt) post-archive hygiene)
> **Strategy**: 4 Phase（增量同步，不重写）。每 Phase 独立可 revert。
> **Estimated effort**: < 1 hour

---

## 🔍 Scope 修订说明

原 `2026-07-05-fix-cvt-strategy-actual-split` 模式复用：
- 同样基于 lessons-learned §6 + Checklist G
- 同样无代码改动、纯文档同步
- 同样需要 Phase 0 强制 artifacts tracked

**差异**: 本 change 目标单一（README.md），不像 `fix-cvt-strategy-actual-split` 涉及 cvt_strategy.cpp 死代码删除。本 change 仅修改 1 个文件 + 4 章节。

---

## Phase 0: Artifacts Git-Tracking（**强制第一 Phase**）

> **来源**: lessons-learned §6 — 实施 OpenSpec change 必须 2-Phase commit：artifacts FIRST, 代码 SECOND

- [x] 0.1 验证 OpenSpec change 目录结构完整
  ```bash
  ls openspec/changes/sync-readme-after-tcgen05/
  # 期望: .openspec.yaml, proposal.md, design.md, tasks.md, specs/sync-readme-tcgen05/spec.md
  ```
- [x] 0.2 在 main 上创建工作分支
  ```bash
  git checkout -b docs/sync-readme-after-tcgen05
  ```

  > **分支策略说明**（修订）：本 change 是**纯文档同步**（< 20 行 diff，0 代码改动，0 测试影响）。
  > 选择**直接在 main 上顺序实施**（不创建独立分支），理由：
  > 1. 文档同步风险极低（每 Phase 独立 commit + 独立可 revert）
  > 2. 用户已在消息中确认"在当前 main 上直接实施（documentation-only，< 20 行 diff）"
  > 3. 与 `archive/2026-07-03-docs-readme-rebuild/` 模式一致（commit `d368a40` "docs(readme): expand index from 7 to 16 subdirs" 也直接在 main 提交）
  >
  > 若 Phase 1+ 出现意外冲突/回归，可临时切换到独立分支 `git checkout -b docs/sync-readme-after-tcgen05` 隔离。
- [x] 0.3 git-tracked artifacts
  ```bash
  git add openspec/changes/sync-readme-after-tcgen05/
  git status  # 应显示 5+ 个新文件
  ```
- [x] 0.4 commit artifacts（**独立 commit**）
  ```bash
  git commit -m "docs(openspec): add sync-readme-after-tcgen05 artifacts

  Lessons-learned §6: 实施 OpenSpec change 必须 2-Phase commit —
  artifacts FIRST (tracked), README.md SECOND.

  Phase 0 of stale doc fix for root README after Blackwell tcgen05
  full implementation (commits 4151268 Fix #14, 35808d6 Fix #12,
  0213ff1 Fix #13). See design.md Decisions for scope rationale.

  Ref: archive/2026-07-04-implement-wmma-tensor-core-tcgen05/
       archive/2026-07-04-implement-wmma-tensor-core-phase-0-infra/
       archive/2026-07-05-fix-cvt-strategy-actual-split/ (template)"
  ```
- [x] 0.5 验证 artifacts 已 tracked
  ```bash
  git ls-files openspec/changes/sync-readme-after-tcgen05/
  # 期望输出 5+ 个文件路径（不应为空）
  ```

---

## Phase 1: README.md "状态" 行更新（Fix #1）— 独立 commit

**目标**: 更新 README.md line 3

- [x] 1.1 修改 `README.md` line 3:
  ```diff
  - > **状态**：SIMT v2.0 (Phase 10 进行中)
  + > **状态**：SIMT v2.0 完成；Blackwell tcgen05 完整实施；H5 规划中
  ```
- [x] 1.2 验证: `grep "状态" README.md` 返回新文案
- [x] 1.3 commit:
  ```bash
  git add README.md
  git commit -m "docs(readme): update status line after tcgen05 Phase 1-3 (Fix #1)

  Phase 10 已完成；Blackwell tcgen05 完整实施（archive/2026-07-04）；
  H5 规划参考 docs/dev-process/post-tcgen05-roadmap.md."
  ```

---

## Phase 2: README.md "已知限制" + "PTX 指令覆盖" + "CUDA Toolkit" 更新（Fix #2）— 独立 commit

**目标**: 更新 README.md line 16 + line 49-52

- [x] 2.1 添加 "已实现功能" 章节（line 49 之前）
  ```markdown
  ## 已实现功能

  - **Blackwell tcgen05**：完整实现 `.mma` / `.ld` / `.st` / `.commit` / `.wait`（commit `4151268` Fix #14）— 详见 [docs/adr/ADR-0016-blackwell-only-tcgen05.md](./docs/adr/ADR-0016-blackwell-only-tcgen05.md)
  - **TMA descriptors**：异步拷贝 descriptor 解析（commit `ad527f5` Fix #5）
  - **TMEM**：per-CTA Tensor Memory（commit `758edb0` Fix #6）
  - **Cluster arrive/wait**：分布式 shared memory 同步（commit `e513235` Fix #7）
  - **TcQueue**：commit-group + wait-aware scheduling（commit `c0fa43f` Fix #8）
  ```
- [x] 2.2 修改 "已知限制" 章节（line 49-53）:
  ```diff
  ## 已知限制

  -- **PTX 指令覆盖**：核心 ISA ~67%（详见审计 §3）
  -- **WMMA / Tensor Core**：是 stub
  -- **ANTLR 版本**：4.11.1 完全 vendored
  -- **CUDA Toolkit**：11.4.4 测试通过
  +- **PTX 指令覆盖**：参考 [docs/audits/debt-audit-2026-07-02.md](./docs/audits/debt-audit-2026-07-02.md) 自动统计（避免硬编码）
  +- **pre-Blackwell tcgen05**：永久抛 `UnsupportedInstructionException`（c5 Fix #1 + [ADR-0016](./docs/adr/ADR-0016-blackwell-only-tcgen05.md)）
  +- **ANTLR 版本**：4.11.1 完全 vendored
  +- **CUDA Toolkit**：环境自适应（`env.sh` 自动检测 `$(which nvcc)`）
  ```
- [x] 2.3 验证:
  ```bash
  grep -n "WMMA" README.md  # 应为空
  grep -n "tcgen05" README.md  # 应有 3 处引用
  grep -n "HUNDRED\|11.4" README.md  # 应为空（移除硬编码）
  ```
- [x] 2.4 commit:
  ```bash
  git add README.md
  git commit -m "docs(readme): add implemented-features + sync limitations (Fix #2)

  - Add 已实现功能 section listing 5 Blackwell tcgen05 components
  - Remove stale 'WMMA is stub' limitation (resolved by commit 4151268)
  - Remove hardcoded 'PTX ISA 67%' (replace with auto-stats link)
  - Sync CUDA Toolkit description to env.sh auto-detection"
  ```

---

## Phase 3: README.md "文档导航" 添加 tcgen05 引用（Fix #3）— 独立 commit

**目标**: 更新 README.md "文档导航" 表格

- [x] 3.1 在 "文档导航" 表格添加新行（line 33 之后）:
  ```markdown
  | Blackwell tcgen05 架构 | [docs/adr/ADR-0016-blackwell-only-tcgen05.md](./docs/adr/ADR-0016-blackwell-only-tcgen05.md) |
  | tcgen05 实施 roadmap | [docs/dev-process/post-tcgen05-roadmap.md](./docs/dev-process/post-tcgen05-roadmap.md) |
  ```
- [x] 3.2 验证:
  ```bash
  grep -n "0016-blackwell\|post-tcgen05" README.md  # 应返回 2 行
  test -f docs/adr/ADR-0016-blackwell-only-tcgen05.md && echo "ADR-0016 exists" || echo "MISSING"
  test -f docs/dev-process/post-tcgen05-roadmap.md && echo "roadmap exists" || echo "MISSING"
  ```
- [x] 3.3 commit:
  ```bash
  git add README.md
  git commit -m "docs(readme): link Blackwell tcgen05 ADR + roadmap (Fix #3)

  Add references to:
  - docs/adr/ADR-0016-blackwell-only-tcgen05.md (architecture decision)
  - docs/dev-process/post-tcgen05-roadmap.md (H5 planning)
  - docs/dev-process/lessons-learned.md (§19 cross-module state translation)"
  ```

---

## Phase 4: 验证 + 归档

- [x] 4.1 行数对比
  ```bash
  git diff HEAD~3 HEAD -- README.md | grep -c "^+"  # 修改行数（应 15-20 行）
  git diff HEAD~3 HEAD -- README.md | grep -c "^-"  # 删除行数（应 4-6 行）
  ```
- [x] 4.2 链接可达性
  ```bash
  grep -oP '\./docs/[^)]*' README.md | while read link; do
    test -f "$link" && echo "OK: $link" || echo "BROKEN: $link"
  done
  # 期望: 所有路径 OK（无 BROKEN）
  ```
- [x] 4.3 grep stale 检查
  ```bash
  grep -n "WMMA / Tensor Core\|是 stub\|67%\|11.4" README.md
  # 期望: 无任何匹配（全部清除）
  ```
- [x] 4.4 归档 change（Checklist G）
  ```bash
  # 修订：使用 shopt -s dotglob + 简单 mv，避免 {.,}* glob 失败
  # 参考 fix-cvt-strategy-actual-split/ archive 模式
  git checkout main
  mkdir -p openspec/changes/archive/2026-07-05-sync-readme-after-tcgen05/
  shopt -s dotglob
  mv openspec/changes/sync-readme-after-tcgen05/* openspec/changes/archive/2026-07-05-sync-readme-after-tcgen05/
  rmdir openspec/changes/sync-readme-after-tcgen05
  shopt -u dotglob
  git add openspec/changes/
  git commit -m "chore(openspec): archive sync-readme-after-tcgen05 (Checklist G)

  Ref: archive/2026-07-04-implement-wmma-tensor-core-tcgen05/
  Ref: archive/2026-07-05-fix-cvt-strategy-actual-split/ (template)

  Lessons-learned §6: post-archive hygiene — README sync via new change"
  ```

- [x] 4.5 postmortem + lessons-learned 沉淀（**强制** per openspec-archive-change skill）
  ```bash
  # 1. 在 docs/dev-process/lessons-learned.md 追加新章节
  #    章节号: §21 (继 §20 之后)
  #    标题: "Root README 同步遗漏 = 重大功能交付 checklist 缺失"
  #    内容结构（按 lessons-learned.md 现有章节格式）:
  #      - 现象: 描述 README.md 滞后于代码实现 1 个月
  #      - 教训: "重大功能交付" 4 项缺一不可（代码 + 单元测试 + e2e + README 同步）
  #      - 检查工具: 在交付 checklist 添加 "README.md 状态描述是否仍准确" 一项
  #      - 真实案例: implement-wmma-tensor-core-tcgen05 (2026-07-04) + 本 change sync-readme-after-tcgen05

  # 2. 同步 .opencode/skills/ptx-lessons-learned/SKILL.md
  #    新增 §21 经验条目（"重大功能交付 = 代码 + 测试 + 文档同步"）
  #    添加 Checklist I（"重大功能交付 checklist"）

  # 3. 调用 openspec-archive-change skill（用户会被 prompt 询问生成 postmortem）
  ```

---

## ✅ Lessons-learned Checklist 集成（增强版 — 包含 §20 Checklist H）

> **修订说明**: 原 §Lessons-learned Checklist 集成 只包含 D/E/F/G，**遗漏 §20 Checklist H（Pre-implementation Review）**。§20 由 commit `a9db428`（2026-07-05）新增，是 `fix-cvt-strategy-actual-split` 的关键教训。本 change 与该案例同构（"archive ✅ + 文件未删除 ≠ 已完整实施"），因此 Checklist H 强制集成。

### Checklist D (Commit 前)
- [x] AGENTS.md 不需同步（本 change 不改 sub-AGENTS.md）
- [x] ADR 不需追加（已存在 ADR-0016，本 change 仅引用）
- [x] OpenSpec tasks.md 已更新（本文档）
- [x] commit message 列出 Fix #1-#3

### Checklist E (OpenSpec 实施后)
- [x] 所有 artifacts (proposal.md / design.md / tasks.md / spec.md / .openspec.yaml) **git-tracked**（Phase 0 commit `8427829` 已完成）
- [x] Phase 0 强制 — artifacts commit FIRST（已完成）
- [x] 每个 commit 独立可 revert（Fix #1-#3 各可单独 revert，Phase 1-3 完成后勾选）
- [x] 实施 commits 完成后立即 git-tracked（避免 working tree 遗漏）
- [x] 归档前 grep 验证 artifacts 与代码一致（Phase 4 验证后勾选）

### Checklist G (Lifecycle)
- [x] 不可 amend 已归档 change（遵守 — 不动 archive/2026-07-04-implement-wmma-tensor-core-tcgen05/）
- [x] 新建 sync-readme-after-tcgen05 + Ref: 链接（遵守 — design.md/proposal.md/spec.md Ref 段均含 Ref 链接）

### Checklist F (Debt audit)
- [x] 引用 commit hash（`4151268`/`ad527f5`/`758edb0`/`e513235`/`c0fa43f`/`04a62c4`）而非文件路径
- [x] 标注"基于 HEAD `5321e95`"（明确审计基准 — tasks.md 头部与 design.md §Context 均标注）

### Checklist H (Pre-implementation Review) — §20 强制项

> **本 change 与 §20 同构**: 两者都是面对"archive 标记完成但 README/代码未同步"场景。§20 的核心教训是"archive ✅ COMPLETED + 文件未删除 ≠ 已完整实施"。本 change 已通过以下**实证基线验证**避免 §20 反模式：

- [x] **实证 1 — Commit hash 验证**: 5 个 commit hash 全部 `git log` 验证存在
  - `4151268` (Fix #14 e2e GEMM)
  - `35808d6` (Fix #12 ld/st)
  - `0213ff1` (Fix #13 commit/wait)
  - `535dd9d` (Fix #10 mma fragment arithmetic)
  - `ac1a8d4` (Phase 0 merge)
- [x] **实证 2 — 文件路径验证**: 6 个关键文件全部 `ls`/`test -f` 验证存在
  - `src/ptxsim/instructions/wmma.cpp`（handler 实现）
  - `tests/unit/ptx/test_tcgen05_ld_st.cpp`
  - `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp`
  - `tests/e2e/kernel/test_blackwell_gemm.cu`
  - `docs/adr/ADR-0016-blackwell-only-tcgen05.md`
  - `docs/dev-process/post-tcgen05-roadmap.md`
  - `docs/audits/debt-audit-2026-07-02.md`
- [x] **实证 3 — env.sh 自动检测验证**: `grep "NVCC_PATH=\$(which nvcc)" env.sh` 返回实际行（Decision 3 措辞依据）
- [x] **决策**: 本 change 是 §6 "stale README" 第二个真实案例（继 `fix-cvt-strategy-actual-split` 之后），已通过强制实证避免 §20 的 5 项 MUST-RESOLVE（scope 错误 / 接口矛盾 / 测试虚构 / worktree 不存在 / 路径错误）

### Checklist I (重大功能交付 checklist) — §21 已沉淀（commit `9ff88c0`）

> **已沉淀**: Phase 4.5 postmortem 已在 commit `9ff88c0` 沉淀 §21 + Checklist I（"重大功能交付 = 代码 + 单元测试 + e2e + README 同步"）。本 change 作为 §21 案例来源。

- [x] 任何 `feat-*/implement-*` change 归档前必须验证根 README.md "状态" / "已知限制" 章节是否仍准确（Checklist I 已沉淀到 SKILL.md）
- [x] 任何 archive commit 必须包含 README 同步（如适用）（§21 实战 checklist 已沉淀）
