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

- [ ] 0.1 验证 OpenSpec change 目录结构完整
  ```bash
  ls openspec/changes/sync-readme-after-tcgen05/
  # 期望: .openspec.yaml, proposal.md, design.md, tasks.md, specs/sync-readme-tcgen05/spec.md
  ```
- [ ] 0.2 在 main 上创建工作分支
  ```bash
  git checkout -b docs/sync-readme-after-tcgen05
  ```
- [ ] 0.3 git-tracked artifacts
  ```bash
  git add openspec/changes/sync-readme-after-tcgen05/
  git status  # 应显示 5+ 个新文件
  ```
- [ ] 0.4 commit artifacts（**独立 commit**）
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
- [ ] 0.5 验证 artifacts 已 tracked
  ```bash
  git ls-files openspec/changes/sync-readme-after-tcgen05/
  # 期望输出 5+ 个文件路径（不应为空）
  ```

---

## Phase 1: README.md "状态" 行更新（Fix #1）— 独立 commit

**目标**: 更新 README.md line 3

- [ ] 1.1 修改 `README.md` line 3:
  ```diff
  - > **状态**：SIMT v2.0 (Phase 10 进行中)
  + > **状态**：SIMT v2.0 完成；Blackwell tcgen05 完整实施；H5 规划中
  ```
- [ ] 1.2 验证: `grep "状态" README.md` 返回新文案
- [ ] 1.3 commit:
  ```bash
  git add README.md
  git commit -m "docs(readme): update status line after tcgen05 Phase 1-3 (Fix #1)

  Phase 10 已完成；Blackwell tcgen05 完整实施（archive/2026-07-04）；
  H5 规划参考 docs/dev-process/post-tcgen05-roadmap.md."
  ```

---

## Phase 2: README.md "已知限制" + "PTX 指令覆盖" + "CUDA Toolkit" 更新（Fix #2）— 独立 commit

**目标**: 更新 README.md line 16 + line 49-52

- [ ] 2.1 添加 "已实现功能" 章节（line 49 之前）
  ```markdown
  ## 已实现功能

  - **Blackwell tcgen05**：完整实现 `.mma` / `.ld` / `.st` / `.commit` / `.wait`（commit `4151268` Fix #14）— 详见 [docs/adr/0016-blackwell-only-tcgen05.md](./docs/adr/0016-blackwell-only-tcgen05.md)
  - **TMA descriptors**：异步拷贝 descriptor 解析（commit `ad527f5` Fix #5）
  - **TMEM**：per-CTA Tensor Memory（commit `758edb0` Fix #6）
  - **Cluster arrive/wait**：分布式 shared memory 同步（commit `e513235` Fix #7）
  - **TcQueue**：commit-group + wait-aware scheduling（commit `c0fa43f` Fix #8）
  ```
- [ ] 2.2 修改 "已知限制" 章节（line 49-53）:
  ```diff
  ## 已知限制

  -- **PTX 指令覆盖**：核心 ISA ~67%（详见审计 §3）
  -- **WMMA / Tensor Core**：是 stub
  -- **ANTLR 版本**：4.11.1 完全 vendored
  -- **CUDA Toolkit**：11.4.4 测试通过
  +- **PTX 指令覆盖**：参考 [docs/audits/debt-audit-2026-07-02.md](./docs/audits/debt-audit-2026-07-02.md) 自动统计（避免硬编码）
  +- **pre-Blackwell tcgen05**：永久抛 `UnsupportedInstructionException`（c5 Fix #1 + [ADR-0016](./docs/adr/0016-blackwell-only-tcgen05.md)）
  +- **ANTLR 版本**：4.11.1 完全 vendored
  +- **CUDA Toolkit**：环境自适应（`env.sh` 自动检测 `$(which nvcc)`）
  ```
- [ ] 2.3 验证:
  ```bash
  grep -n "WMMA" README.md  # 应为空
  grep -n "tcgen05" README.md  # 应有 3 处引用
  grep -n "HUNDRED\|11.4" README.md  # 应为空（移除硬编码）
  ```
- [ ] 2.4 commit:
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

- [ ] 3.1 在 "文档导航" 表格添加新行（line 33 之后）:
  ```markdown
  | Blackwell tcgen05 架构 | [docs/adr/0016-blackwell-only-tcgen05.md](./docs/adr/0016-blackwell-only-tcgen05.md) |
  | tcgen05 实施 roadmap | [docs/dev-process/post-tcgen05-roadmap.md](./docs/dev-process/post-tcgen05-roadmap.md) |
  ```
- [ ] 3.2 验证:
  ```bash
  grep -n "0016-blackwell\|post-tcgen05" README.md  # 应返回 2 行
  test -f docs/adr/0016-blackwell-only-tcgen05.md && echo "ADR-0016 exists" || echo "MISSING"
  test -f docs/dev-process/post-tcgen05-roadmap.md && echo "roadmap exists" || echo "MISSING"
  ```
- [ ] 3.3 commit:
  ```bash
  git add README.md
  git commit -m "docs(readme): link Blackwell tcgen05 ADR + roadmap (Fix #3)

  Add references to:
  - docs/adr/0016-blackwell-only-tcgen05.md (architecture decision)
  - docs/dev-process/post-tcgen05-roadmap.md (H5 planning)
  - docs/dev-process/lessons-learned.md (§19 cross-module state translation)"
  ```

---

## Phase 4: 验证 + 归档

- [ ] 4.1 行数对比
  ```bash
  git diff HEAD~3 HEAD -- README.md | grep -c "^+"  # 修改行数（应 15-20 行）
  git diff HEAD~3 HEAD -- README.md | grep -c "^-"  # 删除行数（应 4-6 行）
  ```
- [ ] 4.2 链接可达性
  ```bash
  grep -oP '\./docs/[^)]*' README.md | while read link; do
    test -f "$link" && echo "OK: $link" || echo "BROKEN: $link"
  done
  # 期望: 所有路径 OK（无 BROKEN）
  ```
- [ ] 4.3 grep stale 检查
  ```bash
  grep -n "WMMA / Tensor Core\|是 stub\|67%\|11.4" README.md
  # 期望: 无任何匹配（全部清除）
  ```
- [ ] 4.4 归档 change（Checklist G）
  ```bash
  git checkout main
  git merge --no-ff docs/sync-readme-after-tcgen05
  mkdir -p openspec/changes/archive/2026-07-05-sync-readme-after-tcgen05/
  git mv openspec/changes/sync-readme-after-tcgen05/{.,}* openspec/changes/archive/2026-07-05-sync-readme-after-tcgen05/ 2>/dev/null || true
  # 移动隐藏文件
  shopt -s dotglob
  mv openspec/changes/sync-readme-after-tcgen05/* openspec/changes/archive/2026-07-05-sync-readme-after-tcgen05/
  git add openspec/changes/
  git commit -m "chore(openspec): archive sync-readme-after-tcgen05 (Checklist G)

  Ref: archive/2026-07-04-implement-wmma-tensor-core-tcgen05/
  Ref: archive/2026-07-05-fix-cvt-strategy-actual-split/ (template)

  Lessons-learned §6: post-archive hygiene — README sync via new change"
  ```

---

## ✅ Lessons-learned Checklist 集成

### Checklist D (Commit 前)
- [x] AGENTS.md 不需同步（本 change 不改 sub-AGENTS.md）
- [x] ADR 不需追加（已存在 ADR-0016，本 change 仅引用）
- [x] OpenSpec tasks.md 已更新（本文档）
- [x] commit message 列出 Fix #1-#3

### Checklist E (OpenSpec 实施后)
- [x] 所有 artifacts (proposal.md / design.md / tasks.md / spec.md / .openspec.yaml) **git-tracked**
- [x] Phase 0 强制 — artifacts commit FIRST
- [x] 每个 commit 独立可 revert（Fix #1-#3 各可单独 revert）
- [x] 实施 commits 完成后立即 git-tracked（避免 working tree 遗漏）
- [x] 归档前 grep 验证 artifacts 与代码一致

### Checklist G (Lifecycle)
- [x] 不可 amend 已归档 change（遵守）
- [x] 新建 sync-readme-after-tcgen05 + Ref: 链接（遵守）

### Checklist F (Debt audit)
- [x] 引用 commit hash（`4151268`/`ad527f5`/`758edb0`/`e513235`/`c0fa43f`）而非文件路径
- [x] 标注"基于 HEAD `5321e95`"（明确审计基准）
