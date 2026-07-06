# Tasks: Fix tcgen05 Grammar LL(*) Conflict + Migrate Old Tests

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 3 specs in [specs/](specs/)
> **范围**: 4 atomic commits,每步独立可 revert(per `ptx-lessons-learned` §3)
> **Lessons-learned**: Checklist E(artifacts tracked) + Checklist G(OpenSpec lifecycle)

## 0. Pre-Implementation Review(强制 FIRST)

> **来源**: `ptx-lessons-learned` §7 + Checklist H — 实施 OpenSpec change 前必跑

- [ ] 0.1 跑 Metis pre-implementation review 子代理,验证:
  - [ ] 0.1.1 `wc -l src/grammar/ptxInstructions.g4 src/grammar/ptxLexer.g4` 数字
  - [ ] 0.1.2 验证 `tcgen05Qual` 规则 16+ alternations(per change-1 design.md)
  - [ ] 0.1.3 验证 `tests/ptx/tcgen05_alloc.ptx tcgen05_mma.ptx` 当前 fail(`mismatched input '.all'`)
  - [ ] 0.1.4 验证 2 个旧测试引用 `S_WMMA`/`makeWmmaInstr`/`WmmaType`
  - [ ] 0.1.5 验证 `Q_TCGEN05_*` 4 stub 位置(`include/ptx_ir/ptx_qualifier.def:193-197`)
  - [ ] 0.1.6 跑 `./tests/ptx/test_all_ptx.sh` 记录 baseline(2 fail)
  - [ ] 0.1.7 Metis 输出 `GO` 或 `⚠️ CONDITIONAL` 后继续

- [ ] 0.2 基线 worktree(per `ptx-lessons-learned` §4):
  - [ ] 0.2.1 `git worktree add .worktrees/baseline-grammar-fix -b feat/fix-tcgen05-grammar-mr3 main`
  - [ ] 0.2.2 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
  - [ ] 0.2.3 `cd build && ctest --output-on-failure` 验证 baseline

## 1. Artifacts Tracking(commit 1,per `ptx-lessons-learned` §6 + Checklist E)

- [ ] 1.1 在 main 上创建分支:`git checkout -b feat/fix-tcgen05-grammar-mr3`
- [ ] 1.2 `git add openspec/changes/fix-tcgen05-grammar-mr3/{proposal.md,design.md,tasks.md,specs/}`
- [ ] 1.3 `git status` 验证 5 个文件全部 staged(proposal + design + tasks + 3 specs)
- [ ] 1.4 `git ls-files openspec/changes/fix-tcgen05-grammar-mr3/` 验证非空
- [ ] 1.5 `git commit -m "docs(openspec): add fix-tcgen05-grammar-mr3 artifacts (ADR-0016)"`
- [ ] 1.6 NOTE:此 commit 独立可 revert(删除 openspec/changes/fix-tcgen05-grammar-mr3/)

## 2. Phase 1: Grammar 修复(commit 2,atomic)

> **MUST**: 修复后立即验证 grammar 编译 + 2 fixture PASS

### 2.1 诊断 LL(*) 冲突根因

- [ ] 2.1.1 读 `src/grammar/ptxInstructions.g4` 当前 `tcgen05Qual` 规则(16+ alternatives)
- [ ] 2.1.2 跑 `antlr4 src/grammar/ptxParser.g4` 单独编译,查看 ANTLR 警告/错误
- [ ] 2.1.3 记录根因(per `ptx-grammar-modification` skill)

### 2.2 修复 grammar

- [ ] 2.2.1 调整 `tcgen05Qual` alternatives 顺序(高歧义子集前置:`KIND COLONCOLON` `TCGEN_CTA_GROUP COLONCOLON` `SHARED COLONCOLON` 等)
- [ ] 2.2.2 调整 `typeSpecifier` 位置:从 `tcgen05Qual` 移除,移至主规则 `tcgen05Inst` 显式位置
- [ ] 2.2.3 调整 `src/grammar/ptxLexer.g4` token 顺序(若需要,优先长匹配)

### 2.3 Build + 验证

- [ ] 2.3.1 `cmake --build build --target GenerateParser` 验证 ANTLR 重新生成
- [ ] 2.3.2 `cmake --build build` 验证编译
- [ ] 2.3.3 `./tests/ptx/test_all_ptx.sh` 验证 2 个现有 fixture(`tcgen05_alloc.ptx` + `tcgen05_mma.ptx`)PASS
- [ ] 2.3.4 `ctest -L "unit|integration" --output-on-failure` 验证零回归
- [ ] 2.3.5 `git diff --stat` 验证 diff 在 `src/grammar/` 范围内(不触及 IR 或 handler)

### 2.4 Commit

- [ ] 2.4.1 `git add src/grammar/`
- [ ] 2.4.2 `git commit -m "fix(grammar): resolve tcgen05 LL(*) prediction conflict (ADR-0016, Change-1 MR-3)"`
- [ ] 2.4.3 验证:commit 独立可 revert(`git revert HEAD` 后 grammar baseline 仍工作)

## 3. Phase 2: 补全 .ptx fixtures(commit 3,atomic)

> **MUST**: 11 个新 fixture 必须全部通过 `test_all_ptx.sh`

### 3.1 创建 11 个新 fixtures(基于 PTX ISA 8.6 §9.7.16)

- [ ] 3.1.1 创建 `tests/ptx/tcgen05_dealloc.ptx`
- [ ] 3.1.2 创建 `tests/ptx/tcgen05_relinquish.ptx`
- [ ] 3.1.3 创建 `tests/ptx/tcgen05_ld.ptx`
- [ ] 3.1.4 创建 `tests/ptx/tcgen05_st.ptx`
- [ ] 3.1.5 创建 `tests/ptx/tcgen05_cp.ptx`
- [ ] 3.1.6 创建 `tests/ptx/tcgen05_cp_multicast.ptx`
- [ ] 3.1.7 创建 `tests/ptx/tcgen05_mma_block_scale.ptx`
- [ ] 3.1.8 创建 `tests/ptx/tcgen05_mma_ws.ptx`
- [ ] 3.1.9 创建 `tests/ptx/tcgen05_commit.ptx`
- [ ] 3.1.10 创建 `tests/ptx/tcgen05_wait.ptx`
- [ ] 3.1.11 创建 `tests/ptx/tcgen05_fence.ptx`

### 3.2 Build + 验证

- [ ] 3.2.1 `cmake --build build` 验证编译
- [ ] 3.2.2 `./tests/ptx/test_all_ptx.sh` 验证 13/13 PASS(2 现有 + 11 新)
- [ ] 3.2.3 `ls tests/ptx/tcgen05_*.ptx | wc -l` 应输出 13

### 3.3 Commit

- [ ] 3.3.1 `git add tests/ptx/`
- [ ] 3.3.2 `git commit -m "test(ptx): add 11 tcgen05.* PTX fixtures (ADR-0016)"`
- [ ] 3.3.3 验证:commit 独立可 revert(`git revert HEAD` 后 fixtures 消失但 grammar 仍工作)

## 4. Phase 3: 旧测试迁移 + stub 删除(commit 4,atomic)

> **MUST**: 旧测试仍 PASS(behavior 不变,仅 IR 命名空间更新)

### 4.1 迁移 test_tcgen05_mma_sync.cpp

- [ ] 4.1.1 读 `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp`
- [ ] 4.1.2 替换 `S_WMMA` → `S_TCGEN05_MMA`
- [ ] 4.1.3 替换 `makeWmmaInstr(WmmaType::WMMA_MMA, ...)` → `makeTcgen05Instr(Tcgen05OpKind::MMA, ...)`
- [ ] 4.1.4 替换 `WmmaType::WMMA_MMA` → `Tcgen05OpKind::MMA`
- [ ] 4.1.5 替换 `Qual::WMMA_INSTR` 或 `S_WMMA` 的 stmt type 检查(若有)

### 4.2 迁移 test_tcgen05_ld_st_commit.cpp

- [ ] 4.2.1 读 `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp`
- [ ] 4.2.2 替换 5 处 `S_WMMA` → 对应 `S_TCGEN05_MMA/LD/ST/COMMIT/WAIT`
- [ ] 4.2.3 替换 5 处 `makeWmmaInstr` → `makeTcgen05Instr`
- [ ] 4.2.4 替换 5 处 `WmmaType::WMMA_*` → `Tcgen05OpKind::*`

### 4.3 删除 4 个 Q_TCGEN05_* stub

- [ ] 4.3.1 读 `include/ptx_ir/ptx_qualifier.def:193-197`
- [ ] 4.3.2 删除 4 行:
  - `X(Q_TCGEN05_LD, TCGEN05_LD, ".tcgen05_ld")`
  - `X(Q_TCGEN05_ST, TCGEN05_ST, ".tcgen05_st")`
  - `X(Q_TCGEN05_COMMIT, TCGEN05_COMMIT, ".tcgen05_commit")`
  - `X(Q_TCGEN05_WAIT, TCGEN05_WAIT, ".tcgen05_wait")`
- [ ] 4.3.3 验证:`grep "Q_TCGEN05" include/ptx_ir/ptx_qualifier.def` 仅 0 行(其他 Q_TCGEN_* 保留)

### 4.4 Build + 验证

- [ ] 4.4.1 `cmake --build build` 验证编译
- [ ] 4.4.2 `ctest -R tcgen05 -V` 验证 2 个旧测试 PASS
- [ ] 4.4.3 `ctest -L "unit|integration" --output-on-failure` 验证零回归
- [ ] 4.4.4 `./tests/ptx/test_all_ptx.sh` 验证 13/13 fixtures 仍 PASS

### 4.5 Commit

- [ ] 4.5.1 `git add tests/integration/tcgen05/ include/ptx_ir/ptx_qualifier.def`
- [ ] 4.5.2 `git commit -m "test(refactor): migrate tcgen05 old tests to S_TCGEN05_* namespace (ADR-0016, Change-1 MR-4)"`
- [ ] 4.5.3 验证:commit 独立可 revert(`git revert HEAD` 后旧测试仍工作)

## 5. Phase 4: Archive(commit 5,per Checklist G)

- [ ] 5.1 跑 `openspec archive fix-tcgen05-grammar-mr3 --yes`
- [ ] 5.2 跑 `ctest --output-on-failure` + `test_all_ptx.sh` 最终验证
- [ ] 5.3 跑 `openspec status` 确认 change 已 archive
- [ ] 5.4 `git add openspec/changes/archive/`
- [ ] 5.5 `git commit -m "chore(openspec): archive fix-tcgen05-grammar-mr3 (ADR-0016)"`
- [ ] 5.6 验证:`git log --oneline | head -7` 显示 5 个 atomic commits

## 6. Final Validation

- [ ] 6.1 `./scripts/sanity.sh` 全量验证
- [ ] 6.2 `./scripts/sanity.sh --ptx` PTX 语法验证
- [ ] 6.3 `cd build && ctest --output-on-failure` 全量测试
- [ ] 6.4 验证:`git log --oneline feat/fix-tcgen05-grammar-mr3` 显示 5 个 atomic commits

## Risks & Mitigations Recap

| Risk | Mitigation in Tasks |
|------|---------------------|
| **R1**: Grammar 修复后 ANTLR 错误 | Task 2.3.1-2.3.2 立即验证 |
| **R2**: 13 fixtures 中某些无法匹配 PTX 规范 | Task 3.2.2 跑 test_all_ptx.sh,失败单独调试 |
| **R3**: 旧测试迁移 behavior 变化 | Task 4.4.2 跑 ctest -R tcgen05 -V 验证 |
| **R4**: 删除 stub 后 wmma.cpp 编译失败 | Task 4.4.1 立即验证 |

## Out-of-Scope Reminder(per [proposal.md](proposal.md))

- ❌ 不实施 tcgen05 handler(change-3b scope)
- ❌ 不修改 wmma.cpp 中 5 个 execute_tcgen05_(change-3b scope)
- ❌ 不实现 cp.async.bulk.tensor(独立 follow-up)
- ❌ 不实现 cta_group::2 distributed_smem(独立 follow-up)
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8
