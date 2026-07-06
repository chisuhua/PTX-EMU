## Context

Change-1 (archived `2026-07-06-implement-tcgen05-syntax-ir`) 建立独立 tcgen05 命名空间后,Metis pre-implementation review 发现 5 个 MUST-RESOLVE 项,其中 **MR-3(grammar LL(*) 冲突)** 和 **MR-4(旧测试迁移)** 推迟到本 change。

当前状态(`test_all_ptx.sh` 输出 33/36 PASS):
- `tests/ptx/tcgen05_alloc.ptx`:FAIL — `mismatched input '.all' expecting ':'`
- `tests/ptx/tcgen05_mma.ptx`:FAIL — 同样 LL(*) 冲突
- 2 个旧 `tests/integration/tcgen05/test_tcgen05_*.cpp`:仍用 `S_WMMA`/`makeWmmaInstr`/`WmmaType`
- 4 个 `Q_TCGEN05_LD/ST/COMMIT/WAIT` stub qualifiers:仍存在(因 wmma.cpp 依赖)

目标状态(本 change 完成后):
- `test_all_ptx.sh` 35/36(2 个 tcgen05 fixtures PASS,仅 pre-existing `atom_cas_basic.ptx` 失败)
- 2 个旧测试改用 `S_TCGEN05_*` + `makeTcgen05Instr` + `Tcgen05OpKind`
- 4 个 `Q_TCGEN05_*` stub 删除

## Goals / Non-Goals

**Goals**: 修复 grammar LL(*) + 补全 11 .ptx fixtures + 迁移 2 个旧测试 + 删除 4 个 stub。

**Non-Goals**: 不实施任何 handler(change-3b)、不修改 wmma.cpp 中 5 个 execute_tcgen05_(change-3b)、不实现 cp.async.bulk.tensor(独立 follow-up)。

## Decisions

### D1: Grammar 修复策略 — predicate + alternatives 重组

**选项 A**: 在 `tcgen05Qual` 规则添加 ANTLR `{<pred>}?` 谓词(per-context guard) — 拒绝,ANTLR 4.11.1 predicate 性能差且增加维护负担

**选项 B**: 拆分 `tcgen05Qual` 为 `tcgen05PreQualifier`(无歧义子集) + `tcgen05PostQualifier`(歧义子集) — 拒绝,过度拆分

**选项 C(采纳)**: 调整 `tcgen05Qual` alternatives 顺序,把高歧义子集(`.cta_group::N`、`.kind::X`)前置 + 添加 `typeSpecifier?` 显式类型后缀 + 调整 lexer token 顺序(最长匹配优先) — 最小侵入,保留 X-Macro 结构

**理由**: ANTLR 的 LL(*) 冲突通常可通过 alternatives 排序 + 消除 ambiguity 解决(per `ptx-grammar-modification` skill 经验)。无需 predicate 或规则拆分。

### D2: 13 fixtures 的来源 — 基于 PTX ISA 8.6 §9.7.16 规范

**采纳**: 13 个 .ptx fixtures 全部根据 NVIDIA PTX ISA 8.6 §9.7.16 规范手写,**不依赖 cuobjdump**(无 GPU 访问)。

**理由**: 与 change-1 已有 2 个 fixture 的风格一致(hand-written PTX snippets,无 cuobjdump)。

### D3: 旧测试迁移策略 — 同语义 IR 重写

**采纳**: 旧测试的 `S_WMMA`+`makeWmmaInstr(WmmaType::WMMA_MMA, ...)` 等价于新 `S_TCGEN05_MMA`+`makeTcgen05Instr(Tcgen05OpKind::MMA, ...)`,**直接替换**。

**拒绝**: 保留旧测试的 wmma 路径(违反用户 "避免 WMMA/wmma 名字" 要求)。

### D4: Q_TCGEN05_* stub 删除时机 — 在 wmma.cpp 不再依赖后

**采纳**: 本 change **Phase 3 旧测试迁移 commit 中**删除 4 个 stub,因为 wmma.cpp 不再用(`S_WMMA` 未删除但测试不直接调 wmma.cpp)。

**理由**: 旧测试不调 wmma.cpp,只调 `makeTcgen05Instr` → 删除 stub 安全。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| **R1**: Grammar 修复后 ANTLR 生成新错误(lexer token 顺序) | Phase 1 跑 `cmake --build build --target GenerateParser` 立即验证 |
| **R2**: 13 fixtures 中某些无法 100% 匹配 PTX ISA 规范(罕见语法) | Phase 2 跑 `test_all_ptx.sh`,失败的 fixture 单独调试 |
| **R3**: 旧测试迁移后 behavior 变化(因 wmma.cpp 仍存在) | Phase 3 跑 `ctest -R tcgen05 -V` 验证 behavior 一致 |
| **R4**: 删除 4 个 Q_TCGEN05_* stub 后,wmma.cpp 编译失败 | Phase 3 跑 `cmake --build build` 验证 |

## Migration Plan

### Phase 1: Grammar 修复(1 commit)

1. 读 `src/grammar/ptxInstructions.g4` 当前 `tcgen05Qual` 规则
2. 调整 alternatives 顺序(高歧义前置)
3. 调整 `src/grammar/ptxLexer.g4` token 顺序(若需要)
4. 跑 `cmake --build build --target GenerateParser` 验证
5. 跑 `./tests/ptx/test_all_ptx.sh` 验证 2 个现有 fixture PASS
6. 跑 `cmake --build build && ctest -L "unit|integration" --output-on-failure` 验证零回归
7. commit

### Phase 2: 补全 .ptx fixtures(1 commit)

1. 创建 11 个新 .ptx fixtures(基于 PTX ISA §9.7.16 规范)
2. 跑 `./tests/ptx/test_all_ptx.sh` 验证 13/13 PASS
3. commit

### Phase 3: 旧测试迁移 + stub 删除(1 commit)

1. 编辑 `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp`(`S_WMMA` → `S_TCGEN05_MMA` 等)
2. 编辑 `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp`(5 处迁移)
3. 删除 `include/ptx_ir/ptx_qualifier.def` 4 个 `Q_TCGEN05_*` stub
4. 跑 `cmake --build build` 验证编译
5. 跑 `ctest -R tcgen05 -V` 验证 behavior
6. commit

### Phase 4: Archive(1 commit,per Checklist G)

1. 跑 `openspec archive fix-tcgen05-grammar-mr3 --yes`
2. 跑 `ctest --output-on-failure` + `test_all_ptx.sh` 最终验证
3. commit archive 目录

### 回退策略

- 任意 commit 失败:`git revert HEAD` 回到上一个 good state
- 整体失败:`git reset --hard <pre-change-sha>`(丢失本 change,需备份 working tree)

## Open Questions

- **Q1**: 13 fixtures 是否覆盖所有 12 个 Blackwell tcgen05 指令族?
  - 答: 是(13 fixtures 覆盖 12 sub-op 族,`tcgen05_cp` + `tcgen05_cp_multicast` 是 1 族 cp 的 2 个变体)
- **Q2**: 是否需要 E2E 测试?
  - 答: 不需要(本 change 仅 grammar + tests,handler 实现在 change-3b)
