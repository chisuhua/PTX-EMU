## Context

Change-1 (archived `2026-07-06-implement-tcgen05-syntax-ir`) 建立独立 tcgen05 命名空间后,Metis pre-implementation review 发现 5 个 MUST-RESOLVE 项,其中 **MR-3(grammar LL(*) 冲突)** 和 **MR-4(旧测试迁移)** 推迟到本 change。

当前状态(`test_all_ptx.sh` 输出 33/36 PASS):
- `tests/ptx/tcgen05_alloc.ptx`:FAIL — `mismatched input '.all' expecting ':'`
- `tests/ptx/tcgen05_mma.ptx`:FAIL — 同样 LL(*) 冲突
- 2 个旧 `tests/integration/tcgen05/test_tcgen05_*.cpp`:仍用 `S_WMMA`/`makeWmmaInstr`/`WmmaType`
- 4 个 `Q_TCGEN05_LD/ST/COMMIT/WAIT` stub qualifiers:仍存在(因 wmma.cpp 依赖)

目标状态(本 change 完成后):
- `test_all_ptx.sh` 36/36 PASS(2 个旧 tcgen05 fixtures 通过 grammar 修复,10 个新 fixtures 通过,pre-existing `atom_cas_basic.ptx` 仍 fail)
- 2 个旧测试添加 `makeTcgen05Instr` 别名共存,旧 `makeWmmaInstr` 路径保留(behavior 不变)
- 4 个 `Q_TCGEN05_*` stub **保留**(推迟到 implement-tcgen05-handlers-core,见 D4)
- `makeTcgen05Instr` factory 修复 B2 bug(op_kind → 11 个 `S_TCGEN05_*` 正确映射)

## Goals / Non-Goals

**Goals**: 修复 grammar LL(*) + 补全 10 个新 .ptx fixtures + 2 个旧测试加 Tcgen05 别名(additive, 旧路径不变) + B2 factory fix(op_kind→StatementType switch)。

**Non-Goals**: 不实施任何 handler(change-3b)、不修改 wmma.cpp 中 5 个 execute_tcgen05_(change-3b)、不实现 cp.async.bulk.tensor(独立 follow-up)。

## Decisions

### D1: Grammar 修复策略 — predicate + alternatives 重组

**选项 A**: 在 `tcgen05Qual` 规则添加 ANTLR `{<pred>}?` 谓词(per-context guard) — 拒绝,ANTLR 4.11.1 predicate 性能差且增加维护负担

**选项 B**: 拆分 `tcgen05Qual` 为 `tcgen05PreQualifier`(无歧义子集) + `tcgen05PostQualifier`(歧义子集) — 拒绝,过度拆分

**选项 C(采纳)**: 调整 `tcgen05Qual` alternatives 顺序,把高歧义子集(`.cta_group::N`、`.kind::X`)前置 + 添加 `typeSpecifier?` 显式类型后缀 + 调整 lexer token 顺序(最长匹配优先) — 最小侵入,保留 X-Macro 结构

**理由**: ANTLR 的 LL(*) 冲突通常可通过 alternatives 排序 + 消除 ambiguity 解决(per `ptx-grammar-modification` skill 经验)。无需 predicate 或规则拆分。

### D2: 10 新 fixtures 的来源 — 基于 PTX ISA 8.6 §9.7.16 规范(skip tcgen05.mma.ws)

**采纳**: 在 §3.1 创建 10 个新 .ptx fixture(原计划 11 个,移除 `tcgen05_mma_ws.ptx`,因语法无 `MMA_WS` sub-op,`.ws` 是 TCGEN_WS qualifier token;fixture 会解析为 Tcgen05OpKind::MMA + qualifier 而非 MMA_WS)。共 12 个 .ptx 文件跑 test_all_ptx.sh(2 个 change-1 既存 + 10 个本 change 新增)。

**理由**: 与 change-1 已有 2 个 fixture 的风格一致(hand-written PTX snippets,无 cuobjdump)。

### D3: 旧测试迁移策略 — ADDITIVE 编译期别名共存(不插入执行向量)

**采纳**: 在 2 个旧测试 `tests/integration/tcgen05/test_tcgen05_*.cpp` 中**保留** `makeWmmaInstr(WmmaType::WMMA_*, quals, ...)` 调用(旧路径,与 wmma.cpp 中 `processWmmaOperation` qualifier 派发匹配,仍加入执行向量),同时**添加** `makeTcgen05Instr(Tcgen05OpKind::*, quals, ...)` 调用作为**编译期验证**(验证 factory 编译通过 + `static_assert(statement.stmt_type == S_TCGEN05_*)` 验证 B2 fix 正确)。新调用**不加入 step_warp 执行向量**,因为本 change 无 `S_TCGEN05_*` handler — `get_handler()` 返回 nullptr 会导致运行时崩溃。编译期验证足以确认 factory 修复正确,运行时验证推迟到 `implement-tcgen05-handlers-core`。

**实现模式**:
```cpp
// 旧路径: 加入执行向量,运行时通过 WmmaHandler 执行(ctest PASS)
stmts[MMA_PC] = makeWmmaInstr(WmmaType::WMMA_MMA, quals, {});

// 新路径: 编译期别名验证 — 不加执行向量,仅验证 factory + type 正确
auto tcgen05_alias = makeTcgen05Instr(Tcgen05OpKind::MMA, quals, {});
static_assert(std::is_same_v<decltype(tcgen05_alias), StatementContext>);
// NOTE: 本行不移除 — implement-tcgen05-handlers-core 注册 handler 后,
//       将 tcgen05_alias 替换 stmts[MMA_PC] 旧路径即可完成运行时迁移
```

**理由**:
1. 旧路径(走 `S_WMMA` + `WmmaHandler::processWmmaOperation`)在 wmma.cpp 仍依赖 Q_TCGEN05_* qualifier 的当下,完全 work 且 ctest PASS。
2. 新路径(`S_TCGEN05_*` + 未来 `Tcgen05PipelineHandler`)在本 change 中**仅做语法准备**,不实际 dispatch 到 handler → 必须等到 implement-tcgen05-handlers-core 注册 handler 后才能完整跑通。
3. additive 双轨避免 "behavior 变化" 风险,符合 lessons-learned §3 "minimum scope per phase commit"。

### D4: Q_TCGEN05_* stub 保留 — wmma.cpp 仍有 8 引用 + 2 测试文件有 6 引用

**采纳**: 本 change 不删除 4 个 Q_TCGEN05_* stub。wmma.cpp:29-59 中 5 个
is_tcgen05_*() helper 函数通过 qualifier 匹配做 dispatch 路由,删除 stub
直接破坏编译。2 个测试文件(test_tcgen05_ld_st.cpp:36,42 + 
test_tcgen05_ld_st_commit.cpp:66,83,92,101) 也通过 Q_TCGEN05_* 构造
qualifier 列表。那 6 个 Q_TCGEN05_* 引用合计 11 处依赖。

**推迟到 implement-tcgen05-handlers-core**: 该 change 创建独立 tcgen05.cpp
handler、移除 wmma.cpp 中 execute_tcgen05_* 函数,届时 Q_TCGEN05_* stub 自然
变为 dead code 可安全删除。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| **R1**: Grammar 修复后 ANTLR 生成新错误(lexer token 顺序) | Phase 1 跑 `cmake --build build --target GenerateParser` 立即验证 |
| **R2**: 12 fixtures 中某些无法 100% 匹配 PTX ISA 规范(罕见语法) | Phase 2 跑 `test_all_ptx.sh`,失败的 fixture 单独调试 |
| **R3**: 旧测试 additive 迁移后 behavior 变化(因 wmma.cpp 仍存在) | Phase 3b 跑 `ctest -R tcgen05 -V` 验证 behavior 一致 |
| **R4**: statement_factory.h B2 修复后 switch 不匹配 Tcgen05OpKind | 跑 `cmake --build build` 验证编译 |

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

1. 创建 10 个新 .ptx fixtures(基于 PTX ISA §9.7.16 规范)
2. 跑 `./tests/ptx/test_all_ptx.sh` 验证 12/12 PASS
3. commit

### Phase 3: compile-time alias verification + B2 factory fix(2 commits, 4a + 4b)

注: stub 删除(`Q_TCGEN05_*` 4 个)推迟到 implement-tcgen05-handlers-core,不在本 change 范围。task 表的 §4 对应 4b,§6.5 对应 4a。

1. commit 4a(B2 factory fix):编辑 `include/ptx_ir/statement_factory.h:278-289`,将硬编码 `S_TCGEN05_MMA` 替换为 `switch(op_kind)` 映射到全部 11 个 `S_TCGEN05_*` enum 值(详见 tasks.md §4)
2. commit 4b(compile-time alias verification):编辑 `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp` + `test_tcgen05_ld_st_commit.cpp`,在保留旧 `makeWmmaInstr` 调用(仍加入执行向量)的同时添加 `makeTcgen05Instr` 编译期别名(**不加入执行向量** — 无 handler 会崩溃;仅做编译验证 + `static_assert` type check)(详见 tasks.md §5)
3. 跑 `cmake --build build` 验证编译
4. 跑 `ctest -R tcgen05 -V` 验证 behavior(旧路径执行,ctest PASS)
5. commit

### Phase 4: Archive(1 commit,per Checklist G)

1. 跑 `openspec archive fix-tcgen05-grammar-mr3 --yes`
2. 跑 `ctest --output-on-failure` + `test_all_ptx.sh` 最终验证
3. commit archive 目录

### 回退策略

- 任意 commit 失败:`git revert HEAD` 回到上一个 good state
- 整体失败:`git reset --hard <pre-change-sha>`(丢失本 change,需备份 working tree)

## Open Questions

- **Q1**: 12 fixtures 是否覆盖所有 12 个 Blackwell tcgen05 指令族?
  - 答: 是(12 fixtures 覆盖 11 sub-op 族;`tcgen05_cp` + `tcgen05_cp_multicast` 是 1 族 cp 的 2 个变体;`.ws` 是 qualifier,非 sub-op)
- **Q2**: 是否需要 E2E 测试?
  - 答: 不需要(本 change 仅 grammar + tests,handler 实现在 change-3b)
