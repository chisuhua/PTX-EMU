# Tasks: Blackwell tcgen05 独立命名空间(ANTLR + IR)

> **架构依据**: [ADR-0016](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) Accepted
> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + [specs/](specs/)
> **范围**: 6 atomic commits,每个独立可 revert (per `ptx-lessons-learned` §3)
> **测试覆盖**: 13 .ptx 端到端 + 5 单元 + 5 集成 + 2 迁移 = 25 个新测试目标

## 0. Pre-Implementation Review (强制 FIRST)

> **来源**: `ptx-lessons-learned` §7 + Checklist H — 实施 OpenSpec change 前必跑

- [ ] 0.1 跑 Metis pre-implementation review 子代理,验证 proposal 关键假设:
  - [ ] 0.1.1 验证 `wc -l src/grammar/*.g4 include/ptx_ir/ptx_op.def include/ptx_ir/ptx_qualifier.def` 数字与 proposal 估算一致
  - [ ] 0.1.2 验证 `grep -rn "S_WMMA" src/ include/ tests/` 列出所有引用点(本 change 需全部迁移)
  - [ ] 0.1.3 验证 `grep -rn "tcgen05" src/grammar/` 零输出(确认本 change 是从零开始)
  - [ ] 0.1.4 验证 `grep -rn "Q_TCGEN05" include/ptx_ir/ptx_qualifier.def` 列出 4 个 stub
  - [ ] 0.1.5 验证 `tests/ptx/test_all_ptx.sh` 当前 100% 通过(baseline)
  - [ ] 0.1.6 Metis 输出 `GO` 或 `⚠️ CONDITIONAL + MUST-RESOLVE` 全清后继续

- [ ] 0.2 基线 worktree 建立(per `ptx-lessons-learned` §4 + Checklist B):
  - [ ] 0.2.1 `git worktree add .worktrees/baseline-tcgen05-syntax -b feat/implement-tcgen05-syntax-ir main`
  - [ ] 0.2.2 `.worktrees/baseline-tcgen05-syntax` 下:`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
  - [ ] 0.2.3 `cd build && ctest --output-on-failure` 验证 baseline 全绿
  - [ ] 0.2.4 记录 baseline 测试数量(对比本 change 完成后)

## 1. Artifacts Tracking(commit 1,per `ptx-lessons-learned` §6 + Checklist E)

> **MUST FIRST**:避免 lessons-learned §6 模式 — artifacts working tree 遗漏

- [ ] 1.1 在 main 上创建分支:`git checkout -b feat/implement-tcgen05-syntax-ir`
- [ ] 1.2 `git add openspec/changes/implement-tcgen05-syntax-ir/{proposal.md,design.md,tasks.md,specs/tcgen05-grammar/spec.md,specs/tcgen05-ir-types/spec.md,specs/tcgen05-parse-tests/spec.md}`
- [ ] 1.3 `git status` 验证 6 个文件全部 staged
- [ ] 1.4 `git ls-files openspec/changes/implement-tcgen05-syntax-ir/` 验证非空
- [ ] 1.5 `git commit -m "docs(openspec): add implement-tcgen05-syntax-ir artifacts (ADR-0016)"`
- [ ] 1.6 NOTE:此 commit 独立可 revert(删除 openspec/changes/implement-tcgen05-syntax-ir/)

## 2. ANTLR Grammar(commit 2,atomic)

> **MUST**:每个步骤 30 分钟内完成 + 完成后立即验证

### 2.1 Lexer Tokens

- [ ] 2.1.1 编辑 `src/grammar/ptxLexer.g4`,在末尾追加 tcgen05 token block(per `design.md` Decision 1)
- [ ] 2.1.2 验证:6 主指令 + 11 sub-op + 25 qualifier tokens(per `specs/tcgen05-grammar/spec.md` Requirement 1)
- [ ] 2.1.3 重新生成 ANTLR parser:`cmake --build build --target GenerateParser`
- [ ] 2.1.4 验证:零 ANTLR 生成错误
- [ ] 2.1.5 验证:`./tests/ptx/test_all_ptx.sh` 100% 通过(无回归)

### 2.2 Parser Rules

- [ ] 2.2.1 编辑 `src/grammar/ptxInstructions.g4`:
  - [ ] 2.2.1.1 删除 `matrixInst: wmmaInst;`(line 426-428)
  - [ ] 2.2.1.2 删除 `wmmaInst / wmmaOp / wmmaLayout / wmmaShape / wmmaKind` 5 个旧规则
  - [ ] 2.2.1.3 删除 `tcgenInst: stBulkInst;`(line 463-465)
  - [ ] 2.2.1.4 新增 `tcgen05Inst` 完整语法 + 8 个子规则(per `design.md` Decision 1 + specs)
  - [ ] 2.2.1.5 新增 `matrixInst: tcgen05Inst;`(替代旧定义)
- [ ] 2.2.2 编辑 `src/grammar/ptxOperands.g4`,新增 tcgen05 qualifier 处理
- [ ] 2.2.3 重新生成:`cmake --build build --target GenerateParser`
- [ ] 2.2.4 验证:零 ANTLR 生成错误
- [ ] 2.2.5 验证:`./tests/ptx/test_all_ptx.sh` 100% 通过
- [ ] 2.2.6 NOTE:此 commit 期间 grammar 通过 + 旧测试通过,但 `S_WMMA` 仍存在(commit 3 删除)

### 2.3 Grammar Documentation

- [ ] 2.3.1 更新 `src/grammar/AGENTS.md`:
  - [ ] 2.3.1.1 添加 `tcgen05*` 规则说明(lexer + parser)
  - [ ] 2.3.1.2 删除 `wmma*` 旧规则引用
  - [ ] 2.3.1.3 添加 ANTLR 4.11.1 兼容说明
- [ ] 2.3.2 验证:`git diff src/grammar/AGENTS.md` 改动合理

### 2.4 Commit

- [ ] 2.4.1 `git add src/grammar/`
- [ ] 2.4.2 `git commit -m "feat(grammar): add TCGEN05 lexer + tcgen05 parser rules (ADR-0016)"`
- [ ] 2.4.3 验证:commit 独立可 revert(`git revert HEAD` 后 grammar baseline 仍工作)

## 3. IR 命名空间(commit 3,atomic)

> **MUST**:wmma.cpp 仍引用 `S_WMMA`,本 commit 期间 wmma.cpp 可能编译失败(预期内,commit 4 修复)

### 3.1 StatementType Enums

- [ ] 3.1.1 编辑 `include/ptx_ir/ptx_op.def`:
  - [ ] 3.1.1.1 删除 `S_WMMA` 整行(line 127)
  - [ ] 3.1.1.2 新增 11 个 `X(S_TCGEN05_*, ..., TCGEN05_INSTR, tcgen05)` 条目
- [ ] 3.1.2 验证:`grep -c "S_TCGEN05" include/ptx_ir/ptx_op.def` 应为 11

### 3.2 Qualifier Enums

- [ ] 3.2.1 编辑 `include/ptx_ir/ptx_qualifier.def`:
  - [ ] 3.2.1.1 删除 4 个 stub:`Q_TCGEN05_LD/ST/COMMIT/WAIT`(line 194-197)
  - [ ] 3.2.1.2 新增 ~25 个 `Q_*` 条目(per `specs/tcgen05-ir-types/spec.md` Requirement 4)
- [ ] 3.2.2 验证:`grep -c "Q_CTA_GROUP\|Q_KIND\|Q_F16\|Q_BF16" include/ptx_ir/ptx_qualifier.def` 应为 25+

### 3.3 Tcgen05Instr Struct + Enums

- [ ] 3.3.1 编辑 `include/ptx_ir/statement_context.h`:
  - [ ] 3.3.1.1 新增 `enum class Tcgen05OpKind { ALLOC, DEALLOC, ..., FENCE };`(11 个值)
  - [ ] 3.3.1.2 新增 `enum class Tcgen05Dtype { F16, BF16, ..., INVALID };`(10 个值)
  - [ ] 3.3.1.3 新增 `struct Tcgen05Instr { ... };`(8 字段,per `design.md` Decision 3)
- [ ] 3.3.2 编辑 `include/ptx_ir/statement_factory.h`:
  - [ ] 3.3.2.1 新增 `makeTcgen05Instr(...)` 工厂函数
  - [ ] 3.3.2.2 删除 `makeWmmaInstr(...)` 工厂函数

### 3.4 Handler Dispatch

- [ ] 3.4.1 编辑 `src/ptxsim/instruction_handlers.cpp`:
  - [ ] 3.4.1.1 添加 `IMPLEMENT_TCGEN05_INSTR_HANDLER(Name)` macro
  - [ ] 3.4.1.2 删除 `IMPLEMENT_WMMA_INSTR_HANDLER(Name)` macro
- [ ] 3.4.2 验证:`grep -rn "IMPLEMENT_WMMA_INSTR_HANDLER" src/` 应为零

### 3.5 Build Validation

- [ ] 3.5.1 验证:`cmake --build build` 期望 wmma.cpp 编译失败(S_WMMA 引用问题,预期内)
- [ ] 3.5.2 NOTE:wmma.cpp 错误是预期的,commit 4 修复

### 3.6 Commit

- [ ] 3.6.1 `git add include/ptx_ir/ src/ptxsim/instruction_handlers.cpp`
- [ ] 3.6.2 `git commit -m "feat(ir): add S_TCGEN05_* StatementType + Tcgen05Instr + Qualifier (ADR-0016)"`
- [ ] 3.6.3 验证:commit 独立可 revert(`git revert HEAD` 回到 wmma-only 状态)

## 4. Parser/Visitor 迁移(commit 4,atomic)

> **MUST**:同步修复 wmma.cpp 的 `S_WMMA` 引用 + 旧测试(tcgen05/)的引用

### 4.1 Visitor Rename

- [ ] 4.1.1 `git mv src/ptx_parser/ptx_visitor_wmma.cpp src/ptx_parser/ptx_visitor_tcgen05.cpp`
- [ ] 4.1.2 编辑 `ptx_visitor_tcgen05.cpp`:
  - [ ] 4.1.2.1 X-Macro 改用 `S_TCGEN05_*` 枚举
  - [ ] 4.1.2.2 改用 `makeTcgen05Instr(...)` 工厂
  - [ ] 4.1.2.3 添加 chinese 注释说明命名空间分离
- [ ] 4.1.3 编辑 `src/CMakeLists.txt`(若有)或 `src/ptx_parser/CMakeLists.txt`:
  - [ ] 4.1.3.1 同步 `ptx_visitor_wmma.cpp` → `ptx_visitor_tcgen05.cpp` 引用
  - [ ] 4.1.3.2 验证:`grep "ptx_visitor_wmma" src/**/CMakeLists.txt` 应为零

### 4.2 PtxListener

- [ ] 4.2.1 编辑 `src/ptx_parser/ptx_parser.cpp`:
  - [ ] 4.2.1.1 line 751: `ctx->WMMA()` 改 `ctx->TCGEN05()`
  - [ ] 4.2.1.2 line 761-771:WMMA 分发改 TCGEN05 分发
  - [ ] 4.2.1.3 line 784:`statementType = S_TCGEN05_*` 替代 `S_WMMA`
  - [ ] 4.2.1.4 添加 chinese 注释说明命名空间分离
- [ ] 4.2.2 验证:`grep -n "S_WMMA\|WMMA_INSTR" src/ptx_parser/ptx_parser.cpp` 应为零

### 4.3 Pipeline Handler

- [ ] 4.3.1 编辑 `src/ptxsim/instruction_base.cpp`:
  - [ ] 4.3.1.1 新增 `Tcgen05PipelineHandler` 类(参照 `WmmaPipelineHandler` 模式)
  - [ ] 4.3.1.2 3 阶段方法:`prepareOperands / executeOperation / commitResults`
  - [ ] 4.3.1.3 `executeOperation` 调用 `processTcgen05Operation(...)` 而非 `processWmmaOperation(...)`
  - [ ] 4.3.1.4 注册 `S_TCGEN05_*` 11 个枚举到 dispatcher
- [ ] 4.3.2 验证:`grep -rn "processTcgen05Operation" src/ptxsim/` 应为新增项

### 4.4 wmma.cpp 过渡期安全网

- [ ] 4.4.1 编辑 `src/ptxsim/instructions/wmma.cpp`:
  - [ ] 4.4.1.1 `WmmaHandler::processWmmaOperation` 入口临时 throw `UnsupportedInstructionException`
  - [ ] 4.4.1.2 添加 chinese 注释:"过渡期安全网,change-3 重写"
  - [ ] 4.4.1.3 NOTE:此 throw 保护 silent no-op 不回归(per design.md Decision 5.1)
- [ ] 4.4.2 验证:wmma.cpp 编译通过

### 4.5 旧测试迁移(最小修改)

- [ ] 4.5.1 编辑 `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp`:
  - [ ] 4.5.1.1 `S_WMMA` 改 `S_TCGEN05_MMA`
  - [ ] 4.5.1.2 `makeWmmaInstr(...)` 改 `makeTcgen05Instr(Tcgen05OpKind::MMA, ...)`
  - [ ] 4.5.1.3 `WmmaType::WMMA_MMA` 改 `Tcgen05OpKind::MMA`
- [ ] 4.5.2 编辑 `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp`:
  - [ ] 4.5.2.1 5 个 `S_WMMA` 改对应 `S_TCGEN05_*`
  - [ ] 4.5.2.2 5 个 `makeWmmaInstr(...)` 改 `makeTcgen05Instr(...)`
  - [ ] 4.5.2.3 5 个 `WmmaType::WMMA_*` 改 `Tcgen05OpKind::*`
- [ ] 4.5.3 验证:`grep -rn "S_WMMA\|makeWmmaInstr\|WmmaType" tests/integration/tcgen05/` 应为零

### 4.6 Build + Test Validation

- [ ] 4.6.1 `cmake --build build` 验证:全量编译通过
- [ ] 4.6.2 `cd build && ctest -L "unit|integration" --output-on-failure` 验证:零回归
- [ ] 4.6.3 NOTE:wmma.cpp 旧 e2e 测试可能因 throw 而失败(预期,本 change 不修 e2e)

### 4.7 Commit

- [ ] 4.7.1 `git add src/ptx_parser/ src/ptxsim/ tests/integration/tcgen05/`
- [ ] 4.7.2 `git commit -m "feat(parser): rename ptx_visitor_wmma → ptx_visitor_tcgen05 + dispatch (ADR-0016)"`
- [ ] 4.7.3 验证:commit 独立可 revert(`git revert HEAD` 回到 wmma 路径)

## 5. Tests(commit 5,atomic)

> **MUST**:12 .ptx 端到端 + 5 单元 + 5 集成 = 22 个新测试目标

### 5.1 PTX 端到端 Fixtures

- [ ] 5.1.1 创建 `tests/ptx/tcgen05_alloc.ptx`
- [ ] 5.1.2 创建 `tests/ptx/tcgen05_dealloc.ptx`
- [ ] 5.1.3 创建 `tests/ptx/tcgen05_relinquish.ptx`
- [ ] 5.1.4 创建 `tests/ptx/tcgen05_ld.ptx`
- [ ] 5.1.5 创建 `tests/ptx/tcgen05_st.ptx`
- [ ] 5.1.6 创建 `tests/ptx/tcgen05_cp.ptx`
- [ ] 5.1.7 创建 `tests/ptx/tcgen05_cp_multicast.ptx`
- [ ] 5.1.8 创建 `tests/ptx/tcgen05_mma.ptx`
- [ ] 5.1.9 创建 `tests/ptx/tcgen05_mma_block_scale.ptx`
- [ ] 5.1.10 创建 `tests/ptx/tcgen05_mma_ws.ptx`
- [ ] 5.1.11 创建 `tests/ptx/tcgen05_commit.ptx`
- [ ] 5.1.12 创建 `tests/ptx/tcgen05_wait.ptx`
- [ ] 5.1.13 创建 `tests/ptx/tcgen05_fence.ptx`
- [ ] 5.1.14 编辑 `tests/ptx/test_all_ptx.sh`,注册 13 个新 fixtures
- [ ] 5.1.15 验证:`./tests/ptx/test_all_ptx.sh` 13 个新 fixtures 100% 通过

### 5.2 单元测试

- [ ] 5.2.1 创建 `tests/unit/ptx_ir/test_tcgen05_qualifier.cpp`(~30 LoC,验证 ~25 Q_* 枚举)
- [ ] 5.2.2 创建 `tests/unit/ptx_ir/test_tcgen05_opkind.cpp`(验证 11 OpKind 枚举)
- [ ] 5.2.3 创建 `tests/unit/ptx_ir/test_tcgen05_dtype.cpp`(验证 10 Dtype 枚举)
- [ ] 5.2.4 创建 `tests/unit/ptx_ir/test_tcgen05_statement_factory.cpp`(验证 makeTcgen05Instr 工厂)
- [ ] 5.2.5 创建 `tests/unit/ptx_ir/test_tcgen05_instr_struct.cpp`(验证 Tcgen05Instr 字段)
- [ ] 5.2.6 编辑 `tests/unit/CMakeLists.txt`,注册 5 个新测试 + 标签 `unit;tcgen05`

### 5.3 集成测试

- [ ] 5.3.1 创建 `tests/integration/parser/test_tcgen05_mma_parse.cpp`(端到端 parse → IR)
- [ ] 5.3.2 创建 `tests/integration/parser/test_tcgen05_ld_parse.cpp`(验证 num_regs 字段)
- [ ] 5.3.3 创建 `tests/integration/parser/test_tcgen05_st_parse.cpp`
- [ ] 5.3.4 创建 `tests/integration/parser/test_tcgen05_commit_parse.cpp`(验证 mbarrier qualifier)
- [ ] 5.3.5 创建 `tests/integration/parser/test_tcgen05_wait_parse.cpp`(验证 .load/.store 子操作)
- [ ] 5.3.6 编辑 `tests/integration/CMakeLists.txt`,注册 5 个新测试 + 标签 `integration;tcgen05;grammar`

### 5.4 Build + Test Validation

- [ ] 5.4.1 `cmake --build build` 验证:零编译错误
- [ ] 5.4.2 `./tests/ptx/test_all_ptx.sh` 验证:13 新 fixtures + 现有 fixtures 全绿
- [ ] 5.4.3 `cd build && ctest -L "unit;tcgen05" --output-on-failure` 验证:5 单元测试全绿
- [ ] 5.4.4 `cd build && ctest -L "integration;tcgen05" --output-on-failure` 验证:5 集成 + 2 迁移 = 7 测试全绿
- [ ] 5.4.5 `cd build && ctest --output-on-failure` 验证:全量测试无回归

### 5.5 Documentation

- [ ] 5.5.1 编辑 `src/ptxsim/instructions/AGENTS.md`:
  - [ ] 5.5.1.1 添加 `tcgen05.cpp` 目录说明(handler 实现在 change-3)
  - [ ] 5.5.1.2 标注 wmma.cpp 临时 throw 状态(change-3 重写)
- [ ] 5.5.2 编辑 `tests/AGENTS.md`(若有):
  - [ ] 5.5.2.1 添加 `tcgen05` 标签说明
  - [ ] 5.5.2.2 更新测试目录结构

### 5.6 Commit

- [ ] 5.6.1 `git add tests/ src/ptxsim/instructions/AGENTS.md tests/AGENTS.md`
- [ ] 5.6.2 `git commit -m "test(ptx): add 13 tcgen05.* PTX fixtures + unit/integration tests (ADR-0016)"`
- [ ] 5.6.3 验证:commit 独立可 revert(`git revert HEAD` 回到 wmma-only 状态,测试目标消失)

## 6. Documentation Sync + Archive(commit 6,per `ptx-lessons-learned` Checklist I + G)

> **MUST**:根 AGENTS.md + ADR 追加 + lessons-learned §22 + 文档同步

### 6.1 Root AGENTS.md

- [ ] 6.1.1 编辑根 `AGENTS.md` 已知限制表:
  - [ ] 6.1.1.1 标注 "tcgen05 语法已独立命名空间" (change-1 状态)
  - [ ] 6.1.1.2 标注 "pre-Blackwell tcgen05 永久抛异常"(保留 ADR-0016 行为)
  - [ ] 6.1.1.3 标注 "tcgen05 handler 实现在 change-3"

### 6.2 ADR Update

- [ ] 6.2.1 编辑 `docs/adr/ADR-0016-blackwell-only-tcgen05.md`:
  - [ ] 6.2.1.1 在 "更新记录" 追加段落: "2026-07-XX — tcgen05 独立命名空间 (commit <sha>)"
  - [ ] 6.2.1.2 引用本 change: "Ref: openspec/changes/implement-tcgen05-syntax-ir/"
  - [ ] 6.2.1.3 NOTE:不修改决策本身,仅追加更新记录(per ADR lifecycle)

### 6.3 Lessons-Learned

- [ ] 6.3.1 编辑 `docs/dev-process/lessons-learned.md`,追加 §22:
  - [ ] 6.3.1.1 标题: "已实施但未清理模式"
  - [ ] 6.3.1.2 案例:`implement-wmma-tensor-core-tcgen05` 走 wmma 路径 + qualifier 注入
  - [ ] 6.3.1.3 教训:ANTLR grammar 必须有独立关键字(避免 wmma 命名空间混淆)
  - [ ] 6.3.1.4 检查命令:`grep "tcgen05" src/grammar/*.g4` 应非空
  - [ ] 6.3.1.5 真实案例:本 change 是该模式的应用

### 6.4 Archive

- [ ] 6.4.1 验证:`cd build && ctest --output-on-failure` 全量通过
- [ ] 6.4.2 验证:`./tests/ptx/test_all_ptx.sh` 全量通过
- [ ] 6.4.3 验证:`grep -rn "wmma" src/ptxsim/instructions/wmma.cpp | grep -v "throw" | grep -v "comment"` 应仅剩 throw 兜底
- [ ] 6.4.4 跑 `openspec archive implement-tcgen05-syntax-ir --yes`(per Checklist G)
- [ ] 6.4.5 验证:`openspec/changes/archive/2026-07-XX-implement-tcgen05-syntax-ir/` 创建
- [ ] 6.4.6 `git add openspec/changes/archive/`
- [ ] 6.4.7 `git commit -m "chore(openspec): archive implement-tcgen05-syntax-ir (ADR-0016)"`
- [ ] 6.4.8 验证:commit 独立可 revert(`git revert HEAD` 回到 working tree 状态,artifacts 重新 active)

## 7. Final Validation

- [ ] 7.1 `./scripts/sanity.sh` 全量验证(per AGENTS.md §TDD 流程)
- [ ] 7.2 `./scripts/sanity.sh --ptx` PTX 语法验证
- [ ] 7.3 `cd build && ctest --output-on-failure` 全量测试
- [ ] 7.4 验证:`git log --oneline feat/implement-tcgen05-syntax-ir` 显示 6 个 atomic commits
- [ ] 7.5 验证:`git log --all --oneline -- openspec/changes/implement-tcgen05-syntax-ir/` 显示完整生命周期
- [ ] 7.6 提议 Change-2:`openspec new change "extend-blackwell-tcgen05-infra"`(基础设施补全)

## Risks & Mitigations Recap

| Risk | Mitigation in Tasks |
|---|---|
| **R1**: 删除 `S_WMMA` 后 wmma.cpp 编译失败 | 任务 3.5.1 明确预期错误 + 任务 4.4 临时 throw 兜底 |
| **R2**: ANTLR 变更导致现有 PTX 解析回归 | 任务 2.1.5 + 2.2.5 验证 `./tests/ptx/test_all_ptx.sh` |
| **R3**: 新 qualifier 命名冲突 | 任务 3.2.2 验证 `grep -c` |
| **R4**: `tcgen05.alloc/dealloc/relinquish` 语法错误 | 任务 5.1.1-5.1.3 用真实 PTX 验证 |
| **R5**: `ptx_visitor_wmma.cpp` rename 漏改 | 任务 4.1.3 验证 CMakeLists |
| **R6**: PTX ISA 9.x 未覆盖 | 任务 0.1.4 验证 baseline,9.x 留待后续 change |
| **R7**: 测试覆盖率不足 | 13 fixtures + 5 单元 + 5 集成 = 25 测试目标 |
| **R8**: pre-Blackwell 行为破坏 | 任务 4.4 临时 throw,change-3 真正实现 |
| **R9**: ANTLR 4.11.1 不支持 | 任务 0.2 baseline 验证 |

## Out-of-Scope Reminder(per `design.md` Non-Goals)

- ❌ Handler 实际实现(change-3 scope)
- ❌ TMA/TMEM/cluster/TcQueue 基础设施审计(change-2 scope)
- ❌ 删除 wmma 路径(change-4 scope)
- ❌ 修改 `src/ptxsim/instructions/wmma.cpp` 实际 handler(change-3 scope)
- ❌ `cuTensorMapEncodeTiled` host API 拦截(候选 ADR-0017,后续 change)
- ❌ `tensormap.replace` device-side 更新(后续 change)
- ❌ sm_120 sparse / FP4 / mxfp8(后续 change per ADR-0016)
- ❌ 修改 ANTLR 版本(per ptx-lessons-learned §4)
- ❌ 性能对标(仅 functional correctness)
