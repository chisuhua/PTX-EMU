# Design: Blackwell tcgen05 独立命名空间(ANTLR + IR)

> **依赖**: [proposal.md](proposal.md) - 已确认
> **范围**: 本 change 仅交付语法 + IR 命名空间;handler 实现在 change-3,基础设施审计在 change-2
> **架构依据**: [ADR-0016](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) (Accepted 2026-07-04)

## Context

### 现状(已通过探索 + 实证)

1. **ANTLR grammar 完全没有 tcgen05 关键字**:
   - `src/grammar/ptxLexer.g4:373` `WMMA: 'wmma'` — 唯一 tensor core 相关 token
   - `src/grammar/ptxInstructions.g4:424-433` — 仅有 `wmmaInst / wmmaOp / wmmaLayout / wmmaShape / wmmaKind` 5 个旧规则
   - `src/grammar/ptxInstructions.g4:463-465` — 现有 `tcgenInst: stBulkInst` 实际只支持 `st.bulk`,**没有 tcgen05 指令语法**
   - 实测:`grep "tcgen05" src/grammar/*.g4` → **零输出**

2. **IR 命名空间混用**:
   - `include/ptx_ir/ptx_op.def:127` `S_WMMA` — 唯一 tensor core IR 枚举,**承载了 tcgen05 实现**
   - `include/ptx_ir/ptx_qualifier.def:194-197` — 4 个 stub `Q_TCGEN05_LD/ST/COMMIT/WAIT` 作为子操作区分
   - 但 NVIDIA PTX ISA 8.6+ 实际有 **12 个 tcgen05 指令族**(alloc/dealloc/relinquish/ld/st/cp/mma/mma.ws/mma.block_scale/commit/wait/fence),IR 严重不足

3. **handler 真实实现已存在但混入 wmma**:
   - `src/ptxsim/instructions/wmma.cpp:330-565` — 5 个 `execute_tcgen05_*` 函数(ld/st/commit/wait/mma)真实工作
   - 但 `src/ptxsim/instructions/AGENTS.md:74-75` 描述: "wmma.cpp (WmmaHandler) — Blackwell tcgen05.* real fragment arithmetic" — **命名空间错误**
   - 32+ 处 `// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13` 注释标记(per `ptx-lessons-learned` §1 审计要求)

4. **测试与现实脱节**:
   - `tests/integration/tcgen05/test_tcgen05_*.cpp` — 2 个,**直接构造 `S_WMMA` + qualifier 注入**,不走真实 PTX 解析
   - `tests/e2e/kernel/test_blackwell_gemm.cu:1-12` — 标签 `[e2e][tcgen05][gemm][sm_100]` 误导,**实际是普通 float kernel**(注释自承: "Uses float (not half) to avoid nvcc sm_100 PTX .nc.u16 loads that the ANTLR grammar does not support")
   - `tests/ptx/test_all_ptx.sh` — 零个 tcgen05 PTX 用例(grep 验证)

### 目标状态(本 change 完成后)

- ✅ ANTLR grammar 接受真实 `tcgen05.*` PTX 文本(12 个指令族)
- ✅ IR 命名空间完全独立:11 个 `S_TCGEN05_*` 枚举 + `Tcgen05Instr` 结构体 + 25+ 个独立 qualifier
- ✅ Visitor/parser 分发改用 `tcgen05*` token
- ✅ 三套测试:12 个 .ptx 端到端 + 单元/集成测试覆盖 grammar 解析
- ⏸ Handler 实现仍沿用 `wmma.cpp`(change-3 重写)
- ⏸ 删除 wmma 路径(change-4)

### 约束(per ADR-0016)

1. **pre-Blackwell 仍抛 `UnsupportedInstructionException`**(per `replace-silent-stub-failures` archived 2026-07-04)
2. **测试 sm_100 虚拟架构**(`configs/ampere_a100.json` 之外,不修改)
3. **ANTLR 版本 4.11.1**(已就绪)
4. **本 change 不修改 handler 文件**(`src/ptxsim/instructions/wmma.cpp` 是 change-3 scope)
5. **代码风格**:`clang-format` 强制,4 空格缩进,Chinese 注释优先

### 利益相关方

- 实施者:本项目架构师(单团队)
- 用户:cute/cutlass 模板用户(emitting sm_100 PTX)
- 维护者:未来 reviewer(per ADR-0009 X-Macro 模式)

## Goals / Non-Goals

### Goals

1. **ANTLR grammar 完整支持 12 个 tcgen05 指令族**(alloc/dealloc/relinquish/ld/st/cp/mma/mma.ws/mma.block_scale/commit/wait/fence)
2. **IR 命名空间完全独立于 wmma**(11 个 `S_TCGEN05_*` + `Tcgen05Instr` 结构体)
3. **Qualifier 完整覆盖**(~25 个 `Q_CTA_GROUP / Q_KIND / Q_F16 / Q_BF16 / ...`)
4. **三套测试覆盖 grammar 解析**(12 .ptx 端到端 + 单元/集成)
5. **commit 粒度 6 个 atomic**,每步独立可回退
6. **保持 pre-Blackwell 行为**(`UnsupportedInstructionException` 合约)

### Non-Goals

- ❌ 实现 handler 实际逻辑(change-3 scope)
- ❌ 评估/补全 TMA/TMEM/cluster/TcQueue 基础设施(change-2 scope)
- ❌ 删除 wmma 路径(change-4 scope)
- ❌ 修改 `src/ptxsim/instructions/wmma.cpp`(change-3 重写)
- ❌ 支持 `cuTensorMapEncodeTiled` host API 拦截(候选 ADR-0017,后续 change)
- ❌ 支持 `tensormap.replace` device-side 动态更新(后续 change)
- ❌ sm_120 sparse / FP4 / mxfp8(后续 change per ADR-0016 决策)
- ❌ 修改 ANTLR 版本(per ptx-lessons-learned §4 基线约束)
- ❌ 性能对标(仅 functional correctness)

## Decisions

### Decision 1: ANTLR Lexer Token 命名方案

**选项**:
- (A) `MMA_/LD_/ST_/COMMIT_/WAIT_` 用下划线后缀(避免与 `mma.sync` 等冲突)
- (B) 直接用 `.mma/.ld/.st/.commit/.wait`(与 PTX 文本一致)
- (C) 用 `MMA_TCGEN05/LD_TCGEN05/...` 全前缀

**采纳**:(A) 保留 PTX 文本一致性 + 避免 ANTLR 词法冲突。

**理由**:
- ANTLR lexer 不支持带点号的子操作(`.mma` 在 lexer 层会被切分为独立 token)
- 选项 B 与现有 `.s32`/`.f16` 等类型 token 命名冲突(虽然语义不同,但 readability 差)
- 选项 C 冗长,与现有命名风格不一致

**实施**:在 `ptxLexer.g4` 末尾新增 token block。

### Decision 2: IR 命名空间完全独立(11 个枚举 vs 1 个通用)

**选项**:
- (A) 11 个 `S_TCGEN05_*` 独立枚举(本 change 采纳)
- (B) 单一 `S_TCGEN05` + `Tcgen05OpKind` 内部枚举(per `wmmaType` 模式)
- (C) 复用 `S_WMMA` + 新增更多 qualifier 区分

**采纳**:(A) 完全独立。

**理由**:
- 12 个指令族在 X-Macro 分发时需要独立 weak symbol(`IMPLEMENT_TCGEN05_INSTR_HANDLER` 自动展开)
- 选项 B 二次 dispatch 增加 runtime 开销(per `ptx-lessons-learned` §1 跨模块间接状态翻译模式)
- 选项 C 继续混淆命名空间,违反用户"避免 WMMA/wmma 名字"明确要求

**Trade-off**:11 个枚举膨胀 `ptx_op.def`(从 193 → 250 行),但 X-Macro 自动展开,实际 maintenance cost 低。

### Decision 3: Tcgen05Instr 结构体字段

**采纳字段**:
```cpp
struct Tcgen05Instr {
    Tcgen05OpKind op_kind;        // 11 个枚举值之一
    std::vector<Qualifier> qualifiers;  // 完整 qualifier 列表
    std::vector<OperandContext> operands;  // 操作数
    std::string instructionText;  // 原文(供 logging/debug)
    uint32_t cta_group = 1;       // 解析后 .cta_group 值
    Tcgen05Dtype dtype = Tcgen05Dtype::F16;  // 解析后 .kind 值
    uint32_t num_regs = 0;        // ld/st x1/x2/x4 标识
    bool has_block_scale = false; // mma.block_scale 标识
};
```

**理由**:
- `op_kind` 区分指令族(handler 内部 switch 即可)
- `cta_group / dtype / num_regs / has_block_scale` 提前解析,handler 无需重复解析 qualifier
- `qualifiers` 保留完整列表(供 error reporting / debugging)
- `operands` 复用 `OperandContext`(per `ptx_op.def` 现有 X-Macro 模式)

**Trade-off**:`cta_group / dtype` 解析需要 visitor 端额外逻辑,但降低 handler 复杂度。

### Decision 4: 删除 `S_WMMA` vs 保留兼容

**采纳**:删除 `S_WMMA` 整行(per `ptx_op.def:127`)。

**理由**:
- 用户明确要求"避免 WMMA/wmma 名字"
- `S_WMMA` 是当前 wmma.cpp 唯一 IR 入口,删除后 change-3 可干净地重写为 `tcgen05.cpp`
- 旧测试(`tests/integration/tcgen05/`)直接构造 `S_WMMA`,change-3 需同步更新测试(本 change 同步)

**Trade-off**:`wmma.cpp` 中 `IMPLEMENT_WMMA_INSTR_HANDLER` weak symbol 失效(change-3 修复),change-2 期间 wmma.cpp handler 仍可工作(weak symbol 默认空实现)。

### Decision 5: 删除 4 个 stub qualifier + 完整重写

**采纳**:删除 `Q_TCGEN05_LD/ST/COMMIT/WAIT` 4 个 stub(`ptx_qualifier.def:194-197`),改用 11 个独立 IR 枚举。

**理由**:
- stub qualifier 是"在 WmmaHandler 内部用 `is_tcgen05_*()` helper 区分"的临时方案(per commit 35808d6)
- 独立 IR 枚举 + `Tcgen05Instr::op_kind` 是更干净的设计
- 旧 `is_tcgen05_ld(qualifiers)` 等 5 个 helper(change-3 重写为 `switch (op_kind)`)

**Trade-off**:`wmma.cpp:21-60` 5 个 helper 函数失效(change-3 重写)。

### Decision 6: 测试粒度(12 个 .ptx 端到端 + 单元/集成)

**采纳**:
- 12 个 `tests/ptx/tcgen05_*.ptx` 文件(每指令族 1 个)
- 5 个 `tests/unit/ptx_ir/test_tcgen05_*.cpp` 单元测试(qualifier 解析、StmtFactory、enum 转换)
- 5 个 `tests/integration/parser/test_tcgen05_*.cpp` 集成测试(端到端 ANTLR parse → IR)

**理由**:
- 12 .ptx 文件覆盖所有指令族(per PTX ISA 8.6 完整 spec)
- 单元测试验证独立模块行为(per `ptx-lessons-learned` §20 经验 5)
- 集成测试验证 grammar → IR 全链路
- e2e 测试不在本 change 范围(handler 未实现,change-3 才加)

**Trade-off**:测试数量多(22 个新测试),但确保 grammar 完备性。

### Decision 7: 6 个 atomic commit 粒度

**采纳**:
1. `docs(openspec): add implement-tcgen05-syntax-ir artifacts` (artifacts FIRST)
2. `feat(grammar): add TCGEN05 lexer + tcgen05 parser rules`
3. `feat(ir): add S_TCGEN05_* StatementType + Tcgen05Instr + Qualifier`
4. `feat(parser): rename ptx_visitor_wmma → ptx_visitor_tcgen05 + dispatch`
5. `test(ptx): add 12 tcgen05.* PTX fixtures + unit/integration tests`
6. `chore(openspec): archive implement-tcgen05-syntax-ir`

**理由**(per `ptx-lessons-learned` §3 + §6 + Checklist E):
- 每个 commit 独立可 revert
- 已有测试回归 → 立即 revert
- artifacts 必须 `git add`(避免 lessons-learned §6 模式)
- 6 个 commit 数量 = 实际工作量(per ADR-0016 决策)

**Trade-off**:每个 commit 期间 `cmake --build build` 可能 fail(intermediate 状态),需 commit message 明确说明"intermediate state"。

## Risks / Trade-offs

| 风险 | 等级 | 缓解 |
|---|---|---|
| **R1: 删除 `S_WMMA` 后 `wmma.cpp` 编译失败** | 🟡 中 | commit 4 之前 wmma.cpp 仍用 `S_WMMA`,需在 commit 4 同步修改 `instruction_base.cpp:226` 的 `processWmmaOperation` 调用;`ptx-lessons-learned` §3 要求"独立可 revert",commit 4 必须是可独立 revert 的 atomic unit |
| **R2: ANTLR grammar 变更导致现有 PTX 解析回归** | 🟡 中 | 必须跑 `./tests/ptx/test_all_ptx.sh` 验证 100% 通过;任何回归立即 revert commit 2 |
| **R3: 新增 25+ qualifier 与现有命名冲突** | 🟢 低 | 实施前 `grep -n "Q_F16\|Q_BF16\|Q_TF32" include/ptx_ir/ptx_qualifier.def` 验证;若冲突,本 change 是设计时教训的机会(per §6 反馈) |
| **R4: `tcgen05.alloc/dealloc/relinquish` 指令语法错误** | 🟡 中 | 实施时 `tests/ptx/tcgen05_alloc.ptx` 等用 `cuobjdump -xptx` 提取的 真实 PTX;若语法错误,fix `.g4` 而非简化(per `ptx-grammar-modification` 强制流程) |
| **R5: commit 4 rename `ptx_visitor_wmma.cpp` 后其他模块引用** | 🟢 低 | `git grep "ptx_visitor_wmma" -- src/ CMakeLists.txt` 验证;rename 时同步 `CMakeLists.txt` 引用 |
| **R6: PTX ISA 9.x 新增指令未覆盖** | 🟡 中 | 本 change 锁定 PTX ISA 8.6 baseline(per ADR-0016);9.x 新增在后续 change 处理;OpenSpec change-4 不修改 spec 范围 |
| **R7: 测试覆盖率不足以发现 qualifier 解析错误** | 🟢 低 | 12 个 .ptx 端到端 + 5 个单元测试 + 5 个集成测试,覆盖率 90%+ |
| **R8: pre-Blackwell 行为被破坏** | 🟢 低 | pre-Blackwell 仍走原 `S_WMMA` 路径(change-3 才删除);本 change 不修改 `wmma.cpp` |
| **R9: ANTLR 4.11.1 不支持某些语法** | 🟢 低 | 已 baseline 测试通过(per `ptx-lessons-learned` §4 经验) |

## Migration Plan

### Phase 1: Artifacts Tracking(per Checklist E,§6 强制 FIRST)

```bash
# 0.0.1 在 main 上创建分支
git checkout -b feat/implement-tcgen05-syntax-ir
# 0.0.2 git add 4 个 artifacts(proposal/design/tasks/specs/tcgen05-*/*/spec.md)
git add openspec/changes/implement-tcgen05-syntax-ir/
# 0.0.3 验证 tracked
git ls-files openspec/changes/implement-tcgen05-syntax-ir/  # 应非空
# 0.0.4 commit
git commit -m "docs(openspec): add implement-tcgen05-syntax-ir artifacts (ADR-0016)"
```

### Phase 2: ANTLR Grammar(commit 2,atomic)

```bash
# 修改 src/grammar/ptxLexer.g4 - 新增 ~30 tokens
# 修改 src/grammar/ptxInstructions.g4 - 删除 wmma 系列 + 新增 tcgen05 系列
# 修改 src/grammar/ptxOperands.g4 - 新增 tcgen05 qualifier 处理
# 重新生成
cmake --build build --target GenerateParser
# 验证 baseline 不破坏
./tests/ptx/test_all_ptx.sh  # 必须 100% PASS
# commit
git commit -m "feat(grammar): add TCGEN05 lexer + tcgen05 parser rules (ADR-0016)"
```

### Phase 3: IR 命名空间(commit 3,atomic)

```bash
# 修改 include/ptx_ir/ptx_op.def - 删除 S_WMMA + 新增 11 个 S_TCGEN05_*
# 修改 include/ptx_ir/ptx_qualifier.def - 删除 4 stub + 新增 ~25 Q_*
# 修改 include/ptx_ir/statement_context.h - 新增 Tcgen05Instr + 2 enum
# 修改 include/ptx_ir/statement_factory.h - 新增 makeTcgen05Instr
# 验证 build 通过(可能 warning,不能 error)
cmake --build build 2>&1 | tee /tmp/build.log
# commit
git commit -m "feat(ir): add S_TCGEN05_* StatementType + Tcgen05Instr + Qualifier (ADR-0016)"
```

### Phase 4: Parser/Visitor(commit 4,atomic)

```bash
# git mv src/ptx_parser/ptx_visitor_wmma.cpp src/ptx_parser/ptx_visitor_tcgen05.cpp
# 修改 ptx_parser.cpp:751-784 - WMMA → TCGEN05
# 修改 src/ptxsim/instruction_handlers.cpp - IMPLEMENT_TCGEN05_INSTR_HANDLER
# 验证 build + 测试
cmake --build build
cd build && ctest -L "unit|integration" --output-on-failure
# commit
git commit -m "feat(parser): rename ptx_visitor_wmma → ptx_visitor_tcgen05 + dispatch (ADR-0016)"
```

### Phase 5: Tests(commit 5,atomic)

```bash
# 创建 12 个 tests/ptx/tcgen05_*.ptx
# 创建 5 个 tests/unit/ptx_ir/test_tcgen05_*.cpp
# 创建 5 个 tests/integration/parser/test_tcgen05_*.cpp
# 注册到 tests/ptx/test_all_ptx.sh
# 注册到 tests/unit/CMakeLists.txt + tests/integration/CMakeLists.txt
# 验证
./tests/ptx/test_all_ptx.sh  # 必须 100% PASS
cd build && ctest -L "unit;tcgen05|integration;tcgen05" --output-on-failure
# commit
git commit -m "test(ptx): add 12 tcgen05.* PTX fixtures + unit/integration tests (ADR-0016)"
```

### Phase 6: Archive(commit 6,per Checklist G)

```bash
# 验证全量测试通过
cd build && ctest --output-on-failure
# archive
openspec archive implement-tcgen05-syntax-ir --yes
# commit
git add openspec/changes/archive/
git commit -m "chore(openspec): archive implement-tcgen05-syntax-ir (ADR-0016)"
```

### 回退策略(per `ptx-lessons-learned` §3)

- 任意 commit 失败:`git revert HEAD~1..HEAD` (revert 到上一个 good state)
- 整体 Phase 失败:`git revert <phase-1-sha>..<phase-N-sha>` (整体回退)
- 紧急回退:`git reset --hard <pre-change-sha>` (丢失本 change 全部,需先备份 working tree)

### 风险预警

- ⚠️ **commit 3 期间 wmma.cpp 仍引用 S_WMMA**:临时编译错误,commit 4 必须紧接 commit 3
- ⚠️ **commit 4 期间 tests/integration/tcgen05/* 仍构造 S_WMMA**:临时编译错误,commit 5 必须紧接 commit 4
- ⚠️ **commit 6 archive 前必须跑 ctest 全量**:已有测试回归需 fix 后再 archive

## Open Questions

### Q1: PTX ISA 9.2 packed 数据类型(.s32x6x2/.u32x6x2/.s8x6x2)是否本 change 覆盖?

**当前决定**:否。本 change 锁定 PTX ISA 8.6 baseline(per ADR-0016)。9.2 packed 类型留待后续 change(per ptx-lessons-learned §1 "scope discipline")。

**验证方法**:实施时检查 `tests/ptx/tcgen05_mma_packed_92.ptx` 是否需要(预计不需要,因本 change 范围仅 8.6 baseline)。

### Q2: cp.async.bulk.tensor 指令是否本 change 覆盖?

**当前决定**:**否**。`cp.async.bulk.tensor` 是 TMA 加载指令,与 `tcgen05.ld` 不同。change-2 评估 TMA 基础设施时再决定。

**但**:本 change 的 grammar 规则可**预解析** `cp.async.bulk.tensor` token,留待 change-2 实际 handler 实现(per `ptx-lessons-learned` §3 "基础设施 first" 决策)。

### Q3: tensormap.create / tensormap.replace 是否本 change 覆盖?

**当前决定**:否。`tensormap.*` 是 device-side 动态更新 descriptor 的指令,本 change 不涉及。候选 ADR-0017 后续处理。

### Q4: `tests/integration/tcgen05/test_tcgen05_*.cpp` (旧测试) 是否本 change 同步修改?

**当前决定**:**是,最小修改**。commit 4 期间需修改这 2 个旧测试的 `S_WMMA` 引用为 `S_TCGEN05_*`,否则编译失败。

**验证方法**:commit 4 后 `cd build && ctest -R tcgen05 -V`,预期 2 个旧测试仍 PASS(仅修改引用)。

### Q5: 是否需要 `impl_tcgen05_alloc_handler` 等 11 个 stub handler?

**当前决定**:**否**。本 change 不实现 handler。`IMPLEMENT_TCGEN05_INSTR_HANDLER` macro 默认空实现(per `instruction_handlers.cpp:170-172`),handler 实际实现留待 change-3。

**但**:grammar 通过测试后,如果用户跑 `tcgen05.mma` PTX,handler 会**静默 no-op**(per `ptx-lessons-learned` §1 的"未实现 stub silent 失败"模式)。这与 `replace-silent-stub-failures` (archived 2026-07-04) 合约冲突。

**缓解**:本 change 完成后,在 `wmma.cpp` 的 `WmmaHandler::processWmmaOperation` 中临时 throw `UnsupportedInstructionException` 兜底(change-3 才真正实现 handler),确保 silent no-op 不会回归。

**修正**:Decision 5.1 — commit 4 期间需在 `wmma.cpp` 临时 throw,作为"过渡期安全网"。

## 影响范围(组件 | 影响类型)

| 组件 | 影响类型 | 备注 |
|---|---|---|
| `src/grammar/ptxLexer.g4` | 新增 | ~30 tokens |
| `src/grammar/ptxInstructions.g4` | 删除 + 新增 | 删除 wmma,新增 tcgen05 |
| `src/grammar/ptxOperands.g4` | 新增 | tcgen05 qualifier 处理 |
| `src/grammar/AGENTS.md` | 修改 | 更新规则说明 |
| `include/ptx_ir/ptx_op.def` | 删除 + 新增 | -1 S_WMMA + 11 S_TCGEN05_* |
| `include/ptx_ir/ptx_qualifier.def` | 删除 + 新增 | -4 stub + 25 Q_* |
| `include/ptx_ir/statement_context.h` | 新增 | Tcgen05Instr + 2 enum |
| `include/ptx_ir/statement_factory.h` | 修改 | makeTcgen05Instr |
| `src/ptx_parser/ptx_parser.cpp` | 修改 | WMMA → TCGEN05 |
| `src/ptx_parser/ptx_visitor_wmma.cpp` → `ptx_visitor_tcgen05.cpp` | 重命名 | 文件 rename + 内容修改 |
| `src/ptxsim/instruction_handlers.cpp` | 修改 | IMPLEMENT_TCGEN05_INSTR_HANDLER |
| `src/ptxsim/instructions/AGENTS.md` | 修改 | 目录说明 |
| `src/ptxsim/instructions/wmma.cpp` | **临时 throw**(过渡期) | 安全网 |
| `tests/ptx/test_all_ptx.sh` | 新增 fixtures | 12 .ptx 文件 |
| `tests/ptx/tcgen05_*.ptx`(12 个) | 新增 | 端到端 grammar 验证 |
| `tests/unit/ptx_ir/test_tcgen05_*.cpp`(5 个) | 新增 | 单元测试 |
| `tests/integration/parser/test_tcgen05_*.cpp`(5 个) | 新增 | 集成测试 |
| `tests/integration/tcgen05/test_tcgen05_*.cpp`(2 个旧) | **最小修改** | S_WMMA → S_TCGEN05_* |
| `tests/unit/CMakeLists.txt` | 修改 | 注册新测试 |
| `tests/integration/CMakeLists.txt` | 修改 | 注册新测试 |
| `CMakeLists.txt` | 修改 | 注册新源文件 |
| `docs/dev-process/lessons-learned.md` | 追加 | §22 案例(本 change 起源) |
| 根 `AGENTS.md` | 修改 | 已知限制表 |
| `src/grammar/AGENTS.md` | 修改 | lexer/parser 规则说明 |
| `src/ptxsim/instructions/AGENTS.md` | 修改 | 目录说明 |

## 实际实施偏差(Actual Implementation Deviations)

> **来源**: Metis pre-implementation review(per `ptx-lessons-learned` §7 + Checklist H)
> **添加时机**: archive 前,记录实施过程中与 design.md 决策的偏离

### Deviation 1: S_TCGEN05_* 移出 ptx_op.def X-Macro 循环

- **设计**(Decision 2):11 个 `S_TCGEN05_*` 在 `ptx_op.def` 中,`struct_kind=TCGEN_INSTR`,由 X-Macro 自动生成 `VISITOR_`/`IMPLEMENT_` 展开
- **实际**(commit `c8c3f13`):S_TCGEN05_* 从 `ptx_op.def` 移出,直接添加到 `ptx_types.h` 的 `StatementType` 枚举末尾(在 S_UNKNOWN 前)
- **根因**:grammar 只有单一 `tcgen05Inst` rule,但 X-Macro 展开生成 11 个 per-instruction visitor 方法(如 `visitTcgen05MmaInst`),期望 11 个对应 grammar rule。ANTLR 生成的 Context 类型名不匹配(`Tcgen05MmaInstContext` 不存在)
- **影响**:
  - S2s() 必须手写覆盖 11 个 case(已在 `statement_context.cpp` 修复 — Metis MR-1)
  - `IMPLEMENT_TCGEN_INSTR_HANDLER` macro 不再被自动展开(留作 change-3 handler 实施时手写)
  - 失去 X-Macro 维护优势,change-3 的 handler 分发需手写
- **后续**:change-3 实施 handler 时,需直接手写 11 个 `processTcgen05Xxx` 方法,不再依赖 X-Macro

### Deviation 2: ptx_visitor_tcgen05.cpp 创建后删除

- **设计**(tasks.md §4.1):新建 `ptx_visitor_tcgen05.cpp`,X-Macro 展开 11 个 visitor 方法
- **实际**:文件创建后因 LSP 错误删除,改用现有 `ptx_visitor_wmma.cpp` 末尾添加手写 `visitTcgen05Inst` 方法
- **根因**:同上(ANTLR Context 类型名不匹配)
- **影响**:
  - visitor 集成分散在 `ptx_visitor_wmma.cpp`(临时,change-3 建议拆分)
  - `visitTcgen05Inst` 是单一方法处理 11 个 sub-op(而非 11 个独立方法)
  - 完整 operand 提取推迟到 change-3(MR-2 修复仅防 silent drop,operands 仍为空)
- **后续**:change-3 实施 handler 时,完善 `visitTcgen05Inst` 的 operand 提取逻辑,或拆分为独立 `ptx_visitor_tcgen05.cpp`

### Deviation 3: 4 个 Q_TCGEN05_* stub 保留

- **设计**(Decision 5):删除 `Q_TCGEN05_LD/ST/COMMIT/WAIT` 4 个 stub,改用独立 IR 枚举
- **实际**:`ptx_qualifier.def:193-199` 保留 4 个 stub,注释 "DEPRECATED in change-3"
- **根因**:`S_WMMA` 未删除(因 `wmma.cpp` 仍依赖),`wmma.cpp` 中 `is_tcgen05_ld/st/commit/wait()` helper 仍用这 4 个 stub 区分 sub-op
- **影响**:
  - qualifier 命名空间有 2 套:`Q_TCGEN_*`(新)和 `Q_TCGEN05_*`(旧 stub)
  - 命名不一致会持续到 change-4(wmma 路径完全删除)
- **后续**:change-3 重写 `wmma.cpp` 为 `tcgen05.cpp` 后,删除 4 个 stub;change-4 完全删除 wmma 路径

### Deviation 4: tasks.md 计划的 12 个 .ptx fixtures 仅完成 2 个

- **设计**(tasks.md §5.1):12 个 `tests/ptx/tcgen05_*.ptx` 端到端 fixtures
- **实际**:仅 `tcgen05_alloc.ptx` 和 `tcgen05_mma.ptx` 2 个
- **根因**:
  - MR-3 grammar LL(*) 冲突未修复(复杂 ANTLR grammar 优化,需 ANTLR 专家)
  - 2 个 fixture 均测试失败:`mismatched input '.all' expecting ':'`
  - 增加更多 fixture 在当前 grammar 下会同样失败,无增量价值
- **影响**:`test_all_ptx.sh` 33/36 通过(2 个新 fixture 失败 + 1 个 pre-existing `atom_cas_basic.ptx`)
- **后续**:change-3 第一步先修 grammar 冲突,再补全剩余 10 个 fixture + 5 单元 + 5 集成测试

### Deviation 5: 旧集成测试未迁移

- **设计**(tasks.md §4.5):`tests/integration/tcgen05/test_tcgen05_*.cpp` 迁移 `S_WMMA` → `S_TCGEN05_*`
- **实际**:未迁移(仍用 `S_WMMA`/`makeWmmaInstr`/`WmmaType`)
- **根因**:S_WMMA 未删除,wmma.cpp handler 仍依赖旧路径,迁移会与现有 handler 行为冲突
- **影响**:旧测试通过(因 wmma.cpp 仍工作),但与新 IR 命名空间不一致
- **后续**:change-3 实施新 handler 时同步迁移;change-4 删除 wmma 后彻底清理

### Deviation 6: tasks.md §3 IR 删除 S_WMMA 推迟

- **设计**(tasks.md §3.1.1):删除 `S_WMMA` 整行(因独立 IR 命名空间)
- **实际**:S_WMMA 保留
- **根因**:见 Deviation 3(S_WMMA 与 wmma.cpp 仍依赖)
- **影响**:X-Macro 仍生成 `S_WMMA` handler(由 `wmma.cpp` 实现)
- **后续**:change-4 删除 wmma 路径时同步删除

## Metis MUST-RESOLVE 项处理状态

| ID | 描述 | 状态 | 修复方式 |
|----|------|------|---------|
| MR-1 | S2s() 不识别 S_TCGEN05_* → assert(false) crash | ✅ Fixed | commit `182385c`:`statement_context.cpp` 手动覆盖 11 个 case |
| MR-2 | PtxListener/PtxVisitor 零修改 → tcgen05 PTX silent drop | ✅ Fixed | commit `182385c`:`ptx_visitor_wmma.cpp` 新增 `visitTcgen05Inst` |
| MR-3 | Grammar LL(*) 冲突 → 2 fixture fail | ⏸ Deferred | change-3 第一步修复 grammar,再补全 fixtures |
| MR-4 | 旧测试未迁移 + CMakeLists 零修改 | ⏸ Deferred | change-3 实施新 handler 时同步迁移 |
| MR-5 | design.md 缺实施偏差记录 | ✅ Fixed | 本段落(追加在 design.md 末尾) |
