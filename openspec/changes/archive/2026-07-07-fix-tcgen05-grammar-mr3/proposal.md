# Fix tcgen05 Grammar LL(*) Conflict + Migrate Old Tests (Change-1 MR-3 + MR-4)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **前置 change**: `archive/2026-07-06-implement-tcgen05-syntax-ir` (Change-1, archived)
> **跨 Change 拆分**: 本 change 是 4-change 路线图的第 3a 步,handler 实施的**硬前置**
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §6(artifacts-first)+ §7(Pre-impl review)

## Why

Change-1(archived `2026-07-06-implement-tcgen05-syntax-ir`)在 Metis pre-implementation review 中发现 5 个 MUST-RESOLVE 项:
- **MR-1**(已修):S2s crash → `commit 182385c`
- **MR-2**(已修):silent drop → `commit 182385c`
- **MR-3**(**本 change 修复**):grammar LL(*) 冲突 → 2 个 .ptx fixture fail(`tcgen05_alloc.ptx`, `tcgen05_mma.ptx` 在 `test_all_ptx.sh` 报错 `mismatched input '.all' expecting ':'`)
- **MR-4**(**本 change 修复**):2 个旧集成测试仍用 `makeWmmaInstr`/`WmmaType`,与新 IR 命名空间不兼容
- **MR-5**(已修):documentation → `commit 220e712`

handler 实施(change-3b)硬依赖 grammar 正确性,故本 change 是 **change-3b 的强制前置**。

**本 change 解决 MR-3 + MR-4**,完成后 grammar 100% 通过 `test_all_ptx.sh`,2 个旧测试迁移到新 IR 命名空间。

## What Changes

### 新增

| 文件 | 范围 |
|------|------|
| `tests/ptx/tcgen05_{dealloc,relinquish,ld,st,cp,cp_multicast,mma_block_scale,commit,wait,fence}.ptx` | 10 个新 .ptx fixtures(总 12) |
| `tests/ptx/tcgen05_{alloc,mma}.ptx` | **修复现有 2 个**(grammar LL(*) 冲突导致 fail) |

### 修改

| 文件 | 范围 |
|------|------|
| `src/grammar/ptxInstructions.g4` | 修复 `tcgen05Qual` 规则的 LL(*) 预测冲突(2 fixture fail 根因) |
| `src/grammar/ptxLexer.g4` | 若 token 顺序冲突,调整最长匹配优先级 |
| `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp` | 添加 `makeTcgen05Instr` 编译期别名验证(不加入执行向量) + `WmmaType::WMMA_MMA` → `Tcgen05OpKind::MMA` |
| `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp` | 添加 5 个编译期别名验证 + 5 个 `makeWmmaInstr` 引用 + 5 个 `WmmaType` 引用 |
| `include/ptx_ir/ptx_qualifier.def` | **保留** 4 个 `Q_TCGEN05_LD/ST/COMMIT/WAIT` stub(推迟到 [implement-tcgen05-handlers-core](../implement-tcgen05-handlers-core/),见 design.md D4;wmma.cpp 仍有 8 引用 + 2 测试文件 6 引用) |

### 不修改(范围外,留待后续 change)

- ❌ `src/ptxsim/instructions/wmma.cpp` 中的 5 个 `execute_tcgen05_*` 函数(change-3b scope)
- ❌ `src/ptx_parser/ptx_visitor_wmma.cpp` 中 `visitTcgen05Inst` 的 operand 提取完善(change-3b scope,因为 handler 需要 qualifiers)
- ❌ 任何 handler 实现(change-3b scope)
- ❌ 删除 `S_WMMA`/`WmmaInstr`/`WmmaType`(change-4 scope)
- ❌ 删除整个 `wmma.cpp` 文件(change-4 scope)

## Non-Goals

### 显式拒绝

- ❌ 不实现 `cp.async.bulk.tensor.*`(TMA 加载指令)→ 独立 follow-up change `implement-cp-async-bulk-tensor`
- ❌ 不实现 tcgen05 handler(change-3b scope)
- ❌ 不修改 4 个底层基础设施子系统(Change-2 scope,先审计)
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8
- ❌ 不实现 `cta_group::2` distributed_smem

### 范围限制

- 仅 grammar fix + 旧测试迁移,**无新功能**
- 仅 f16 mma(其他 dtype 留待 change-3b 或 follow-up)
- 性能对标不要求
- 5 handler 留待 change-3b

## Goals

### Phase 1: 修复 grammar(1-2 个 commit)

1. 修复 `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` 的 LL(*) 预测冲突
2. 2 个现有 .ptx fixture(`tcgen05_alloc.ptx`, `tcgen05_mma.ptx`)在 `test_all_ptx.sh` 中 PASS
3. 跑 `cmake --build build --target GenerateParser && cmake --build build` 验证 ANTLR 重新生成无错
4. 跑 `ctest -L "unit|integration" --output-on-failure` 验证零回归

### Phase 2: 补全 .ptx fixtures(1 commit)

1. 创建 10 个新 .ptx fixtures(基于 Change-1 specs `tcgen05-parse-tests` 的 scenarios)
2. 所有 12 个 fixtures 在 `test_all_ptx.sh` 中 PASS
3. 注册到 `test_all_ptx.sh`

### Phase 3: 编译期别名验证(1 commit)

1. `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp` 添加 `makeTcgen05Instr` 编译期别名(不加入执行向量,仅编译 + `static_assert` 验证)
2. `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp` 添加 `makeTcgen05Instr` 编译期别名(同上)
3. 保留 4 个 `Q_TCGEN05_*` stub qualifiers(推迟到 [implement-tcgen05-handlers-core](../implement-tcgen05-handlers-core/),见 design.md D4;wmma.cpp 仍有 8 引用 + 2 测试文件 6 引用)
4. 旧测试仍 PASS(behavior 不变,旧路径执行;编译期别名仅验证 factory 正确)
5. `ctest -R tcgen05 -V` 验证

### Phase 4: Archive(1 commit,per Checklist G)

1. 跑 `openspec archive fix-tcgen05-grammar-mr3 --yes`
2. 跑 `cd build && ctest --output-on-failure` 全量验证
3. 跑 `./tests/ptx/test_all_ptx.sh` 全量验证
4. commit archive 目录

## Capabilities

### New Capabilities

- `tcgen05-grammar-fix`:grammar LL(*) 冲突修复(spec 范围不变,补到 `tcgen05-grammar` spec)
- `tcgen05-fixtures`:12 个 .ptx 端到端 fixtures(补到 `tcgen05-parse-tests` spec)
- `tcgen05-old-test-migration`:2 个旧集成测试添加编译期别名验证(补到 `tcgen05-parse-tests` spec)

### Modified Capabilities

- `tcgen05-grammar`:spec 修订(MR-3 冲突已修)
- `tcgen05-parse-tests`:spec 修订(12 fixtures 全 PASS,旧测试已迁移)
- `tcgen05-ir-types`:spec 修订(Q_TCGEN05_* 4 stub 保留,推迟到 implement-tcgen05-handlers-core)

## Impact

### 影响的代码(预计)

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `src/grammar/ptxInstructions.g4` | 修改(grammar fix) | ±80 |
| `src/grammar/ptxLexer.g4` | 修改(可能) | ±20 |
| `tests/ptx/tcgen05_*.ptx`(10 个新 + 2 个修) | 新增/修复 | +220 |
| `tests/integration/tcgen05/test_tcgen05_*.cpp`(2 个迁移) | 修改 | ±60 |
| `include/ptx_ir/ptx_qualifier.def` | 不修改(4 stub 保留,推迟到 implement-tcgen05-handlers-core) | 0 |
| **总计** | | **+376** |

### 影响的依赖

- `ptx-grammar-modification` skill(强制 TDD 流程,修改 .g4 前必跑 baseline)
- `cuobjdump -xptx` 工具(若 E2E 需真实 PTX,本 change 不需要)

### 不影响的依赖

- `src/ptxsim/memory/*`, `src/ptxsim/cluster/*`, `src/ptxsim/async/*`(Change-2 scope)
- `src/ptxsim/instructions/wmma.cpp`(change-3b scope)
- grammar 之外的 IR 类型(Change-1 已完成)

### 影响的文档

- `docs/adr/0016-blackwell-only-tcgen05.md`(追加本 change archive commit 引用)
- `openspec/changes/archive/2026-07-XX-fix-tcgen05-grammar-mr3/`(archive)
- `openspec/specs/tcgen05-{grammar,parse-tests,ir-types}/spec.md`(修订)

## Design-Time Checklist (Lessons-Learned)

### 函数审计完整性(此 change 主要改 grammar + tests)

- [x] Baseline 函数清单:`tcgen05Qual` 规则 16+ alternations(per Change-1 design.md)
- [x] 现有测试数量已修正:2 fixtures fail,11 fixtures 缺
- [x] 跨模块状态翻译:无(本 change 不动 handler)
- [x] invariant 清单:ANTLR grammar 必须 deterministic(LL(*) 冲突违反此)

### 多 Phase 推进(4 个 atomic commits)

- [x] Phase 1: grammar fix(独立 commit,先跑 baseline 验证)
- [x] Phase 2: 补全 fixtures(独立 commit,基于 Phase 1 修复)
- [x] Phase 3: 旧测试迁移(独立 commit,handler 不动)
- [x] Phase 4: archive(独立 commit,per Checklist G)
- [x] 基线 worktree 计划:`.worktrees/baseline-grammar-fix`(per `ptx-lessons-learned` §4)
- [x] 失败处理策略:已有测试回归 → 立即 revert 该 Phase

### 文档同步

- [x] ADR 追加段落已规划
- [x] tasks.md 任务规划(待实施前)
- [x] archive 路径已列出

### 实施前必跑(per `ptx-lessons-learned` §7)

- [ ] **Metis pre-implementation review**:验证 grammar fix 方案、fixture 列表
- [ ] 跑 `./tests/ptx/test_all_ptx.sh` 记录 baseline(2 fail)
- [ ] 跑 `grep -c "TEST_CASE" tests/integration/tcgen05/test_tcgen05_*.cpp` 记录旧测试结构
- [ ] 跑 `grep "Q_TCGEN05" include/ptx_ir/ptx_qualifier.def` 确认 4 stub 位置

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| Change-1 (archive/2026-07-06) | **fix-tcgen05-grammar-mr3** | implement-tcgen05-handlers-core (change-3b) |

- **Change-1 → 本 change**:依赖 Change-1 建立的 `S_TCGEN05_*` 命名空间 + `Tcgen05Instr` struct + `makeTcgen05Instr` factory
- **本 change → change-3b**:handler 实施依赖 grammar 100% 正确(否则 handler 测试的失败可能归因错)
- **本 change → change-3d**:6 个剩余 handler 依赖 grammar(已修) + 旧测试已迁移
- **本 change → Change-2**:审计可独立进行(不需要等本 change)
