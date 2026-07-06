# Implement Blackwell tcgen05 Handlers (5 instructions + full test coverage)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **前置 changes**: 
>   - `archive/2026-07-06-implement-tcgen05-syntax-ir/` (Change-1, archived)
>   - `extend-blackwell-tcgen05-infra` (Change-2, pending)
> **4-Change 拆分**: 本 change 是第 3 步(共 4 步),实施 5 个 tcgen05 指令的真实 handler 实现
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §7(Pre-impl review)

## Why

Change-1 建立了独立 tcgen05 命名空间(grammar + IR,11 个 `S_TCGEN05_*` 枚举 + `Tcgen05Instr` struct),Change-2 审计并补全 4 个底层子系统(TMA/TMEM/Cluster/TcQueue)。但 **`src/ptxsim/instructions/wmma.cpp` 中的 `execute_tcgen05_*` 函数仍沿用 wmma 命名空间**(commit `35808d6` archive),与新 IR 不兼容。

本 change 将:
1. 修复 Change-1 MR-3(grammar LL(*) 冲突)— 实施 handler 前必须先修
2. 迁移 2 个旧集成测试(`S_WMMA` → `S_TCGEN05_*`,Change-1 MR-4 推迟项)
3. 新建 `src/ptxsim/instructions/tcgen05.cpp`(替代 `wmma.cpp` 中的 tcgen05 部分),实施 5 个真实 handler
4. 三套测试:补全 10 个 .ptx fixtures + 5 单元 + 5 集成 + 1 真实 E2E kernel
5. 删除 `wmma.cpp` 中的 tcgen05 部分(但保留 pre-Blackwell 路径直到 Change-4)

## What Changes

### 新增

| 文件 | 范围 |
|------|------|
| `src/ptxsim/instructions/tcgen05.cpp` | 5 个 `execute_tcgen05_*` 函数(MMA/LD/ST/COMMIT/WAIT),从 wmma.cpp 提取并适配新 IR |
| `tests/ptx/tcgen05_{alloc,dealloc,relinquish,ld,st,cp,cp_multicast,mma,mma_block_scale,mma_ws,commit,wait,fence}.ptx` | 13 个 .ptx fixtures(Change-1 仅 2 个) |
| `tests/unit/ptx_ir/test_tcgen05_*.cpp` | 5 个单元测试(qualifier/opkind/dtype/stmt_factory/instr_struct) |
| `tests/integration/parser/test_tcgen05_*_parse.cpp` | 5 个集成测试(mma/ld/st/commit/wait 端到端 parse → IR) |
| `tests/e2e/kernel/test_tcgen05_real.cu` | 1 个真实 CUDA kernel E2E(用 cuobjdump 提取的 tcgen05 PTX) |

### 修改

| 文件 | 范围 |
|------|------|
| `src/grammar/ptxInstructions.g4` | **修复 Change-1 MR-3**:grammar LL(*) 冲突 |
| `src/grammar/ptxLexer.g4` | 移除冲突 token 或调整顺序 |
| `src/ptx_parser/ptx_visitor_wmma.cpp` | 完善 `visitTcgen05Inst` 的 operand 提取(MR-2 推迟项) |
| `src/ptxsim/instructions/wmma.cpp` | 移除 tcgen05 相关代码(保留 pre-Blackwell 路径) |
| `src/ptxsim/instructions/AGENTS.md` | 添加 `tcgen05.cpp` 说明 |
| `src/ptxsim/CMakeLists.txt` | 注册新源文件 |
| `tests/unit/CMakeLists.txt` | 注册新单元测试 |
| `tests/integration/CMakeLists.txt` | 注册新集成测试 + 迁移旧测试 |
| `tests/e2e/CMakeLists.txt` | 注册新 E2E kernel |
| `tests/integration/tcgen05/test_tcgen05_*.cpp`(2 个旧) | 迁移 `S_WMMA` → `S_TCGEN05_*`(Change-1 MR-4) |
| `include/ptx_ir/ptx_qualifier.def` | 删除 4 个 Q_TCGEN05_* stub(Change-1 Deviation 3) |
| `include/ptx_ir/ptx_types.h` | 验证 S_TCGEN05_* enum 仍正确 |
| 根 `AGENTS.md` | 更新已知限制表(tcgen05 handler 已实现) |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 追加 Phase 1-3 archive commit 引用 |

### 不修改(范围外,留待 Change-4)

- ❌ 删除 `S_WMMA` 枚举(Change-4 scope)
- ❌ 删除 `WmmaInstr` struct(Change-4 scope)
- ❌ 删除 `src/ptxsim/instructions/wmma.cpp` 整个文件(Change-4 scope)
- ❌ 修改 4 个基础设施子系统(Change-2 scope)
- ❌ 修改 grammar 之外的 IR 类型(Change-1 已完成)

## Goals

### Phase 1: 修复 grammar(Change-1 MR-3,独立 commit)

1. 修复 `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` 的 LL(*) 冲突
2. 13 个 .ptx fixtures 全部 `test_all_ptx.sh` 通过
3. `ctest --output-on-failure` 零回归

### Phase 2: 迁移旧测试(Change-1 MR-4,独立 commit)

1. `tests/integration/tcgen05/test_tcgen05_*.cpp` 改用 `S_TCGEN05_*` + `Tcgen05OpKind` + `makeTcgen05Instr`
2. 旧测试仍 PASS(behavior 不变,仅 IR 命名空间)
3. 4 个 Q_TCGEN05_* stub 删除(因 wmma.cpp 不再用)

### Phase 3: 实施 handler(独立 commit,核心工作)

1. `src/ptxsim/instructions/tcgen05.cpp` 5 个函数真实实现
2. 每个 handler 有 per-`// UNVERIFIED-AGAINST-HARDWARE` 注释(per ADR-0016)
3. 32 lane × 8x4 fragment arithmetic(mma)
4. 128-byte TMA desc → TMEM slot 0 拷贝(ld/st)
5. `tc_queue.commit(1)` + cluster arrive(commit)
6. `tc_queue.wait(warp, 0, 1)` + cluster wait(wait)

### Phase 4: 三套测试(独立 commit)

1. 5 单元测试(PER instruction, ≥5 assertions each)
2. 5 集成测试(端到端 PTX → IR → handler 调用)
3. 1 真实 E2E kernel(用 cuobjdump 提取的 tcgen05.mma GEMM)
4. 13 .ptx fixtures 全部通过(grammar 修复 + handler 实施)

### Phase 5: 文档 + Archive(独立 commit)

1. `src/ptxsim/instructions/AGENTS.md` 更新
2. 根 `AGENTS.md` 更新已知限制表
3. ADR-0016 追加 Phase 1-3 commit 引用
4. archive

## Non-Goals

### 显式拒绝(per ADR-0016 锁定)

- ❌ 不实现 `cp.async.bulk.tensor.*`(TMA 加载指令,留待 follow-up change)
- ❌ 不实现 `tensormap.create/replace` host API 拦截
- ❌ 不修改 4 个基础设施子系统(Change-2 scope)
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8
- ❌ 不实现 `cta_group::2` distributed_smem(留待 cta_group::2 follow-up)

### 范围限制

- 仅 f16 mma(per ADR-0016 Phase 1 scope)
- 5 个 handler(MMA/LD/ST/COMMIT/WAIT),其他 6 个(ALLOC/DEALLOC/RELINQUISH/CP/FENCE/MMA_WS)留待 Change-3.5 follow-up
- 性能对标不要求(仅 functional correctness)
- S2s "UNVERIFIED" 注释保留(handler 与硬件验证由后续 change 处理)

## Capabilities

### New Capabilities

- `tcgen05-handlers`: 5 个 Blackwell 指令的真实 handler 实现(mma/ld/st/commit/wait)
- `tcgen05-handler-tests`: 三套测试(单元/集成/E2E)覆盖新 handler
- `tcgen05-grammar-fix`: 修复 Change-1 MR-3(LL(*) 冲突)
- `tcgen05-old-test-migration`: 2 个旧集成测试从 WMMA 迁移到 TCGEN05 命名空间

### Modified Capabilities

- `wmma-tensor-core`: 移除 tcgen05 相关段落(handler 已在 tcgen05 中),但保留 pre-Blackwell 路径
- `tcgen05-grammar`: 修复 MR-3 冲突(spec 范围不变)
- `tcgen05-parse-tests`: 补全 10 个缺失的 .ptx fixtures(从 2 → 13)

## Impact

### 影响的代码(预计)

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `src/grammar/ptxInstructions.g4` | 修改(grammar fix) | ±50 |
| `src/grammar/ptxLexer.g4` | 修改(可能) | ±20 |
| `src/ptxsim/instructions/tcgen05.cpp` | 新增 | +500 |
| `src/ptxsim/instructions/wmma.cpp` | 修改(移除 tcgen05) | -200 |
| `src/ptx_parser/ptx_visitor_wmma.cpp` | 修改(operands) | +50 |
| `tests/ptx/tcgen05_*.ptx`(11 个新) | 新增 | +200 |
| `tests/unit/ptx_ir/test_tcgen05_*.cpp`(5 个) | 新增 | +200 |
| `tests/integration/parser/test_tcgen05_*.cpp`(5 个) | 新增 | +250 |
| `tests/e2e/kernel/test_tcgen05_real.cu`(1 个) | 新增 | +150 |
| `tests/integration/tcgen05/test_tcgen05_*.cpp`(2 个旧) | 迁移 | ±50 |
| `include/ptx_ir/ptx_qualifier.def` | 修改(删除 stub) | -4 |
| 多个 CMakeLists.txt | 注册新文件 | +30 |
| `docs/adr/0016-*.md` + AGENTS.md | 文档 | +30 |
| **总计** | | **+1226** |

### 影响的依赖

- `oracle-prompting` skill(若 TmaDescriptor 偏移需硬件 dump 验证)
- `ptx-debug` skill(handler 调试)
- `three-mode-testing` skill(三套测试)
- `cuobjdump -xptx` 工具(若 E2E 需真实 PTX)

### 不影响的依赖

- `src/ptxsim/memory/*`(Change-2 scope)
- `src/ptxsim/cluster/*`(Change-2 scope)
- `src/ptxsim/async/*`(Change-2 scope)
- grammar 之外的 IR 类型

### 影响的文档

- 根 `AGENTS.md`(已知限制表)
- `src/ptxsim/instructions/AGENTS.md`(目录说明)
- `docs/adr/0016-blackwell-only-tcgen05.md`(更新记录)
- `docs/dev-process/lessons-learned.md`(可选 §24 新案例)

## Design-Time Checklist (Lessons-Learned)

### 函数审计完整性

- [x] Baseline 函数清单:`wmma.cpp` 中 5 个 `execute_tcgen05_*` 函数
- [x] 锁点审计:5 个函数均无锁调用(纯计算)
- [x] 跨模块状态翻译:handler 调 `cta->tmem()` / `cta->tma_descriptor_store()` / `cta->tc_queue()` / `cta->cluster_context()`
- [x] invariant 清单:per-warp ordering、CTA 隔离、commit-group counter 原子性

### 多 Phase 推进(5 个 atomic commits)

- [x] Phase 1: grammar fix(独立 commit,先跑 baseline 验证)
- [x] Phase 2: 旧测试迁移(独立 commit,handler 不动)
- [x] Phase 3: handler 实施(核心,最复杂)
- [x] Phase 4: 三套测试(独立 commit)
- [x] Phase 5: 文档 + archive(独立 commit,per Checklist G)
- [x] 基线 worktree 计划:`.worktrees/baseline-tcgen05-handlers`
- [x] 失败处理策略:已有测试回归 → 立即 revert 该 Phase

### 文档同步

- [x] AGENTS.md 同步项已列出
- [x] ADR 追加段落已规划
- [x] tasks.md 任务规划(待实施前)

### 实施前必跑(per `ptx-lessons-learned` §7)

- [ ] **Metis pre-implementation review**:验证 grammar fix 方案、handler 实现范围
- [ ] 验证 `wc -l src/ptxsim/instructions/wmma.cpp`(约 564 行,需 -200 移除 tcgen05)
- [ ] 验证 Change-2 baseline 测试通过(若 Change-2 已 archive)
- [ ] 验证 cuobjdump -xptx 可用(若 E2E 需真实 PTX)
