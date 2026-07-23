# Implement Blackwell tcgen05 Core Handlers (5 instructions + golden-value tests)

> **架构依据**: [ADR-0016](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) Accepted
> **前置 changes**:
>   - `archive/2026-07-06-implement-tcgen05-syntax-ir` (Change-1, archived)
>   - `fix-tcgen05-grammar-mr3` (Change-3a, pending) — **硬前置**(grammar 必须 100% 正确)
>   - `extend-blackwell-tcgen05-infra` (Change-2, pending) — **软前置**(审计报告 ≥L2)
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §7(Pre-impl review)+ Metis MR-2(已拆分为 core/extended)

## Why

Change-1 建立了独立 tcgen05 命名空间(grammar + IR),Change-3a 修复 grammar + 迁移旧测试,Change-2 审计基础设施。**但 `wmma.cpp` 中 4 个 `execute_tcgen05_*` 函数(ld/st/commit/wait)+ 1 个 inline mma handler(line 352)**仍沿用 wmma 命名空间,本 change 将其实施为独立 `tcgen05.cpp` handler(共 5 个 `processTcgen05Xxx` 函数)。

**5 个核心 handler**(per ADR-0016 Phase 1+2):
1. `tcgen05.mma` — 32 lane × 8x4 f16 fragment arithmetic(per `wmma.cpp:374-420`)
2. `tcgen05.ld` — 128 字节 TMA desc → TMEM slot 0(per `wmma.cpp:423-461`)
3. `tcgen05.st` — 128 字节 TMEM slot 0 → TMA desc(per `wmma.cpp:463-500`)
4. `tcgen05.commit` — `tc_queue().commit(1)` + cluster arrive(per `wmma.cpp:502-532`)
5. `tcgen05.wait` — `tc_queue().wait(warp, 0, 1)` + cluster wait(per `wmma.cpp:534-565`)

**Metis MR(本 change 必须解决)**:
- MR-1(已修):S2s crash
- MR-2(已修):silent drop
- MR-3(Change-3a 修):grammar LL(*)
- MR-4(Change-3a 修):旧测试迁移
- MR-5(已修):documentation
- **新发现**:"真实实现" acceptance criteria 未定义 → 本 change design.md 明确定义

## What Changes

### 新增

| 文件 | 范围 |
|------|------|
| `src/ptxsim/instructions/tcgen05.cpp` | 5 个 `processTcgen05Xxx` 函数,从 wmma.cpp 提取并适配新 IR |
| `tests/unit/ptx/test_tcgen05_qualifier.cpp` | Qualifier 枚举单元测试 |
| `tests/unit/ptx/test_tcgen05_opkind.cpp` | Tcgen05OpKind 枚举单元测试 |
| `tests/unit/ptx/test_tcgen05_dtype.cpp` | Tcgen05Dtype 枚举单元测试 |
| `tests/unit/ptx/test_tcgen05_statement_factory.cpp` | makeTcgen05Instr 工厂单元测试 |
| `tests/unit/ptx/test_tcgen05_instr_struct.cpp` | Tcgen05Instr struct 字段单元测试 |
| `tests/integration/tcgen05/test_tcgen05_mma_parse.cpp` | mma 端到端 parse → IR 集成测试 |
| `tests/integration/tcgen05/test_tcgen05_ld_parse.cpp` | ld 集成测试(验证 num_regs 字段) |
| `tests/integration/tcgen05/test_tcgen05_st_parse.cpp` | st 集成测试 |
| `tests/integration/tcgen05/test_tcgen05_commit_parse.cpp` | commit 集成测试(验证 mbarrier qualifier) |
| `tests/integration/tcgen05/test_tcgen05_wait_parse.cpp` | wait 集成测试(验证 .load/.store) |

> **2026-07-07 修正**:Day 1 验证发现 `tests/unit/ptx_ir/` 和 `tests/integration/parser/` 目录不存在,实际目录是 `tests/unit/ptx/` 和 `tests/integration/tcgen05/`。
| `tests/e2e/kernel/test_tcgen05_mma_gemm.cu` | 1 个真实 CUDA kernel E2E(用 cuobjdump 提取的 tcgen05.mma GEMM) |

### 修改

| 文件 | 范围 |
|------|------|
| `src/ptx_parser/ptx_visitor_wmma.cpp` | 完善 `visitTcgen05Inst` 的 operand 提取(MR-2 推迟项) |
| `src/ptxsim/instructions/wmma.cpp` | 移除 tcgen05 相关代码(保留 pre-Blackwell 路径直到 Change-4) |
| `src/ptxsim/instructions/AGENTS.md` | 添加 `tcgen05.cpp` 说明 |
| `src/ptxsim/CMakeLists.txt` | 注册 `tcgen05.cpp` |
| `tests/unit/CMakeLists.txt` | 注册 5 个新单元测试 |
| `tests/integration/CMakeLists.txt` | 注册 5 个新集成测试 |
| `tests/e2e/CMakeLists.txt` | 注册新 E2E kernel |
| 根 `AGENTS.md` | 更新已知限制表(tcgen05 handler 已实现) |
| `docs/adr/ADR-0016-blackwell-only-tcgen05.md` | 追加 Phase 1-2 archive commit 引用 |

### 不修改(范围外)

- ❌ 删除 `S_WMMA` 枚举(Change-4 scope)
- ❌ 删除 `WmmaInstr` struct(Change-4 scope)
- ❌ 删除整个 `wmma.cpp` 文件(Change-4 scope)
- ❌ 修改 4 个基础设施子系统(Change-2 scope)
- ❌ 修改 grammar(Change-3a scope)
- ❌ 修改 IR 类型(Change-1 已完成)
- ❌ 实施其他 6 个 handler(change-3d: ALLOC/DEALLOC/RELINQUISH/CP/FENCE/MMA_WS)
- ❌ 不实现 `cp.async.bulk.tensor.*`(独立 follow-up `implement-cp-async-bulk-tensor`)
- ❌ 不实现 `cta_group::2` distributed_smem(独立 follow-up)
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8

## Non-Goals

### 显式拒绝

- ❌ 不实现 sm_120 sparse / FP4 / mxfp8(per ADR-0016 锁定)
- ❌ 不实现 `cta_group::2` distributed_smem(per ADR-0016 Open Question #2)
- ❌ 不实现 `cp.async.bulk.tensor.*`(TMA 加载指令 → 独立 follow-up)
- ❌ 不实现其他 6 个 handler(change-3d scope)

### 范围限制

- 仅 f16 mma(per ADR-0016 Phase 1 scope)
- 仅 5 个 core handler(MMA/LD/ST/COMMIT/WAIT)
- 性能对标不要求(仅 functional correctness)
- S2s "UNVERIFIED" 注释保留(handler 数值正确性由 golden-value test 验证,不是硬件对比)

## Goals

### Phase 1: 完善 visitor operand 提取(1 commit)

1. `visitTcgen05Inst` 完整提取 qualifiers + operands(MR-2 推迟项)
2. 跑 `ctest -L "integration;tcgen05" -V` 验证 5 集成测试 PASS
3. 跑 `./tests/ptx/test_all_ptx.sh` 验证 13 fixtures 仍 PASS(无 regression)

### Phase 2: 实施 5 个 handler(1-2 commits,核心)

1. `src/ptxsim/instructions/tcgen05.cpp` 5 个 `processTcgen05Xxx` 函数
2. 每个 handler 有 per-`// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16` 注释(per ADR-0016)
3. **Acceptance Criteria**(本 change 明确,Metis E.1 修复):
   - **mma**:32 lane × 8x4 fragment,对比 **golden value**(per `tests/ptx/reference/tcgen05_mma_golden.h`,**来源:复用 `wmma.cpp:374-420` 现有 inline mma + PTX ISA §9.7.16 手算**——Cutlass 3.x 在 Day 1 验证时确认环境不可用,见 design.md D1 修正)
   - **ld**:已知 128 字节 blob → 验证 TMEM slot 0 内容 byte-by-byte
   - **st**:已知 TMEM slot 0 → 验证 128 字节 dest byte-by-byte
   - **commit**:验证 `tc_queue.commit_count` + cluster `arrive_count` 增加
   - **wait**:验证 warp `is_blocked` 转换(commit 后 wait 解除 blocked)
4. 跑 `ctest -L "unit;tcgen05" -V` 验证 5 单元测试 PASS
5. 跑 `ctest -L "integration;tcgen05" -V` 验证 5 集成测试 PASS
6. 跑 `cmake --build build` 全量编译通过
7. 跑 `cd build && ctest --output-on-failure` 验证零回归

### Phase 3: E2E 真实 kernel(1 commit)

1. `tests/e2e/kernel/test_tcgen05_mma_gemm.cu` 用 cuobjdump 提取的真实 tcgen05.mma GEMM PTX
2. 验证 E2E 测试通过(对比 host 端 reference GEMM)
3. 注册到 `tests/e2e/CMakeLists.txt`

### Phase 4: 文档同步(1 commit)

1. 根 `AGENTS.md` 更新已知限制表
2. `src/ptxsim/instructions/AGENTS.md` 添加 `tcgen05.cpp` 说明
3. ADR-0016 追加 Phase 1-2 commit 引用
4. `docs/dev-process/lessons-learned.md` 可选 §24 新案例(若发现新模式)

### Phase 5: Archive(1 commit,per Checklist G + I)

1. 跑 `openspec archive implement-tcgen05-handlers-core --yes`
2. 跑 `cd build && ctest --output-on-failure` 全量验证
3. 跑 `./tests/ptx/test_all_ptx.sh` 验证
4. 跑 `cd build && ctest -L "e2e;tcgen05" -V` 验证 E2E

## Capabilities

### New Capabilities

- `tcgen05-handlers-core`:5 个核心 Blackwell 指令的真实 handler 实现(mma/ld/st/commit/wait)+ golden-value 测试
- `tcgen05-handler-tests-core`:5 单元 + 5 集成 + 1 E2E 测试覆盖 core handler

### Modified Capabilities

- `tcgen05-parse-tests`:spec 修订(operand 提取完善)

## Impact

### 影响的代码(预计)

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `src/ptxsim/instructions/tcgen05.cpp` | 新增 | +500 |
| `src/ptxsim/instructions/wmma.cpp` | 修改(移除 tcgen05) | -200 |
| `src/ptx_parser/ptx_visitor_wmma.cpp` | 修改(operand 提取) | +50 |
| `tests/unit/ptx/test_tcgen05_*.cpp`(5 个) | 新增 | +200 |
| `tests/integration/tcgen05/test_tcgen05_*_parse.cpp`(5 个) | 新增 | +250 |
| `tests/e2e/kernel/test_tcgen05_mma_gemm.cu`(1 个) | 新增 | +150 |
| `tests/ptx/reference/tcgen05_mma_golden.h` | 新增(golden values) | +100 |
| 多个 CMakeLists.txt | 注册 | +30 |
| `docs/adr/ADR-0016-*.md` + AGENTS.md | 文档 | +30 |
| **总计** | | **+1110** |

### 影响的依赖

- `ptx-debug` skill(handler 调试)
- `three-mode-testing` skill(三套测试)
- `cuobjdump -xptx` 工具(E2E 真实 PTX)
- ~~Cutlass 3.x `SM100_MMA_F16_F16_F32`~~(Day 1 验证不可用,已废弃)
- `wmma.cpp:374-420` 现有 inline mma + PTX ISA §9.7.16 手算(实际 golden value 来源,per design.md D1 修正)

### 不影响的依赖

- `src/ptxsim/memory/*`, `src/ptxsim/cluster/*`, `src/ptxsim/async/*`(Change-2 scope,假设 ≥L2)
- `src/grammar/*`(Change-3a scope,假设已修)
- 其他 6 个 handler(change-3d scope)

### 影响的文档

- 根 `AGENTS.md`(已知限制表)
- `src/ptxsim/instructions/AGENTS.md`(目录说明)
- `docs/adr/ADR-0016-blackwell-only-tcgen05.md`(更新记录)
- `docs/dev-process/lessons-learned.md`(可选 §24)

## Design-Time Checklist (Lessons-Learned)

### 函数审计完整性

- [x] Baseline 函数清单:`wmma.cpp` 中 **4 个 `execute_tcgen05_*`** 函数(行 321/323/325/327,定义 + 423/463/502/534 实现)+ **1 个 inline mma handler**(line 352 + line 374-420)——**不是 5 个 execute 函数**,Day 1 验证修正
- [x] 锁点审计:5 个函数均无锁调用(纯计算)
- [x] 跨模块状态翻译:handler 调 `cta->tmem()` / `cta->tma_descriptor_store()` / `cta->tc_queue()` / `cta->cluster_context()`
- [x] invariant 清单:per-warp ordering、CTA 隔离、commit-group counter 原子性
- [x] **Metis E.1 修复**:"真实实现" acceptance criteria 明确定义(见 Phase 2)

### 多 Phase 推进(5 个 atomic commits)

- [x] Phase 1: visitor operand 提取(独立 commit)
- [x] Phase 2: 5 handler 实施(1-2 commits)
- [x] Phase 3: E2E kernel(独立 commit)
- [x] Phase 4: 文档(独立 commit)
- [x] Phase 5: archive(独立 commit,per Checklist G)
- [x] 基线 worktree 计划:`.worktrees/baseline-tcgen05-handlers-core`
- [x] 失败处理策略:已有测试回归 → 立即 revert 该 Phase

### 文档同步(per Checklist I)

- [x] 根 AGENTS.md 同步项已列出
- [x] ADR 追加段落已规划
- [x] Golden value 来源已明确(Day 1 验证:Cutlass 3.x **环境不可用** → 改用 `wmma.cpp:374-420` 现有 inline mma + PTX ISA §9.7.16 手算,见 design.md D1 + D7)

### 实施前必跑(per `ptx-lessons-learned` §7)

- [ ] **Metis pre-implementation review**:验证 handler 实现范围、golden value 来源
- [ ] 验证 `wc -l src/ptxsim/instructions/wmma.cpp`(约 564 行,需 -200 移除 tcgen05)
- [ ] 验证 Change-3a 已 archive(grammar 100% 正确)
- [ ] 验证 Change-2 已 archive(基础设施 readiness ≥L2)
- [ ] 验证 cuobjdump -xptx 可用(若 E2E 需真实 PTX)
- [ ] 跑 `ctest -L "unit;memory|unit;cluster|unit;async"` 确认 baseline

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| Change-1 (archive) | **implement-tcgen05-handlers-core** | tcgen05-docs-and-archive (change-3c) |
| fix-tcgen05-grammar-mr3 (Change-3a) | | implement-tcgen05-handlers-extended (change-3d) |
| extend-blackwell-tcgen05-infra (Change-2) | | cleanup-wmma-namespace (Change-4) |

- **Change-1 → 本 change**:依赖 `S_TCGEN05_*` IR 命名空间 + `Tcgen05Instr` struct
- **Change-3a → 本 change**:**硬前置**(handler 实施依赖 grammar 100% 正确)
- **Change-2 → 本 change**:软前置(审计报告 ≥L2 才能保证 handler 不会被基础设施 bug 阻塞)
- **本 change → change-3c**:handler 已实施,docs 同步
- **本 change → change-3d**:5 core handler 已实施,扩展 6 handler 可以独立进行
- **本 change → Change-4**:wmma.cpp 移除 tcgen05 部分(保留 pre-Blackwell 路径直到 Change-4)

## 本 change 特有设计决策(per Metis F.2)

**决策 D1:Golden value 来源**
- 优选:`tests/ptx/reference/tcgen05_mma_golden.h` 从 **`wmma.cpp:374-420` 现有 inline mma** + **PTX ISA §9.7.16 手算**(2026-07-07 修正:Cutlass 3.x 环境不可用)
- 备选:从 **PTX ISA §9.7.16 规范** 提取(per IEEE 754 + 8x4 矩阵乘定义)
- 拒绝:不依赖 `cuobjdump -xptx` 输出(需真实 GPU,当前无访问)
- 备选:从 **PTX ISA §9.7.16 规范** 提取(per IEEE 754 + 8x4 矩阵乘定义)
- 拒绝:不依赖 `cuobjdump -xptx` 输出(需真实 GPU,当前无访问)

**决策 D2:handler 性能 vs correctness 优先级**
- 优先:functional correctness(对比 golden value)
- 拒绝:cycle-accurate timing(per ADR-0016 "性能对标不要求")

**决策 D3:operand 提取完成度**
- 完整提取:qualifiers + operands(虽然 handler 可能不用,但为 change-3d 6 handler 铺路)
- 拒绝:仅提取 handler 立即需要的部分(技术债)

**决策 D4:handler 文件拆分粒度**
- 单文件 `tcgen05.cpp`(5 handler 集中,易 revert 整体)
- 备选:每 handler 一文件(`tcgen05_mma.cpp` 等)— 拒绝,过度拆分

**决策 D5:wmma.cpp 保留的边界**
- **删除**:5 个 `execute_tcgen05_*` 函数 + `is_tcgen05_*()` 5 个 helper
- **保留**:`pre-Blackwell` `wmma.mma.sync.*` 路径(per ADR-0016 锁定,pre-Blackwell 永久 throw)
- **不动**:`S_WMMA` 枚举 + `WmmaInstr` struct + `makeWmmaInstr` 工厂(Change-4 scope)
