# Implement Extended tcgen05 Handlers (6 remaining: ALLOC/DEALLOC/RELINQUISH/CP/FENCE/MMA_WS)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **前置 changes**(全部必须 archive 后才能执行本 change):
>   - `archive/2026-07-06-implement-tcgen05-syntax-ir` (Change-1, archived)
>   - `fix-tcgen05-grammar-mr3` (Change-3a, pending) — **硬前置**(grammar 必须 100% 正确)
>   - `extend-blackwell-tcgen05-infra` (Change-2, pending) — **软前置**(审计 ≥L2)
>   - `implement-tcgen05-handlers-core` (Change-3b, pending) — **强前置**(5 core handler 已实施)
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ Metis A.4(6 handler 不能继续"留待 follow-up" — 必须 propose)

## Why

Change-3b 实施 5 个 **core handler**(MMA/LD/ST/COMMIT/WAIT),但 Blackwell `tcgen05.*` 指令族**实际有 11 个**(per PTX ISA §9.7.16),剩 6 个**未实现**:

| 指令 | 描述 | 复杂度 | 依赖 |
|------|------|--------|------|
| `tcgen05.alloc` | 分配 TMEM 槽位 | 中 | CTAContext + Tmem |
| `tcgen05.dealloc` | 释放 TMEM 槽位 | 中 | CTAContext + Tmem |
| `tcgen05.relinquish_alloc_permit` | 释放 allocate permit(线程专用) | 低 | WarpState |
| `tcgen05.cp` | 共享内存 → TMEM 拷贝(非 TMA) | 高 | CTAContext + Smem + Tmem |
| `tcgen05.fence` | 同步原语(::before/after_thread_sync) | 中 | WarpScheduler |
| `tcgen05.mma.ws` | weight-stationary MMA 变种 | 高 | Tcgen05Instr + WarpContext(与 .mma 共享 fragment 算术) |

这 6 个 handler **在 grammar + IR 中已存在**(`S_TCGEN05_ALLOC/DEALLOC/RELINQUISH/CP/MMA_WS/FENCE` enum, `Tcgen05OpKind` 已含),但无 handler 实现 → **dead code**。

**Metis A.4 修复**:Change-3 原 Non-Goals 写 "留待 Change-3.5 follow-up" 但 Change-3.5 未 propose。本 change 正式承担这 6 个 handler 的实施责任。

**本 change 是可选的** — 4-change 路线图核心交付(per ADR-0016)是 5 core handler(mma/ld/st/commit/wait),这 6 个是 **full completeness**。

## What Changes

### 新增

| 文件 | 范围 |
|------|------|
| `src/ptxsim/instructions/tcgen05_alloc.cpp` | alloc + dealloc + relinquish_alloc_permit handler(3 个) |
| `src/ptxsim/instructions/tcgen05_cp.cpp` | cp handler(SMEM → TMEM 拷贝) |
| `src/ptxsim/instructions/tcgen05_fence.cpp` | fence handler(::before/after_thread_sync) |
| `src/ptxsim/instructions/tcgen05_mma_ws.cpp` | mma.ws handler(weight-stationary 变种) |
| `tests/unit/ptx_ir/test_tcgen05_extended_opkind.cpp` | 6 OpKind 单元测试 |
| `tests/integration/parser/test_tcgen05_extended_parse.cpp` | 6 集成测试(parse → IR) |
| `tests/e2e/kernel/test_tcgen05_alloc.cu` | alloc E2E(用 cuobjdump 提取的 tcgen05.alloc) |
| `tests/e2e/kernel/test_tcgen05_cp.cu` | cp E2E(用 cuobjdump 提取的 tcgen05.cp) |

### 修改

| 文件 | 范围 |
|------|------|
| `src/ptxsim/instructions/tcgen05.cpp`(Change-3b 新增) | 添加 6 个 processTcgen05Xxx 函数 |
| `src/ptxsim/CMakeLists.txt` | 注册 4 个新源文件 |
| `tests/unit/CMakeLists.txt` | 注册新单元测试 |
| `tests/integration/CMakeLists.txt` | 注册新集成测试 |
| `tests/e2e/CMakeLists.txt` | 注册新 E2E kernel |
| `src/ptxsim/instructions/AGENTS.md` | 更新 `tcgen05.cpp` 包含 extended handler |
| 根 `AGENTS.md` | 已知限制表标注 "tcgen05 11/11 handler 已实现" |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 追加 "11 handler 全实现" 标记 |

### 不修改(范围外)

- ❌ 5 core handler(Change-3b scope,已完成)
- ❌ 移除 `S_WMMA` 枚举(Change-4 scope)
- ❌ 删除 `wmma.cpp`(Change-4 scope)
- ❌ 修改 4 个基础设施子系统(Change-2 scope,已完成审计)
- ❌ 修改 grammar(Change-3a scope,已完成)
- ❌ 不实现 `cp.async.bulk.tensor.*`(独立 follow-up `implement-cp-async-bulk-tensor`)
- ❌ 不实现 `cta_group::2` distributed_smem(独立 follow-up)
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8(per ADR-0016 锁定)

## Non-Goals

### 显式拒绝

- ❌ 不修改 5 core handler(Change-3b 已完成)
- ❌ 不实现 sm_120 sparse / FP4 / mxfp8(per ADR-0016 锁定)
- ❌ 不实现 `cp.async.bulk.tensor.*`(TMA 加载指令 → 独立 follow-up)
- ❌ 不实现 `cta_group::2` distributed_smem(per ADR-0016 Open Question #2)
- ❌ 不删除 wmma.cpp 整体(Change-4 scope)

### 范围限制

- 仅 6 个 extended handler(per PTX ISA §9.7.16 完整列表)
- 仅 f16 mma(其他 dtype 留待 follow-up)
- 性能对标不要求(per ADR-0016)
- 优先级:本 change **可选** — 4-change 路线图核心(mma/ld/st/commit/wait)不依赖此

## Goals

### Phase 1: 实施 alloc/dealloc/relinquish(1 commit,简单)

1. `src/ptxsim/instructions/tcgen05_alloc.cpp` 3 个 handler
2. **Acceptance**:
   - **alloc**:指定 `num_cols` 槽位 → CTAContext.tmem 分配成功
   - **dealloc**:释放 CTAContext.tmem 槽位
   - **relinquish**:WarpState 的 allocate_permit 释放(per CTA-specialized warp)
3. 跑 `ctest -R tcgen05_alloc -V` 验证
4. 跑 `./tests/ptx/test_all_ptx.sh` 仍 PASS(13 fixtures)

### Phase 2: 实施 cp(1 commit,中等)

1. `src/ptxsim/instructions/tcgen05_cp.cpp` 1 个 handler
2. **Acceptance**:128 字节从 SMEM → TMEM 拷贝(byte-by-byte 验证)
3. 跑 `ctest -R tcgen05_cp -V` 验证
4. E2E `test_tcgen05_cp.cu` 验证

### Phase 3: 实施 fence(1 commit,简单)

1. `src/ptxsim/instructions/tcgen05_fence.cpp` 1 个 handler
2. **Acceptance**:
   - `::before_thread_sync` → 等待所有 warp 到达 fence
   - `::after_thread_sync` → 等待 fence 前的指令完成
3. 跑 `ctest -R tcgen05_fence -V` 验证

### Phase 4: 实施 mma.ws(1 commit,复杂)

1. `src/ptxsim/instructions/tcgen05_mma_ws.cpp` 1 个 handler
2. **Acceptance**:与 mma 共享 fragment 算术 + weight-stationary 布局差异
3. **Golden value**:从 PTX ISA §9.7.16 规范提取(weight-stationary 数据流)
4. 跑 `ctest -R tcgen05_mma_ws -V` 验证

### Phase 5: 文档同步(1 commit)

1. 根 `AGENTS.md` 已知限制表:tcgen05 → 11/11 handler 已实现
2. `src/ptxsim/instructions/AGENTS.md`:`tcgen05.cpp` 包含 11 handler
3. ADR-0016 更新记录:追加本 change archive commit 引用

### Phase 6: Archive(1 commit,per Checklist G)

1. 跑 `openspec archive implement-tcgen05-handlers-extended --yes`
2. 跑 `cd build && ctest --output-on-failure` 全量验证
3. 跑 `./tests/ptx/test_all_ptx.sh` 验证
4. commit archive 目录

## Capabilities

### New Capabilities

- `tcgen05-handlers-extended`:6 个 extended Blackwell 指令的真实 handler 实现(alloc/dealloc/relinquish/cp/fence/mma_ws)
- `tcgen05-handler-tests-extended`:6 单元 + 6 集成 + 2 E2E 测试

### Modified Capabilities

- `tcgen05-grammar`:spec 修订(11/11 handler 已实现)
- `tcgen05-ir-types`:spec 修订(11/11 enum 已使用)
- `tcgen05-parse-tests`:spec 修订(13 fixtures + extended parse 测试 PASS)

## Impact

### 影响的代码(预计)

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `src/ptxsim/instructions/tcgen05_alloc.cpp` | 新增 | +150 |
| `src/ptxsim/instructions/tcgen05_cp.cpp` | 新增 | +200 |
| `src/ptxsim/instructions/tcgen05_fence.cpp` | 新增 | +100 |
| `src/ptxsim/instructions/tcgen05_mma_ws.cpp` | 新增 | +250 |
| `src/ptxsim/instructions/tcgen05.cpp` | 修改(添加 6 function dispatch) | +80 |
| `tests/unit/ptx_ir/test_tcgen05_extended_opkind.cpp` | 新增 | +80 |
| `tests/integration/parser/test_tcgen05_extended_parse.cpp` | 新增 | +150 |
| `tests/e2e/kernel/test_tcgen05_{alloc,cp}.cu` | 新增(2 个) | +250 |
| 多个 CMakeLists.txt | 注册 | +30 |
| `AGENTS.md` + `AGENTS.md` 子文件 + ADR | 文档 | +30 |
| **总计** | | **+1320** |

### 影响的依赖

- `ptx-debug` skill(handler 调试)
- `three-mode-testing` skill(三套测试)
- `cuobjdump -xptx` 工具(E2E 真实 PTX)

### 不影响的依赖

- 5 core handler(Change-3b scope)
- 4 个基础设施子系统(Change-2 scope)
- grammar(Change-3a scope)
- `S_WMMA` / `wmma.cpp`(Change-4 scope)

### 影响的文档

- 根 `AGENTS.md`(已知限制表)
- `src/ptxsim/instructions/AGENTS.md`(目录说明)
- `docs/adr/0016-blackwell-only-tcgen05.md`(更新记录)

## Design-Time Checklist (Lessons-Learned)

### 函数审计完整性

- [x] Baseline 函数清单:6 个 extended handler(spec 已明确)
- [x] 锁点审计:6 个 handler 均无锁调用(纯计算 + 资源管理)
- [x] 跨模块状态翻译:
  - alloc/dealloc: `cta->tmem()` 分配/释放
  - cp: `cta->smem()` + `cta->tmem()`(per CTA)
  - fence: `warp->barrier_module()` 同步
  - mma.ws: 与 mma 共享 fragment 算术 + 不同 layout
  - relinquish: `warp->set_allocate_permit(false)`(per CTA-specialized)
- [x] invariant 清单:per-CTA 资源隔离、weight-stationary layout 正确性

### 多 Phase 推进(6 个 atomic commits,per handler 独立)

- [x] Phase 1: alloc/dealloc/relinquish(独立 commit,简单优先)
- [x] Phase 2: cp(独立 commit,中等)
- [x] Phase 3: fence(独立 commit,简单)
- [x] Phase 4: mma.ws(独立 commit,复杂最后)
- [x] Phase 5: 文档(独立 commit)
- [x] Phase 6: archive(独立 commit,per Checklist G)
- [x] 基线 worktree 计划:`.worktrees/baseline-tcgen05-handlers-extended`
- [x] 失败处理策略:已有测试回归 → 立即 revert 该 Phase

### 文档同步

- [x] AGENTS.md 同步项已列出
- [x] ADR 追加段落已规划

### 实施前必跑(per `ptx-lessons-learned` §7)

- [ ] **Metis pre-implementation review**:验证 6 handler 实现范围、优先级排序
- [ ] 验证 Change-3b 已 archive(5 core handler 已实施)
- [ ] 验证 Change-3a 已 archive(grammar 100%)
- [ ] 验证 Change-2 已 archive(infra ≥L2)
- [ ] 跑 `ctest -L "unit;tcgen05" -V` 确认 baseline
- [ ] 跑 `./tests/ptx/test_all_ptx.sh` 确认 13 fixtures PASS

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| Change-1 (archive) | **implement-tcgen05-handlers-extended** | (无 — 5th change 可选) |
| fix-tcgen05-grammar-mr3 (Change-3a) | | |
| implement-tcgen05-handlers-core (Change-3b) | | |
| extend-blackwell-tcgen05-infra (Change-2) | | |

- **Change-1 → 本 change**:依赖 `S_TCGEN05_*` 11 enum(6 extended 已在 enum 中)
- **Change-3a → 本 change**:硬前置(grammar 必须 100% 正确)
- **Change-3b → 本 change**:强前置(5 core handler 提供 `dispatchTcgen05Xxx` 模式可复用)
- **Change-2 → 本 change**:软前置(infra ≥L2)
- **本 change 是可选的** — 4-change 路线图核心不依赖此,5 core handler 即可交付

## 本 change 特有设计决策(per Metis F.2)

**决策 D1:handler 文件拆分粒度**
- **拆分**:4 个独立文件(alloc.cpp/cp.cpp/fence.cpp/mma_ws.cpp)— 每个文件单一职责
- 备选:单文件 `tcgen05.cpp` 包含所有 11 handler — 拒绝,与 Change-3b 的 `tcgen05.cpp`(5 core)合并
- 理由:5 core 在 Change-3b 已合并,extended 6 个单独文件避免单文件过大

**决策 D2:handler 实施优先级**
- **简单优先**:alloc/dealloc/relinquish(无 fragment 算术)→ fence(同步原语)→ cp(SMEM-TMEM 拷贝)→ mma.ws(fragment 算术)
- 理由:先建立 confidence,再处理复杂 case
- 拒绝:按 PTX ISA §9.7.16 顺序实施(无 learning curve 价值)

**决策 D3:mma.ws vs mma 共享**
- **共享 fragment 算术**:mma.ws 复用 Change-3b 的 mma handler,只在 layout 上差异
- 备选:完全独立实现 — 拒绝,代码重复
- 理由:`// UNVERIFIED-AGAINST-HARDWARE` 注释需在共享部分标注一次

**决策 D4:alloc/dealloc 资源管理**
- **per-CTA**:TMEM 槽位由 CTAContext 拥有,warp/thread 只读
- 备选:per-warp — 拒绝,违反 NVIDIA 硬件架构(per-CTA shared resource)
- 理由:与 `tmem.h` 现有 256 slot 槽位管理一致

**决策 D5:relinquish 语义**
- **per-warp permit**:每个 warp 有自己的 allocate permit,relinquish 后其他 warp 可 alloc
- 备选:per-CTA 单一 permit — 拒绝,违反 NVIDIA 硬件(CTA-specialized 场景)
- 理由:per `docs/adr/0016-*.md` 描述的 CTA-specialized warp 场景

**决策 D6:cp SMEM 源**
- **`.shared::cta` 源**:cp 只支持 per-CTA shared memory(per PTX ISA)
- 备选:`.shared::cluster` — 拒绝,cluster cp 需要 distributed_smem(本 change scope 外)
- 理由:cluster cp 需 Change `implement-cta-group-2-dist-smem`(独立 follow-up)
