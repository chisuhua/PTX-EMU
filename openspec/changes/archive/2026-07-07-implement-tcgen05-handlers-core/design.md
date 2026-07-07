## Context

Change-1 (archive) 建立独立 tcgen05 命名空间 + IR,Change-3a 修复 grammar,Change-2 审计基础设施。本 change 实施 5 个核心 handler(mma/ld/st/commit/wait)从 `wmma.cpp` 提取并适配新 IR。

目标:`src/ptxsim/instructions/tcgen05.cpp` 5 个真实实现 + 5 单元 + 5 集成 + 1 E2E kernel 测试。**Metis E.1 修复**:`真实实现` acceptance criteria 明确定义为对比 golden value,不是对比硬件。

## Goals / Non-Goals

**Goals**: 5 handler 实施 + 三套测试 + visitor operand 提取完善 + wmma.cpp 移除 tcgen05 部分。

**Non-Goals**: 不实施其他 6 个 handler(change-3d)、不实现 cp.async.bulk.tensor(独立 follow-up)、不删除 wmma.cpp(Change-4)、不实现 cta_group::2 / sm_120。

## Decisions

### D1: Golden value 来源 — 复用 `wmma.cpp` 现有 inline mma + PTX ISA §9.7.16 手算(2026-07-07 修正)

**采纳**: Cutlass 3.x `SM100_MMA_F16_F16_F32` 在 Day 1 验证时确认**环境内不可用**(`/usr/local/cutlass`, `/opt/cutlass`, `~/cutlass` 均不存在;`find` 无任何 SM100_MMA 命中)。改用以下两源结合:

1. **复用现有 inline mma handler**:`src/ptxsim/instructions/wmma.cpp:374-420` 已有 `tcgen05.mma` fragment arithmetic 实现(line 352 inline 走原路径)。**该实现是当前 PTX-EMU 模拟器产生所有 tcgen05.mma 输出的来源**,其输出天然"自洽"。
2. **PTX ISA §9.7.16 规范手算**:对 fragment arithmetic 公式(8x4 矩阵乘 f16 → f32)按 IEEE 754 + NV 文档计算 reference 值,与 (1) 对比验证 functional correctness。

**golden value 存放**:`tests/ptx/reference/tcgen05_mma_golden.h` —— 由(1) inline mma handler 当前输出提取 + (2) PTX ISA 手算双重交叉验证。

**拒绝**:
- 不依赖 `cuobjdump -xptx`(需真实 GPU,环境内无访问)。
- 不依赖外部 Cutlass 包(环境未安装)。

### D7: handler 数量澄清 — 4 execute + 1 inline(2026-07-07 修正)

**Day 1 验证事实**:`wmma.cpp` 实际只含 **4 个 `execute_tcgen05_*` 函数**:
- `execute_tcgen05_ld` (line 321/423)
- `execute_tcgen05_st` (line 323/463)
- `execute_tcgen05_commit` (line 325/502)
- `execute_tcgen05_wait` (line 327/534)

`tcgen05.mma` **没有独立的 `execute_tcgen05_mma` 函数**——它在 `wmma.cpp:352` inline 走 `if (!is_tcgen05_mma_f16(qualifiers))` 路径(line 374-420)。

**最终 5 个 handler 在新 `tcgen05.cpp` 中**:
- `processTcgen05Mma` —— **新独立函数**(从 inline handler 提取,**不是从 execute 函数提取**)
- `processTcgen05Ld` —— 从 `execute_tcgen05_ld` 提取
- `processTcgen05St` —— 从 `execute_tcgen05_st` 提取
- `processTcgen05Commit` —— 从 `execute_tcgen05_commit` 提取
- `processTcgen05Wait` —— 从 `execute_tcgen05_wait` 提取

**wmma.cpp 清理范围(精确版)**:
- 删除:4 个 `execute_tcgen05_*` 函数(ld/st/commit/wait)+ 5 个 `is_tcgen05_*()` helper(mma/ld/st/commit/wait)
- 提取:inline mma handler(line 352 + line 374-420)→ 新 `tcgen05.cpp::processTcgen05Mma`
- 保留:pre-Blackwell `wmma.mma.sync.*` 路径(per ADR-0016 锁定,`UnsupportedInstructionException`)

**Golden value 来源(2026-07-07 修正)**:复用 `wmma.cpp:374-420` 现有 inline mma fragment arithmetic + PTX ISA §9.7.16 规范手算(见 D1 完整描述)。**Cutlass 3.x 在 Day 1 验证时确认环境不可用**。

**拒绝**: cuobjdump -xptx 真实硬件(无 GPU 访问)。

### D2: handler 文件拆分 — 单文件 `tcgen05.cpp`

**采纳**: 5 core handler 在单文件 `tcgen05.cpp`(易整体 revert)。

**拒绝**: 拆分 `tcgen05_mma.cpp` 等 5 个文件(过度拆分)。

### D3: wmma.cpp 边界 — 移除 4 个 execute_tcgen05_*(ld/st/commit/wait),保留 pre-Blackwell

> **2026-07-07 修正**:原 D3 "删除 5 个 execute_tcgen05_*" 错误,实际只有 4 个。

**采纳**: 删除 **4 个** `execute_tcgen05_*` 函数(ld/st/commit/wait)+ 5 个 `is_tcgen05_*()` helper(mma/ld/st/commit/wait)。**保留** pre-Blackwell `wmma.mma.sync.*` 路径(per ADR-0016 锁定,pre-Blackwell 永久 throw)。**inline mma handler**(line 352 + 374-420)提取到 `tcgen05.cpp::processTcgen05Mma`(per D7)。

**不动**:`S_WMMA` 枚举 / `WmmaInstr` struct / `makeWmmaInstr` 工厂(Change-4 scope)。

### D4: visitor operand 提取 — 完整提取

**采纳**: 完整提取 qualifiers + operands,为 change-3d 6 handler 铺路。

**拒绝**: 仅提取 handler 立即需要的部分(技术债,change-3d 需重做)。

### D5: 文件粒度 — 5 commits,handler 实施为 1-2 commits

**采纳**: Phase 1 visitor + Phase 2 handlers(可拆 2 commits)+ Phase 3 E2E + Phase 4 docs + Phase 5 archive = 5 commits。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| **R1**: fragment arithmetic 与硬件不一致 | Golden value 来自 `wmma.cpp:374-420` inline mma + PTX ISA §9.7.16 手算,验证 f16 → f32 正确性(per D1 修正) |
| **R2**: visitor 提取 qualifier 错误 | 跑 ctest -L "integration;tcgen05" 验证 |
| **R3**: wmma.cpp 删除后旧测试 fail | Phase 3 前先迁移旧测试(change-3a) |
| **R4**: E2E 真实 PTX 不可用 | 用 hand-written 真实 PTX 风格(per change-1 已有 2 fixture) |

## Migration Plan

### Phase 1: visitor operand 提取(1 commit)

1. 编辑 `src/ptx_parser/ptx_visitor_wmma.cpp` 完善 `visitTcgen05Inst` 提取 qualifiers + operands
2. 跑 `ctest -L "integration;tcgen05" -V` 验证

### Phase 2: 5 handler 实施(1-2 commits)

> **2026-07-07 修正**:Day 1 验证发现 `wmma.cpp` 实际只有 4 个 `execute_tcgen05_*`(ld/st/commit/wait),mma 是 inline handler。详细见 D7。

1. 新建 `src/ptxsim/instructions/tcgen05.cpp` 5 个 `processTcgen05Xxx` 函数(mma/ld/st/commit/wait)
2. 4 个(ld/st/commit/wait)从 `wmma.cpp:321-534` 提取并适配新 IR(使用 `Tcgen05Instr::op_kind` 而非 Q_TCGEN05_* qualifier)
3. 1 个(mma)从 `wmma.cpp:352 + 374-420` inline handler 提取并提升为独立 `processTcgen05Mma` 函数
4. 每个 handler 有 per-element UNVERIFIED 注释
5. 新建 `tests/ptx/reference/tcgen05_mma_golden.h`(per D1:从 wmma.cpp inline mma + PTX ISA §9.7.16 手算)
6. 编辑 `src/ptxsim/instructions/wmma.cpp`:
   - 删除 4 个 `execute_tcgen05_*`(ld/st/commit/wait)
   - 删除 5 个 `is_tcgen05_*()` helper(mma/ld/st/commit/wait)
   - 移除 inline mma handler(line 352 + 374-420,已迁移)
   - 保留 pre-Blackwell `wmma.mma.sync.*` 路径(per ADR-0016)
7. 编辑 `src/ptxsim/CMakeLists.txt` 注册新文件
8. 跑 `ctest -L "unit;tcgen05|integration;tcgen05" -V` 验证
9. 跑 `cmake --build build` 验证

### Phase 3: E2E kernel(1 commit)

1. 新建 `tests/e2e/kernel/test_tcgen05_mma_gemm.cu`(用 cuobjdump 提取或 hand-written 真实风格)
2. 注册到 `tests/e2e/CMakeLists.txt`
3. 跑 `ctest -L "e2e;tcgen05" -V` 验证

### Phase 4: 文档(1 commit)

1. 根 `AGENTS.md` 更新已知限制表(tcgen05 5 core handler 已实现)
2. `src/ptxsim/instructions/AGENTS.md` 添加 `tcgen05.cpp` 说明
3. ADR-0016 追加本 change commit 引用

### Phase 5: Archive(per Checklist G)

`openspec archive` + commit archive。

## Open Questions

无(handler 实施范围明确,golden value 来源已确定)。
