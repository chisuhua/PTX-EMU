# Backfill tcgen05.cp Test Coverage and Exception Cleanup

> **架构依据**: [ADR-0016](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) Accepted
> **关联提交**: `178457d` (`feat(handlers): implement tcgen05.cp smem→tmem (ADR-0016, Phase 2, Oracle Q4-B/Q2-A)`)
> **设计时教训**: `ptx-lessons-learned` §3 (分 Phase commit) + §6/§7 (OpenSpec artifact 完整性 + Metis pre-implementation review) + §9 (bare string lexer token 禁用)

## Why

Commit `178457d` 实现了 `tcgen05.cp` handler (SMEM → TMEM 拷贝)，但交付时**未附带任何针对该 handler 的测试**。知识图 `tests_for` 查询确认 `processTcgen05Cp` 与 `processTcgen05Operation` 均无测试命中；`tests/` 目录下也找不到 `tcgen05_cp` 或 `processTcgen05Cp` 相关用例。这违反项目 AGENTS.md 对新增指令必须覆盖 unit / integration / e2e 三层测试的要求。

此外，handler 在异常类型上存在不一致：缺少 WarpContext / CTAContext / `cta_group::2` 时抛 `UnsupportedInstructionException`，而 `cta->sharedMemSpace == nullptr` 时却抛 `std::runtime_error`。统一异常类型有助于调用方一致处理"不支持的执行环境"。

## What Changes

### 新增测试

| 文件 | 范围 |
|------|------|
| `tests/unit/tcgen05/test_tcgen05_cp.cpp` | 单元测试：smem offset 提取、异常路径、边界检查 |
| `tests/integration/tcgen05/test_tcgen05_cp.cpp` | 集成测试：通过 `execute_warp_instruction` 驱动真实 SMEM → TMEM 拷贝 |
| `tests/e2e/kernel/test_tcgen05_cp.cu` | E2E kernel：使用真实 `tcgen05.cp` PTX 走完整执行流程（如可用）|

### 修改代码

| 文件 | 范围 |
|------|------|
| `src/ptxsim/instructions/tcgen05_cp.cpp` | 将 `sharedMemSpace == nullptr` 的 `std::runtime_error` 改为 `UnsupportedInstructionException` |
| `src/ptxsim/instructions/tcgen05_cp.cpp` | 为 `kDestSlot=0`、shape qualifier 解析、register offset 解析等 placeholder 添加 `TODO(Phase 3)` 跟踪注释 |
| 多个 `CMakeLists.txt` | 注册新增测试目标 |
| `src/ptxsim/instructions/AGENTS.md` | 更新 `tcgen05.cp` 测试覆盖状态 |
| 根 `AGENTS.md` | 同步 tcgen05 已知限制表 |

### 不修改(范围外)

- ❌ 不改 `tcgen05.cp` 的核心拷贝逻辑（已合入 `178457d`）
- ❌ 不改 `tcgen05.cp` 的 `cta_group::2` 异常语义（仅统一异常类型）
- ❌ 不实现 `cp.async.bulk.tensor.*` 或 `cta_group::2` distributed smem（独立 follow-up）
- ❌ 不修改 5 core handler 或其他 extended handler

## Goals

### Phase 1: 单元测试与异常清理（1 commit）

1. 新增 `tests/unit/tcgen05/test_tcgen05_cp.cpp`
   - 测试 `extract_smem_offset_placeholder` 对 immediate / register / non-shared 地址的返回值
   - 测试 `cta_group::2` 抛出 `UnsupportedInstructionException` 且消息含 `ADR-0018`
   - 测试 `sharedMemSpace == nullptr` 抛出 `UnsupportedInstructionException`
   - 测试 `smem_offset + Tmem::kSlotSize > sharedMemBytes` 抛出异常
2. 将 `tcgen05_cp.cpp` 中的 `std::runtime_error` 改为 `UnsupportedInstructionException`
3. 添加 `TODO(Phase 3 of implement-tcgen05-handlers-extended)` 跟踪注释
4. 跑 `ctest -R unit_tcgen05_cp -V` 验证
5. **commit**: `test(tcgen05): add unit tests for tcgen05.cp and unify exception type (ADR-0016)`

### Phase 2: 集成测试（1 commit）

1. 新增 `tests/integration/tcgen05/test_tcgen05_cp.cpp`
   - 使用 `ptxsim::testing` 工具构造指令序列
   - 通过 `execute_warp_instruction` 驱动 `tcgen05.cp`
   - 验证 SMEM 128 字节写入 TMEM slot 0
   - 验证越界访问抛出异常
2. 跑 `ctest -R integration_tcgen05_cp -V` 验证
3. **commit**: `test(tcgen05): add integration test for tcgen05.cp (ADR-0016)`

### Phase 3: E2E / 文档同步（1 commit）

1. 若可行，新增 `tests/e2e/kernel/test_tcgen05_cp.cu` 并用 `nvcc -ptx` 提取真实 PTX；否则在文档中说明跳过原因
2. 更新 `src/ptxsim/instructions/AGENTS.md` 中 `tcgen05.cp` 覆盖状态
3. 更新根 `AGENTS.md` 已知限制表
4. 跑 `./scripts/sanity.sh` 全量验证
5. **commit**: `test(tcgen05): add e2e kernel and update AGENTS for tcgen05.cp (ADR-0016)`

### Phase 4: Archive（1 commit）

1. 跑 `openspec archive tcgen05-cp-test-coverage-and-exception-cleanup --yes`
2. 跑 `cd build && ctest --output-on-failure` 全量验证
3. 跑 `./tests/ptx/test_all_ptx.sh` 验证
4. commit archive 目录

## Capabilities

### New Capabilities

- `tcgen05-cp-test-coverage`: 为 `tcgen05.cp` handler 补全单元、集成、E2E 三层测试，并统一异常类型。

### Modified Capabilities

- （无 spec 级行为变更；仅实现既有 `tcgen05-handler-test-coverage` 能力对 `tcgen05.cp` 的覆盖要求。）

## Impact

### 影响的代码

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `tests/unit/tcgen05/test_tcgen05_cp.cpp` | 新增 | +120 |
| `tests/integration/tcgen05/test_tcgen05_cp.cpp` | 新增 | +150 |
| `tests/e2e/kernel/test_tcgen05_cp.cu` | 新增（可选） | +80 |
| `src/ptxsim/instructions/tcgen05_cp.cpp` | 修改 | +5 / -3 |
| 多个 `CMakeLists.txt` | 注册 | +20 |
| `src/ptxsim/instructions/AGENTS.md` | 文档 | +5 |
| 根 `AGENTS.md` | 文档 | +3 |
| **总计** | | **~+380** |

### 影响的依赖

- `three-mode-testing` skill（三层测试规范）
- `ptx-lessons-learned` skill（设计时 checklist）
- `cuobjdump -xptx` 工具（E2E 真实 PTX 提取）

### 不影响的依赖

- 5 core handler（已完成）
- 其他 extended handler（alloc/dealloc/relinquish/fence/mma_ws）
- grammar 与 IR 类型

## Design-Time Checklist (Lessons-Learned)

### 函数审计完整性

- [x] Baseline 函数：`src/ptxsim/instructions/tcgen05_cp.cpp` 中的 `processTcgen05Cp`、`extract_smem_offset_placeholder`、`throw_cta_group_2`
- [x] 锁审计：本 change 不引入新锁，仅复用现有 `cta->tmem()` 读接口；无递归锁风险
- [x] 跨模块状态翻译：
  - `tcgen05.cp` 读取 `cta->sharedMemSpace`（只读拷贝）→ 写入 `cta->tmem()`（`Tmem::write`）
  - 不修改 `ThreadContext::state` / `WarpState` 等调度器状态
- [x] invariant 清单：
  - 不改变 `cta->sharedMemBytes` 语义
  - 不破坏 `Tmem::kSlotSize` 与 `Tmem::kSlotCount` 一致性（`static_assert` 已存在）
  - 越界检查仍抛出异常

### 多 Phase 推进（4 个独立 commits）

- [x] Phase 1: 单元测试 + 异常清理（独立 commit）
- [x] Phase 2: 集成测试（独立 commit）
- [x] Phase 3: E2E + 文档同步（独立 commit）
- [x] Phase 4: archive（独立 commit，per Checklist G）
- [x] 基线 worktree 计划：
  ```bash
  git worktree add .worktrees/baseline-tcgen05-cp 178457d
  cd .worktrees/baseline-tcgen05-cp
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  cd build && ctest --output-on-failure
  ```
- [x] 失败处理策略：任何已有测试回归 → 立即 revert 该 Phase，不混入后续 commit

### 文档同步

- [x] `src/ptxsim/instructions/AGENTS.md` 同步项已列出
- [x] 根 `AGENTS.md` 同步项已列出
- [x] ADR-0016 不需要追加段落（本 change 是测试补充，不是架构决策）

### 实施前必跑（per `ptx-lessons-learned` §7）

- [ ] 验证 `178457d` 已合入当前分支
- [ ] 跑 `ctest -R tcgen05 -V` 确认 baseline 通过
- [ ] 跑 `./tests/ptx/test_all_ptx.sh` 确认 PTX 语法测试 PASS

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| `implement-tcgen05-handlers-extended` (Phase 2, commit `178457d`) | **tcgen05-cp-test-coverage-and-exception-cleanup** | 无（独立测试补充） |

- **上游 → 本 change**：依赖 `tcgen05.cp` handler 已存在且可编译
- **本 change → 下游**：无；仅提高测试覆盖率，不阻塞其他功能

## 本 change 特有设计决策

**决策 D1：异常类型统一**
- 将 `sharedMemSpace == nullptr` 的 `std::runtime_error` 改为 `UnsupportedInstructionException`
- 理由：与缺少 WarpContext / CTAContext / `cta_group::2` 的语义一致，均为"当前执行环境不支持该指令"
- 备选：保留 `std::runtime_error` — 拒绝，会导致调用方捕获不一致

**决策 D2：目标 slot 硬编码处理**
- 当前代码将 `kDestSlot` 硬编码为 0，这是 `178457d` 中已声明的 Phase 2 placeholder
- 本 change 不改动该行为，仅添加 `TODO(Phase 3)` 注释指向 `implement-tcgen05-handlers-extended`
- 理由：避免在同一 commit 中混合"测试补充"和"功能扩展"，保持 Phase 独立可回退

**决策 D3：测试范围限定**
- 单元测试仅覆盖 `extract_smem_offset_placeholder` 的 immediate 路径；register / symbolic 路径作为 placeholder 被显式验证为返回 0
- 集成测试构造一个真实 warp + CTA，验证 128 字节 SMEM → TMEM 拷贝
- E2E 视 `nvcc` 对 `tcgen05.cp` 的支持情况而定；如不可行，在 `tests/e2e/kernel/CMakeLists.txt` 中跳过并说明
