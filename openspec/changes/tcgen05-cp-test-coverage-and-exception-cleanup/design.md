## Context

Commit `178457d` 在 `src/ptxsim/instructions/tcgen05_cp.cpp` 中实现了 `tcgen05.cp` handler，用于将 per-CTA shared memory 中的 128 字节数据拷贝到 TMEM slot 0。该 handler 已被正确接入 `Tcgen05Handler::processTcgen05Operation` dispatch，但提交时未附带任何直接测试。当前代码存在以下问题：

1. **测试缺口**：`processTcgen05Cp` 没有任何单元、集成或 E2E 测试命中；`tests/integration/tcgen05/test_tcgen05_dispatch.cpp` 仅验证 `S_TCGEN05_CP` 在 dispatch 表中非空，未执行 handler 逻辑。
2. **异常不一致**：`cta->sharedMemSpace == nullptr` 时抛 `std::runtime_error`，而缺少 WarpContext / CTAContext / `cta_group::2` 时均抛 `UnsupportedInstructionException`。
3. **Placeholder 缺乏跟踪**：目标 TMEM slot 硬编码为 0、shape qualifier 未解析、register offset 未解析等中期状态仅有文字注释，无 `TODO` 或 issue 指向后续 Phase。

本 change 是一次**测试补充 + 代码清理**，不扩展 `tcgen05.cp` 的功能语义。

## Goals / Non-Goals

**Goals:**
- 为 `tcgen05.cp` 添加 unit / integration / e2e 三层测试，覆盖正常路径、异常路径和边界条件。
- 统一 `tcgen05.cp` 的异常类型为 `UnsupportedInstructionException`。
- 为已知的 placeholder 添加显式 `TODO(Phase 3 of implement-tcgen05-handlers-extended)` 跟踪注释。
- 更新 `src/ptxsim/instructions/AGENTS.md` 和根 `AGENTS.md` 中 `tcgen05.cp` 的覆盖状态。

**Non-Goals:**
- 不修改 `tcgen05.cp` 的核心拷贝逻辑（`std::memcpy` 路径）。
- 不实现 destination slot 解析、shape qualifier 提取或 register offset 解析。
- 不实现 `cta_group::2` distributed smem 或 `cp.async.bulk.tensor.*`。
- 不修改其他 tcgen05 handler 的测试或实现。

## Decisions

**D1: 单元测试直接实例化内部 helper 函数**
- `extract_smem_offset_placeholder` 和 `throw_cta_group_2` 位于匿名 namespace，无法被外部测试直接调用。
- **方案**：将这两个 helper 提升到 `ptxsim` 命名空间（或新增公开测试接口），使其可被 `tests/unit/tcgen05/test_tcgen05_cp.cpp` 调用。
- **备选**：通过 `processTcgen05Cp` 间接测试 — 拒绝，会导致单元测试需要构造完整的 `ThreadContext` / `WarpContext` / `CTAContext`，引入无关依赖。
- **理由**：直接测试 helper 函数能更精确地覆盖异常路径，且不影响现有 API 稳定性（helper 仅用于 `tcgen05_cp.cpp` 内部）。

**D2: 集成测试复用 `ptxsim::testing` 工具**
- **方案**：参考 `tests/integration/tcgen05/test_alloc_dealloc_relinquish.cpp` 和 `tests/integration/divergence/test_divergence_sync_convergence.cpp`，使用 `ptxsim::testing::step_warp` 和 `ptxsim::testing::make_*` 构造指令序列。
- **备选**：直接调用 `processTcgen05Cp` 并手动构造 `Tcgen05Instr` — 拒绝，集成测试应验证真实执行路径而非直接调用 handler。
- **理由**：通过 `execute_warp_instruction` 驱动能同时验证 dispatch 和 handler 行为，与项目集成测试原则一致。

**D3: E2E 测试视 nvcc 支持情况而定**
- **方案**：尝试用 `nvcc -ptx` 编译包含 `tcgen05.cp` 的 `.cu` 文件；若 ptxas 不支持 `tcgen05.cp`（常见情况），则跳过 E2E 并记录原因。
- **备选**：强制使用 fixture PTX 文件 — 拒绝，项目 E2E 测试优先使用真实 CUDA 编译；fixture 应留在 integration 层。
- **理由**：避免为不可用的 E2E 测试引入虚假失败，同时保留未来扩展可能。

**D4: 异常类型统一**
- **方案**：将 `cta->sharedMemSpace == nullptr` 的 `std::runtime_error` 改为 `UnsupportedInstructionException`。
- **备选**：保持现状 — 拒绝， inconsistent exception 会导致上层调用方（如 `ThreadContext::execute_thread_instruction`）捕获逻辑不一致。
- **理由**：`UnsupportedInstructionException` 是项目表示"当前环境不支持该指令"的专用异常，语义最准确。

**D5: 仅添加 TODO 注释，不改动 placeholder 行为**
- **方案**：为 `kDestSlot=0`、shape qualifier 解析、register offset 解析添加 `TODO(Phase 3 of implement-tcgen05-handlers-extended)` 注释。
- **备选**：在测试补充 change 中实现 slot 解析 — 拒绝，超出本 change 范围，且会引入新的设计决策（需要 Oracle / ADR 讨论）。
- **理由**：保持 Phase 独立可回退，测试补充 change 不混入功能扩展。

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 单元测试暴露内部 helper 后，未来重构需同步更新测试 | 低 | helper 逻辑简单且稳定；测试使用高层级断言，不依赖实现细节 |
| 集成测试需要构造完整 warp/cta，可能引入 setup 开销 | 低 | 复用 `ptxsim::testing` 现有工具，避免重复造轮子 |
| E2E 测试因 nvcc 不支持 `tcgen05.cp` 而无法添加 | 低 | 在 `tests/e2e/kernel/CMakeLists.txt` 中显式跳过并说明原因；integration 测试已覆盖核心路径 |
| 异常类型变更可能影响已有调用方 | 低 | 当前 `tcgen05.cp` 为新增 handler，无外部调用方依赖 `std::runtime_error`；`UnsupportedInstructionException` 是更通用基类 |

## Migration Plan

不适用。本 change 是向后兼容的测试补充和代码清理，不引入破坏性变更。

## Open Questions

1. `nvcc` 对 `tcgen05.cp` 的 PTX 输出是否可用于 E2E 测试？需在 Phase 3 实施前验证。
2. 是否应将 `extract_smem_offset_placeholder` 和 `throw_cta_group_2` 提取到独立头文件供测试复用，还是仅在 `tcgen05_cp.cpp` 内改为 `namespace ptxsim`？
3. 现有 `tcgen05-handler-test-coverage` spec 是否应被更新以引用 `tcgen05.cp` 的覆盖，还是保持独立 capability？
