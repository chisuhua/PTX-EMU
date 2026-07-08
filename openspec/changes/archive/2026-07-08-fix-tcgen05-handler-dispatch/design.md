# Wire tcgen05 Handlers to Dispatch Pipeline — Design

> **架构依据**: [ADR-0016](../../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **Spec 来源**: [openspec/specs/tcgen05-ir-types/spec.md](../../specs/tcgen05-ir-types/spec.md) — 设计意图已锁定,本 design 落实实施细节
> **前置 change**: `implement-tcgen05-handlers-core` (df6dde7)

## Context

### 现状问题

`implement-tcgen05-handlers-core` (commit `df6dde7`, archived 2026-07-07) 交付了 5 个 `processTcgen05*` 自由函数:

| 函数 | 文件位置 | 字节数 | 副作用分类 |
|------|----------|--------|-----------|
| `processTcgen05Mma` | `tcgen05.cpp:311-374` | ~64 行 | PURE(TMEM only) |
| `processTcgen05Ld` | `tcgen05.cpp:383-420` | ~38 行 | PURE(TMA + TMEM) |
| `processTcgen05St` | `tcgen05.cpp:429-465` | ~37 行 | PURE(TMEM + TMA) |
| `processTcgen05Commit` | `tcgen05.cpp:473-502` | ~30 行 | SIDE-EFFECT(通过 `TcQueue::commit` 释放 waiter) |
| `processTcgen05Wait` | `tcgen05.cpp:510-540` | ~31 行 | SIDE-EFFECT(直接 stall lane 0 + cluster wait) |

但**没有 dispatch 路径**能到达它们:

```
PTX `tcgen05.mma` → ANTLR 解析 → PtxVisitor::visitTcgen05Inst
   → makeTcgen05Instr (statement_factory.h:278-305)
   → StatementContext{type=S_TCGEN05_MMA, data=Tcgen05Instr}
   → WarpContext::execute_warp_instruction (warp_context.cpp:369)
   → ThreadContext::_execute_once (thread_context.cpp:101)
   → handler = InstructionFactory::get_handler(statement.type)   ← 返 nullptr
   → set_state(EXIT)                                            ← 死代码
```

### 现状代码事实

| 组件 | 当前 | 来源 |
|------|------|------|
| `S_TCGEN05_*` enum 定义 | 在 `ptx_types.h:28-38` 手工添加(11 行) | 跨过 X-Macro |
| X-Macro 注册 | `ptx_op.def:129-136` 显式排除 | 注释解释原因 |
| `InstructionFactory::initialize` | 只循环 `ptx_op.def`,不注册 `S_TCGEN05_*` | `instruction_factory.cpp:10-28` |
| `_execute_once` nullptr fallback | `set_state(EXIT)` + stderr 噪音 | `thread_context.cpp:142-146` |
| `IMPLEMENT_TCGEN_INSTR_HANDLER` 宏 | `instruction_handlers.cpp:172` 转发到 `IMPLEMENT_SIMPLE_HANDLER` — 仅推进 PC | 从未通过 X-Macro 触发 |
| `DECLARE_TCGEN_INSTR_HANDLER` 宏 | `instruction_handlers.h:130` 转发到 `DECLARE_SIMPLE_HANDLER` | 同上 |
| Tcgen05 完整 IR (`Tcgen05Instr` struct) | `statement_context.h:189-199` | already-exists |
| `makeTcgen05Instr` factory | `statement_factory.h:278-305` | already-exists |
| Tcgen05 调度规范定义 | `specs/tcgen05-ir-types/spec.md:114-143` | 已规定 `Tcgen05PipelineHandler` 模式 |

### 约束

- **ADR-0016 §Decision 1**: Blackwell-only (sm_100+). pre-Blackwell 永久抛异常
- **ADR-0016 §Decision 5**: tcgen05.* grammar 单 rule + 单一 dispatch point
- **ptx-lessons-learned**: 必须分 Phase commit,任何测试回归立即 revert 该 Phase
- **现有已注册 handler 类**: `S_WMMA`(`ppx_op.def:127` 注册 → `handler_map[S_WMMA] = new WmmaHandler()` 已在用) — 本 change 应类比此模式

## Goals / Non-Goals

**Goals**:
1. 5 个 `processTcgen05*` 函数通过 dispatch 真实可达
2. 复用 `WmmaPipelineHandler` 3-阶段流水线模式(prepare/execute/commit)
3. 保留所有 fragment arithmetic 逻辑(不重写已测试的算法)
4. 现有 E2E `test_blackwell_gemm.cu` 在新路径下仍能产生稳定输出
5. 集成测试 + 单元测试 100% 验证 dispatch 路径

**Non-Goals**:
- ❌ 不实现 alloc/dealloc/cp/mma_ws/fence(留给 Change-3d `implement-tcgen05-handlers-extended`)
- ❌ 不修改 `Tcgen05Instr` struct 定义
- ❌ 不修改 grammar
- ❌ 不为 commit/wait 引入新 mutex 设计 — 沿用现有 `TcQueue` 互斥机制(per ptx-barrier-mechanism)

## Decisions

### D1: 注册机制 — 走 X-Macro 而非显式

**采纳**: 通过 `ptx_op.def` 的 X-Macro 注册(对齐 `S_WMMA` 模式),让 `InstructionFactory::initialize()` 自动注册。

**理由**:
- `S_WMMA` 已在用此模式(`ptx_op.def:127` + 11 行未注册的 S_TCGEN05)
- 一致性更强(本 change 不引入新注册机制)
- `Tcgen05Handler::processTcgen05Operation` 用 `__attribute__((weak))` 默认实现,允许 `tcgen05.cpp` 提供强覆盖 — 与 `S_WMMA` 完全同构

**拒绝 — 选项 B(显式 `handler_map[S_TCGEN05_*] = ...`)**:
- 会绕过 X-Macro,引入第二套注册机制
- 需要修改 `InstructionFactory::initialize()` 加显式块
- 与现有架构风格冲突

### D2: 适配层 — 镜像 `WmmaPipelineHandler`

**采纳**: 创建 `Tcgen05PipelineHandler`,与 `WmmaPipelineHandler` 同形 (`instruction_base.cpp:213-237` 的 3 阶段模板)。

```cpp
class Tcgen05PipelineHandler : public PipelineHandler {
public:
    bool prepareOperands(ThreadContext*, StatementContext&) override;
    bool executeOperation(ThreadContext*, StatementContext&) override;
    bool commitResults(ThreadContext*, StatementContext&) override;
};
```

**拒绝 — 选项(单一 `Tcgen05Handler : InstructionHandler` 直接覆盖 `ExecPipe`)**:
- 会绕过 pipeline acquire/collect/commit 基础设施
- 失去对 operand 生命周期的统一管理
- 与 S_WMMA 模式冲突

**`prepareOperands` 关键步骤**:
```cpp
bool Tcgen05PipelineHandler::prepareOperands(ThreadContext *context, StatementContext &stmt) {
    const Tcgen05Instr &instr = std::get<Tcgen05Instr>(stmt.data);
    // MMA/LD/ST 至少 1 个 operand,COMMIT/WAIT/FENCE 0 个
    if (instr.operands.empty()) return true;
    if (!acquireAllOperands(context, instr.operands, instr.qualifiers,
                            static_cast<int>(instr.operands.size()))) {
        return false;
    }
    context->collect_operands(stmt, instr.operands, &(instr.qualifiers));
    return true;
}
```

### D3: 现有函数处理 — 改造为类方法

**采纳**: 现有 5 个 `processTcgen05Xxx` 自由函数改造为 `Tcgen05Handler::processTcgen05Operation(context, ops, qualifiers, instr)` 方法体的 `switch (instr.op_kind)` 分发。

**为什么需要改造**:
- X-Macro 期望 `IMPLEMENT_TCGEN05_INSTR_HANDLER` 生成的 stub 是 `Tcgen05Handler::processTcgen05Operation` 方法
- 现有自由函数签名 `void(ThreadContext*, const Tcgen05Instr&)` 不匹配 pipeline 接口

**改造示例**:
```cpp
// 现行 signature:
//   void processTcgen05Mma(ThreadContext* ctx, const Tcgen05Instr& instr)
// 改造后:
//   void Tcgen05Handler::processTcgen05Operation(
//       ThreadContext *ctx, void **operands,
//       const std::vector<Qualifier> &qualifiers,
//       const Tcgen05Instr &instr) {
//     if (instr.op_kind != Tcgen05OpKind::MMA) return; // 不在 dispatch 路径
//     // ...原 processTcgen05Mma 逻辑不变,但用 instr.op_kind 分发
//   }
```

### D4: 测试策略 — 集成路径 + 单元 pipeline

**采纳**: 双层测试覆盖

| 测试层 | 文件 | 验证点 |
|--------|------|--------|
| 类型二(集成) | `tests/integration/tcgen05/test_tcgen05_dispatch.cpp`(新建子目录) | 5 个 op_kind 的完整 dispatch 路径不触发 EXIT |
| 类型一(单元) | `tests/unit/ptx_ir/test_tcgen05_pipeline_handler.cpp` | 3 阶段 pipeline(prepare/execute/commit)单独验证 |

**为什么双层**:
- 集成测试:验证 dispatch table 真正注册了(`get_handler` 不返 nullptr),且 `ExecPipe` 被调用
- 单元测试:验证 pipeline 基础设施 3 阶段顺序正确 + operand 收集/释放

### D5: E2E 注释更新 — `test_blackwell_gemm.cu:11`

**采纳**: 更新现有 `tests/e2e/kernel/test_blackwell_gemm.cu:11` 的注释:

```diff
- // Uses float (not half) to avoid nvcc sm_100 PTX .nc.u16 loads that the ANTLR grammar does not support
+ // Uses float (not half) to avoid nvcc sm_100 PTX grammar limits.
+ // tcgen05.mma itself is wired to dispatch (commit df6dde7 + fix-tcgen05-handler-dispatch).
```

**理由**: 现有注释暗示 tcgen05 根本没实现,实际是"dispatch 没接好"。修好 dispatch 后,应反映"handler 已实现,f16 受 grammar 限制而非 dispatcher 限制"。

## Risks / Trade-offs

| ID | Risk | Level | Mitigation |
|----|------|-------|------------|
| R1 | Phase 1 `ptx_op.def` 修改破坏 X-Macro 展开 | 🟡 中 | 编译验证;若 `S_TCGEN05_*` enum 重定义 → 用 C++ guard |
| R2 | 现有函数的 `(void)instr;` 模式忽略 IR 字段(cta_group/dtype/num_regs) | 🟡 中 | Phase 3 测试**显式**断言 multi-variant dispatch(MMA .sp/.block_scale) |
| R3 | E2E `test_blackwell_gemm.cu` 数值从 skip→real 变化 | 🟡 中 | 在 Phase 3 commit message 中记录 baseline vs new;若 GEMM 数值不对,Phase 3 整体 revert |
| R4 | `_execute_once` nullptr fallback 仍存在 → 静默 EXIT | 🟢 低 | 保留作为防御;未来新指令若忘注册,显式 set_state(EXIT) 而非未定义行为 |
| R5 | wmma 与 tcgen05 同时在 X-Macro 中并存 → 编译冲突 | 🟢 低 | 不同 enum 值,不同 struct_kind(`WMMA_INSTR` vs `TCGEN05_INSTR`),不同 macro(`IMPLEMENT_WMMA_INSTR_HANDLER` vs `IMPLEMENT_TCGEN05_INSTR_HANDLER`) |
| R6 | Phase 2 stub 抛异常让现有测试 fail | 🟡 中 | Phase 2 后**禁止**archive;必须 Phase 3 立即跟进 |
| R7 | `TcQueue::commit` 释放 waiter 路径涉及 `advance_thread_pc` → DUAL STATE MECHANISM 风险 | 🟡 中 | Phase 3 集成测试**专门**断言:`commit` 后 lane 的 `is_blocked=false` + `pc` 推进 |
| R8 | commit/wait handler 阻塞/释放语义在 dispatcher 中首次启用 → 可能暴露现有 BUG | 🟡 中 | per ptx-lessons-learned §3:**任何已有测试回归 → 立即 revert 该 Phase,不混入后续 commit** |

## Migration Plan

### Phase 1: 结构性变更(commit 1 — 无功能,仅 CMake/X-Macro 调整)

**Step 1.1** — 修改 `include/ptx_ir/ptx_op.def`:
- 删除 line 129-136 注释
- 在 line 127 后追加 11 个 X-Macro 条目:
```cpp
X(S_TCGEN05_ALLOC,      tcgen05.alloc,      Tcgen05,    1, TCGEN05_INSTR, tensor)
X(S_TCGEN05_DEALLOC,    tcgen05.dealloc,    Tcgen05,    1, TCGEN05_INSTR, tensor)
X(S_TCGEN05_RELINQUISH, tcgen05.relinquish, Tcgen05,    1, TCGEN05_INSTR, tensor)
X(S_TCGEN05_LD,         tcgen05.ld,         Tcgen05,    2, TCGEN05_INSTR, tensor)
X(S_TCGEN05_ST,         tcgen05.st,         Tcgen05,    2, TCGEN05_INSTR, tensor)
X(S_TCGEN05_CP,         tcgen05.cp,         Tcgen05,    3, TCGEN05_INSTR, tensor)
X(S_TCGEN05_MMA,        tcgen05.mma,        Tcgen05,    4, TCGEN05_INSTR, tensor)
X(S_TCGEN05_MMA_WS,     tcgen05.mma.ws,     Tcgen05,    4, TCGEN05_INSTR, tensor)
X(S_TCGEN05_COMMIT,     tcgen05.commit,     Tcgen05,    0, TCGEN05_INSTR, tensor)
X(S_TCGEN05_WAIT,       tcgen05.wait,       Tcgen05,    0, TCGEN05_INSTR, tensor)
X(S_TCGEN05_FENCE,      tcgen05.fence,      Tcgen05,    0, TCGEN05_INSTR, tensor)
```

**Step 1.2** — 修改 `include/ptx_ir/ptx_types.h`:
- 删除 line 28-38 的 11 行手工 S_TCGEN05_* enum(由 X-Macro 自动产生,可能需要放在 X-Macro 循环之后以防重复)

⚠️ **`enum_val` 在 `ptx_types.h` 的 X-Macro 是 `#define X(enum_val, ...)` 形式**: 11 个 S_TCGEN05_* 在 `ptx_op.def` 加 X 条目后,`ptx_types.h` 的 X-Macro 循环会自动产生它们。需要删除 line 28-38 的手工列举。

**Step 1.3** — 验证:
- `cmake --build build` 编译验证(无功能变化,死代码仍死代码)
- `ctest --output-on-failure` 跑全量 baseline(170+/170+ PASS)

### Phase 2: 适配层(commit 2 — stub 抛异常但 dispatcher 接好)

**Step 2.1** — `include/ptxsim/instruction_handlers.h`:
- 替换 `DECLARE_TCGEN_INSTR_HANDLER`(line 130) 为 `DECLARE_TCGEN05_INSTR_HANDLER`,改为继承 `Tcgen05PipelineHandler`,声明 `processTcgen05Operation`

**Step 2.2** — `src/ptxsim/instruction_handlers.cpp`:
- 替换 `IMPLEMENT_TCGEN_INSTR_HANDLER`(line 172) 为 `IMPLEMENT_TCGEN05_INSTR_HANDLER`,在 `processTcgen05Operation` 中抛 `UnsupportedInstructionException`(stub)

**Step 2.3** — `include/ptxsim/instruction_base.h`:
- 新增 `Tcgen05PipelineHandler` 类声明(3-阶段接口)

**Step 2.4** — `src/ptxsim/instruction_base.cpp`:
- 新增 `Tcgen05PipelineHandler` 3 阶段实现:
  - `prepareOperands`: 委托 `GenericPipelineHandler::prepareOperands`(复用 `acquireAllOperands` + `collect_operands`)— 注意 `Tcgen05Instr.operands` 可能是空(COMMIT/WAIT/FENCE),需早返回
  - `executeOperation`: 调用 `processTcgen05Operation(context, &operand_collected[0], qualifiers, std::get<Tcgen05Instr>(stmt.data))`
  - `commitResults`: 若 `instr.operands` 非空且首元素是 dst,调用 `context->commit_operand(stmt, instr.operands[0], instr.qualifiers)`,然后 `releaseAllOperands`

**Step 2.5** — 验证:
- 编译通过(新 stub 抛异常但不破坏编译)
- 预期现有测试**可能 fail**(因为新 dispatcher 现在会抛异常而非静默 EXIT)
- ❌ **若 Phase 2 后所有 ctest PASS 是错的**:可能是 dispatcher 根本没调到(检查 stderr 噪音是否消失)

### Phase 3: 现有 handler 适配(commit 3 — 启用真功能)

**Step 3.1** — `src/ptxsim/instructions/tcgen05.cpp`:
- 删除 5 个自由函数定义
- 添加 `namespace ptxsim { void Tcgen05Handler::processTcgen05Operation(...) { switch (instr.op_kind) { ... } } }` 方法
- **关键**: 必须保留原 fragment arithmetic 逻辑逐字不变(per ADR-0016 已 frozen)

**Step 3.2** — 验证:
- 编译通过
- 跑 `fix-tcgen05-test-coverage-gaps` 中的 7 个新测试(dead-code coverage→真路径)— **预期全部 PASS**(这些测试已为 future dispatcher 接入做了准备)
- 跑 `tests/e2e/kernel/test_blackwell_gemm.cu` — 记录 skip→real 的差异
- 若 E2E 数值与 baseline 不一致:**整体 revert Phase 3**,修复后重新尝试

### Phase 4: 测试 + 文档(commit 4)

**Step 4.1** — `tests/integration/tcgen05/test_tcgen05_dispatch.cpp`(新建):
```cpp
#include "ptxsim/testing/scheduler_utils.h"
#include "ptxsim/testing/instruction_helpers.h"

using ptxsim::testing::step_warp;

TEST_CASE("tcgen05.mma dispatch reaches handler", "[integration][tcgen05][dispatch]") {
    SMContext sm(4, 128, 4096, 0);
    // ... 构造 mma 指令序列
    REQUIRE(step_warp(warp, stmts) != EXIT);  // 验证不再触发 EXIT
}
```

**Step 4.2** — `tests/unit/ptx_ir/test_tcgen05_pipeline_handler.cpp`(新建):
- 构造 mock `Tcgen05Instr`(operands, qualifiers, op_kind)
- 调用 `Tcgen05Handler{}::prepareOperands / executeOperation / commitResults`
- 验证 3-阶段顺序

**Step 4.3** — CMakeLists 注册:
- `tests/integration/CMakeLists.txt`:新增子目录 `tcgen05/`
- `tests/unit/CMakeLists.txt`:注册 `unit_ptx_ir_tcgen05_pipeline_handler`

**Step 4.4** — 文档:
- 删除 `src/ptxsim/instructions/AGENTS.md` 中 "dead code" 标注
- 根 `AGENTS.md` 已知限制表:删除 "dispatch 死代码" 行
- 更新 `tests/e2e/kernel/test_blackwell_gemm.cu:11` 注释

**Step 4.5** — Archive:
- `ctest --output-on-failure` 最终 PASS(177+/177+,包括新加的 2 个测试)
- `openspec archive fix-tcgen05-handler-dispatch --yes`
- 提交并 git push

## Implementation Order Diagram

```
Phase 0  .worktrees/baseline-dispatch-fix 创建 + 全量测试 baseline
  │
  ▼
Phase 1  ptx_op.def + ptx_types.h X-Macro 注册(无功能变化)
  │     └─ 编译 + baseline test 验证
  ▼
Phase 2  适配层(stub 抛异常)
  │     └─ 编译验证;预期测试已开始变化
  ▼
Phase 3  现有 handler 适配(真功能启用)
  │     └─ 测试 + E2E;若 fail → 整体 revert
  ▼
Phase 4  新测试 + 文档 + archive
```

## Backward Compatibility & Rollback

- **API surface**: 无变化(都是内部 dispatch,无外部 API 影响)
- **行为变化**: `S_TCGEN05_*` 指令从"静默 EXIT"变为"实际执行" — 这是 bug fix,不是兼容性 break
- **E2E 输出变化**: 任何之前因 EXIT 而失败的数值,现在有真实结果 — 可能在某些测试中产生 baseline diff
- **回滚**: 每个 Phase 都是独立 commit → `git revert <commit-hash>` 回滚到上一个 commit(per ptx-lessons-learned §3)

## Open Questions

| Q# | Question | Resolved In |
|----|----------|-------------|
| Q1 | 11 个 X-Macro 的 `op_count` 字段如何处理(MMA=4, COMMT=0, FENCE=0)? | 按 ADR-0016 §D5 (operand count 反映 generic operand 数量,零操作数变体=0) |
| Q2 | `Tcgen05PipelineHandler` 是直接继承 `PipelineHandler` 还是 `WmmaPipelineHandler`? | 直接 `PipelineHandler`(避免 `S_WMMA` 路径上的 acquire/collect 副作用被默认行为污染) |
| Q3 | `IMPLEMENT_TCGEN_INSTR_HANDLER` 旧宏是否删除? | 是 — line 172 必须替换(留下会引起 linker confusion) |
| Q4 | 若 Phase 3 启用后某个历史 GEMM E2E 测试 fail,如何诊断? | ptx-barrier-mechanism 技能(commit/wait 间接副作用是 STA/SYNC 风格风险点) |

## 影响范围表格

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptx_ir/ptx_op.def` | 直接修改 | 11 行 X-Macro + 删除注释 |
| `include/ptx_ir/ptx_types.h` | 直接修改 | 删除 11 行手工 enum |
| `include/ptxsim/instruction_handlers.h` | 直接修改 | 替换宏 |
| `src/ptxsim/instruction_handlers.cpp` | 直接修改 | 替换宏 |
| `include/ptxsim/instruction_base.h` | 新增 | `Tcgen05PipelineHandler` |
| `src/ptxsim/instruction_base.cpp` | 新增 | 3-阶段实现 |
| `src/ptxsim/instructions/tcgen05.cpp` | 直接修改 | 自由函数 → 类方法(逻辑不变) |
| `tests/integration/tcgen05/` (新) | 新增 | dispatch 集成测试 |
| `tests/unit/ptx_ir/test_tcgen05_pipeline_handler.cpp` | 新增 | 单元 pipeline 测试 |
| `tests/integration/CMakeLists.txt` | 新增子目录注册 | +5 行 |
| `tests/unit/CMakeLists.txt` | 新增测试注册 | +5 行 |
| `tests/e2e/kernel/test_blackwell_gemm.cu` | 注释更新 | 1 行 |
| `src/ptxsim/instructions/AGENTS.md` | 删除 dead code 行 | -3 行 |
| 根 `AGENTS.md` | 已知限制表更新 | -1 行 |
| `openspec/specs/tcgen05-ir-types/spec.md` | 状态更新(req implemented) | 注释 |
