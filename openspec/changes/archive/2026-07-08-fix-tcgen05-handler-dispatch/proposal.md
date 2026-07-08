# Wire tcgen05 Handlers to Dispatch Pipeline

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **前置 spec**: [openspec/specs/tcgen05-ir-types/spec.md](../../specs/tcgen05-ir-types/spec.md) — 已设计完整的 `Tcgen05PipelineHandler` 模式,本 change **落实**该 spec
> **前置 change**: `implement-tcgen05-handlers-core` (archived @ df6dde7) — 写了 5 个 `processTcgen05*` 自由函数但**未接入 dispatch**(死代码)
> **依赖 change**: `fix-tcgen05-test-coverage-gaps` — 提供 dead-code coverage 测试基础设施(独立 commit 后,死代码不再"死代码",那些测试成为真实路径验证)
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §4(基线 worktree)+ §7(Pre-impl review)

## Why

`implement-tcgen05-handlers-core` (commit `df6dde7`) 实现了 5 个 `processTcgen05*` handler(在 `tcgen05.cpp:311-540`, `src/ptxsim/instructions/tcgen05.cpp`),**但从未接入 dispatch 管道**。具体事实:

1. `S_TCGEN05_*` 枚举值定义在 `ptx_types.h:28-38`(在 X-Macro 循环之后)
2. `ptx_op.def:129-136` 显式注释:`S_TCGEN05_*` 排除在 X-Macro 之外
3. `InstructionFactory::initialize()` 只从 `ptx_op.def` 注册 handler
4. `grep -rn "processTcgen05" src/ptxsim/ | grep -v tcgen05.cpp` 返回 0 结果(handler 是死代码)
5. 任何含 `tcgen05.*` 指令的 PTX 执行:`_execute_once()` 第 143 行 `get_handler` 返回 `nullptr` → `set_state(EXIT)` → lane 静默终止

**影响**:
1. E2E 测试 `tests/e2e/kernel/test_blackwell_gemm.cu` 中所有 tcgen05 指令实际并未执行 tcgen05 路径 → 现有 GEMM 测试的真值性可疑
2. `fix-tcgen05-test-coverage-gaps` 中的单元测试是 "dead code coverage"(依赖 handler 头文件 + 直接调用) — 在 dispatch 修好后,这些测试将真正走生产路径
3. 任何后续 Blackwell kernel (.mma, .commit, .wait 等) 在 PTX-EMU 中立即崩溃(per-lane EXIT)

## What Changes

### 修改 `src/ptx_ir/ptx_op.def`

**恢复 X-Macro 注册**(删除 line 129-136 注释,加入 11 个 X 条目):

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

### 修改 `include/ptx_ir/ptx_types.h`

**删除 X-Macro 之后的手工添加**的 `S_TCGEN05_*`(lines 28-38) —— 它们现在通过 X-Macro 注册。

⚠️ **BREAKING**: `ptx_types.h:28-38` 当前是手工枚举,且与 `ptx_op.def` 注释矛盾。本 change 同时删除这 11 行,改由 `ptx_op.def:129` 处的 X-Macro 提供。

### 修改 `include/ptxsim/instruction_handlers.h`

```cpp
// 替换 line 130 的 IMPLEMENT_TCGEN_INSTR_HANDLER 为 IMPLEMENT_TCGEN05_INSTR_HANDLER
#define DECLARE_TCGEN05_INSTR_HANDLER(Name) \
    class Name##Handler : public Tcgen05PipelineHandler { \
    public: \
        void processTcgen05Operation(ThreadContext *context, void **operands, \
                                     const std::vector<Qualifier> &qualifiers, \
                                     const Tcgen05Instr &instr) override; \
    };
```

### 修改 `src/ptxsim/instruction_handlers.cpp`

**替换 line 172 的 `IMPLEMENT_TCGEN_INSTR_HANDLER`**(目前只是 `IMPLEMENT_SIMPLE_HANDLER` 转发,只推进 PC),改为:

```cpp
#define IMPLEMENT_TCGEN05_INSTR_HANDLER(Name) \
    __attribute__((weak)) void Name##Handler::processTcgen05Operation( \
        ThreadContext *context, void **operands, \
        const std::vector<Qualifier> &qualifiers, const Tcgen05Instr &instr) { \
        (void)context; (void)operands; (void)qualifiers; (void)instr; \
        throw UnsupportedInstructionException("tcgen05.*", "not implemented"); \
    }
```

### 新增 `src/ptxsim/instruction_base.h` / `instruction_base.cpp`

**新增 `Tcgen05PipelineHandler`**(镜像 `WmmaPipelineHandler` 3 阶段流水线):

```cpp
class Tcgen05PipelineHandler : public PipelineHandler {
public:
    bool prepareOperands(ThreadContext*, StatementContext&) override;
    bool executeOperation(ThreadContext*, StatementContext&) override;
    bool commitResults(ThreadContext*, StatementContext&) override;
};
```

### 修改 `src/ptxsim/instructions/tcgen05.cpp`

**替换 5 个 `processTcgen05*` 自由函数** 为 `Tcgen05Handler::processTcgen05Operation(...)` 方法(以 `op_kind` switch 分发),保留原有逻辑(per ADR-0016)。

### 修改 `src/ptxsim/instruction_factory.cpp`

无需修改 —— X-Macro 展开自动注册 `handler_map[S_TCGEN05_*] = new Tcgen05Handler()`。

### 修改 `AGENTS.md` + 各种 `src/ptxsim/instructions/AGENTS.md`

- 移除"dispatch 死代码"标注
- 添加新 constraint:禁止在 `ptx_op.def` 之后手工添加 StatementType enum

### 不修改(范围外)

- ❌ 不修改 `tcgen05.cpp` 中 5 个函数的 fragment arithmetic 逻辑(已在 df6dde7 完成)
- ❌ 不修改 grammar(grammar 已有 `tcgen05Inst` 单一 rule)
- ❌ 不修改 `Tcgen05Instr` struct 定义(已在 tcgen05-ir-types 中)
- ❌ 不实现 alloc/dealloc/cp/mma_ws/fence(留给 Change-3d `implement-tcgen05-handlers-extended`)

## Non-Goals

- ❌ 不实现 6 extended handler(Change-3d scope)
- ❌ 不修改 `processTcgen05*` 内部算法——只做接口适配
- ❌ 不修改 `tests/e2e/kernel/test_blackwell_gemm.cu` 的硬编码 f32 模式(f16 修复在 Change-3d)

## Goals

### Phase 1: 基础设施(1 commit)

1. `ptx_op.def`:删除 line 129-136 注释 + 加入 11 个 X 条目
2. `ptx_types.h`:删除 line 28-38 手工枚举(由 X-Macro 自动产生)
3. 编译验证(无功能变化,死代码仍死代码)
4. 跑全量测试(170+/170+ baseline)

### Phase 2: 适配层(1 commit)

1. `instruction_handlers.h`:增加 `DECLARE_TCGEN05_INSTR_HANDLER`
2. `instruction_handlers.cpp`:替换 `IMPLEMENT_TCGEN_INSTR_HANDLER` → `IMPLEMENT_TCGEN05_INSTR_HANDLER`(带异常抛出的 stub)
3. `instruction_base.h/.cpp`:新增 `Tcgen05PipelineHandler`(复用 `acquireAllOperands` + `collect_operands` + `commit_operand`)
4. `tcgen05.cpp`:改造 5 个自由函数为 `Tcgen05Handler::processTcgen05Operation(context, ops, qualifiers, instr)` 方法体,以 `switch (instr.op_kind)` 分发
5. 编译验证

### Phase 3: 测试 + 验证(1 commit)

1. 添加 `tests/integration/tcgen05/test_tcgen05_dispatch.cpp`:用 `ptxsim::testing::step_warp` 驱动完整路径,验证 5 个 op_kind 不再触发 `set_state(EXIT)`
2. 添加 `tests/unit/ptx_ir/test_tcgen05_pipeline_handler.cpp`:验证 3 阶段流水线(prepare/execute/commit)正确调用
3. 运行已有 E2E `test_blackwell_gemm.cu` —— 此测试现在**真正**走到 `processTcgen05*` handler 路径(预期可能有数值差异,因为之前是 skip → EXIT 路径)
4. 修正 `test_blackwell_gemm.cu` 注释(从"ANTLR grammar 限制"改为"handler 在 df6dde7 实现 + 本 change 接入 dispatch")

### Phase 4: 文档 + Archive(1 commit)

1. 移除 `src/ptxsim/instructions/AGENTS.md` 中"dead code"标注
2. 根 `AGENTS.md` 已知限制表更新:删除 "dispatch 死代码" 行
3. Archive

## Capabilities

### New Capabilities
- **`tcgen05-handler-dispatch`**(`openspec/specs/tcgen05-handler-dispatch/spec.md`):定义 tcgen05 handler 通过 dispatch 管道可访问、3-阶段 pipeline 路由、handler_map 注册、`set_state(EXIT)` fallback 不再触发的契约。
  - 这是落实 [openspec/specs/tcgen05-ir-types/spec.md](../../specs/tcgen05-ir-types/spec.md) 中已经声明的 "shall" 意图(spec line 25 `X(S_TCGEN05_*, ..., TCGEN05_INSTR, tcgen05)` 与 line 114 `IMPLEMENT_TCGEN05_INSTR_HANDLER`、`line 126-143` 的 `Tcgen05PipelineHandler` requirement)

### Modified Capabilities
- 无(`tcgen05-ir-types` 不修改,只是在 archive 后从 "shall" 标注为 "implemented",由 sync-specs 自动处理)

## Impact

### 影响的代码(预计)

| 文件 | 变更类型 | LoC 估计 |
|------|---------|---------|
| `include/ptx_ir/ptx_op.def` | 修改(+13 行 -7 行) | +6 |
| `include/ptx_ir/ptx_types.h` | 修改(-11 行) | -11 |
| `include/ptxsim/instruction_handlers.h` | 修改(替换宏) | +10 |
| `src/ptxsim/instruction_handlers.cpp` | 修改(替换宏) | +15 |
| `include/ptxsim/instruction_base.h` | 新增 `Tcgen05PipelineHandler` | +25 |
| `src/ptxsim/instruction_base.cpp` | 新增 3-阶段实现 | +60 |
| `src/ptxsim/instructions/tcgen05.cpp` | 自由函数 → 类方法(等价规模) | +10 / -10 |
| `tests/integration/tcgen05/CMakeLists.txt` | 新增子目录 | +15 |
| `tests/integration/tcgen05/test_tcgen05_dispatch.cpp` | 新增 | +120 |
| `tests/unit/ptx_ir/test_tcgen05_pipeline_handler.cpp` | 新增 | +80 |
| `tests/integration/CMakeLists.txt` | 注册新测试 | +5 |
| `tests/unit/CMakeLists.txt` | 注册新测试 | +5 |
| `src/ptxsim/instructions/AGENTS.md` + 根 AGENTS.md | 更新 | +10 |
| **总计** | | **+360** |

### 影响的依赖

- `tests/ptx/test_all_ptx.sh`(12 个 tcgen05 PTX fixtures 之前仅过语法层 → 现在也跑 handler 层)
- `tests/e2e/kernel/test_blackwell_gemm.cu`(隐式行为变化 — 实际执行 handler 而非 skip)
- `openspec/specs/tcgen05-ir-types/spec.md`(Requirements 现在是 "implemented" 而非 "shall")

### 不影响的依赖

- `tests/ptx/test_all_ptx.sh`(语法测试不变,只检查 ANTLR 是否能解析 PTX 字符串,不执行 handler)
- 其他 handler 模块(arithmetic/memory/control-flow 等)
- `Tcgen05Instr` struct 定义(继承自 Change-3b)

## Design-Time Checklist (Lessons-Learned)

### 函数审计完整性

- [x] Baseline: 5 `processTcgen05*` 自由函数已在 `tcgen05.cpp:311-540`(df6dde7)
- [x] **跨模块状态翻译**: `TcQueue::commit/wait` 在 commit/wait handler 中通过 `advance_thread_pc` 释放等待 lane(per ptx-barrier-mechanism,DUAL STATE MECHANISM 注意事项)
- [x] **invariant 清单**: prepare/execute/commit 3 阶段必须完整执行(不可跳过)— 失败 policy:在 `executeOperation` 返回 false 时 `set_state(EXIT)`

### 多 Phase 推进(4 atomic commits)

- [x] Phase 1: ptx_op.def + ptx_types.h 调整(独立 commit,仅结构性变更,无功能)
- [x] Phase 2: 适配层(独立 commit,无功能 — stub 抛异常)
- [x] Phase 3: 测试 + 验证(独立 commit,允许功能启用但测试先发)
- [x] Phase 4: 文档 + archive(独立 commit)
- [x] **基线 worktree**: `.worktrees/baseline-dispatch-fix`(在 Phase 0.2 创建)
- [x] **失败处理策略**: 任何 Phase 测试失败 → revert 该 Phase,**不**修其他 Phase 后再 commit(per ptx-lessons-learned §3)

### 文档同步

- [x] `src/ptxsim/instructions/AGENTS.md` "dead code" 标注待删除(Phase 4)
- [x] 根 `AGENTS.md` 已知限制表:删除"dispatch 死代码"行(Phase 4)
- [x] `tests/e2e/kernel/test_blackwell_gemm.cu:11` 注释待更新(Phase 3.4)
- [x] **lessons-learned 钩子**(Phase 4 完成时检查):若发现新模式(如 "future change 写了函数但忘了接入 dispatch"),沉淀到 `ptx-lessons-learned`

### ADR 合规性

ADR-0016 §Decision 1-5 已为 `tcgen05.*` 定义了 architecture(Blackwell-only、单 `tcgen05Inst` rule、不兼容 pre-Blackwell)。本 change 完全遵循,无新架构决策。

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| `implement-tcgen05-handlers-core` (df6dde7) | **fix-tcgen05-handler-dispatch** | `implement-tcgen05-handlers-extended`(alloc/dealloc/cp/mma_ws/fence) |
| `tcgen05-ir-types` spec (已存在) | | |
| `fix-tcgen05-test-coverage-gaps`(独立) | | |

- **Change-3b (handlers-core) → 本 change**: 5 handler 函数 + IR types
- **本 change → Change-3d (handlers-extended)**: 提供完整的 dispatch + pipeline 基础设施,新 handler 可复用 `Tcgen05Handler::processTcgen05Operation` 模式
- **本 change ↔ fix-tcgen05-test-coverage-gaps**: 后者的 dead-code coverage 单元测试在 dispatcher 修好后**自动升级**为真实路径测试(同一份代码)
- **不依赖** `cleanup-wmma-namespace`(独立 change)

## 关键决策摘要

| 决策 | 选择 | 理由 |
|------|------|------|
| D1: 接入方式 | 走 `ptx_op.def` X-Macro(对齐 `S_WMMA` 模式) | 与现有架构一致;自动注册 11 个 handler;改动小 |
| D2: 适配层 | `Tcgen05PipelineHandler` 镜像 `WmmaPipelineHandler` | 复用 `acquireAllOperands`/`commit_operand` 基础 |
| D3: 现有函数处理 | 改造为类方法 `Tcgen05Handler::processTcgen05Operation` | 保留 fragment arithmetic(不可重写) |
| D4: 测试策略 | 集成测试(完整路径) + 单元测试(pipeline 单独验证) | 满足 TDD 三类测试要求 |
| D5: back-compat 风险 | E2E `test_blackwell_gemm.cu` 注释更新,数值可能变化 | skip→real 路径变更,记录风险 |

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| R1: Phase 1 修改 `ptx_op.def` 引发 X-Macro 展开错误 | 编译验证 + baseline worktree 对比 |
| R2: 5 个原有函数的 instr 字段被忽略(cta_group/dtype/num_regs) — 适配后应读 | Phase 3 测试中**显式断言**变体 dispatch 路径 |
| R3: E2E test_blackwell_gemm.cu 数值从 skip→handler 可能有差异 | 标注"GEMM E2E 测试本身是 reference baseline" → 在 commit message 中记录 |
| R4: wmma 的 `IMPLEMENT_WMMA_INSTR_HANDLER` 仍存在 — 影响 dispatch? | X-Macro 二者都注册到 `handler_map`(`S_WMMA` 与 `S_TCGEN05_*` 是不同 enum 值,无冲突) |
| R5: `_execute_once` EXIT 路径不再触发 — 可能掩盖未来未注册指令 | 保留 nullptr fallback(per design D5)— 仍 set_state(EXIT),仅不再 stderr 噪音 |
| R6: Phase 2 抛异常导致现有测试 fail | Phase 2 后**必须**立即进入 Phase 3,**不**允许 Phase 4 archive 前留下异常路径 |

## Lessons-Learned 应用

来自 `ptx-lessons-learned`:
- **§3 分 Phase commit**: 4 atomic commits,任一失败立即 revert(本 change 已规划)
- **§4 基线 worktree**: `.worktrees/baseline-dispatch-fix` 在 Phase 0.2 创建
- **§7 Pre-impl review**: 本 proposal 已通过 Metis 审视(独立 sessions)确认 5 个事实(proposal.md 关键事实清单)
- **NEW (如发现新模式)**: "future change 写了函数但忘了接入 dispatch" — 可作为新的 lessons-learned 条目
