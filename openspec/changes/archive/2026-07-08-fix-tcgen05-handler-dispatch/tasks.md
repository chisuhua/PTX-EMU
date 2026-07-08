# Tasks: Wire tcgen05 Handlers to Dispatch Pipeline

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + [specs/tcgen05-handler-dispatch/spec.md](specs/tcgen05-handler-dispatch/spec.md)
> **前置 change**: `implement-tcgen05-handlers-core` (df6dde7)
> **范围**: 4 atomic commits(4 phases per design.md)
> **关键约束**: 任何 Phase 失败 → 立即 revert 该 Phase(per ptx-lessons-learned §3)

## 0. Pre-Implementation Review

### 0.1 验证假设(全部必做,失败则 STOP)

- [ ] 0.1.1 `ls src/ptxsim/instructions/tcgen05.cpp` 确认存在
- [ ] 0.1.2 `grep -c "void processTcgen05" src/ptxsim/instructions/tcgen05.cpp` 确认正好 5 个函数
- [ ] 0.1.3 `grep "S_TCGEN05_.*=" include/ptx_ir/ptx_types.h` 应显示 11 个手工定义(line 28-38)
- [ ] 0.1.4 `grep "X(S_TCGEN05_" include/ptx_ir/ptx_op.def` **必须为空**(确认尚未注册)
- [ ] 0.1.5 `grep -rn "processTcgen05" src/ptxsim/ | grep -v tcgen05.cpp` **必须为空**(handler 是死代码)
- [ ] 0.1.6 `ls include/ptxsim/instructions/tcgen05.h` —— 由 `fix-tcgen05-test-coverage-gaps` 创建的 forward declaration 头文件应已存在
- [ ] 0.1.7 `ls include/ptxsim/instruction_base.h && grep "class PipelineHandler" include/ptxsim/instruction_base.h` 确认基类存在
- [ ] 0.1.8 `grep "class WmmaPipelineHandler" src/ptxsim/instruction_base.cpp` 确认 WMMA pipeline 可参考
- [ ] 0.1.9 `grep "Tcgen05PipelineHandler\|Tcgen05Handler" src/ptxsim/ include/ptxsim/` **必须为空**(确认尚未实现)
- [ ] 0.1.10 `cd build && ctest -N 2>&1 | wc -l` 确认 baseline 测试数(参考点)

### 0.2 基线 worktree(per ptx-lessons-learned §4)

- [ ] 0.2.1 `git worktree add .worktrees/baseline-dispatch-fix main`
- [ ] 0.2.2 `cd .worktrees/baseline-dispatch-fix && . env.sh`
- [ ] 0.2.3 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
- [ ] 0.2.4 `cd build && ctest --output-on-failure` 验证 170+/170+ baseline PASS
- [ ] 0.2.5 `cd .. && ./tests/ptx/test_all_ptx.sh` 验证 PTX 语法 baseline
- [ ] 0.2.6 记录 baseline `ctest` 输出数字(用于 Phase 4 验证)

### 0.3 创建 worktree

- [ ] 0.3.1 `git checkout -b feat/fix-tcgen05-handler-dispatch`
- [ ] 0.3.2 `git worktree add .worktrees/fix-tcgen05-handler-dispatch feat/fix-tcgen05-handler-dispatch`

## 1. Artifacts Tracking(commit 1)

- [ ] 1.1 `cd .worktrees/fix-tcgen05-handler-dispatch`
- [ ] 1.2 `git add openspec/changes/fix-tcgen05-handler-dispatch/`
- [ ] 1.3 `git commit -m "docs(openspec): add fix-tcgen05-handler-dispatch artifacts (ADR-0016)"`

## 2. Phase 1: X-Macro 注册(commit 2 — 无功能变化,仅结构)

### 2.1 修改 `include/ptx_ir/ptx_op.def`

- [ ] 2.1.1 删除 line 129-136 的注释块(7 行注释)
- [ ] 2.1.2 在 `X(S_WMMA, ...)` 之后(line 128 附近)插入 11 个 `S_TCGEN05_*` X-Macro 条目:
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

### 2.2 修改 `include/ptx_ir/ptx_types.h`

- [ ] 2.2.1 删除 line 28-38 的 11 行手工 `S_TCGEN05_* = ?,` enum 定义
- [ ] 2.2.2 若需要,确认 enum 现在只从 X-Macro 产生

⚠️ **依赖检查**: `ptx_types.h` 中 S_TCGEN05_* enum 由 `ptx_op.def` 的 X-Macro 展开产生。验证方法:`grep -c "S_TCGEN05_" include/ptx_ir/ptx_types.h` 应等于 `0`(因为 enum 名只在 `ptx_op.def` 中存在,在 `ptx_types.h` 是展开为值)

### 2.3 验证(此 Phase 不应有功能变化)

- [ ] 2.3.1 `cmake --build build` 编译验证
- [ ] 2.3.2 `cd build && ctest --output-on-failure` 全量 PASS(等于 baseline 数字)
- [ ] 2.3.3 `./tests/ptx/test_all_ptx.sh` 验证 12 个 tcgen05 PTX fixtures 仍 PASS
- [ ] 2.3.4 `git commit -m "refactor(ptxir): register 11 S_TCGEN05_* via X-Macro (ADR-0016, structural)"`

## 3. Phase 2: 适配层(commit 3 — stub 抛异常)

⚠️ **进入 Phase 2 之前,确保 Phase 1 已成功 commit 且 baseline 仍稳定**

### 3.1 修改 `include/ptxsim/instruction_handlers.h`

- [ ] 3.1.1 找到并删除 line 130 附近的 `DECLARE_TCGEN_INSTR_HANDLER` 宏
- [ ] 3.1.2 添加新宏:
   ```cpp
   #define DECLARE_TCGEN05_INSTR_HANDLER(Name) \
       DECLARE_SIMPLE_HANDLER(Name) \
       class Name##Handler : public Tcgen05PipelineHandler { \
       public: \
           void processTcgen05Operation(ThreadContext *context, void **operands, \
                                        const std::vector<Qualifier> &qualifiers, \
                                        const Tcgen05Instr &instr) override; \
       };
   ```
   **⚠️ NOTE**: 检查 `instruction_handlers.h` 是否已经 `#include` 了 `ptx_ir/tcgen05_instr.h`,若无则需要添加

### 3.2 修改 `src/ptxsim/instruction_handlers.cpp`

- [ ] 3.2.1 删除 line 172 附近的 `IMPLEMENT_TCGEN_INSTR_HANDLER`(目前是 `IMPLEMENT_SIMPLE_HANDLER` 转发)
- [ ] 3.2.2 添加新宏(在 `IMPLEMENT_WMMA_INSTR_HANDLER` 附近):
   ```cpp
   #define IMPLEMENT_TCGEN05_INSTR_HANDLER(Name) \
       IMPLEMENT_SIMPLE_HANDLER(Name) \
       __attribute__((weak)) void Name##Handler::processTcgen05Operation( \
           ThreadContext *context, void **operands, \
           const std::vector<Qualifier> &qualifiers, \
           const Tcgen05Instr &instr) { \
           (void)context; (void)operands; (void)qualifiers; (void)instr; \
           throw UnsupportedInstructionException(#Name, \
               "tcgen05 handler stub: real implementation in tcgen05.cpp"); \
       }
   ```

### 3.3 新增 `include/ptxsim/instruction_base.h`

- [ ] 3.3.1 添加 `Tcgen05PipelineHandler` 类声明:
   ```cpp
   class Tcgen05PipelineHandler : public PipelineHandler {
   public:
       bool prepareOperands(ThreadContext *context, StatementContext &stmt) override;
       bool executeOperation(ThreadContext *context, StatementContext &stmt) override;
       bool commitResults(ThreadContext *context, StatementContext &stmt) override;
   };
   ```
   位置:在 `WmmaPipelineHandler` 附近

### 3.4 新增 `src/ptxsim/instruction_base.cpp`

- [ ] 3.4.1 添加 `Tcgen05PipelineHandler` 3-阶段实现:
   ```cpp
   bool Tcgen05PipelineHandler::prepareOperands(ThreadContext *context, StatementContext &stmt) {
       const Tcgen05Instr &instr = std::get<Tcgen05Instr>(stmt.data);
       if (instr.operands.empty()) return true;
       // 复用 GenericPipelineHandler 的 acquire/collect 路径
       return GenericPipelineHandler::prepareOperands(context, stmt);
   }
   bool Tcgen05PipelineHandler::executeOperation(ThreadContext *context, StatementContext &stmt) {
       const Tcgen05Instr &instr = std::get<Tcgen05Instr>(stmt.data);
       // 把 free function 调用的方式转交给子类实现
       // 注意:processTcgen05Operation 是 virtual,子类的 Tcgen05Handler::processTcgen05Operation 会被调用
       this->processTcgen05Operation(context, &(context->operand_collected[0]), instr.qualifiers, instr);
       return true;
   }
   bool Tcgen05PipelineHandler::commitResults(ThreadContext *context, StatementContext &stmt) {
       // COMMIT/WAIT/FENCE 的 operands 为空,跳过 commit_operand
       // MMA/LD/ST 的 operand[0] 是 dst,调用 commit_operand(模式参考 GenericPipelineHandler::commitResults)
       WmmaInstr &instr = std::get<WmmaInstr>(stmt.data);  // 错误用法 - 应该是 Tcgen05Instr
       // 改成 const Tcgen05Instr &instr = std::get<Tcgen05Instr>(stmt.data);
       // ...
   }
   ```
   **⚠️ NOTE**: 实施时需仔细对应 `GenericPipelineHandler::prepareOperands` / `commitResults` 的实际签名(参考 instruction_base.cpp 中 WmmaPipelineHandler 的写法)

### 3.5 验证

- [ ] 3.5.1 `cmake --build build` 编译验证
- [ ] 3.5.2 `cd build && ctest --output-on-failure` —— **必须记录**:phase 2 引入 stub 抛 `UnsupportedInstructionException`,预期会**触发部分测试失败**(因为之前静默 EXIT 现在会显式抛异常)
- [ ] 3.5.3 **失败处理**: 若测试失败但失败原因**仅是** "UnsupportedInstructionException" 而非真值问题 → 接受(Phase 3 会修复);若失败原因是 GEMM 数值不对或 dispatch 跳错 handler → 立即 `git revert <commit>` 整体回滚 Phase 2
- [ ] 3.5.4 `git commit -m "refactor(ptxsim): add Tcgen05PipelineHandler + stub for S_TCGEN05_* dispatch (ADR-0016)"`

## 4. Phase 3: 现有 handler 适配(commit 4 — 启用真功能)

⚠️ **进入 Phase 3 之前,确保 Phase 2 stub 异常路径可控**

### 4.1 修改 `include/ptxsim/instructions/tcgen05.h`(已由 fix-tcgen05-test-coverage-gaps 创建)

- [ ] 4.1.1 验证头文件存在: `cat include/ptxsim/instructions/tcgen05.h`
- [ ] 4.1.2 **追加** `Tcgen05Handler` 类声明(在头文件中):
   ```cpp
   namespace ptxsim {
   class Tcgen05Handler : public Tcgen05PipelineHandler {
   public:
       void processTcgen05Operation(ThreadContext *context, void **operands,
                                    const std::vector<Qualifier> &qualifiers,
                                    const Tcgen05Instr &instr) override;
   };
   }  // namespace ptxsim
   ```

### 4.2 修改 `src/ptxsim/instructions/tcgen05.cpp`

- [ ] 4.2.1 删除 5 个 `void processTcgen05Xxx(ThreadContext*, const Tcgen05Instr&)` 自由函数定义(line 311-540)
- [ ] 4.2.2 添加 `Tcgen05Handler::processTcgen05Operation` 方法:
   ```cpp
   void Tcgen05Handler::processTcgen05Operation(
       ThreadContext *context, void **operands,
       const std::vector<Qualifier> &qualifiers,
       const Tcgen05Instr &instr) {
       (void)operands; (void)qualifiers;  // 当前实现未用
       switch (instr.op_kind) {
       case Tcgen05OpKind::MMA:
           // 原 processTcgen05Mma body 不变,只复用 instr 的其它属性
           // 实际从 context 读取 / 写入 TMEM
           break;
       case Tcgen05OpKind::LD:
           // 原 processTcgen05Ld body
           break;
       case Tcgen05OpKind::ST:
           // 原 processTcgen05St body
           break;
       case Tcgen05OpKind::COMMIT:
           // 原 processTcgen05Commit body
           break;
       case Tcgen05OpKind::WAIT:
           // 原 processTcgen05Wait body
           break;
       default:
           throw UnsupportedInstructionException("tcgen05.*",
               "op_kind " + std::to_string(static_cast<int>(instr.op_kind)) +
               " not yet implemented (per ADR-0016, deferred but wired)");
       }
   }
   ```

⚠️ **CRITICAL**: 5 个 case 的函数体必须保留原 fragment arithmetic 逻辑 **逐字不变**(per ADR-0016)。可以将原 processTcgen05Xxx 函数体的内容剪切到对应 case 中。

### 4.3 验证

- [ ] 4.3.1 `cmake --build build` 编译验证
- [ ] 4.3.2 `cd build && ctest --output-on-failure` —— **关键验证**:现在 5 个核心 op_kind 应不再抛异常,ALL/Cp/MMA_WS/FENCE 仍抛异常(预期)
- [ ] 4.3.3 **跑 fix-tcgen05-test-coverage-gaps 的 7 个新测试**:`ctest -L "integration;ptx;tcgen05"` + `ctest -L "unit;ptx_ir;tcgen05"` —— **必须仍 PASS**(这些测试从 dead-code 升级为真路径)
- [ ] 4.3.4 **跑 E2E**:`ctest -L "e2e;tcgen05" -V` ——记录与 baseline 的数值差异
- [ ] 4.3.5 **若 e2e_blackwell_gemm 数值变化**:在 commit message 中记录 baseline vs new 数值,但**不**revert(行为变化是预期的 fix)
- [ ] 4.3.6 **若 e2e_blackwell_gemm 失败原因**是 segfault/crash(非数值)→ 立即 `git revert <commit>` 整体回滚 Phase 3,debug
- [ ] 4.3.7 `git commit -m "feat(ptxsim): wire 5 processTcgen05Xxx handlers to Tcgen05Handler::processTcgen05Operation dispatch (ADR-0016)"`

## 5. Phase 4: 测试 + 文档 + archive(commit 5)

### 5.1 创建 `tests/integration/tcgen05/`(新子目录)

- [ ] 5.1.1 `mkdir -p tests/integration/tcgen05`
- [ ] 5.1.2 创建 `tests/integration/tcgen05/CMakeLists.txt`:
   ```cmake
   include_directories(${CMAKE_SOURCE_DIR}/include)
   add_catch_test(integration_tcgen05_dispatch
       test_tcgen05_dispatch.cpp
   )
   set_tests_properties(integration_tcgen05_dispatch PROPERTIES
       LABELS "integration;tcgen05;dispatch")
   ```
- [ ] 5.1.3 注册到 `tests/integration/CMakeLists.txt`:添加 `add_subdirectory(tcgen05)`

### 5.2 创建 `tests/integration/tcgen05/test_tcgen05_dispatch.cpp`

- [ ] 5.2.1 创建测试文件,关键测试场景:
   ```cpp
   #include <catch_amalgamated.hpp>
   #include "ptxsim/testing/scheduler_utils.h"
   #include "ptxsim/testing/instruction_helpers.h"

   using ptxsim::testing::step_warp;

   TEST_CASE("S_TCGEN05_MMA dispatch reaches processTcgen05Operation", "[integration][tcgen05][dispatch]") {
       // 构造 mma 指令序列 + warp
       // step_warp 后检查 lane state != EXIT
   }

   TEST_CASE("All 11 S_TCGEN05_* dispatch correctly (no nullptr handler)", "[integration][tcgen05][dispatch]") {
       // 程序化遍历 S_TCGEN05_* 11 个值,确认 get_handler 全非 nullptr
   }
   ```

### 5.3 创建 `tests/unit/ptx_ir/test_tcgen05_pipeline_handler.cpp`

- [ ] 5.3.1 创建测试文件,验证 3-阶段 pipeline:
   ```cpp
   TEST_CASE("Tcgen05Handler::prepareOperands with zero operands", "[unit][ptx_ir][tcgen05][pipeline]") {
       // 构造带空 operands 的 Tcgen05Instr(COMMIT-like)
       // 验证 prepareOperands 返 true 不 crash
   }
   TEST_CASE("Tcgen05Handler::executeOperation dispatches by op_kind", "[unit][ptx_ir][tcgen05][pipeline]") {
       // 构造 MMA instr
       // 调用 executeOperation
       // 验证被路由到 MMA 分支(可通过副作用追踪)
   }
   ```

### 5.4 CMakeLists 注册新测试

- [ ] 5.4.1 `tests/unit/CMakeLists.txt`:注册 `unit_ptx_ir_tcgen05_pipeline_handler`

### 5.5 文档更新

- [ ] 5.5.1 编辑 `src/ptxsim/instructions/AGENTS.md`:删除"dead code" 段落(若有)
- [ ] 5.5.2 编辑根 `AGENTS.md`:已知限制表删除"5 core handler dispatch 死代码"行(若 fix-tcgen05-test-coverage-gaps 之前已加)
- [ ] 5.5.3 更新 `tests/e2e/kernel/test_blackwell_gemm.cu:11` 注释:从 grammar 限制改为"handler 在 df6dde7 + fix-tcgen05-handler-dispatch 实现"

### 5.6 Archive

- [ ] 5.6.1 `cd build && ctest --output-on-failure` 最终验证(至少 178+/178+ PASS,数字 ≥ baseline)
- [ ] 5.6.2 `./tests/ptx/test_all_ptx.sh` 验证 PTX 语法 baseline
- [ ] 5.6.3 `openspec archive fix-tcgen05-handler-dispatch --yes`
- [ ] 5.6.4 `git add openspec/changes/archive/`
- [ ] 5.6.5 `git commit -m "chore(openspec): archive fix-tcgen05-handler-dispatch (ADR-0016)"`

## Final Validation

- [ ] 6.1 `git log --oneline | head -6` 显示 5 atomic commits
- [ ] 6.2 `cd build && ctest --output-on-failure` 全量测试 PASS
- [ ] 6.3 `ctest -L "integration;tcgen05;dispatch" -V` 显示新测试 PASS
- [ ] 6.4 `ctest -L "unit;ptx_ir;tcgen05" -V` 显示 pipeline 测试 PASS
- [ ] 6.5 `ctest -L "e2e;tcgen05" -V` 显示 E2E 仍 PASS(数值可能变)
- [ ] 6.6 `./tests/ptx/test_all_ptx.sh` 12 个 tcgen05 fixtures 仍 PASS

## Risks Recap(per design.md)

| Risk | Phase 触发 | Mitigation |
|------|-----------|------------|
| R1 Phase 1 `ptx_op.def` 破坏 X-Macro | Phase 1 | 编译 + baseline test 对比 |
| R2 `(void)instr;` 模式忽略 IR 字段 | Phase 3 | 测试显式 multi-variant dispatch |
| R3 E2E `test_blackwell_gemm` 数值变化 | Phase 3 | commit message 记录 baseline vs new |
| R4 nullptr fallback 消失影响新指令 | Phase 3 | 保留兜底逻辑 |
| R5 wmma/tcgen05 并存编译冲突 | Phase 2 | enum 不同 + macro 不同 |
| R6 Phase 2 stub 抛异常让测试 fail | Phase 2 | Phase 3 立即跟进 |
| R7 `TcQueue::commit` 双状态机制 | Phase 3 | 集成测试**专门**断言状态变化 |
| R8 dispatch 启用暴露历史 BUG | Phase 3 | **任何回归 revert 该 Phase** |

## Lessons-Learned 应用

- **§3 分 Phase commit**: 4 atomic commits 已规划;每个 Phase 独立 commit,独立可 revert
- **§4 基线 worktree**: `.worktrees/baseline-dispatch-fix` 在 Phase 0.2 创建
- **§7 Pre-impl review**: 已通过 explore + analysis 确认 5 个事实
- **§NEW (如发现新模式)**: 若 Phase 3 中确认 dispatch 修好后,`fix-tcgen05-test-coverage-gaps` 的死代码测试自动变为真路径测试 — 这是**新的 lessons-learned 条目**:
  > "若 test change 标注 'dead code coverage',应同时设计 dispatcher 修复 change 让死代码测试自动升级为真路径测试,避免重复工作"
  → 在 .opencode/skills/ptx-lessons-learned/SKILL.md 添加此条

## Lessons-Learned 钩子

- [ ] 7.1 在 Phase 4 完成后,检查 `.opencode/skills/ptx-lessons-learned/SKILL.md` 是否需要加新条目
- [ ] 7.2 在 `docs/dev-process/lessons-learned.md` 添加 Phase 3 中遇到的具体经验(若有)
- [ ] 7.3 创建后续 issue `fix-tcgen05-handler-extended` 跟踪 alloc/dealloc/cp/mma_ws/fence 的实现(Change-3d)
