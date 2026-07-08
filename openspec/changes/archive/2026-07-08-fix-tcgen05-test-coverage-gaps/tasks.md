# Tasks: Fill tcgen05 5 Core Handler Test Coverage Gaps

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec
> **前置 change**: `implement-tcgen05-handlers-core` (archived @ df6dde7)
> **范围**: 5 atomic commits(原 4 个 + 新增的 Phase 0.5 header commit)
> **关键修正(Metis 审查后)**: 路径全部对齐现有目录;新增 header 创建独立 commit;Phase 2 标注 dead-code coverage

## 0. Pre-Implementation Review

### 0.1 验证假设(全部必做,失败则 STOP)

- [ ] 0.1.1 `ls src/ptxsim/instructions/tcgen05.cpp` 确认存在 + `grep -c "processTcgen05" src/ptxsim/instructions/tcgen05.cpp` 确认有 5 个 handler 函数
- [ ] 0.1.2 **CRITICAL** `grep -rn "processTcgen05" src/ptxsim/ | grep -v tcgen05.cpp` —— **必须返回空**,否则 handler 不只是没 dispatch 而是根本没被调用 → Phase 2 单元测试将无法证明 handler 在真实路径有效
- [ ] 0.1.3 `ls tests/integration/ptx/test_*.cpp | wc -l` 确认 ≥ 18(实际目录已有 18 个 PTX 集成测试)
- [ ] 0.1.4 `ls include/ptx_ir/statement_factory.h && grep -n "makeTcgen05Instr" include/ptx_ir/statement_factory.h` 确认 factory helper 存在
- [ ] 0.1.5 `ls tests/e2e/kernel/test_tcgen05_mma_gemm.cu 2>&1` 确认不存在
- [ ] 0.1.6 `ls include/ptxsim/instructions/tcgen05.h 2>&1` —— **必须返回 "No such file"**,否则说明已有声明
- [ ] 0.1.7 `ls tests/reference/ptx_builtin/ 2>&1 | head -3` 确认 `tests/reference/` 是 reference data 根
- [ ] 0.1.8 `cat tests/e2e/kernel/test_blackwell_gemm.cu | head -20` 确认现有 E2E 注释中 f16 的限制说明
- [ ] 0.1.9 `which cuobjdump 2>&1` 决定 Phase 3 路径
- [ ] 0.1.10 `which nvcc && nvcc --version | tail -1` 决定 Phase 3 f16 预验证可行性

### 0.2 基线 worktree

- [ ] 0.2.1 `git worktree add .worktrees/baseline-tcgen05-tests -b feat/fix-tcgen05-test-coverage-gaps main`
- [ ] 0.2.2 `. .worktrees/baseline-tcgen05-tests/env.sh`(设置 env)
- [ ] 0.2.3 `cmake -S .worktrees/baseline-tcgen05-tests -B .worktrees/baseline-tcgen05-tests/build -DCMAKE_BUILD_TYPE=Release && cmake --build .worktrees/baseline-tcgen05-tests/build -j$(nproc)`
- [ ] 0.2.4 `cd .worktrees/baseline-tcgen05-tests/build && ctest --output-on-failure` 验证 baseline(170/170 PASS)
- [ ] 0.2.5 `./.worktrees/baseline-tcgen05-tests/tests/ptx/test_all_ptx.sh` 验证 PTX 语法 baseline(确保无回归)

### 0.3 创建引用 worktree

- [ ] 0.3.1 `git worktree add .worktrees/fix-tcgen05-test-coverage-gaps feat/fix-tcgen05-test-coverage-gaps`

## 1. Artifacts Tracking(commit 1)

- [ ] 1.1 `cd .worktrees/fix-tcgen05-test-coverage-gaps`
- [ ] 1.2 `git add openspec/changes/fix-tcgen05-test-coverage-gaps/`
- [ ] 1.3 `git commit -m "docs(openspec): add fix-tcgen05-test-coverage-gaps artifacts (ADR-0016, Metis 修订)"`

## 2. Phase 0.5: Handler 头文件(commit 2,新增)

### 2.1 创建 `include/ptxsim/instructions/tcgen05.h`

- [ ] 2.1.1 创建 `include/ptxsim/instructions/tcgen05.h`:
   ```cpp
   #pragma once
   // WARNING: 这些函数当前未被 dispatch 路径调用。
   // S_TCGEN05_* 在 ptx_op.def 中显式排除 X-Macro,所以
   // InstructionFactory::get_handler() 返回 nullptr,代码走
   // thread_context.cpp:142-146 的 "No handler found" 路径。
   // 本头文件仅为 dead-code coverage 测试而存在
   // (独立 change `fix-tcgen05-handler-dispatch` 将修复 dispatcher)
   //
   // DEAD-CODE-NOTICE: 关联 issue 记录在
   //   openspec/changes/fix-tcgen05-test-coverage-gaps/design.md D4
   #include "ptxsim/core/thread_context.h"
   #include "ptx_ir/tcgen05_instr.h"
   
   namespace ptxsim {
   void processTcgen05Mma(ThreadContext* ctx, const Tcgen05Instr& instr);
   void processTcgen05Ld(ThreadContext* ctx, const Tcgen05Instr& instr);
   void processTcgen05St(ThreadContext* ctx, const Tcgen05Instr& instr);
   void processTcgen05Commit(ThreadContext* ctx, const Tcgen05Instr& instr);
   void processTcgen05Wait(ThreadContext* ctx, const Tcgen05Instr& instr);
   }  // namespace ptxsim
   ```

### 2.2 验证

- [ ] 2.2.1 `cmake --build build` 验证编译(头文件不应破坏构建)
- [ ] 2.2.2 `ctest --output-on-failure` 验证零回归(头文件不影响行为)
- [ ] 2.2.3 `git commit -m "refactor(ptxsim): add tcgen05.h forward declarations (ADR-0016, dead-code coverage)"`

## 3. Phase 1: 5 集成 parse 测试(commit 3)

**路径修正**: 全部在 `tests/integration/ptx/`(已存在),注册到 `tests/integration/CMakeLists.txt`(已存在)

### 3.1 test_tcgen05_mma_parse.cpp

- [ ] 3.1.1 创建 `tests/integration/ptx/test_tcgen05_mma_parse.cpp`:
   - 构造 `tcgen05.mma.kind::f16.cta_group::1 d, a, b, c;` PTX 文本
   - 用 ANTLR parser 直接解析(参考 `tests/unit/parser/test_extern_function.cpp` 模式)
   - 验证 `Tcgen05Instr.op_kind == Tcgen05OpKind::MMA`
   - 验证 qualifiers: `KIND::F16` + `CTA_GROUP::1`
   - 验证 operands count == 4
   - 标签:`[integration][ptx][tcgen05][parse]`

### 3.2 test_tcgen05_ld_parse.cpp

- [ ] 3.2.1 创建 `tests/integration/ptx/test_tcgen05_ld_parse.cpp`:
   - 构造 `tcgen05.ld.sync.aligned.32x32b.shared::cta.b32 [r0], [r1];` PTX 文本
   - 验证 `op_kind == LD`
   - 验证 qualifiers: SYNC + ALIGNED + SHAPE_32x32b + SHARED_CTA
   - 验证 operands: dst register [r0], src address [r1](count == 2)

### 3.3 test_tcgen05_st_parse.cpp

- [ ] 3.3.1 创建 `tests/integration/ptx/test_tcgen05_st_parse.cpp`:
   - 构造 `tcgen05.st.sync.aligned.32x32b.shared::cta.b32 [r0], [r1];` PTX 文本
   - 验证 `op_kind == ST`
   - 验证 qualifiers + operands 对称 ld

### 3.4 test_tcgen05_commit_parse.cpp

- [ ] 3.4.1 创建 `tests/integration/ptx/test_tcgen05_commit_parse.cpp`:
   - 构造 `tcgen05.commit.cta_group::1;` PTX 文本
   - 验证 `op_kind == COMMIT`
   - 验证 qualifiers: CTA_GROUP::1(可选 mbarrier)
   - 验证 operands count == 0

### 3.5 test_tcgen05_wait_parse.cpp

- [ ] 3.5.1 创建 `tests/integration/ptx/test_tcgen05_wait_parse.cpp`:
   - 构造 `tcgen05.wait::load.cta_group::1;` + `tcgen05.wait::store.cta_group::1;` PTX 文本(分两个 TEST_CASE)
   - 验证 `op_kind == WAIT`
   - 验证 qualifiers: LOAD/STORE + CTA_GROUP::1
   - 验证 operands count == 0

### 3.6 CMakeLists 注册

- [ ] 3.6.1 编辑 `tests/integration/CMakeLists.txt`,在 `ptx/` 子部分追加:
   ```cmake
   add_catch_test(integration_ptx_tcgen05_mma_parse
       ptx/test_tcgen05_mma_parse.cpp
   )
   set_tests_properties(integration_ptx_tcgen05_mma_parse PROPERTIES
       LABELS "integration;ptx;tcgen05;parse;mma")
   # ... 重复 ld/st/commit/wait 各一条
   ```

### 3.7 验证

- [ ] 3.7.1 `cmake --build build` 验证编译
- [ ] 3.7.2 `ctest -L "integration;tcgen05;parse" -V` 验证 5/5 PASS
- [ ] 3.7.3 `ctest --output-on-failure` 验证零回归(170+5 = 175/175)
- [ ] 3.7.4 `git commit -m "test(integration): add 5 tcgen05 parse → IR tests (ADR-0016, tests/integration/ptx/)"`

## 4. Phase 2: Golden value + 死代码覆盖单元测试(commit 4,顺序敏感)

**关键路径修正**: `tests/ptx/reference/` → `tests/reference/ptx_tcgen05/`(与现有 `ptx_builtin/` 同级)

### 4.1 创建头文件(已完成,Phase 0.5)

- ✓ `include/ptxsim/instructions/tcgen05.h` 已存在

### 4.2 创建 golden value 文件

- [ ] 4.2.1 创建 `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h`:
   ```cpp
   #pragma once
   // Hand-computed reference values for tcgen05.mma fragment arithmetic.
   // 来源: PTX ISA §9.7.16 规范手算(8x4 f16×f16→f32 fragment)
   //
   // Inputs:
   //   A[8][1] = {1.0f16, 2.0f16, 3.0f16, 4.0f16, 5.0f16, 6.0f16, 7.0f16, 8.0f16}
   //   B[1][4] = {1.0f16, 2.0f16, 3.0f16, 4.0f16}
   //
   // Expected output C[i][j] = A[i][0] * B[0][j], f16→f32 conversion
   //
   // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16
   // 每元素可被 reviewer 手算复算:
   //   C[0][0] = 1*1 = 1.0, C[0][1] = 1*2 = 2.0, C[0][2] = 1*3 = 3.0, C[0][3] = 1*4 = 4.0
   //   C[1][0] = 2*1 = 2.0, C[1][1] = 2*2 = 4.0, ...
   //   C[7][0] = 8*1 = 8.0, C[7][1] = 8*2 = 16.0, C[7][2] = 8*3 = 24.0, C[7][3] = 8*4 = 32.0
   
   #include <array>
   
   namespace ptxsim::reference::tcgen05 {
   constexpr std::array<float, 32> GOLDEN_MMA_F16_F16_F32 = {
       1.0f,  2.0f,  3.0f,  4.0f,    // i=0: 1*[1..4]
       2.0f,  4.0f,  6.0f,  8.0f,    // i=1: 2*[1..4]
       3.0f,  6.0f,  9.0f, 12.0f,    // i=2: 3*[1..4]
       4.0f,  8.0f, 12.0f, 16.0f,    // i=3: 4*[1..4]
       5.0f, 10.0f, 15.0f, 20.0f,
       6.0f, 12.0f, 18.0f, 24.0f,
       7.0f, 14.0f, 21.0f, 28.0f,
       8.0f, 16.0f, 24.0f, 32.0f,
   };
   }  // namespace ptxsim::reference::tcgen05
   ```

### 4.3 创建 dead-code coverage 单元测试(**新文件,非扩张 cluster 测试**)

- [ ] 4.3.1 创建 `tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp`(新文件):
   ```cpp
   // ============================================================================
   // DEAD-CODE COVERAGE TEST — see design.md D4
   //
   // processTcgen05Mma 当前未被 dispatch 路由调用:
   //   - S_TCGEN05_* 在 ptx_op.def:129-136 中显式排除 X-Macro
   //   - InstructionFactory::get_handler(S_TCGEN05_MMA) 返回 nullptr
   //   - ThreadContext::execute_thread_instruction() 走 "No handler found"
   //     路径(thread_context.cpp:142-146)并 set_state(EXIT)
   //
   // 此测试直接调用 processTcgen05Mma 验证 fragment arithmetic 正确性。
   // 当未来 dispatcher change (`fix-tcgen05-handler-dispatch`) 接入后,
   // 同样的测试断言将验证真实运行路径。
   // ============================================================================
   #include <catch_amalgamated.hpp>
   #include "ptxsim/instructions/tcgen05.h"  // ← 必须 include 头文件
   #include "ptxsim/core/thread_context.h"
   #include "ptx_ir/tcgen05_instr.h"
   #include "reference/ptx_tcgen05/tcgen05_mma_golden.h"
   
   using namespace ptxsim;
   using ptxsim::reference::tcgen05::GOLDEN_MMA_F16_F16_F32;
   
   namespace {
   // 设置最小的 thread/warp/CTA context 让 handler 可调用
   // 参考 tests/unit/memory/test_tmem.cpp:18 的初始化模式
   struct TestEnv {
       std::unique_ptr<CTAContext> cta;
       std::unique_ptr<WarpContext> warp;
       std::unique_ptr<ThreadContext> thread;
       TestEnv() {
           cta = std::make_unique<CTAContext>();
           warp = std::make_unique<WarpContext>(cta.get(), /*lane_id=*/0);
           thread = std::make_unique<ThreadContext>(warp.get(), /*tid=*/0);
           cta->init_tmem(256);  // 每个 CTA 32KB TMEM
       }
   };
   }  // namespace
   
   TEST_CASE("tcgen05.mma golden value (dead-code coverage)", "[unit][ptx_ir][tcgen05][mma][golden]") {
       TestEnv env;
       
       // 构造 Tcgen05Instr:mma.kind::f16.cta_group::1 (4 operands)
       Tcgen05Instr instr{
           .op_kind = Tcgen05OpKind::MMA,
           .qualifiers = { /* KIND::F16 */, /* CTA_GROUP::1 */ },
           .operands = { /* d, a, b, c */ }
       };
       
       // 直接调用 handler(独立于 dispatch)
       processTcgen05Mma(env.thread.get(), instr);
       
       // 读取输出寄存器/TMEM 与 golden value 比对
       // 具体实现取决于 handler 的输出位置(TMEM slot 64..95 或 register)
       for (int i = 0; i < 32; ++i) {
           float actual = env.cta->read_tmem_f32(/*slot=*/64 + i);
           REQUIRE(actual == Catch::Approx(GOLDEN_MMA_F16_F16_F32[i]).epsilon(1e-4));
       }
       
       // 加 1 个 explicit 断言(spec 要求每元素可被 reviewer 复算)
       REQUIRE(env.cta->read_tmem_f32(64) == Catch::Approx(1.0f).epsilon(1e-4));
       REQUIRE(env.cta->read_tmem_f32(/*offset for C[7][3]*/) == Catch::Approx(32.0f).epsilon(1e-4));
   }
   ```

### 4.4 CMakeLists 注册

- [ ] 4.4.1 编辑 `tests/unit/CMakeLists.txt`:
   ```cmake
   add_catch_test(unit_ptx_ir_tcgen05_mma_golden
       ptx_ir/test_tcgen05_mma_golden.cpp
   )
   set_tests_properties(unit_ptx_ir_tcgen05_mma_golden PROPERTIES
       LABELS "unit;ptx_ir;tcgen05;mma;golden")
   ```

### 4.5 验证

- [ ] 4.5.1 `cmake --build build` 验证编译
- [ ] 4.5.2 `ctest -L "unit;tcgen05;mma;golden" -V` 验证测试 PASS
- [ ] 4.5.3 `ctest --output-on-failure` 验证零回归(176/176)
- [ ] 4.5.4 `git commit -m "test(unit+reference): add tcgen05.mma f16×f16→f32 golden value + dead-code coverage test (PTX ISA §9.7.16)"`

## 5. Phase 3: E2E kernel(commit 5,3-tier 降级)

**关键修正**: f16 风险 → 3-tier 降级(原 2-tier → 现 3-tier + Phase 3.0 预验证)

### 5.0 Phase 3 预验证(MUST DO)

- [ ] 5.0.1 `cat > /tmp/test_tcgen05_mma_f16.cu <<'EOF'
   __device__ void test_kernel() {
       // 完整 f16 tcgen05.mma 调用,看 nvcc -ptx 是否能产生合法 PTX
       asm volatile("tcgen05.mma.kind::f16.cta_group::1 [%0], [%1], [%2], [%3];"
                    :: "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
   }
   EOF`
- [ ] 5.0.2 `nvcc -ptx -arch=sm_100 /tmp/test_tcgen05_mma_f16.cu -o /tmp/test.ptx 2>&1`
- [ ] 5.0.3 `grep "tcgen05" /tmp/test.ptx` —— 如果看到合法的 tcgen05.mma .f16 指令,**f16 路径可用** → 走 5.2
- [ ] 5.0.4 如果 5.0.3 失败 → 走 5.3(float 降级模式,复用 `test_blackwell_gemm.cu:11` 注释的模式)
- [ ] 5.0.5 结果记录到 commit message

### 5.1 策略选择

- [ ] 5.1.1 `which cuobjdump 2>&1` —— 如果可用 → 走 5.2(优先);如果不可用 → 走 5.3

### 5.2 cuobjdump 路径(优先,f16 可用时)

- [ ] 5.2.1 `cuobjdump -xptx` 从 Cutlass 3.x GEMM 提取真实 PTX
- [ ] 5.2.2 写入 `tests/e2e/kernel/test_tcgen05_mma_gemm.cu`
- [ ] 5.2.3 注册(见 5.4)

### 5.3 手动构造路径(f16 不可用 或 cuobjdump 不可用时)

- [ ] 5.3.1 手动构造 `tcgen05.mma` + `tcgen05.ld` + `tcgen05.st` + `tcgen05.commit` + `tcgen05.wait` 指令序列
- [ ] 5.3.2 必要时 fallback 到 f32(`test_blackwell_gemm.cu:11` 注释的方式)
- [ ] 5.3.3 写入 `tests/e2e/kernel/test_tcgen05_mma_gemm.cu`
- [ ] 5.3.4 头部注释显式说明采用降级路径:
   ```cpp
   // E2E kernel for tcgen05.mma GEMM (ADR-0016)
   //
   // 路径选择(如实记录):
   //   - 默认:cuobjdump 提取的 Cutlass PTX(f16)
   //   - 降级 1:手写 f16 tcgen05.mma 指令序列(nvcc -ptx 验证合法后)
   //   - 降级 2:f32 GEMM(复用 tests/e2e/kernel/test_blackwell_gemm.cu:11 模式)
   //
   // 当前实际采用: <路径>
   // 原因:<原因,如 f16 ANTLR grammar 限制 / cuobjdump 不可用>
   ```

### 5.4 CMakeLists 注册 + 验证

- [ ] 5.4.1 编辑 `tests/e2e/CMakeLists.txt`:
   ```cmake
   add_catch_test(e2e_tcgen05_mma_gemm
       kernel/test_tcgen05_mma_gemm.cu
   )
   set_tests_properties(e2e_tcgen05_mma_gemm PROPERTIES
       LABELS "e2e;kernel;tcgen05;gemm;sm100")
   ```
- [ ] 5.4.2 `cmake --build build` 验证编译
- [ ] 5.4.3 `ctest -L "e2e;tcgen05" -V` 验证 E2E PASS
- [ ] 5.4.4 `ctest --output-on-failure` 验证零回归(177/177)
- [ ] 5.4.5 `git commit -m "test(e2e): add tcgen05.mma GEMM kernel test (ADR-0016, <路径>)"`

## 6. Phase 4: 文档 + Archive(commit 6)

### 6.1 文档同步

- [ ] 6.1.1 编辑 `src/ptxsim/instructions/AGENTS.md`:
   - 更新 tcgen05.cpp 测试覆盖状态
   - 标注: **"5 core handler 单元测试覆盖(dead-code coverage via tcgen05.h) + 5 integration parse + 1 E2E kernel"**
   - **标注**: dispatcher 死代码问题(独立 change `fix-tcgen05-handler-dispatch`)
- [ ] 6.1.2 编辑根 `AGENTS.md` 已知限制表:
   - 标注 **"5 core handler test coverage 100% (dead code, dispatch 未修复)"**

### 6.2 Archive

- [ ] 6.2.1 `ctest --output-on-failure` 最终验证(177+/177+ PASS)
- [ ] 6.2.2 `openspec archive fix-tcgen05-test-coverage-gaps --yes`
- [ ] 6.2.3 `git add openspec/changes/archive/`
- [ ] 6.2.4 `git commit -m "chore(openspec): archive fix-tcgen05-test-coverage-gaps (ADR-0016, Metis 修订)"`

## Final Validation

- [ ] 7.1 `git log --oneline | head -7` 显示 6 atomic commits(原 4 + 新增 Phase 0.5 commit + 任何 split)
- [ ] 7.2 `ctest --output-on-failure` 全量 177+/177 PASS
- [ ] 7.3 `ctest -L "integration;tcgen05;parse" -V` 显示 5 个新测试
- [ ] 7.4 `ctest -L "unit;tcgen05;mma;golden" -V` 显示 1 个新单元测试
- [ ] 7.5 `ctest -L "e2e;tcgen05" -V` 显示 1 个新 E2E
- [ ] 7.6 `./tests/ptx/test_all_ptx.sh` 验证 12+ PTX tcgen05 fixtures 仍 PASS

## Risks Recap(Metis 修订后)

| Risk | Mitigation |
|------|------------|
| **R1**: Parse 测试不驱动 handler | 设计范围(非 handler test);Phase 2 单元测试补足 dead-code coverage |
| **R2**: Golden value 与硬件不一致 | `UNVERIFIED-AGAINST-HARDWARE` 标注;**每元素可被 reviewer 手算复算**;spec 加具体值断言 |
| **R3**: cuobjdump 不可用 | 降级 1:手动 f16 构造(若 nvcc -ptx 验证合法) |
| **R4**: f16 ANTLR grammar 限制 | 降级 2:f32 复用 `test_blackwell_gemm.cu:11` 模式 |
| **R5**: nvcc -ptx 阶段 f16 失败 | Phase 3.0 预验证 5.0 必须先做,失败立即转降级 |
| **R6**: handler 死代码让测试不可信 | `tcgen05.h` 头注释 + 测试文件头注释 **显式标注** dead-code coverage 性质 |
| **R7**: 未来读者误以为 handler 实际运行 | spec.md 加 requirement 标注 + 根 AGENTS.md 限制表明示 |

## Lessons-Learned 应用(预写 lessons 钩子)

如果本 PR 暴露新模式(例如 dispatcher 死代码已成系统性问题),必须添加:
- [ ] 9.1 在 `.opencode/skills/ptx-lessons-learned/SKILL.md` 增加**"X-Macro 排除项的 dead-code 验证步骤"**
- [ ] 9.2 在 `docs/dev-process/lessons-learned.md` 加 ADR-compliant 失败模式
- [ ] 9.3 创建后置 issue `fix-tcgen05-handler-dispatch` 跟踪 dispatch 修复
