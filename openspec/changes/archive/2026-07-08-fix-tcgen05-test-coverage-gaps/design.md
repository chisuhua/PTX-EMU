# Fill tcgen05 5 Core Handler Test Coverage Gaps — Design

> **架构依据**: [ADR-0016](../../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) Accepted
> **前置 change**: `implement-tcgen05-handlers-core` (archived @ `df6dde7`)
> **核心 Metis 修正**: D1 (helpers) / D2 (path) / D3 (f16 风险) / **D4 (handler 死代码 + dead code coverage 策略)**

## Context

`implement-tcgen05-handlers-core` (commit `df6dde7`, archived 2026-07-07) 交付了 5 个 `processTcgen05Xxx` handler:
- `processTcgen05Mma`
- `processTcgen05Ld`
- `processTcgen05St`
- `processTcgen05Commit`
- `processTcgen05Wait`

**关键背景**: 这些 handler 在 `tcgen05.cpp:311-540` 实现,但在代码库中**没有任何 dispatch 路径调用它们**:
- `S_TCGEN05_*` StatementType 枚举在 `ptx_types.h:28-38` 中定义,**有意排除**在 `ptx_op.def` 的 X-Macro 循环之外(`ptx_op.def:129-136` 注释说明)
- `InstructionFactory::initialize()` 只通过 X-Macro 注册 handler,因此 `S_TCGEN05_*` 在 handler_map 中**无条目**
- `ThreadContext::execute_thread_instruction()` 调用 `InstructionFactory::get_handler()`,返回 `nullptr`,进入 `set_state(EXIT)` 路径(`thread_context.cpp:142-146`)
- `grep -rn "processTcgen05" src/ptxsim/ | grep -v tcgen05.cpp` 返回零结果——handler 是 **dead code**

**已知测试缺口**(实情):
- ✅ 12 个 PTX 语法 fixture 已存在(`tests/ptx/tcgen05_*.ptx`,自动 `test_all_ptx.sh` 覆盖)
- ✅ `tests/unit/cluster/test_cluster_tcgen05_integration.cpp` 已存在(测 cluster arrive)
- ✅ `tests/e2e/kernel/test_blackwell_gemm.cu` 已存在(f32 GEMM)
- ❌ 5 个 `processTcgen05*` 函数的**集成 / 单元**测试均**不存在**
- ❌ tcgen05.mma fragment arithmetic **golden value 不存在**
- ❌ handler **头文件声明**不存在
- ❌ f16 专门 `tcgen05.mma` GEMM E2E 测试不存在(现有用 f32 规避 grammar 限制)

## Goals / Non-Goals

**Goals**:
1. 5 integration parse 测试覆盖 `Tcgen05Instr` IR 字段一致性
2. handler 头文件 `tcgen05.h` 创建(forward declaration)
3. 单元测试直接调用 `processTcgen05Mma` 验证 vs golden value(dead code coverage)
4. E2E kernel 覆盖完整 `tcgen05.mma` 执行路径(f32 优先,f16 降级)
5. 文档同步根 AGENTS.md 与 `src/ptxsim/instructions/AGENTS.md`

**Non-Goals**:
- ❌ 不修复 dispatch 死代码(独立 change;需要 `InstructionFactory` 注册 + handler 适配)
- ❌ 不修改 grammar/IR
- ❌ 不实现新 handler(留给 Change-3d `implement-tcgen05-handlers-extended`)
- ❌ 不追求 cycle-accurate 性能对标

## Decisions

### D1: 集成 parse 测试模式 — **直接 ANTLR parse + factory 验证**

**采纳**:
- 使用 ANTLR parser 直接解析 PTX 字符串
- 使用 `makeTcgen05Instr(op_kind, qualifiers, operands, text)`(在 `include/ptx_ir/statement_factory.h:278` 已存在)
- 验证 `std::get<Tcgen05Instr>(stmt.data)` 的字段:`op_kind`, `qualifiers`, `operands`

**❌ 拒绝原假设**:
- 原 D1 提议 `ptxsim::testing::step_warp` + `make_*` helpers——**这些 helpers 完全不存在**
- `include/ptxsim/testing/instruction_helpers.h`(651 行)只有通用 helper(`make_mov`, `make_atom_*` 等),**无 `make_tcgen05_*` 或 `make_bra_pred` 风格的 tcgen05 构造器**
- 即使 helpers 存在,parse → IR 测试**不应驱动 warp 执行**(那是 dispatcher 的工作)

**依据**: 类似 `tests/unit/parser/test_extern_function.cpp:8` 注释——"完整 ANTLR 解析测试在 integration 层(避免触发 pre-existing parser LSP 错误)"

```cpp
// 示例: mma parse 测试(直接 ANTLR)
TEST_CASE("tcgen05.mma parse → IR", "[integration][ptx][tcgen05][parse]") {
    const std::string ptx =
        "tcgen05.mma.kind::f16.cta_group::1 d, a, b, c;";
    ParserHelper helper;
    auto stmts = helper.parse_string(ptx);
    REQUIRE(stmts.size() == 1);
    auto& instr = std::get<Tcgen05Instr>(stmts[0].data);
    REQUIRE(instr.op_kind == Tcgen05OpKind::MMA);
    REQUIRE(instr.qualifiers.size() >= 2);
}
```

### D2: Golden value 路径 — `tests/reference/ptx_tcgen05/`

**采纳**:
- 路径: `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h`
- **依据**: 现有 `tests/reference/ptx_builtin/` 已是该 reference data 模式(`tests/reference/` 是 reference data 根,非 `tests/ptx/`)

**内容**:
```cpp
namespace ptxsim::reference::tcgen05 {
constexpr std::array<float, 32> GOLDEN_MMA_F16_F16_F32 = {
    // C[i][j] = A[i][0] * B[0][j]   (8x4 fragment, f16→f32)
    // A = [1,2,3,4,5,6,7,8] (as f16), B = [1,2,3,4] (as f16)
    1.0f,  2.0f,  3.0f,  4.0f,    // i=0: 1*[1..4]
    2.0f,  4.0f,  6.0f,  8.0f,    // i=1: 2*[1..4]
    3.0f,  6.0f,  9.0f, 12.0f,
    4.0f,  8.0f, 12.0f, 16.0f,
    5.0f, 10.0f, 15.0f, 20.0f,
    6.0f, 12.0f, 18.0f, 24.0f,
    7.0f, 14.0f, 21.0f, 28.0f,
    8.0f, 16.0f, 24.0f, 32.0f,
};
}  // namespace ptxsim::reference::tcgen05
// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16
```

**附加约束**:
- 每个值必须**可被 reviewer 手算复算**(避免无效值通过 spec)
- 单元测试断言具体值:`REQUIRE(GOLDEN_MMA_F16_F16_F32[0] == Approx(1.0f))` 等等

### D3: E2E kernel f16 风险 + 降级路径

**采纳 (3-tier fallback)**:
1. **优先**: `cuobjdump -xptx` 提取 Cutlass 3.x GEMM 真实 Blackwell PTX → 编译为 `.cu`
2. **降级**: `nvcc -ptx -arch=sm_100` 验证能否生成合法 `tcgen05.mma.kind::f16` PTX;若不行:
3. **深度降级**: 复用 `tests/e2e/kernel/test_blackwell_gemm.cu:11` 的 f32 模式(注释说明 f16 受 ANTLR grammar 限制)

**❌ 拒绝原假设**:
- "强制 f16 GEMM" —— 与现有 `test_blackwell_gemm.cu:11` 注释直接冲突
- ANTLR grammar 对 f16 fragment 加载的支持存在限制(`tests/e2e/kernel/test_blackwell_gemm.cu:11` 记录)

**预验证步骤 (Phase 3.0)**:
```bash
# 必做 — 在写 .cu 前先验证
echo '...tcgen05.mma.kind::f16...' | nvcc -ptx -arch=sm_100 -o /tmp/test.ptx -
```

### D4: handler dead code coverage 策略【新决策】

**采纳**: Phase 2 单元测试**显式标注**为 "dead code coverage":

1. 在 `include/ptxsim/instructions/tcgen05.h` 中声明:
   ```cpp
   namespace ptxsim {
   // 警告: 这些函数当前未被 dispatch 路径调用(S_TCGEN05_* 在 ptx_op.def 中显式排除)
   // 本头文件仅为 dead-code coverage 测试而存在
   void processTcgen05Mma(ThreadContext*, const Tcgen05Instr&);
   void processTcgen05Ld(ThreadContext*, const Tcgen05Instr&);
   void processTcgen05St(ThreadContext*, const Tcgen05Instr&);
   void processTcgen05Commit(ThreadContext*, const Tcgen05Instr&);
   void processTcgen05Wait(ThreadContext*, const Tcgen05Instr&);
   }  // namespace ptxsim
   ```

2. 单元测试 `tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp` 头部必须**显式标注**:
   ```cpp
   // DEAD-CODE COVERAGE TEST
   // processTcgen05Mma 当前未被 dispatch 路由调用(S_TCGEN05_* 在 ptx_op.def 中排除 X-Macro)
   // 此测试直接调用 handler 验证 fragment arithmetic —— 是为未来 dispatcher 集成做准备的占位测试
   ```

3. 测试本身有效但**不反映真实运行路径**——spec 必须标注此限制

**为什么不在本 PR 修复 dispatch**:
- 需要创建 `Tcgen05Handler` 类继承 `InstructionHandler`(或适配现有接口)
- 需要在 `InstructionFactory::initialize()` 中显式注册(绕过 X-Macro)
- 影响 E2E 现有 GEMM 测试(可能从 skip 转为真执行) → 需要独立 change + 完整回归测试
- 超出本 PR "纯测试补充" 范围

**后续 action item**: 创建独立 change `fix-tcgen05-handler-dispatch` 修复此问题

## Risks / Trade-offs

| 风险 | 等级 | 缓解 |
|------|------|------|
| **R1**: Parse 测试因路径错误导致 CMake 注册失败 | 🟢 低 | 已修正:`tests/integration/ptx/` 已验证存在 |
| **R2**: Phase 2 调用 `processTcgen05Mma` 因函数未声明而无法编译 | 🟢 低 | 已添加头文件 `tcgen05.h`(D4) |
| **R3**: Phase 2 单元测试因 TMEM 设置复杂而失败 | 🟡 中 | 复用 `tests/unit/memory/test_tmem.cpp:18` 的 `Tmem` 初始化模式 |
| **R4**: E2E `test_tcgen05_mma_gemm.cu` 因 f16 ANTLR 限制编译失败 | 🟡 中 | D3 三层降级(f16→cuobjdump→f32 复用),Phase 3.0 预验证 |
| **R5**: 现有 170+ 测试因 Phase 改动产生回归 | 🟢 低 | 严格遵守 tasks.md 0.2 基线 + 6.2 全量验证 |
| **R6**: Golden value 注释 "UNVERIFIED-AGAINST-HARDWARE" 误导未来 reviewer | 🟡 中 | D2 约束"每元素可被 reviewer 手算复算"+ spec 加具体值断言 |
| **R7**: handler 死代码让 Phase 3 E2E 永远过/不过 | 🟢 低 | E2E 测试通过 `cudaLaunchKernel` 间接触发 dispatcher,即使 handler 未注册也会得到 "no handler found" 路径;需 Phase 3 预验证 PTX 是否合法 |
| **R8**: spec.md grep-only 验证场景让任何实现都过 | 🟢 低 | 改为具体值断言(见 spec 修订) |

## Migration Plan

### Phase 1: 5 集成 parse 测试(commit 1)

5 个 `.cpp` 文件 → `tests/integration/ptx/` + `tests/integration/CMakeLists.txt` 注册

**实现模式**:
```cpp
// 每个文件结构相同
#include <ptx_ir/parser_helper.h>
#include <ptx_ir/statement_factory.h>
TEST_CASE("tcgen05.<op> parse → IR", "[integration][ptx][tcgen05][parse]") {
    const std::string ptx = "...";
    auto stmts = parse_ptx_string(ptx);
    REQUIRE(stmts.size() == 1);
    auto& instr = std::get<Tcgen05Instr>(stmts[0].data);
    // verify op_kind, qualifiers, operands
}
```

### Phase 2: handler 头文件 + Golden value(commit 2)

1. `include/ptxsim/instructions/tcgen05.h`(顺序:先创建)
2. `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h`
3. `tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp`(头标注 `DEAD-CODE COVERAGE TEST`)

### Phase 3: E2E kernel(commit 3,可降级)

1. Phase 3.0 预验证:`nvcc -ptx -arch=sm_100` 是否生成合法 f16 tcgen05.mma PTX
2. 按 D3 三层降级选择路径
3. `.cu` 文件 → `tests/e2e/kernel/` + `tests/e2e/CMakeLists.txt` 注册

### Phase 4: 文档 + Archive(commit 4)

1. 更新 `src/ptxsim/instructions/AGENTS.md` + 根 `AGENTS.md`
2. 标注 **"5 handler 单元测试覆盖 + dispatch 死代码独立 change 处理"**
3. Archive

## Open Questions

| # | 问题 | 解决时机 |
|---|------|----------|
| Q1 | `cuobjdump` 是否可用? | Phase 3 启动时验证 |
| Q2 | `nvcc -ptx` 是否生成合法 f16 tcgen05.mma PTX? | Phase 3.0 必做 |
| Q3 | 修复 dispatch 的独立 change 谁来做? | 后续 cycle(与 Change-3d 同步启动) |
| Q4 | golden value 数值是否符合 NVIDIA 实际硬件? | UNVERIFIED 标注;后续如有 hardware 验证可校准 |

## Files Created / Modified(预估)

```
include/
  ptxsim/instructions/tcgen05.h                                    [NEW]

tests/integration/ptx/                                              [EXISTING DIR]
  test_tcgen05_mma_parse.cpp                                        [NEW]
  test_tcgen05_ld_parse.cpp                                         [NEW]
  test_tcgen05_st_parse.cpp                                         [NEW]
  test_tcgen05_commit_parse.cpp                                     [NEW]
  test_tcgen05_wait_parse.cpp                                       [NEW]

tests/unit/ptx_ir/                                                  [EXISTING DIR]
  test_tcgen05_mma_golden.cpp                                       [NEW]

tests/reference/ptx_tcgen05/                                        [NEW DIR — but consistent with existing tests/reference/]
  tcgen05_mma_golden.h                                              [NEW]

tests/e2e/kernel/                                                   [EXISTING DIR]
  test_tcgen05_mma_gemm.cu                                          [NEW]

tests/integration/CMakeLists.txt                                    [+25]
tests/e2e/CMakeLists.txt                                            [+10]
tests/unit/CMakeLists.txt                                           [+10]
src/ptxsim/instructions/AGENTS.md                                   [+5]
AGENTS.md (root)                                                    [+5]
```
