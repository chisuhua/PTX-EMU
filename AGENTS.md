# PTX-EMU Agent Instructions

> **PTX 模拟器**: C++20/CUDA PTX 模拟器，ANTLR4 解析 PTX，fake libcudart.so 拦截 CUDA runtime
> **C++ 标准**: C++20（含 CUDA 代码）
> **ANTLR 版本**: 4.11.1（antlr-4.11.1-complete.jar）
> **Generated**: 2026-07-03 | **Commit**: 4b9d6e1 | **Branch**: main

---

## 🛑 OpenSpec 流程 + 经验沉淀（最高优先级）

**任何 OpenSpec change 的 propose/apply/archive 阶段都强制应用项目经验沉淀**。

### 必读（OpenSpec 流程开始前）

- **Skill**: [`.opencode/skills/ptx-lessons-learned/SKILL.md`](.opencode/skills/ptx-lessons-learned/SKILL.md) — 16 个核心经验 + 4 个 checklist + 失败模式速查表
- **完整文档**: [`docs/dev-process/lessons-learned.md`](docs/dev-process/lessons-learned.md) — 具体案例 + 代码片段 + 长篇解释
- **互补**: [`docs/dev-process/debugging-strategy.md`](docs/dev-process/debugging-strategy.md)（问题分类与快速验证）

### 强制集成点

| OpenSpec 阶段 | 必查 | 来源 |
|-------------|------|------|
| **propose** | Design-Time Checklist（4 项）+ Proposal 模板增强 | `.opencode/skills/openspec-propose/SKILL.md` |
| **apply** | 基线 worktree + Checklist A（迁移）+ D（commit 前）+ 失败处理纪律 | `.opencode/skills/openspec-apply-change/SKILL.md` |
| **archive** | **强制 Prompt 询问生成 postmortem**（用户必须明确选择） | `.opencode/skills/openspec-archive-change/SKILL.md` |
| **adr-compliance-check** | Cross-Check 5 个 lessons-learned 失败模式（A-E） | `.opencode/skills/adr-compliance-check/SKILL.md` |

### 4 条最常犯的错误（详见 ptx-lessons-learned）

1. **跨模块间接状态翻译**：迁移函数时漏掉看似冗余的 `set_state(BAR_SYNC)`，因为下一模块的 `sync_to_warp_state()` 才把它翻译为 `is_blocked = true`。**必须行级 Diff，不只比对主要逻辑**。
2. **递归锁死锁**：持锁方法调用同锁的其他 public 方法 = `std::mutex` 死锁。**互斥量需要"使用同一锁的所有代码路径"做集中审计**。
3. **复杂迁移必须分 Phase commit**：每个 Phase 独立可回退。**任何已有测试回归 → 立即 revert 该 Phase，不混入后续 commit**。
4. **基线 worktree 是最低成本保险**：重大重构前 1 分钟建立，节省数小时争论"这个失败是基线的还是我的"。

> **沉淀元规则**: 新发现的 bug 模式必须立即写入 `ptx-lessons-learned` + `lessons-learned.md` + 相关 ADR postmortem。这是防止"经验随归档而消失"的强制钩子。

---

## 🛑 PTX 语法修改流程（最高优先级）

**当检测到 ANTLR 解析错误时，必须先运行测试分类，再行动。**

### 错误分类（先运行测试！）

```bash
# 1. 运行测试获取错误
cd build && ctest -R <test_name> -V 2>&1 | tail -50

# 2. 分类错误
echo <输出> | grep -E "missing|mismatched|no viable|extraneous|ANTLR"
# → 有输出 = ANTLR 解析错误 → 走语法修复流程
# → 无输出 = 运行时错误   → 走 ptx-debug 流程
```

### 语法修复检查清单

```
□ 1. 加载技能：.opencode/skills/ptx-grammar-modification/SKILL.md
□ 2. 阅读 docs/ptx/ 对应章节
□ 3. 运行基线：./tests/ptx/test_all_ptx.sh（不是 ctest！）
□ 4. cuobjdump -xptx 提取真实 PTX → 复制到 tests/ptx/
□ 5. 修改 .g4 → cmake --build build --target GenerateParser
□ 6. ./tests/ptx/test_all_ptx.sh 全部通过才能交付
```

### ❌ 禁止

- 用 `ctest` 代替 `./tests/ptx/test_all_ptx.sh`
- 未读 docs/ptx/ 就改语法
- 未加测试用例就修语法
- 测试未全通过就标完成
- 手动编辑 `build/antlr4_generated_src/` 中的生成文件

---

## 🎯 技能触发

### 全局技能 (~/.config/opencode/skills/)

| 触发关键词 | 技能 |
|-----------|------|
| `segfault`, `SIGSEGV`, `core dumped`, `死锁`, `内存泄漏` | cpp-debug |
| `CMakeLists.txt`, `link error`, `undefined reference` | cmake-manage |
| `模块依赖`, `架构图` | cpp-architecture |
| `现代 C++`, `智能指针` | cpp-modernize |
| CUDA/kernel/nsys/ncu/cuda-gdb | cuda-ptx |

### 项目技能 (.opencode/skills/)

| 触发场景 | 技能文件 |
|---------|---------|
| ANTLR 解析错误 | `.opencode/skills/ptx-grammar-modification/SKILL.md` |
| 运行时错误/SegFault | `.opencode/skills/ptx-debug/SKILL.md` |
| 屏障机制问题 | `.opencode/skills/ptx-barrier-mechanism/SKILL.md` |
| 指令执行/PC 问题 | `.opencode/skills/ptx-instruction-pipeline/SKILL.md` |
| PTX 加载慢/序列化 | `.opencode/skills/ptxir-serialization/SKILL.md` |
| 咨询 Oracle | `.opencode/skills/oracle-prompting/SKILL.md` |
| 测试回归定位 | `.opencode/skills/regression-bisect/SKILL.md` |
| 状态值异常审计 | `.opencode/skills/state-modification-audit/SKILL.md` |
| 生成 PTX 测试 | `.opencode/skills/three-mode-testing/SKILL.md` |
| **项目经验沉淀**（强制）| **`.opencode/skills/ptx-lessons-learned/SKILL.md`** |

### 技能调用关系

```
ptx-debug (入口)
  ├─ regression-bisect (测试回归 → 找 root cause)
  │   ├─ state-modification-audit (值被覆盖 → 交叉引用)
  │   └─ oracle-prompting (咨询 Oracle → 防幻觉)
  ├─ ptx-instruction-pipeline (PC/ExecPipe 问题)
  │   └─ ptx-barrier-mechanism (屏障问题)
  ├─ ptx-grammar-modification (ANTLR 解析错误)
  └─ cpp-debug (C++ 崩溃/内存)
```

### ⚠️ 测试失败决策树

```
用户："修复 test_X"
    ↓
先运行测试，看错误输出
    ↓
ANTLR 解析错误？──→ ptx-grammar-modification
SegFault/崩溃？   ──→ ptx-debug + cpp-debug
逻辑错误/断言？   ──→ ptx-debug
编译/链接错误？   ──→ cmake-manage + cpp-debug
```

---

## 构建

```bash
# 1. 设置环境（必须！设置 CUDA_PATH、CLASSPATH、LD_LIBRARY_PATH）
. env.sh

# 2. 配置 + 构建（Release）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# 快捷方式
./build.sh          # Debug 模式，自动 . env.sh

# 构建特定目标
cmake --build build --target cudart     # fake libcudart.so
cmake --build build --target ptxsim     # PTX 模拟引擎
cmake --build build --target ptx_parser # PTX 解析器
cmake --build build --target ptxir      # PTXIR 序列化库（ptxir_writer + ptxir_reader）

# 重生成 ANTLR 解析器（改 .g4 后）
cmake --build build --target GenerateParser
```

**依赖**: CMake ≥ 3.15, CUDA Toolkit, GCC, Java, ccache（可选自动启用）

**输出目录**: `build/bin/` (可执行文件), `build/lib/` (共享库)
`libcudart.so` 自动软链到项目根目录 `lib/`

---

## 测试

```bash
# 全部测试
cd build && ctest

# 按标签分组
ctest -L mini              # Mini 基础测试
ctest -L "unit;barrier"    # 类型一：屏障相关单元测试
ctest -L "integration"     # 类型二：所有指令序列集成测试
ctest -L "e2e"             # 类型三：所有 CUDA Kernel E2E 测试

# 单个测试（verbose）
ctest -R unit_barrier_module -V

# 运行单个 benchmark（项目根目录）
make -C build RAY

# PTX 语法全量测试（非 ctest！）
./tests/ptx/test_all_ptx.sh
```

**测试框架**: Catch2（`tests/catch_amalgamated.hpp`）
**CUDA 测试编译**: `-keep` 保留中间 PTX，`sm_100` 虚拟架构

---

## TDD 开发流程

**强制执行**: 所有功能实现和缺陷修复必须遵循 TDD 三阶段流程。

### 三阶段流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. 测试先行  │ → │  2. 实现代码  │ → │  3. 验证     │
│  Write Test │    │  Implement  │    │  Sanity     │
└─────────────┘    └─────────────┘    └─────────────┘
```

| 阶段 | 操作 | 验证方式 |
|------|------|---------|
| **1. 测试先行** | 编写失败的测试用例 | `./scripts/sanity.sh --quick` 确认失败 |
| **2. 实现代码** | 编写通过测试的实现 | `./scripts/sanity.sh --quick` 确认通过 |
| **3. 验证** | 运行完整 sanity 检查 | `./scripts/sanity.sh` 无回归 |

### Sanity 脚本用法

```bash
# 完整 sanity 检查（推荐开发时使用）
./scripts/sanity.sh

# 快速检查（仅关键 bug 修复）
./scripts/sanity.sh --quick

# 仅 PTX 语法测试
./scripts/sanity.sh --ptx

# 详细输出
./scripts/sanity.sh --verbose

# 运行特定标签
cd build && ctest -L "exec_mask|simt_entry" -V
```

### 测试标签速查

> **命名约定**：ctest 目标名带类型前缀 `unit_` / `integration_` / `e2e_`（commit `ab55e06`），避免 ctest 命名冲突。完整标签为 `unit;<subject>` 或 `integration;<subject>`。

| 标签 | 覆盖范围 | ctest 目标示例 |
|------|---------|-------------|
| `unit;exec_mask` / `unit;simt_entry` | BUG-001 exec_mask 恢复 / BUG-002 SIMT stack exit | `unit_exec_mask`, `unit_simt_stack_entry` |
| `unit;active_mask` | ISSUE-004 active_mask 一致性 | `unit_active_mask_consistency` |
| `unit;barrier` / `unit;wbar` | 屏障同步、reconvergence | `unit_barrier_module`, `unit_warp_barrier`, `unit_barrier_reconvergence` |
| `unit;ptx;integer/float/bitwise/cvt/ld_st/cvta` | PTX 指令 | `unit_ptx_integer`, `unit_ptx_float`, `unit_ptx_bitwise`, `unit_ptx_cvt`, `unit_ptx_ld_st`, `unit_ptx_cvta` |
| `unit;memory` | 内存分配、边界检查 | `unit_memory_bounds`, `unit_memory_manager_legacy` |
| `integration;barrier` / `integration;wbar` | 屏障指令序列 | `integration_barrier_module`, `integration_warp_barrier` |
| `integration;divergence` | 分歧与 reconvergence | `integration_divergence_sync_standalone`, `integration_divergence_sync_convergence`, `integration_nested_divergence`, `integration_post_barrier_divergence` |
| `integration;simt` / `integration;sync` / `integration;pc` | SIMT/同步/PC 指令序列 | `integration_simt`, `integration_sync_mechanism`, `integration_pc_management` |
| `e2e;barrier` / `e2e;cfg` | 完整 kernel 端到端 | `e2e_barrier_warp_sync`, `e2e_test3_cfg_full` |

### 测试分类规范

**所有测试按目录物理分类到 `tests/unit/`、`tests/integration/`、`tests/e2e/` 三个子目录，对应下面三种类型。从不同层次验证功能：**

| 类型 | 目录 | 标签前缀 | 测试目标 |
|------|------|---------|---------|
| 类型一：直接单元测试 | `tests/unit/` | `unit;...` | 核心数据结构/算法 |
| 类型二：指令序列集成测试 | `tests/integration/` | `integration;...` | 指令执行流程 |
| 类型三：CUDA Kernel E2E 测试 | `tests/e2e/` | `e2e;...` | 完整 kernel 端到端 |

> **历史变更**：2026-06 起的三类测试目录重构（commit `ab55e06`）将原本混在一起的 `.cpp` 文件按类型物理分类到三个子目录；同时为 `add_catch_test` 目标统一加上了 `unit_` / `integration_` / `e2e_` 前缀以避免 ctest 命名冲突。原 `tests/three_mode_testing/` 下的 E2E 旧实现已迁移至 `tests/integration/` 与 `tests/e2e/` 的对应子目录。
>
> **`tests/archive/` 状态（2026-07 更新）**：作为未来归档旧测试的占位目录（`.gitkeep`），当前无文件。相关历史归档清理通过 commits `88e1526`（多文件删除）和 `c86d0ea`（integration 归档）完成。详见 [`tests/archive/.gitkeep`](tests/archive/.gitkeep)。

#### 类型一：直接单元测试（Direct Unit Test）

直接测试核心数据结构和算法，不涉及执行流程。源文件位于 `tests/unit/`。

**特征**：
- 直接实例化 `WarpBarrier`、`CTABarrier`、`BarrierModule` 等类
- 调用类的方法验证行为
- 无需 PTX 解析或指令执行

**示例**（`tests/unit/barrier/test_barrier_module.cpp`，ctest 名 `unit_barrier_module`）：
```cpp
#include "ptxsim/barrier/warp_barrier.h"
using namespace ptxsim;
TEST_CASE("WarpBarrier initialization", "[barrier][warp_barrier]") {
    WarpBarrier wb;
    wb.init(0xFFFF0000, 21, 20);
    REQUIRE(wb.is_initialized() == true);
    REQUIRE(wb.get_participation_mask() == 0xFFFF0000);
    REQUIRE(wb.get_reconvergence_pc() == 21);
}
```

**适用场景**：数据结构逻辑、状态机、工具函数

#### 类型二：指令序列集成测试（Instruction Sequence Test）

使用 `ptxsim/testing/` 提供的测试辅助函数构建指令序列并驱动执行。源文件位于 `tests/integration/`。**禁止在测试代码里重新实现 `step_warp`、`make_*`、`setup_pred` 等基础函数** —— 一律复用 `ptxsim::testing` 命名空间下的工具（详见 `include/ptxsim/testing/`）。

**核心原则**（以 `tests/integration/divergence/test_divergence_sync_convergence.cpp` 头部注释为准）：

1. **所有 PC 变化通过 `execute_warp_instruction` → 指令执行管道驱动** —— 测试不直接改写 PC。
2. **路径由 `ptxsim::testing::step_warp()` 完全模拟调度器决策**（算法见 `sm_context.cpp:250-264`），返回本步执行的 PC。
3. **测试不干预调度器选择，只验证其选择是否正确** —— `step_warp` 内部封装了"取 lane 分组 → 找最低非阻塞 PC → 执行 → 循环至 reconvergence"全过程。
4. **predicate 通过 `RegisterBankManager` 设置 per-lane 值** —— 使用 `ptxsim::testing::setup_pred(w, mask)` / `set_predicate_per_lane(w, lane, val)`。
5. **分歧由 `handle_branch` 自动处理** —— 测试代码不直接 push/pop SIMT stack，只读取 `warp->get_simt_stack().depth()` 等结果状态。

**分类规则**：如测试需要手动设置 SIMT stack/PC 状态（如两级分歧 back-edge），则该测试**不是指令序列集成测试**，应归入 `tests/unit/`（单元测试允许直接 `execute_warp_instruction`）。

**特征**：
- 头文件：`ptxsim/testing/scheduler_utils.h`、`ptxsim/testing/instruction_helpers.h`、`ptxsim/testing/predicates.h`（按需包含）
- 指令构造：`ptxsim::testing::make_mov()` / `make_mov_imm()` / `make_bra()` / `make_bra_pred()` / `make_bar_sync()` / `make_nop()` / `make_ret()` 等
- 路径推进：`step_warp(warp, stmts) → int pc`（单步驱动调度器 + 执行管道）
- 谓词设置：`setup_pred(warp, lane_mask)` 或 `set_predicate_per_lane(warp, lane_id, value)`
- 验证维度：执行后 PC（`get_thread_pc(lane)`）、`active_mask`、`get_lanes_by_pc()` 分组、SIMT stack 深度、寄存器值

**示例**（`tests/integration/divergence/test_divergence_sync_convergence.cpp`，ctest 名 `integration_divergence_sync_convergence`）：
```cpp
#include "ptx_ir/statement_factory.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/testing/scheduler_utils.h"   // step_warp
#include "ptxsim/testing/instruction_helpers.h" // make_bra_pred / make_nop / ...
#include "ptxsim/testing/predicates.h"        // setup_pred

using ptxsim::testing::step_warp;
using ptxsim::testing::setup_pred;

// 1) 构建指令序列：35 条含一个 @%p1 bra + 汇聚点
static std::vector<StatementContext> build_instrs(std::map<std::string,int>& l2pc) {
    std::vector<StatementContext> v(NUM_STMTS);
    for (auto& s : v) s = ptxsim::testing::make_nop();
    v[BRANCH_PC] = ptxsim::testing::make_bra_pred("L__BB0_4", "%p1", false, CONV_PC);
    v[BRA_UNI_PC] = ptxsim::testing::make_bra("L__BB0_3");
    v[27]         = ptxsim::testing::make_ret();
    l2pc["L__BB0_4"] = PATH_B_TARGET;
    l2pc["L__BB0_3"] = CONV_PC;
    return v;
}

// 2) 通过 setup_pred 给分歧谓词设置 per-lane 值（low 16 lanes 走 Path B）
auto v = build_instrs(l2pc);
SMContext sm(4, 128, 4096, 0);
WarpContext* w = setup(sm, v, l2pc);
setup_pred(w, 0x0000FFFFu);  // 原则 4

// 3) step_warp 驱动执行：调度器选择 + 指令执行全自动
CHECK(step_warp(w, v) == BRANCH_PC);             // 抵达分歧点
REQUIRE(w->get_simt_stack().depth() == 1);        // 验证 handle_branch 起效（原则 5）
// 4) 只验证调度器在汇聚点的切换是否正确，不干预其决策（原则 3）
while (step_warp(w, v) != CONV_PC) { /* drain Path A */ }
CHECK(w->get_warp_state().threads[16].is_blocked); // Path A 阻塞在汇聚点
CHECK(step_warp(w, v) == PATH_B_TARGET);           // 调度器切至 Path B
```

**适用场景**：调度器选择验证、PC 推进、分歧/汇聚、SIMT stack 边界条件、barrier 后控制流、`active_mask` 一致性

**反模式**：
- ❌ 在测试里手写 `step_warp` 循环逻辑（应使用 `ptxsim::testing::step_warp`）
- ❌ 直接调用 `warp->execute_warp_instruction()` 绕过调度器（应通过 `step_warp` 间接调用）
- ❌ 手写 `setp` + 寄存器赋值来构造谓词（应使用 `setup_pred` / `set_predicate_per_lane`）
- ❌ 直接 `push_simt_stack()` / `pop_simt_stack()` 干预分歧（应观察 `handle_branch` 后的状态；如必须手动 push，则该测试属于 `unit/` 而非 `integration/`）

#### 类型三：CUDA Kernel E2E 测试（End-to-End Test）

编译真实 CUDA kernel（`.cu` 源文件），提取 PTX，通过模拟器完整执行。源文件位于 `tests/e2e/kernel/`。

**特征**：
- 使用 `nvcc -ptx` 编译 CUDA 源文件（`-keep --no-compress` 保留中间 PTX）
- `cudaLaunchKernel()` 触发完整执行流程（通过 fake `libcudart.so` 拦截）
- 验证内存输出或函数行为

**示例**（`tests/e2e/kernel/test_barrier_warp_sync.cu`，ctest 名 `e2e_barrier_warp_sync`）：
```cuda
#include "ptxsim/execution_types.h"
#include "ptxsim/sm_context.h"
__global__ void kernel_barrier_sync(int* output, int num_threads) {
    __shared__ int shared_data[32];
    int tid = threadIdx.x;
    shared_data[tid] = tid + 1;
    __syncthreads();    // 由模拟器翻译为 S_BAR / S_BAR_WARP_SYNC
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < num_threads; i++) sum += shared_data[i];
        output[0] = sum;  // host 端读取验证
    }
}
```

**适用场景**：完整功能验证、PTX 解析集成、运行时行为

---

**测试覆盖率检查清单**：

| 新增功能/修复 Bug | 类型一（单元） | 类型二（指令序列） | 类型三（E2E） |
|-----------------|--------------|------------------|--------------|
| 新数据结构 | ✅ 必须 | ✅ 推荐 | ✅ 推荐 |
| 新指令实现 | ✅ 必须 | ✅ 必须 | ✅ 必须 |
| Bug 修复 | ✅ 必须 | ✅ 必须 | ✅ 必须 |
| 性能优化 | ✅ 可选 | ✅ 推荐 | ✅ 推荐 |

**CMake 添加规则**：

> **命名约束**：ctest 目标名必须带类型前缀（`unit_` / `integration_` / `e2e_`），且唯一。新增测试时遵循该约定。

```cmake
# 类型一：直接单元测试（tests/unit/<area>/CMakeLists.txt）
add_catch_test(unit_barrier_module
    barrier/test_barrier_module.cpp
)
set_tests_properties(unit_barrier_module PROPERTIES LABELS "unit;barrier")

# 类型二：指令序列集成测试（tests/integration/<area>/CMakeLists.txt）
add_catch_test(integration_simt_stack_entry
    simt/test_simt_stack_entry_integrated.cpp
)
set_tests_properties(integration_simt_stack_entry PROPERTIES LABELS "integration;simt_stack")

# 类型三：CUDA Kernel E2E 测试（tests/e2e/kernel/CMakeLists.txt）
add_catch_test(e2e_barrier_warp_sync
    kernel/test_barrier_warp_sync.cu
)
set_tests_properties(e2e_barrier_warp_sync PROPERTIES LABELS "e2e;barrier")
```

### ❌ 禁止

- 未写测试就实现功能
- 测试未失败就实现（Red 阶段必须有失败）
- 提交前不运行 `./scripts/sanity.sh`
- 用 `ctest` 代替 `./tests/ptx/test_all_ptx.sh` 做 PTX 语法测试

---

## 代码风格

- **格式化**: clang-format（`.clang-format`, BasedOnStyle=LLVM, IndentWidth=4, ColumnLimit=80）
- **命名**: 文件 snake_case | 函数 camelCase | 类 PascalCase | 变量 camelCase
- **PTX 指令**: 全小写（`mov`, `add`, `ld`, `st`）
- **头文件**: `#ifndef`/`#define`/`#endif` 守卫
- **提交前**: `clang-format -i <file>`

---

## 架构速览

### 执行层次

```
GPUContext (全局内存, SM 列表)
  └── SMContext (资源, warp 调度器, 屏障)
        └── CTAContext (warps, shared/local 内存)
              └── WarpContext (32 线程, 活跃掩码, 分歧)
                    └── ThreadContext (寄存器, 条件码, PC)
```

### 执行流

1. `__cudaRegisterFatBinary` → `cuobjdump` 提取 PTX → ANTLR4 解析 → 填充 PtxContext
2. `cudaLaunchKernel` → PtxInterpreter → 构建符号表 → 提交 KernelLaunchRequest
3. GPUContext → SMContext → CTAContext → WarpContext 分发
4. ThreadContext::execute_thread_instruction() → InstructionFactory 分发

### 核心目录

| 目录 | 内容 |
|------|------|
| `src/ptx_ir/` | IR 类型、操作数/语句上下文 |
| `src/ptxir/` | PTXIR 序列化库（ptxir_writer + ptxir_reader） |
| `src/ptx_parser/` | PTXVisitor, CFGBuilder |
| `src/ptxsim/core/` | GPU/SM/CTA/Warp/Thread 上下文 |
| `src/ptxsim/instructions/` | PTX 指令实现 |
| `src/ptxsim/memory/` | 内存抽象 (SimpleMemory, SharedMemoryManager) |
| `src/ptxsim/register/` | 寄存器抽象 |
| `src/cudart/` | fake libcudart.so (CUDA runtime 替代) |
| `src/grammar/` | ptxLexer.g4, ptxParser.g4 |
| `configs/` | GPU 架构 JSON + 调试 INI |
| `include/` | 公共头文件 |

---

## 添加 PTX 指令

1. 更新 `include/ptx_ir/ptx_op.def`（X-Macro 模式）
2. 在 `src/ptxsim/instructions/` 实现 handler
3. 如需，更新 `src/grammar/ptxParser.g4`
4. `cmake --build build --target GenerateParser`
5. 添加测试

### X-Macro 模式

```cpp
#define X(name, ...) process_##name(__VA_ARGS__);
#include "ptx_op.def"
#undef X
```

---

## 调试与日志

```bash
# 使用调试脚本选择配置
./scripts/debug-run.sh {配置名} ./build/bin/程序

# 可用配置：debug, verbose, memory, instruction, perf
```

配置文件：`configs/config.ini` 或 `configs/debug_config.ini`
组件: emu, exec, mem, reg, thread, func
级别: trace, debug, info, warning, error, fatal

详见 `docs/debugging_guide.md`

---

## 已知限制

| 类别 | 状态 |
|------|------|
| WMMA/Tensor Core | Blackwell `tcgen05.*` 已实现（`feat/implement-tcgen05-handlers` change, Phase 1-3）。pre-Blackwell 永久抛 `UnsupportedInstructionException` + `PTX_ERROR_EMU`（c5 Fix #1 + [ADR-0016](docs/adr/0016-blackwell-only-tcgen05.md)）。 |
| Atomic 操作 | 无真正原子性 (stub) |
| Hopper (sm_90+) cluster | cluster 抽象未实现 — 实施中（[ADR-0016](docs/adr/0016-blackwell-only-tcgen05.md) Phase 0.3） |
| Event/Stream API | fake 返回（不同步） |
| 函数调用 | 未完全实现 |
| Multi-PTX cubins | 累加所有 sections + `PTX_WARN_EMU` 警告 section 数量（parser-completeness Fix #2 + c5 Fix #3）。潜在风险：不同 section 可能定义同名符号（warning 告知用户检查）。`ptx_parser.cpp:60` 与 `cubin_utils.cpp` 行为对齐 |
| `assert(false)` | 多处 → 遇未处理代码路径会崩溃 |

### 安全假设

- 基本算术/逻辑指令 ✓
- 内存操作 (ld/st): global/shared/local ✓
- 控制流 (bra, ret) ✓
- Ampere (sm_80) 及更早架构 ✓

---

## 重要文件

| 文件 | 说明 |
|------|------|
| `include/ptx_ir/ptx_op.def` | PTX 指令定义 (X-Macro) |
| `src/grammar/ptxLexer.g4` / `ptxParser.g4` | ANTLR 语法 |
| `src/cudart/cudart_sim.cpp` | CUDA runtime 入口 |
| `src/ptxsim/InstructionHandlers.cpp` | 指令实现 |
| `configs/ampere_a100.json` | 默认 GPU 架构配置 |
| `include/ptxir/ptxir_serialization.h` | PTXIR 序列化 API |
| `tests/ptx/test_all_ptx.sh` | PTX 语法全量测试 |

## 参考文档

- `docs/gpgpu_arch.md` - GPU 执行架构
- `docs/ptx-emu_arch.md` - 系统架构
- `docs/debugging_guide.md` - 调试与日志
- `docs/sm90_100.md` - Hopper/Blackwell 架构
- `.opencode/skills/README.md` - 技能索引与调用关系

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
