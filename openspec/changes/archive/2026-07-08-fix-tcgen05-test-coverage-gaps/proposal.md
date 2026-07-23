# Fill tcgen05 5 Core Handler Test Coverage Gaps

> **架构依据**: [ADR-0016](../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) Accepted
> **前置 change**: `archive/2026-07-07-implement-tcgen05-handlers-core` (Change-3b, archived 2026-07-07 @ `df6dde7`)
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §7(Pre-impl review)
> **Metis 审查修订**: 见底部 "Metis 审查修订记录" — 8 项关键假设错误已修正

## Why

`implement-tcgen05-handlers-core` (archived 2026-07-07, commit `df6dde7`) 交付了 5 个 `processTcgen05Xxx` handler + `wmma.cpp` 缩减。但这些 handler 当前**未接入 dispatch 管道**(`S_TCGEN05_*` 在 `ptx_op.def` 中被显式排除,X-Macro 不注册)——它们是**死代码**。后续 `implement-tcgen05-handlers-extended` 实施新 handler 时,如果直接 call `processTcgen05*`,需要在单元测试层验证 fragment arithmetic 与 IR 解析。

### 当前测试状态(`df6dde7` 时点)

| 测试层 | 已存在 | 缺失 | 备注 |
|--------|--------|------|------|
| **PTX 语法** | 12 个 `.ptx` fixture (`tests/ptx/tcgen05_*.ptx`) | 0 | 已被 `test_all_ptx.sh` 自动覆盖 ✓ |
| **C++ 单元** | 1 个 (`tests/unit/cluster/test_cluster_tcgen05_integration.cpp`, 23 行,仅测 cluster arrive) | 4 个真实 tcgen05 handler 单元测试 | ❌ 缺失 |
| **C++ 集成** | 0 个 parse-→-IR 测试 | 5 个(mma/ld/st/commit/wait) | ❌ 缺失 |
| **E2E kernel** | 1 个 (`tests/e2e/kernel/test_blackwell_gemm.cu`,用 float 而非 f16) | 专门 `tcgen05.mma` GEMM 测试 | ❌ 缺失 |
| **Reference/golden** | 0 个 | `tcgen05.mma` f16×f16→f32 golden | ❌ 缺失 |
| **Handler 头文件** | 0 个 | `include/ptxsim/instructions/tcgen05.h` | ❌ 缺失(handler 无声明) |

**影响**:
1. 5 handler 完全无运行时集成回归覆盖(也未在任何 dispatch 路径中执行)
2. `tcgen05.mma` fragment arithmetic 无 golden value → 即使 handler 被修复也无法验证数值正确性
3. 没有端到端真实 `tcgen05.mma` kernel 验证 → 只在 PTX 语法层测试

**范围限制**: 本 change **不修复** dispatch 死代码问题(超出范围)。Phase 2 直接调用 handler 是"dead code coverage"——验证当 handler 被未来 dispatcher 接入时,其逻辑正确。

## What Changes

### 新增

| 文件 | 范围 | LoC 估计 |
|------|------|---------|
| `include/ptxsim/instructions/tcgen05.h` | 5 handler forward declaration(必备,handler 无声明) | +20 |
| `tests/integration/ptx/test_tcgen05_mma_parse.cpp` | mma parse → IR 集成测试(`tests/integration/ptx/` 为现有目录) | +80 |
| `tests/integration/ptx/test_tcgen05_ld_parse.cpp` | ld 集成测试(验证 num_regs 字段) | +60 |
| `tests/integration/ptx/test_tcgen05_st_parse.cpp` | st 集成测试 | +60 |
| `tests/integration/ptx/test_tcgen05_commit_parse.cpp` | commit 集成测试(验证 mbarrier qualifier) | +60 |
| `tests/integration/ptx/test_tcgen05_wait_parse.cpp` | wait 集成测试(验证 .load/.store) | +60 |
| `tests/e2e/kernel/test_tcgen05_mma_gemm.cu` | 真实 CUDA kernel E2E(优先 f32 复用 `test_blackwell_gemm.cu` 模式) | +150 |
| `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h` | mma fragment golden values(PTX ISA §9.7.16 手算,与 `ptx_builtin/` 同级) | +100 |
| `tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp` | 单测验证 `processTcgen05Mma` 输出 vs golden(1e-4 容差) | +120 |
| `tests/integration/CMakeLists.txt` | 注册 5 个新集成测试(`tests/integration/ptx/` 已存在) | +25 |
| `tests/e2e/CMakeLists.txt` | 注册 E2E kernel | +10 |
| `tests/unit/CMakeLists.txt` | 注册 `unit_tcgen05_mma_golden` | +10 |

### 修改

| 文件 | 范围 |
|------|------|
| `src/ptxsim/instructions/AGENTS.md` | 更新 tcgen05.cpp 测试覆盖状态 |
| 根 `AGENTS.md` | 已知限制表:标注 "5 core handler test coverage 100% (dead code, dispatch 未修复)" |

### 不修改(范围外)

- ❌ 不修改任何 handler 实现(已在 df6dde7 完成)
- ❌ 不修改 grammar/IR(已在 Change-1/3a 完成)
- ❌ 不实现新 handler(Change-3d scope)
- ❌ **不修复 dispatch 死代码**(独立 change;handler 注册路径不在本 PR)
- ❌ 不修改 `tests/integration/parser/`(该目录不存在,与现有模式冲突)

### 修改

| 文件 | 范围 |
|------|------|
| `src/ptxsim/instructions/AGENTS.md` | 更新 tcgen05.cpp 测试覆盖状态 |
| 根 `AGENTS.md` | 已知限制表:标注 "5 core handler test coverage 100%" |

### 不修改(范围外)

- ❌ 不修改任何 handler 实现(已在 df6dde7 完成)
- ❌ 不修改 grammar/IR(已在 Change-1/3a 完成)
- ❌ 不实现新 handler(Change-3d scope)
- ❌ 不修改 CMakeLists.txt 顶层(仅 tests 子目录)

## Non-Goals

- ❌ 不实现 6 extended handler(Change-3d scope)
- ❌ 不修改 `tcgen05.cpp` handler 逻辑
- ❌ 不添加新的 PTX fixture(`tests/ptx/` 已有 13 个 tcgen05 fixtures)
- ❌ 不追求 cycle-accurate 性能对标(per ADR-0016)

## Goals

### Phase 1: 5 集成 parse 测试(1 commit)

1. **预创建目录确认**: `tests/integration/ptx/` 已存在,无需新建
2. 创建 `tests/integration/ptx/test_tcgen05_mma_parse.cpp` — 验证 `tcgen05.mma.kind::f16.cta_group::1` parse → IR(使用 ANTLR parser + `makeTcgen05Instr` factory,**不**用 `ptxsim::testing::step_warp`——helpers 不存在)
3. 创建 `tests/integration/ptx/test_tcgen05_ld_parse.cpp` — 验证 `tcgen05.ld.sync.aligned.shared::cta` parse
4. 创建 `tests/integration/ptx/test_tcgen05_st_parse.cpp` — 验证 `tcgen05.st.sync.aligned.shared::cta` parse
5. 创建 `tests/integration/ptx/test_tcgen05_commit_parse.cpp` — 验证 `tcgen05.commit.cta_group::1` parse
6. 创建 `tests/integration/ptx/test_tcgen05_wait_parse.cpp` — 验证 `tcgen05.wait::load.cta_group::1` parse
7. 每个测试:构造 PTX 字符串 → ANTLR 解析 → verify `Tcgen05Instr` struct fields + Qualifier 解析
8. 标签:`integration;ptx;tcgen05;parse`
9. 注册到 `tests/integration/CMakeLists.txt`(已有,使用 `add_catch_test(integration_ptx_tcgen05_<op>_parse ptx/test_tcgen05_<op>_parse.cpp)`)
10. 跑 `ctest -L "integration;tcgen05;parse" -V` 验证 5/5 PASS

### Phase 2: handler 头文件 + golden value(1 commit,顺序敏感)

1. **先**: 创建 `include/ptxsim/instructions/tcgen05.h`(5 handler forward declaration)
2. **后**: 创建 `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h`(与 `tests/reference/ptx_builtin/` 同级):
   - 来源: PTX ISA §9.7.16 规范手算(8x4 f16×f16→f32 fragment arithmetic)
   - 格式: `constexpr std::array<float, 32> GOLDEN_MMA_F16_F16_F32 = { ... };`
   - 每元素必须可被 reviewer 复算
   - 标注 `// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16`
3. **再**: 创建 `tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp`(新文件,**不**扩张 `test_cluster_tcgen05_integration.cpp`)
   - #include `ptxsim/instructions/tcgen05.h` + `reference/ptx_tcgen05/tcgen05_mma_golden.h`
   - 构造 `ThreadContext`/`WarpContext` stub + 初始化 TMEM
   - 直接调用 `ptxsim::processTcgen05Mma(...)`(dead code 调用,记录到 spec)
   - 验证输出寄存器 vs golden(1e-4 容差)
4. 标签:`unit;ptx_ir;tcgen05;mma;golden`

### Phase 3: E2E kernel(1 commit,可降级)

1. 创建 `tests/e2e/kernel/test_tcgen05_mma_gemm.cu`:
   - **预验证**: `nvcc -ptx` 必须能生成合法 `tcgen05.mma.kind::f16` PTX,否则复用 `test_blackwell_gemm.cu` 的 float 模式(注释:f16 受 ANTLR grammar 限制影响,见 `test_blackwell_gemm.cu:11`)
   - 用 `cuobjdump -xptx` 从 Cutlass 3.x GEMM 提取真实 Blackwell PTX(若可用)
   - 若 cuobjdump 不可用 → 降级为手动构造 `tcgen05.mma` 指令序列
2. 注册到 `tests/e2e/CMakeLists.txt`
3. 标签:`e2e;kernel;tcgen05;gemm;sm100`
4. 跑 `ctest -L "e2e;tcgen05" -V` 验证

### Phase 4: 文档同步 + Archive(1 commit)

1. 更新 `src/ptxsim/instructions/AGENTS.md` 测试覆盖状态
2. 根 `AGENTS.md` 已知限制表更新:标注 **"5 core handler 单元测试覆盖 + dispatch 死代码未修复(独立 change)"**
3. Archive

## Capabilities

### New Capabilities

- `tcgen05-handler-test-coverage`: 5 集成 parse 测试 + handler 头文件 + 1 golden value + 1 E2E kernel

### Modified Capabilities

- 无(纯测试补充,handler 头文件为新增而非修改)

## Impact

### 影响的代码(预计)

| 文件 | 变更类型 | LoC 估计 |
|------|---------|---------|
| `include/ptxsim/instructions/tcgen05.h` | 新增(头文件) | +20 |
| `tests/integration/ptx/test_tcgen05_*_parse.cpp`(5 个) | 新增 | +320 |
| `tests/e2e/kernel/test_tcgen05_mma_gemm.cu` | 新增 | +150 |
| `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h` | 新增(与 `tests/reference/ptx_builtin/` 同级) | +100 |
| `tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp` | 新增 | +120 |
| `tests/integration/CMakeLists.txt` | 修改(+25) | +25 |
| `tests/e2e/CMakeLists.txt` | 修改(+10) | +10 |
| `tests/unit/CMakeLists.txt` | 修改(+10) | +10 |
| `src/ptxsim/instructions/AGENTS.md` + 根 `AGENTS.md` | 修改 | +10 |
| **总计** | | **+765** |

### 影响的依赖

- `cuobjdump -xptx` 工具(Phase 3 E2E,若不可用则降级)
- `nvcc -ptx` 验证 f16 tcgen05.mma PTX 是否合法(Phase 3 预检)
- `ptx-debug` skill(若测试失败调试)
- `three-mode-testing` skill(三套测试)
- `statement_factory.h::makeTcgen05Instr`(Phase 1 + Phase 2 复用)

### 不影响的依赖

- 5 core handler 实现(已 archive,逻辑不变)
- grammar/IR(已 archive)
- `InstructionFactory::initialize()`(dispatch 死代码不在本 PR 范围)

## Design-Time Checklist (Lessons-Learned)

### 函数审计完整性

- [x] Baseline: 5 `processTcgen05Xxx` handler 已存在于 `tcgen05.cpp`(df6dde7),但**无 dispatch 路径**——独立头文件 + 直接调用 = "dead code coverage"
- [x] 跨模块状态翻译: 无(纯测试)
- [x] invariant 清单: parse → IR 字段一致性、golden value 数值正确性(handler 输出)

### 多 Phase 推进(4 atomic commits)

- [x] Phase 1: 5 集成 parse 测试(独立 commit)
- [x] Phase 2: 头文件 + golden value(独立 commit,顺序敏感:头文件先)
- [x] Phase 3: E2E kernel(独立 commit,可选降级)
- [x] Phase 4: 文档 + archive(独立 commit)
- [x] 基线 worktree: `.worktrees/baseline-tcgen05-tests`
- [x] 失败处理策略: 某 Phase 测试失败 → revert 该 Phase

### 文档同步

- [x] AGENTS.md 同步项已列出
- [x] lessons-learned 预留(若发现新模式,如 handler 死代码模式)

### Metis 审查修订记录(本节记录关键假设修正)

| ID | 原文假设 | 修正后 | 原因 |
|----|----------|--------|------|
| F1 | `tests/integration/parser/` | `tests/integration/ptx/` | 原目录不存在,实际为 `tests/integration/ptx/`(18 个现有 PTX 集成测试) |
| F2 | `tests/ptx/reference/` | `tests/reference/ptx_tcgen05/` | 原目录不存在,reference 数据均在 `tests/reference/`(与 `ptx_builtin/` 同级) |
| F3 | `ptxsim::testing::step_warp` + `make_*` helpers | ANTLR parser + `makeTcgen05Instr` factory | helpers 不存在;parse 测试不驱动 warp 执行,不应使用 step_warp |
| F4 | "5 handler 在 dispatch 中可调用" | "5 handler 是 dead code,需独立调用" | `S_TCGEN05_*` 在 `ptx_op.def:129-136` 显式排除 X-Macro,handler 无外部调用者 |
| F5 | 不需要 header(include 即可) | 需新建 `tcgen05.h` | 现有 `include/` 下零声明,无 forward declaration 则 link 失败 |
| F6 | golden 测试放 `test_cluster_tcgen05_integration.cpp` | 新建 `tests/unit/ptx_ir/test_tcgen05_mma_golden.cpp` | 现有 cluster 测试 23 行,与 golden value 无关 |
| F7 | E2E 强制 f16 GEMM | f16 预验证 + 降级到 float(复用 `test_blackwell_gemm.cu:11` 注释) | 现有 E2E 显式回避 f16 受 grammar 限制 |
| F8 | parse 测试用 `add_catch_test` | 同——但目录改 `tests/integration/ptx/` | 原目录不存在 |
| **NEW** | (无此记录) | **spec.md 增加 dead-code-coverage 标注** | Phase 2 直接调用 handler 必须显式标记其未被 dispatch 路由 |

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| `implement-tcgen05-handlers-core` (archived @ df6dde7) | **fix-tcgen05-test-coverage-gaps** | `implement-tcgen05-handlers-extended` (Change-3d) |
| `implement-tcgen05-syntax-ir` (archived) | | |

- **Change-3b → 本 change**: 依赖 `tcgen05.cpp` 5 handler(dead code 状态) + `makeTcgen05Instr` factory(在 `statement_factory.h`)
- **本 change → Change-3d (handlers-extended)**: 提供 5 integration parse 测试基础设施 + handler 头文件 + golden value,新 handler 可复用
- **本 change → 独立 change (dispatch fix)**: 暴露 F4 死代码问题,需独立 change 修复 `InstructionFactory` 注册路径
- **不依赖** Change-4 (cleanup-wmma-namespace) — 测试不碰 wmma

## Golden Value 来源决策

**决策 D1**: 优选 PTX ISA §9.7.16 规范手算(8x4 f16×f16→f32 fragment):
- 公式: `C[i][j] += A[i][k] * B[k][j]` (8 rows × 4 cols matmul)
- 精度: IEEE 754 f16→f32 转换
- 拒绝: 依赖 Cutlass 3.x(可能不可用)
- 注释: 全量 `UNVERIFIED-AGAINST-HARDWARE`

**决策 D2**: E2E kernel 降级策略:
- 优先: `cuobjdump -xptx` 提取真实 Blackwell GEMM PTX
- 降级: 手动构造 `tcgen05.mma` + `tcgen05.ld` + `tcgen05.st` + `tcgen05.commit` + `tcgen05.wait` 指令序列
- 拒绝: 跳过 E2E(会留下覆盖率缺口)