# PTX-EMU Agent Instructions

**Generated:** 2026-07-25 | **Commit:** be746976 | **Branch:** main

PTX ISA 指令级模拟器 — C++20/CUDA, ANTLR4 解析, fake libcudart.so 拦截 CUDA runtime. sm_100 虚拟架构.

## 🛑 最高优先级

- **ptx-lessons-learned**: 任何 OpenSpec change 前必读 `.opencode/skills/ptx-lessons-learned/SKILL.md`（14 核心经验 + 12 checklist）
- **跨模块间接状态翻译**: 迁移函数必须行级 diff, 不漏看似冗余的 `set_state()` — 下模块的 `sync_to_warp_state()` 才翻译为 `is_blocked`
- **递归锁死锁**: 持锁方法调用同锁其他 public 方法 = deadlock. 集中审计所有使用同一锁的代码路径
- **复杂迁移分 Phase commit**: 每个 Phase 独立可回退. 测试回归 → revert 该 Phase, 不混入后续 commit
- **基线 worktree**: 重大重构前 1 分钟建立 `.worktrees/`, 节省数小时争论

## STRUCTURE

```
src/                  # 核心源码
├── cudart/           # fake libcudart.so (__cudaRegisterFatBinary → cudaLaunchKernel)
├── grammar/          # ANTLR4 .g4 (5 split files)
├── ptx_ir/           # IR types + ptx_op.def X-Macro (106 entries)
├── ptx_parser/       # PtxVisitor + CFGBuilder
├── ptxsim/           # 执行引擎
│   ├── barrier/      # BarrierModule + WarpBarrier + CTABarrier
│   ├── core/         # GPU/SM/CTA/Warp/Thread 上下文
│   ├── instructions/ # PTX 指令 handlers (~20 files)
│   └── memory/       # TMA descriptor, TMEM, TmemAllocator
├── ptxir/            # PTXIR 二进制序列化
├── memory/           # SimpleMemory, SharedMemoryManager
└── register/         # RegisterBankManager
include/              # 公共头文件 (mirrors src/)
tests/                # 3 层: unit/integration/e2e (~181 文件, Catch2)
configs/              # GPU JSON + debug INI
bench/                # CUDA 基准测试 (含 CUTLASS/cute headers)
docs/                 # ADR, 架构文档, audits (173 文件)
openspec/             # proposal/design/tasks/specs (413 文件)
scripts/              # sanity.sh, regression.sh, debug-run.sh
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| CUDA runtime 拦截 | `src/cudart/cudart_sim.cpp` |
| PTX 解析 | `src/ptx_parser/` + `src/grammar/` |
| IR + 新指令注册 | `include/ptx_ir/ptx_op.def` (X-Macro) |
| 指令实现 | `src/ptxsim/instructions/` |
| 执行层次 | `src/ptxsim/core/` (GPU→SM→CTA→Warp→Thread) |
| 屏障同步 | `src/ptxsim/barrier/` (BarrierModule) |
| 测试 | `tests/{unit,integration,e2e}/` 对应子目录 |
| PTX 语法测试 | `tests/ptx/test_all_ptx.sh` (NOT ctest!) |
| 架构决策 | `docs/adr/` (23 文件) |
| 技能索引 | `.opencode/skills/README.md` |

## CODE MAP

| Symbol | Type | File | Role |
|--------|------|------|------|
| `__cudaRegisterFatBinary` | Func | `src/cudart/cudart_sim.cpp` | 初始化入口 |
| `cudaLaunchKernel` | Func | `src/cudart/cudart_sim.cpp` | 内核启动 |
| `GPUContext::exe_once()` | Func | `src/ptxsim/core/gpu_context.cpp` | 时钟周期循环 |
| `SMContext::exe_once()` | Func | `src/ptxsim/core/sm_context.cpp` | SM 级执行 |
| `WarpContext::execute_warp_instruction()` | Func | `src/ptxsim/core/warp_context.cpp` | Warp 指令分发 |
| `ThreadContext::execute_thread_instruction()` | Func | `src/ptxsim/core/thread_context.cpp` | 线程级执行 |
| `InstructionFactory::get_handler()` | Func | `src/ptxsim/instruction_factory.cpp` | X-Macro 分派 |
| `BarrierModule` | Class | `include/ptxsim/barrier/barrier_module.h` | 屏障状态机 |
| `WarpBarrier` | Class | `include/ptxsim/barrier/warp_barrier.h` | Per-warp 屏障 |
| `PtxVisitor::visitFunctionDecl()` | Func | `src/ptx_parser/ptx_visitor.cpp` | PTX 解析入口 |

## CONVENTIONS

- **格式化**: clang-format (BasedOnStyle=LLVM, IndentWidth=4, ColumnLimit=80)
- **命名**: 文件 snake_case | 函数 camelCase | 类 PascalCase | 变量 camelCase
- **PTX 指令**: 全小写 (mov, add, ld, st)
- **头文件**: `#ifndef`/`#define`/`#endif` 守卫
- **X-Macro**: `#define X(name) #include "ptx_op.def" #undef X`
- **TDD 强制**: 测试先行 (Red) → 实现 (Green) → sanity.sh 验证
- **ctest 命名**: 必须前缀 `unit_`/`integration_`/`e2e_` (commit `ab55e06`)
- **测试标签**: `<type>;<subject>`, 如 `unit;barrier`

## ANTI-PATTERNS

- ❌ `ctest` 代替 `test_all_ptx.sh` 做 PTX 语法测试
- ❌ 不加载 `ptx-grammar-modification` 技能就改 `.g4`
- ❌ `force_set_pc()` — 用 `set_pc()` + `commit_pc()`
- ❌ 从 WarpContext 调 ThreadContext 方法无锁
- ❌ `set_active_mask()` 做 OR 合并 (ret handler 依赖 overwrite 语义)
- ❌ 新增 `Wbar` struct 使用 — 已全部迁移至 BarrierModule
- ❌ 不加 struct+variant 就加 `ptx_op.def` 条目
- ❌ 不改 reader 就改 PTXIR writer 格式

## COMMANDS

```bash
. env.sh                                                    # 设置环境 (必须!)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build.sh                                                  # Debug 快捷
cmake --build build --target GenerateParser                 # 改 .g4 后重生成
cd build && ctest                                           # 全量测试
./tests/ptx/test_all_ptx.sh                                 # PTX 语法测试 (NOT ctest!)
./scripts/sanity.sh                                         # 分层健康检查
./scripts/regression.sh                                     # 全量回归
```

## CHILD AGENTS.MD

| File | Coverage |
|------|----------|
| `configs/AGENTS.md` | GPU arch JSON + debug INI schema |
| `include/cudart/AGENTS.md` | cpptlm_bridge ABI + public headers |
| `src/cudart/AGENTS.md` | CUDA runtime interception |
| `src/grammar/AGENTS.md` | ANTLR4 grammar modification |
| `src/ptx_ir/AGENTS.md` | IR types + X-Macro + PTXIR |
| `src/ptx_parser/AGENTS.md` | PTX parser + CFG builder |
| `src/ptxsim/AGENTS.md` | Execution engine overview |
| `src/ptxsim/core/AGENTS.md` | GPU→SM→CTA→Warp→Thread contexts |
| `src/ptxsim/instructions/AGENTS.md` | PTX instruction handlers |
| `src/ptxsim/barrier/AGENTS.md` | Barrier state machine |
| `tests/AGENTS.md` | 3-tier test suite |