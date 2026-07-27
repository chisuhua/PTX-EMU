# tests/ — PTX-EMU 3-Tier Test Suite

**~174 Catch2 targets + 46 PTX syntax files. 3-tier physical separation. sm_100 virtual arch.**

## STRUCTURE

```
tests/
├── unit/            # Type 1 — 直接单元测试 (~89 targets)
│   ├── barrier/     #   BarrierModule, WarpBarrier, participation_mask
│   ├── simt/        #   SIMT stack entry, handle_branch, thread PC
│   ├── warp/        #   Warp state, blocked_decrement (standalone: test_warp_context)
│   ├── exec/        #   exec_mask, active_mask, BRA/BRAPRED, ret divergent
│   ├── memory/      #   MemoryManager, TMA descriptor, TMEM, TmemAllocator
│   ├── sm/          #   Streaming admission, exe_once helpers
│   ├── sync/        #   Sync mechanism, barrier PC overwrite, syncthreads
│   ├── pc/          #   PC management
│   ├── ptx/         #   CVT helpers/context/strategies, fma, rsqrt, ld/st
│   ├── ptx_ir/      #   tcgen05 golden values, pipeline handler, extended opkind
│   ├── tcgen05/     #   tcgen05.cp, mma.ws scope
│   ├── cudart/      #   Stream handle, stream destroy, sync loop, memory API
│   ├── cpptlm/      #   Bridge, attach/detach, injection, cosim smoke
│   ├── cluster/     #   Cluster mode, tcgen05 integration
│   ├── async/       #   TcQueue (commit group)
│   ├── common/      #   CC register, parse immediate, scheduler config, latency
│   ├── divergence/  #   Divergence + sync standalone
│   ├── register/    #   Dest register extraction
│   ├── utils/       #   Half-precision utils
│   ├── parser/      #   Multi-PTX, extern function
│   └── testing/     #   Memory test utils self-test
├── integration/     # Type 2 — 指令序列集成测试 (~68 targets)
│   ├── barrier/     #   Barrier lifecycle, divergence scheduling, memory visibility
│   ├── simt/        #   Thread PC
│   ├── divergence/  #   Sync convergence, nested divergence, post-barrier
│   ├── exec/        #   Warp state, lane verification, ldglobal no-hang
│   ├── pc/          #   PC management
│   ├── sync/        #   Sync mechanism, syncthreads pipeline
│   ├── ptx/         #   LD/ST, integer/float arith, CVT, atom, bitwise, tcgen05 parse
│   ├── tcgen05/     #   Dispatch, alloc/dealloc, cp, mma, fence, commit/wait, slot routing
│   ├── memory/      #   Shared memory layout, local memory, dynamic shared memory
│   ├── tma/         #   TMA descriptor + CTAContext
│   ├── tmem/        #   TMEM + CTAContext
│   ├── cluster/     #   Cluster + CTAContext
│   ├── async/       #   TcQueue + CTAContext
│   └── cpptlm/      #   Singleton, async launch, ld/st bridge, mock injection
├── e2e/             # Type 3 — CUDA Kernel E2E 测试 (~17 targets)
│   ├── kernel/      #   CFG, barrier, ldglobal, shared memory, tcgen05 GEMM, flash-attention
│   ├── divergence/  #   Divergence + divergence sync
│   └── cosim/       #   CppTLM co-simulation (vector add, infinite loop ceiling, multi-kernel)
├── ptx/             # PTX 语法测试 (46 .ptx) — 由 test_all_ptx.sh 驱动
├── common/          # ptx_lane_printer.cpp (通用工具, 非测试)
├── instructions/    # test_ptx_bra.cu (遗留指令测试)
├── ptxir/           # PTXIR 序列化 (空目录, 占位)
├── archive/         # 已归档/废弃测试
├── reference/       # 参考 PTX 输出 (ptx_builtin/, ptx_tcgen05/)
├── catch_amalgamated.hpp  # Catch2 单头文件 (~14K lines)
└── catch_amalgamated.cpp  # Catch2 编译单元
```

## WHERE TO LOOK

| 你想要的 | 目录 |
|----------|------|
| BarrierModule/WarpBarrier 直接测试 | `tests/unit/barrier/` |
| 指令序列执行 (step_warp) | `tests/integration/ptx/` |
| 真实 CUDA kernel 端到端 | `tests/e2e/kernel/` |
| PTX 语法解析验证 | `tests/ptx/` (46 .ptx) |
| 集成测试辅助函数 | `include/ptxsim/testing/*.h` |
| 测试工具自测 | `tests/unit/testing/` |

## TEST FRAMEWORK

- **Catch2**: 单头文件 `catch_amalgamated.hpp` + `catch_amalgamated.cpp` (14K lines, 所有测试共用)
- **CUDA arch**: `sm_100` (虚拟架构), `-keep --no-compress` 保留中间 PTX
- **测试辅助函数** (`include/ptxsim/testing/`):
  - `instruction_helpers.h` — `make_mov`, `make_bra_pred`, `make_bar_sync` 等 StatementContext 构造
  - `scheduler_utils.h` — `step_warp()` 单步推进 warp 执行
  - `predicates.h` — `setup_pred` 谓词寄存器设置
  - `assertion_utils.h` — 自定义 Catch2 断言
  - `warp_test_utils.h` — WarpContext 测试夹具
  - `memory_test_utils.h` — 内存操作验证
  - `shared_memory.h` — 共享内存测试辅助
  - `tmem_helpers.h` — TMEM 测试辅助
  - `debug_utils.h` — 调试打印工具

## CONVENTIONS

- **ctest 命名**: 必须前缀 `unit_`/`integration_`/`e2e_` (commit `ab55e06`)
- **测试标签**: `<type>;<subject>`, 如 `unit;barrier`, `integration;divergence`
- **回归测试标签**: 追加 `regression;<BUG-ID>`, 如 `regression;BUG-POSTBARRIER-TWOHALVES`
- **RED PHASE 注释**: 回归测试源文件顶部标注 `RED PHASE: this test MUST FAIL on unpatched code`

### 标签速查表

| 标签 | ctest 命令 | 用途 |
|------|------------|------|
| `unit` | `ctest -L unit` 或 `ctest -R "^unit_"` | 所有单元测试（直接类实例化） |
| `integration` | `ctest -L integration` 或 `ctest -R "^integration_"` | 所有指令序列集成测试 |
| `e2e` | `ctest -L e2e` 或 `ctest -R "^e2e_"` | 所有端到端 CUDA kernel 测试 |
| `unit;barrier` | `ctest -L "unit;barrier"` | 仅屏障单元测试 |
| `integration;divergence` | `ctest -L "integration;divergence"` | 仅同步收敛测试 |
| `regression;<BUG-ID>` | `ctest -L "regression;<BUG-ID>"` | 单个 bug 回归测试 |
- **类型约定**:
  - Type 1 (unit): 直接类实例化, 不经过指令执行, 无 PTX 解析
  - Type 2 (integration): `step_warp()` + `make_*` helper 驱动指令序列
  - Type 3 (e2e): 真实 CUDA kernel 编译 + `cudaLaunchKernel` 拦截
- **PTX 语法测试**: `.ptx` 文件放在 `tests/ptx/`, 用 `test_all_ptx.sh` 驱动

## COMMANDS

```bash
. env.sh && cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build
cd build && ctest --output-on-failure                    # 全量测试
cd build && ctest -L "unit;barrier"                       # 按标签筛选
cd build && ctest -R "integration_ptx_cvt"                # 按名称正则
./tests/ptx/test_all_ptx.sh                               # PTX 语法测试 (NOT ctest!)
```

## ANTI-PATTERNS

- ❌ 用 `ctest` 代替 `test_all_ptx.sh` 做 PTX 语法测试 — 46 个 .ptx 文件由独立脚本驱动
- ❌ ctest 目标名缺少 `unit_`/`integration_`/`e2e_` 前缀
- ❌ 在 Type 1 测试中调用 `step_warp()` 或 `execute_warp_instruction()` — 界限模糊时用 Type 2
- ❌ 在 Type 2 测试中手动 `pushSIMTStack` — 违反 Principle 5, 由 `step_warp()` 自动管理
- ❌ 在 Type 3 测试中绕过 `cudaLaunchKernel` 直接调用 handler — 破坏 end-to-end 语义
- ❌ 测试文件放在类型不匹配的子目录 (如把 e2e 测试放在 `tests/unit/`)
- ❌ 回归测试不标注 RED PHASE — 破坏 TDD 纪律