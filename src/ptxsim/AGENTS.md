# PTX-EMU Execution Engine
**SSOT**: Common conventions (build/test/format/conventions/anti-patterns) live in root AGENTS.md; this file only documents ptxsim-specific content.

PTX 指令解释执行引擎 — 管理 GPU/SM/CTA/Warp/Thread 层次化上下文，分派指令到 handler，协调屏障同步与内存访问。

## STRUCTURE

```
src/ptxsim/
├── atomic/           # AtomicMutex (cross-warp atomicity)
├── async/            # TcQueue (TMA commit-group scheduling)
├── barrier/          # BarrierModule + WarpBarrier + CTABarrier
├── cluster/          # ClusterContext (distributed shared memory sync)
├── core/             # GPU→SM→CTA→Warp→Thread 上下文 (20 files)
├── debug/            # ptx_config.cpp, warp_trace_formatter.cpp
├── instructions/     # PTX 指令 handlers (~18 files + cvt/ subdir)
│   └── cvt/          # cvt 指令策略实现 (6 files)
├── memory/           # TMA descriptor, TMEM, TmemAllocator
├── utils/            # qualifier_utils, half_utils, type_utils
├── instruction_factory.cpp   # X-Macro dispatch (ptx_op.def → handler)
├── instruction_handlers.cpp  # 遗留全局 handler 表
├── instruction_base.cpp      # BaseInstruction 基类
└── register_analyzer.cpp     # 寄存器活性分析
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| 指令分派 | `instruction_factory.cpp` (X-Macro routing via `ptx_op.def`) |
| 新指令 handler | `instructions/<category>.cpp` (snake_case: `process_add`) |
| cvt 指令 | `instructions/cvt/` (策略模式: float↔int↔float) |
| 时钟周期循环 | `core/gpu_context.cpp` → `core/sm_context.cpp` |
| Warp 调度 | `core/warp_scheduler.cpp` |
| 屏障同步 | `barrier/barrier_module.cpp` (统一入口, 2026-06 迁移完成) |
| SIMT 栈 | `core/simt_stack.cpp` + `core/simt_pc_manager.cpp` |
| TMA 异步拷贝 | `memory/tma_descriptor.cpp` + `async/tc_queue.cpp` |
| TMEM | `memory/tmem.cpp` + `memory/tmem_allocator.cpp` |
| Cluster arrive/wait | `cluster/cluster_context.cpp` |
| 跨 warp 互斥 | `atomic/atomic_mutex.cpp` |
| 调试配置 | `debug/ptx_config.cpp` (INI 驱动的断点/日志) |
| 类型/qualifier 工具 | `utils/qualifier_utils.cpp`, `utils/type_utils.cpp` |
| 寄存器分析 | `register_analyzer.cpp` |
| 执行跟踪 | `core/execution_tracer.cpp` + `debug/warp_trace_formatter.cpp` |

## KEY FILES

| File | Role |
|------|------|
| `instruction_factory.cpp` | X-Macro dispatch: `ptx_op.def` → `process_*` handler |
| `instruction_handlers.cpp` | 全局 handler 注册表 (与 factory 共存) |
| `instruction_base.cpp` | Virtual `BaseInstruction` with `execute()` interface |
| `register_analyzer.cpp` | Liveness analysis for register allocation |

## CONVENTIONS

- **Handler 命名**: `process_<op>()` snake_case, 见 `instruction_factory.cpp`
- **屏障路由**: barrier handler 必须通过 `BarrierModule` API (s_bar / s_bar_warp_sync 均已迁移)
- **cvt 策略**: 新增类型转换加 `cvt_strategy.cpp` + `cvt_helpers.cpp`, 不直接写 switch
- **tcgen05 指令**: 全部在 `instructions/tcgen05*.cpp`, 仅在 Blackwell 架构启用
- **TMA descriptor**: 解析逻辑在 `memory/tma_descriptor.cpp`, 调度在 `async/tc_queue.cpp`

## ANTI-PATTERNS

- ❌ WarpContext 中调 ThreadContext 方法不加锁 — 使用 `AtomicMutex` 或上层同步
- ❌ 新增 `Wbar` struct 使用 — 已全部迁移至 `BarrierModule` + `WarpBarrier`
- ❌ `set_active_mask()` 做 OR 合并 — OR 逻辑在 `BarrierModule::release_warp_barrier()`
- ❌ 直接修改 thread PC 绕过 `simt_pc_manager` — 用 `set_pc()` + `commit_pc()`
- ❌ 在 `instruction_handlers.cpp` 加新 handler — 优先用 `instructions/<category>.cpp` + factory
- ❌ 不加 struct+variant 就加 `ptx_op.def` 条目

## COMMANDS

```bash
# 调试 (INI 配置, 见 debug/ptx_config.cpp)
./build/bin/ptx_emu --config configs/debug_config.ini <kernel.ptx>

# 单指令 handler 测试 (integration)
cd build && ctest -R integration_<op_name>

# 全量回归
./scripts/regression.sh
```