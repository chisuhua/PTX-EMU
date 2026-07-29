# PTX Instruction Handlers
**SSOT**: Common conventions (build/test/format/conventions/anti-patterns) live in root AGENTS.md; this file only documents ptxsim/instructions-specific content.

~20 files + cvt/ subdirectory — 每个 handler 通过 `instruction_factory.cpp` 的 X-Macro 分派（`ptx_op.def` → `process_<op>`）。

## STRUCTURE

```
instructions/
├── arithmetic.cpp          # add, sub, neg, abs
├── arithmetic_ext.cpp      # addc, subc, mul24, mad24, fma
├── arithmetic_muldiv.cpp   # mul, div, mad, min, max, rem
├── atomic.cpp              # atom, cas → serialized via AtomicMutex
├── barrier.cpp             # bar.warp.sync, activemask → BarrierModule dispatch
├── bitwise.cpp             # and, or, xor, shl, shr, bfe, popc, clz, not
├── call.cpp                # call, ret (BUG-RETHANG: marks ALL lanes exited), printf
├── comparison.cpp          # setp, selp
├── control.cpp             # bra (SIMT stack branch)
├── data_transfer.cpp       # mov, cvta
├── math.cpp                # sin, cos, sqrt, rcp, lg2, ex2, rsqrt
├── memory.cpp              # ld, st
├── tcgen05.cpp             # 5 processTcgen05Xxx handlers (mma/ld/st/commit/wait)
├── tcgen05_alloc.cpp       # 3 alloc-family (alloc/dealloc/relinquish)
├── tcgen05_cp.cpp          # SMEM→TMEM copy (128-byte transfer)
├── tcgen05_fence.cpp       # Fence no-op marker
├── tcgen05_helpers.cpp     # Shared fragment arithmetic
└── cvt/
    ├── cvt_strategy.{h,cpp}     # Dispatcher + ConversionStrategy interface
    ├── cvt_float_to_float.{h,cpp}  # f32↔f64↔f16
    ├── cvt_float_to_int.{h,cpp}    # .sat / 5 rounding modes / .ftz
    ├── cvt_int_to_float.{h,cpp}    # IntToFloatStrategy
    ├── cvt_int_to_int.{h,cpp}      # IntToIntStrategy
    └── cvt_helpers.{h,cpp}         # 4 helper functions
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| 新指令 handler | 新建 `instructions/<category>.cpp` + 注册 `ptx_op.def` |
| 算术/位运算 | `arithmetic*.cpp` / `bitwise.cpp` (模板函数 + per-handler processOperation) |
| 类型转换 | `cvt/` (策略模式，不直接写 switch) |
| 屏障同步 | `barrier.cpp` → `BarrierModule` API |
| 分支/返回 | `control.cpp` (bra) + `call.cpp` (ret: 全 lane 退出标记) |
| 内存操作 | `memory.cpp` (ld/st) + `atomic.cpp` (atom/cas) |
| 数据移动 | `data_transfer.cpp` (mov/cvta) |
| Blackwell tcgen05 | `tcgen05*.cpp` (仅 Blackwell 架构启用) |
| pre-Blackwell WMMA | 已移除 (commit `0cd2c3eb`) — 永久抛 `UnsupportedInstructionException` 见 `tcgen05.cpp` 注释 "Extracted from wmma.cpp" |

## KEY FILES

| File | Role |
|------|------|
| `barrier.cpp` | 屏障路由 — 必须经过 BarrierModule，不再直接操作 Wbar |
| `call.cpp` | RetHandler 负责全 warp 退出标记 (BUG-RETHANG 修复) |
| `atomic.cpp` | 跨 warp 原子操作 — 通过 AtomicMutex 串行化 |
| `cvt/cvt_strategy.cpp` | cvt 策略模式入口: `select_strategy()` + `convert()` |

## CONVENTIONS

- **Handler 签名**: `void processOperation(ThreadContext*, void** operands, const std::vector<Qualifier>&, ...)` — 通过 `instruction_factory.cpp` 的 X-Macro 注册
- **PC 管理**: `commit_pc()` 是唯一的正常 PC 推进方式；禁止直接修改 thread PC
- **屏障路由**: barrier handler 必须通过 `BarrierModule` API — 不直接操作 Wbar struct
- **cvt 策略**: 新增类型转换加 `cvt/` 子目录策略，不直接写 switch
- **tcgen05**: 全部在 `tcgen05*.cpp`，仅在 Blackwell (`sm_100`) 架构启用

## KNOWN STUBS

- `atomic.cpp`: cross-warp CAS 通过 `AtomicMutex` 串行化，无硬件级原子性保证
- pre-Blackwell WMMA: handler 已移除 (commit `0cd2c3eb`)；调用走 `tcgen05.cpp` 异常路径
  (ADR-0016)

## KNOWN ISSUES

- **BUG-RETHANG (FIXED)**: `RetHandler::processOperation` 必须标记 ALL 32 lanes 为 `Exited`，不能只标记执行中的 lane
- **BUG-POSTBARRIER-TWOHALVES (FIXED)**: `BarrierModule::release_warp_barrier()` 中做 OR arrived_mask，不在 `set_active_mask()` 中做
- **SCOPE-OF-EFFECT**: 指令 handler 只修改自己的操作数，不做（不依赖）非本指令的语义副作用

## COMMANDS

```bash
# 单指令集成测试
cd build && ctest -R integration_<op_name>

# single-thread 指令 handler 测试 (unit)
cd build && ctest -R unit_<op_name>

# 全量回归
./scripts/regression.sh
```