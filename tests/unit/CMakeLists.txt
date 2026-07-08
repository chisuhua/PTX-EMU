# tests/unit/CMakeLists.txt
# Type 1: 直接单元测试
# 直接实例化核心数据结构/类，调用方法验证行为，不经过指令执行流程

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_PTX_COMPILATION ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_ARCHITECTURES "100")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep --no-compress")

set(CATCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/tests)

# ============================================================================
# Barrier 单元测试
# ============================================================================
add_catch_test(unit_barrier_module
    barrier/test_barrier_module.cpp
)
set_tests_properties(unit_barrier_module PROPERTIES LABELS "unit;barrier")

add_catch_test(unit_warp_barrier
    barrier/test_warp_barrier.cpp
)
set_tests_properties(unit_warp_barrier PROPERTIES LABELS "unit;wbar")

# ============================================================================
# barrier-module-lifecycle-tests (OpenSpec change 2026-07-03):
# Direct unit coverage for BarrierModule::release_warp_barrier state
# translation + WarpBarrier lifecycle + participation_mask boundaries.
# Closes the coverage gap from migrate-bar-warp-sync-to-barrier-module
# (review Issue I1). 7 tests across 3 files.
# Refs: lessons-learned.md §1 (cross-module state translation), §19
#       (BUG-POSTBARRIER-TWOHALVES / BUG-RECONVERGENCE-SIMPLEGEMM success)
# ============================================================================
add_catch_test(unit_participation_mask_boundaries
    barrier/test_participation_mask_boundaries.cpp
)
set_tests_properties(unit_participation_mask_boundaries PROPERTIES LABELS "unit;barrier")

add_catch_test(unit_warp_barrier_lifecycle
    barrier/test_warp_barrier_lifecycle.cpp
)
set_tests_properties(unit_warp_barrier_lifecycle PROPERTIES LABELS "unit;barrier;lifecycle")

add_catch_test(unit_barrier_module_release
    barrier/test_barrier_module_release.cpp
)
set_tests_properties(unit_barrier_module_release PROPERTIES LABELS "unit;barrier;regression;BUG-POSTBARRIER-TWOHALVES")

add_catch_test(unit_barrier_reconvergence
    barrier/test_barrier_reconvergence.cpp
)
set_tests_properties(unit_barrier_reconvergence PROPERTIES LABELS "unit;barrier")

add_catch_test(unit_barrier_interaction
    barrier/test_barrier_interaction_integrated.cpp
)
set_tests_properties(unit_barrier_interaction PROPERTIES LABELS "unit;barrier")

add_catch_test(unit_barrier_scenarios_integrated
    barrier/test_barrier_scenarios_integrated.cpp
)
set_tests_properties(unit_barrier_scenarios_integrated PROPERTIES LABELS "unit;barrier;regression")

# P2-1 (re-enabled 2026-06-07): was disabled per KNOWN_ISSUES.md §D1.2
# (SIMTStackEntry/simt_stack not in scope). Fixed by adding the missing
# #include "ptxsim/simt_stack.h" to test_barrier_verification.cpp.
add_catch_test(unit_barrier_verification
    barrier/test_barrier_verification.cpp
)
set_tests_properties(unit_barrier_verification PROPERTIES LABELS "unit;barrier;fence")

# ============================================================================
# Barrier Regression (BUG-CUTE-RMSNORM-BROADCAST-SKIP)
# RED PHASE: Divergent warp (one lane writes sdata, the other 31 skip)
# reconverges at bar.warp.sync. All 32 lanes must be released to
# reconvergence_pc (the broadcast ld.shared PC), not reconvergence_pc+1.
# Without fix: lanes land on reconvergence_pc+1, broadcast ld.shared
# is skipped, downstream st.global writes 0 → output[0]=0 (cute_rmsnorm
# failure mode).
# See: docs/developer-guide/KNOWN_ISSUES.md §"cute_rmsnorm — broadcast-after-barrier skipped"
#      KNOWN_ISSUES.md §"Pre-P0 Baseline Red — cute_rmsnorm"
# ============================================================================
add_catch_test(unit_broadcast_after_barrier
    barrier/test_broadcast_after_barrier.cpp
)
set_tests_properties(unit_broadcast_after_barrier PROPERTIES LABELS "unit;barrier;divergence;regression;BUG-CUTE-RMSNORM-BROADCAST-SKIP")

# BUG-DISPATCH-GATE-LANE0-SKIP (hypothesis test, 2026-06-16):
# Manually injects a stale SIMTStackEntry with reconvergence_pc matching
# the lane 0 st.shared PC. Locks in correct dispatch behavior: the gate
# must NOT block the instruction. If the gate incorrectly blocks (which
# is the cute_rmsnorm hypothesis), this test fails and reproduces the
# broadcast-after-barrier skip bug at the unit level.
add_catch_test(unit_simt_stack_stale_entry_blocks_lane0
    barrier/test_simt_stack_stale_entry_blocks_lane0.cpp
)
set_tests_properties(unit_simt_stack_stale_entry_blocks_lane0 PROPERTIES
    LABELS "unit;barrier;divergence;simt;regression;BUG-DISPATCH-GATE-LANE0-SKIP")

# ============================================================================
# Barrier Regression (BUG-POSTBARRIER-TWOHALVES)
# RED PHASE: Divergent warp splits into two halves that hit the same barrier
# at different times. The second release must NOT lose the first half from
# active_mask. Without fix: active_mask becomes 0000FFFF instead of FFFFFFFF.
# ============================================================================
add_catch_test(unit_post_barrier_two_halves
    barrier/test_post_barrier_two_halves.cpp
)
set_tests_properties(unit_post_barrier_two_halves PROPERTIES LABELS "unit;barrier;divergence;regression;BUG-POSTBARRIER-TWOHALVES")

# ============================================================================
# Barrier Regression (BUG-RECONVERGENCE-SIMPLEGEMM)
# RED PHASE: simpleGEMM kernel — lanes 0-15 take a_tile/b_tile loops while
# lanes 16-31 skip them. Both halves converge at bar.warp.sync. The barrier
# must be initialized with the FULL static_mask (0xFFFFFFFF), not the
# dynamic_mask that excludes lanes still in flight in the divergent paths.
# Without fix: wbar.participation_mask = 0xFFFF0000; only 16 lanes released;
# lanes 0-15 stuck at barrier forever; warp never finishes GEMM main loop.
# ============================================================================
add_catch_test(unit_barrier_divergence_reconvergence_simplegemm
    barrier/test_barrier_divergence_reconvergence_simplegemm.cpp
)
set_tests_properties(unit_barrier_divergence_reconvergence_simplegemm PROPERTIES LABELS "unit;barrier;divergence;regression;BUG-RECONVERGENCE-SIMPLEGEMM")

# ============================================================================
# SIMT 单元测试
# ============================================================================
add_catch_test(unit_simt_stack_entry
    simt/test_simt_stack_entry.cpp
)
set_tests_properties(unit_simt_stack_entry PROPERTIES LABELS "unit;simt_entry")

add_catch_test(unit_simt_stack_entry_integrated
    simt/test_simt_stack_entry_integrated.cpp
)
set_tests_properties(unit_simt_stack_entry_integrated PROPERTIES LABELS "unit;simt_entry")

add_catch_test(unit_simt_stack_catch2
    simt/test_simt_stack_catch2.cpp
)
set_tests_properties(unit_simt_stack_catch2 PROPERTIES LABELS "unit;simt_stack;simt_entry")

add_catch_test(unit_simt_thread_pc
    simt/test_simt_thread_pc.cpp
)
set_tests_properties(unit_simt_thread_pc PROPERTIES LABELS "unit;simt_entry")

add_catch_test(unit_simt_integration
    simt/test_simt_integration.cpp
)
set_tests_properties(unit_simt_integration PROPERTIES LABELS "unit;simt")

add_catch_test(unit_handle_branch_two_level_divergence
    simt/test_handle_branch_two_level_divergence.cpp
)
set_tests_properties(unit_handle_branch_two_level_divergence PROPERTIES LABELS "unit;simt;divergence")

add_catch_test(unit_handle_branch
    simt/test_handle_branch_integration.cpp
)
set_tests_properties(unit_handle_branch PROPERTIES LABELS "unit;branch;simt_entry")

add_catch_test(unit_barrier_simt
    simt/test_barrier_simt_integration.cpp
)
set_tests_properties(unit_barrier_simt PROPERTIES LABELS "unit;barrier;simt")

# ============================================================================
# Warp 单元测试
# 注: test_warp_context.cpp, test_warp_scheduler.cpp, test_sm_context.cpp
#     是 standalone 测试 (有 main 函数), 不用 add_catch_test
# ============================================================================
add_catch_test(unit_warp_state
    warp/test_warp_state.cpp
)
set_tests_properties(unit_warp_state PROPERTIES LABELS "unit;warp;state;simt_entry")

# BUG-001: active_count staleness after WarpContext::decrement_blocked_cycles()
# drains blocked cycles. RED phase: this test MUST FAIL on unpatched code
# (active_count remains 0 → scheduler skips a schedulable warp → hang).
add_catch_test(unit_warp_blocked_decrement
    warp/test_warp_blocked_decrement.cpp
)
set_tests_properties(unit_warp_blocked_decrement PROPERTIES LABELS "unit;warp;regression;BUG-LDGLOBAL-HANG")

# ============================================================================
# Memory 单元测试
# 注: 原始 CMakeLists.txt 中 test_memory_manager.cpp 实际上未被编译
#     (legacy bug: 同一个 test_parse_immediate.cpp 被编译两次 — 一次作为
#      unit_memory_manager_legacy, 另一次作为 unit_parse_immediate)。
#     修复: 拆分为两个独立目标, 各用正确的源文件。
# ============================================================================
add_catch_test(unit_memory_manager
    memory/test_memory_manager.cpp
)
set_tests_properties(unit_memory_manager PROPERTIES LABELS "unit;memory")

add_catch_test(unit_memory_bounds
    memory/test_memory_bounds.cpp
)
set_tests_properties(unit_memory_bounds PROPERTIES LABELS "unit;memory")

# ============================================================================
# Phase 0.1 (Fix #5): TMA TensorMap descriptor parser unit tests.
# Validates 128-byte CUtensorMap opaque descriptor parsing per PTX ISA §9.7.13
# (all byte offsets UNVERIFIED-AGAINST-HARDWARE — see tma_descriptor.h).
# 18 TEST_CASEs cover happy path + dtype/swizzle/interleave/rank variants +
# invalid size/alignment/bounds + TmaDescriptorStore round-trip/isolation/
# overwrite/clear. Consumed by tcgen05.ld in Phase 1-3.
# ============================================================================
add_catch_test(unit_tma_descriptor
    memory/test_tma_descriptor.cpp
)
set_tests_properties(unit_tma_descriptor PROPERTIES LABELS "unit;memory;tma")

# ============================================================================
# Phase 0.2 (Fix #6): Blackwell per-CTA Tensor Memory (TMEM) unit tests.
# 256 slot × 128 byte = 32 KB per CTA — per PTX ISA §9.7.13.
# ≥10 TEST_CASEs cover default-zero, read/write round-trip, cross-CTA
# isolation, bounds enforcement (slot_id, size), clear, non-clobbering
# partial writes. Consumed by tcgen05.* in Phase 1-3.
# ============================================================================
add_catch_test(unit_tmem
    memory/test_tmem.cpp
)
set_tests_properties(unit_tmem PROPERTIES LABELS "unit;memory;tmem")

# ============================================================================
# Phase 0.3 (Fix #7): cluster arrive/wait unit tests.
# ≥10 TEST_CASEs cover constructor validation, single-CTA arrive/wait,
# multi-CTA blocking sync, duplicate arrive rejection,
# wait-before-arrive rejection, cross-cluster isolation.
# Consumed by cta_group::1 (Phase 1-3).
# ============================================================================
add_catch_test(unit_cluster_mode
    cluster/test_cluster_mode.cpp
)
set_tests_properties(unit_cluster_mode PROPERTIES LABELS "unit;cluster")

# wire-cluster-context-to-tcgen05: oracle test for opt-in cluster arrive/wait
add_catch_test(unit_cluster_tcgen05_integration
    cluster/test_cluster_tcgen05_integration.cpp
)
set_tests_properties(unit_cluster_tcgen05_integration PROPERTIES LABELS "unit;cluster;tcgen05")

# ============================================================================
# Phase 0.4 (Fix #8): async tensor core command queue unit tests.
# ≥14 TEST_CASEs cover default construction, commit-group monotonic counter,
# wait() lane blocking + completion_pc capture, commit() per-lane unblock,
# multi-waiter ordering, clear reset, concurrent commit thread-safety,
# and completion_pc drift prevention.
# Consumed by tcgen05.* async commit/wait in Phase 1-3.
# ============================================================================
add_catch_test(unit_tc_queue
    async/test_tc_queue.cpp
)
set_tests_properties(unit_tc_queue PROPERTIES LABELS "unit;async;tc_queue")

# ============================================================================
# Memory Regression Tests (BUG-ST-OVERREAD, BUG-NAME2SHARE)
# RED PHASE: These tests must FAIL on unpatched code.
# ============================================================================
add_catch_test(unit_st_handler_data_size
    memory/test_st_handler_data_size.cpp
)
set_tests_properties(unit_st_handler_data_size PROPERTIES LABELS "unit;memory;regression;BUG-ST-OVERREAD")

add_catch_test(unit_get_memory_addr_shared
    memory/test_get_memory_addr_shared.cpp
)
set_tests_properties(unit_get_memory_addr_shared PROPERTIES LABELS "unit;memory;regression;BUG-NAME2SHARE")

# BUG-VECOP-STALE: vecOp_phy_addrs per-ThreadContext stack regression
# RED PHASE: On the unpatched (FIFO) code, two consecutive VEC acquires
# alias the same buffer; on the fixed code, each acquire pushes a fresh
# entry and the returned pointer references only the just-pushed one.
add_catch_test(unit_vec_op_phy_addrs_stack
    memory/test_vec_op_phy_addrs_stack.cpp
)
set_tests_properties(unit_vec_op_phy_addrs_stack PROPERTIES LABELS "unit;memory;ptx;vec;regression;BUG-VECOP-STALE")

# ============================================================================
# Exec 单元测试
# ============================================================================
add_catch_test(unit_exec_mask
    exec/test_exec_mask.cpp
)
set_tests_properties(unit_exec_mask PROPERTIES LABELS "unit;exec_mask;simt_entry")

add_catch_test(unit_active_mask_consistency
    exec/test_active_mask_consistency.cpp
)
set_tests_properties(unit_active_mask_consistency PROPERTIES LABELS "unit;active_mask")

add_catch_test(unit_exec_integration_h1_h4
    exec/test_exec_integration_h1_h4.cpp
)
set_tests_properties(unit_exec_integration_h1_h4 PROPERTIES LABELS "unit;exec")

add_catch_test(unit_exec_layer_e1_e3
    exec/test_exec_layer_e1_e3.cpp
)
set_tests_properties(unit_exec_layer_e1_e3 PROPERTIES LABELS "unit;exec")

# ============================================================================
# Exec Regression Tests (BUG-BRAPRED)
# RED PHASE: These tests must FAIL on unpatched code.
# ============================================================================
add_catch_test(unit_bra_pred_divergent
    exec/test_bra_pred_divergent.cpp
)
set_tests_properties(unit_bra_pred_divergent PROPERTIES LABELS "unit;exec;regression;BUG-BRAPRED")

# ============================================================================
# Exec Regression Tests (BUG-RETHANG)
# RED PHASE: These tests must FAIL on unpatched code (ret on divergent warp hangs).
# ============================================================================
add_catch_test(unit_ret_handler_divergent
    exec/test_ret_handler_divergent.cpp
)
set_tests_properties(unit_ret_handler_divergent PROPERTIES LABELS "unit;exec;regression;BUG-RETHANG")

# ============================================================================
# PC 单元测试
# ============================================================================
add_catch_test(unit_pc_management
    pc/test_pc_management.cpp
)
set_tests_properties(unit_pc_management PROPERTIES LABELS "unit;pc;scheduler")

add_catch_test(unit_pc_management_advanced
    pc/test_pc_management_advanced.cpp
)
set_tests_properties(unit_pc_management_advanced PROPERTIES LABELS "unit;simt_entry")

# ============================================================================
# SM 单元测试 (BUG-SM-ADMISSION-OVERFLOW: streaming block admission)
# RED PHASE: requires get_total_block_count/get_admitted_block_count/
#            get_pending_block_count on SMContext, plus pending_blocks_
#            queue in add_block (currently silent drop on overflow).
# ============================================================================
add_catch_test(unit_streaming_admission
    sm/test_streaming_admission.cpp
)
set_tests_properties(unit_streaming_admission PROPERTIES LABELS "unit;sm;admission;streaming;regression;BUG-SM-ADMISSION-OVERFLOW")

# ============================================================================
# Sync 单元测试
# ============================================================================
add_catch_test(unit_sync_mechanism
    sync/test_sync_mechanism.cpp
)
set_tests_properties(unit_sync_mechanism PROPERTIES LABELS "unit;sync;simt_entry")

add_catch_test(unit_barrier_pc_overwrite
    sync/test_barrier_pc_overwrite.cpp
)
set_tests_properties(unit_barrier_pc_overwrite PROPERTIES LABELS "unit;barrier")

add_catch_test(unit_syncthreads_direction
    sync/test_syncthreads_direction.cpp
)
set_tests_properties(unit_syncthreads_direction PROPERTIES LABELS "unit;syncthreads")

add_catch_test(unit_syncthreads_test3_repro
    sync/test_syncthreads_test3_repro.cpp
)
set_tests_properties(unit_syncthreads_test3_repro PROPERTIES LABELS "unit;syncthreads;regression")

# ============================================================================
# Logger 单元测试
# ============================================================================
add_catch_test(unit_logger_cudart_component
    logger/test_logger_cudart_component.cpp
)
set_tests_properties(unit_logger_cudart_component PROPERTIES LABELS "unit;logger;cudart")

# ============================================================================
# Common 单元测试
# ============================================================================
# P2-3 (re-enabled 2026-06-07): was disabled per KNOWN_ISSUES.md §D1.2
# (subc_handler undeclared + non-Catch2 format). Fixed by rewriting
# to Catch2 TEST_CASE format with corrected processOperation (camelCase).
add_catch_test(unit_cc_register
    common/test_cc_register.cpp
)
set_tests_properties(unit_cc_register PROPERTIES LABELS "unit;cc;addc;subc")

add_catch_test(unit_specific_bugs_unit
    common/test_specific_bugs_unit.cpp
)
set_tests_properties(unit_specific_bugs_unit PROPERTIES LABELS "unit;regression")

add_catch_test(unit_instruction_latency_table
    common/test_instruction_latency_table.cpp
)
set_tests_properties(unit_instruction_latency_table PROPERTIES LABELS "unit;ptx;latency")

add_catch_test(unit_instruction_latency_config_load
    common/test_instruction_latency_config_load.cpp
)
set_tests_properties(unit_instruction_latency_config_load PROPERTIES LABELS "unit;ptx;latency")

add_catch_test(unit_scheduler_config
    common/test_scheduler_config.cpp
)
set_tests_properties(unit_scheduler_config PROPERTIES LABELS "unit;scheduler")

add_catch_test(unit_parse_immediate
    common/test_parse_immediate.cpp
)
set_tests_properties(unit_parse_immediate PROPERTIES LABELS "unit;parser")

add_catch_test(unit_warp_trace_formatter
    common/test_warp_trace_formatter.cpp
)
set_tests_properties(unit_warp_trace_formatter PROPERTIES LABELS "unit;warp_trace_formatter")

# ============================================================================
# Divergence 单元测试
# ============================================================================
add_catch_test(unit_divergence_sync_standalone_integrated
    divergence/test_divergence_sync_standalone_integrated.cpp
)
set_tests_properties(unit_divergence_sync_standalone_integrated PROPERTIES LABELS "unit;divergence;barrier_sync")

# ============================================================================
# PTX 指令单元测试
# ============================================================================
# T2-6 Step 1: helper extraction unit tests for CVT instruction
# Validates 4 helpers (round_half_to_even / half_to_float / float_to_half /
# should_saturate_uint32) before/after extraction from arithmetic_conversion.cpp
add_catch_test(unit_cvt_helpers
    ptx/test_cvt_helpers.cpp
)
set_tests_properties(unit_cvt_helpers PROPERTIES LABELS "unit;ptx;cvt;helpers")

# T2-6 Sub-task 3: CvtContext + build_context() tests for CVT strategy pattern
# Validates qualifier extraction (dst_bytes/src_bytes/is_float/is_half/is_signed
# and all 11 rounding/saturation flags) into CvtContext struct.
add_catch_test(unit_cvt_context
    ptx/test_cvt_context.cpp
)
set_tests_properties(unit_cvt_context PROPERTIES LABELS "unit;ptx;cvt;context")

# T2-6 Sub-task 4a: FloatToFloatStrategy unit tests
# Locks in behavior of f16/f32/f64 mutual conversions (incl. .sat NaN handling,
# half-precision round-trips) before extraction from GeneralCvtStrategy.
add_catch_test(unit_cvt_float_to_float
    ptx/test_cvt_float_to_float.cpp
)
set_tests_properties(unit_cvt_float_to_float PROPERTIES LABELS "unit;ptx;cvt;strategy;f2f")

# T2-6 Sub-task 4b: IntToFloatStrategy unit tests
# Locks in behavior of s8/s16/s32/s64/u8/u16/u32/u64 -> f16/f32/f64 conversions
add_catch_test(unit_cvt_int_to_float
    ptx/test_cvt_int_to_float.cpp
)
set_tests_properties(unit_cvt_int_to_float PROPERTIES LABELS "unit;ptx;cvt;strategy;i2f")

# T2-6 Sub-task 4c+4d: FloatToIntStrategy unit tests
# Locks in behavior of f16/f32/f64 -> s8/s16/s32/s64/u8/u16/u32/u64 conversions
# incl. .sat saturation and 5 rounding modes (.rn/.rz/.rm/.rp/.rna)
add_catch_test(unit_cvt_float_to_int
    ptx/test_cvt_float_to_int.cpp
)
set_tests_properties(unit_cvt_float_to_int PROPERTIES LABELS "unit;ptx;cvt;strategy;f2i")

# T2-6 Sub-task 4d: IntToIntStrategy unit tests
# Locks in behavior of 4x4 dimension matrix (s8/s16/s32/s64 + u8/u16/u32/u64)
# x 4 (dst_bytes) + .sat + 5 rounding modes
add_catch_test(unit_cvt_int_to_int
    ptx/test_cvt_int_to_int.cpp
)
set_tests_properties(unit_cvt_int_to_int PROPERTIES LABELS "unit;ptx;cvt;strategy;i2i")

# ============================================================================
# PTX Math Instruction Unit Tests (2026-07-01)
# Verify fma.rn.f32 / rsqrt.approx.f32 / ld.global / st.global correctness
# as part of cute_rmsnorm basic-math investigation.
# ============================================================================
add_catch_test(unit_math_fma_rn_f32
    ptx/test_fma_rn_f32.cpp
)
set_tests_properties(unit_math_fma_rn_f32 PROPERTIES LABELS "unit;ptx;math;fma")

add_catch_test(unit_math_rsqrt_approx_f32
    ptx/test_rsqrt_approx_f32.cpp
)
set_tests_properties(unit_math_rsqrt_approx_f32 PROPERTIES LABELS "unit;ptx;math;rsqrt")

add_catch_test(unit_memory_global_f32
    ptx/test_ld_st_global_f32.cpp
)
set_tests_properties(unit_memory_global_f32 PROPERTIES LABELS "unit;ptx;memory;global")

# ---------------------------------------------------------------------------
# WMMA / Tensor Core stub (c5 - replace-silent-stub-failures, Fix #1):
# WmmaHandler::processWmmaOperation now throws UnsupportedInstructionException
# instead of silently no-op'ing. Locks in the explicit-failure contract so any
# regression to silent no-op is caught at unit level.
# Real WMMA / Tensor Core implementation tracked by `implement-wmma-tensor-core`.
# ---------------------------------------------------------------------------
add_catch_test(unit_wmma_not_implemented
    ptx/test_wmma_not_implemented.cpp
)
set_tests_properties(unit_wmma_not_implemented PROPERTIES LABELS "unit;ptx;wmma;stub")

# ---------------------------------------------------------------------------
# Phase 1.1 (Fix #10): tcgen05.mma fragment arithmetic unit test.
# RED phase: this test MUST FAIL until wmma.cpp implements real fragment
# arithmetic (currently throws UnsupportedInstructionException).
# ---------------------------------------------------------------------------
# unit_tcgen05_mma_fragment — migrated to tests/unit/ptx/test_tcgen05_qualifier.cpp (ADR-0016 Phase 2)
# unit_tcgen05_ld_st — migrated to tests/unit/ptx/test_tcgen05_opkind.cpp (ADR-0016 Phase 2)

# ---------------------------------------------------------------------------
# A2 cleanup (2026-06): the 7 tests below all `<<<1, 1>>>` real GPU kernels
# and verify NVIDIA's PTX semantics — they do NOT exercise the PTX-EMU
# simulator. They were moved to tests/reference/ptx_builtin/ as a
# historical "semantic reference". Simulator-driven equivalents live in
# tests/integration/ptx/ (see integration_ptx_ld_st_shared and
# integration_ptx_integer_arith). Disabled per Oracle A2 recommendation.
# ============================================================================
# add_catch_test(unit_ptx_integer
#     ptx/test_ptx_integer.cu
#     ptx/ptx_integer_arith.cu
# )
# set_tests_properties(unit_ptx_integer PROPERTIES LABELS "unit;ptx;integer")
#
# add_catch_test(unit_ptx_float
#     ptx/test_ptx_float.cu
#     ptx/ptx_float_arith.cu
# )
# set_tests_properties(unit_ptx_float PROPERTIES LABELS "unit;ptx;float")
#
# add_catch_test(unit_ptx_extended
#     ptx/test_ptx_extended.cu
#     ptx/ptx_extended_prec.cu
# )
# set_tests_properties(unit_ptx_extended PROPERTIES LABELS "unit;ptx;integer")
#
# add_catch_test(unit_ptx_bitwise
#     ptx/test_ptx_bitwise.cu
#     ptx/ptx_bitwise_shift.cu
# )
# set_tests_properties(unit_ptx_bitwise PROPERTIES LABELS "unit;ptx;bitwise")
#
# add_catch_test(unit_ptx_cvt
#     ptx/test_ptx_cvt.cu
#     ptx/ptx_cvt_arith.cu
# )
# set_tests_properties(unit_ptx_cvt PROPERTIES LABELS "unit;ptx;cvt")
#
# add_catch_test(unit_ptx_ld_st
#     ptx/test_ptx_ld_st.cu
#     ptx/ptx_ld_st.cu
# )
# set_tests_properties(unit_ptx_ld_st PROPERTIES LABELS "unit;ptx;ld_st")
#
# add_catch_test(unit_ptx_cvta
#     ptx/test_ptx_cvta.cu
#     ptx/ptx_cvta.cu
# )
# set_tests_properties(unit_ptx_cvta PROPERTIES LABELS "unit;ptx;cvta")

# ============================================================================
# CUDA Runtime 拦截单元测试 (T1-2)
# 验证 cudaStream/cudaEvent destroy 实现正确
# 审计 §2.2.1 声称 destroy 是 STUB，但 cudart_sim.cpp:692/721 实际有 delete。
# 真实问题（Errata §1.7）: type-unsafe reinterpret_cast + fake sync。
# ============================================================================
add_catch_test(unit_cuda_stream_handle
    cudart/test_cuda_stream_handle.cu
)
set_tests_properties(unit_cuda_stream_handle PROPERTIES LABELS "unit;cudart;stream")

# ============================================================================
# half-precision utility tests (phase3-half-precision-bugfix)
# Validates f16_to_f32 / f32_to_f16 denormal/subnormal path correctness
# in include/ptxsim/utils/half_utils.h.
# ============================================================================
add_catch_test(unit_half_utils
    utils/test_half_utils.cpp
)
set_tests_properties(unit_half_utils PROPERTIES LABELS "unit;half;utils")

# ============================================================================
# half-precision cross-implementation consistency
# Sweeps all 65536 possible half inputs and asserts that half_utils.h::
# f16_to_f32 and cvt_helpers.cpp::half_to_float produce bit-identical output.
# Gate test for T2-6 Step 2 (reuse half_utils.h for cvt_helpers).
# ============================================================================
add_catch_test(unit_half_utils_consistency
    utils/test_half_utils_consistency.cpp
)
set_tests_properties(unit_half_utils_consistency PROPERTIES LABELS "unit;half;utils;consistency;cvt")

# parser-completeness spec: multi-PTX cubin 累加 + warning oracle
add_catch_test(unit_multi_ptx
    parser/test_multi_ptx.cpp
)
set_tests_properties(unit_multi_ptx PROPERTIES LABELS "unit;parser;multi_ptx")

# add-extern-function-declaration spec: extern function 双路径 oracle
add_catch_test(unit_extern_function
    parser/test_extern_function.cpp
)
set_tests_properties(unit_extern_function PROPERTIES LABELS "unit;parser;extern")

# ============================================================================
# tcgen05 dead-code coverage unit test (ADR-0016, fix-tcgen05-test-coverage-gaps)
# Validates golden value file + handler header linkage. Direct handler
# invocation test deferred to fix-tcgen05-handler-dispatch (when dispatch
# is wired, real warp execution will exercise processTcgen05Mma and we
# can compare TMEM output against the same golden values).
# ============================================================================
add_catch_test(unit_ptx_ir_tcgen05_mma_golden
    ptx_ir/test_tcgen05_mma_golden.cpp
)
set_tests_properties(unit_ptx_ir_tcgen05_mma_golden PROPERTIES
    LABELS "unit;ptx_ir;tcgen05;mma;golden")

# ============================================================================
# tcgen05 pipeline handler unit test (ADR-0016, fix-tcgen05-handler-dispatch)
# Validates Tcgen05PipelineHandler 3-stage structure (zero-operand vs
# multi-operand op_kinds). Full pipeline execution covered by
# integration_tcgen05_dispatch + future real-warp tests.
# ============================================================================
add_catch_test(unit_ptx_ir_tcgen05_pipeline_handler
    ptx_ir/test_tcgen05_pipeline_handler.cpp
)
set_tests_properties(unit_ptx_ir_tcgen05_pipeline_handler PROPERTIES
    LABELS "unit;ptx_ir;tcgen05;pipeline")
