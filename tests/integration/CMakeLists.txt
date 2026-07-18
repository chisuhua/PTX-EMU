#tests / integration / CMakeLists.txt
#Type 2 : 指令序列集成测试
#使用 StatementContext + execute_warp_instruction() 验证指令执行

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared) set(CMAKE_CUDA_PTX_COMPILATION ON) set(
    CMAKE_CUDA_SEPARABLE_COMPILATION
        OFF) set(CMAKE_CUDA_ARCHITECTURES
                 "100") set(CMAKE_CUDA_FLAGS
                            "${CMAKE_CUDA_FLAGS} -keep --no-compress")

    set(CATCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR} / tests)

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#Barrier 集成测试
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
        add_catch_test(integration_barrier_module barrier /
                       test_barrier_module_integrated
                           .cpp) set_tests_properties(integration_barrier_module
                                                          PROPERTIES LABELS
                                                      "integration;barrier")

#integration_barrier_scenarios moved to unit /                                 \
    (manual SIMT stack push violates Principle 5)

            add_catch_test(
                integration_barrier_verification barrier /
                test_barrier_verification_integrated
                    .cpp) set_tests_properties(integration_barrier_verification
                                                   PROPERTIES LABELS
                                               "integration;barrier")

                add_catch_test(
                    integration_barrier_divergence_scheduling barrier /
                    test_barrier_divergence_scheduling.cpp)
                    set_tests_properties(
                        integration_barrier_divergence_scheduling
                            PROPERTIES LABELS "integration;barrier;divergence")

                        add_catch_test(integration_warp_barrier barrier /
                                       test_warp_barrier_integrated.cpp)
                            set_tests_properties(
                                integration_warp_barrier PROPERTIES LABELS
                                "integration;wbar;regression")

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#P1 - 3 : Tier 8 cross - component integration test
#(added 2026 - 06 - 07 per docs / superpowers / specs / 2026 - 06 - 07 - ptx - \
  emu - tier8 - design.md)
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
                                add_catch_test(
                                    integration_barrier_full_lifecycle barrier /
                                    test_barrier_full_lifecycle.cpp)
                                    set_tests_properties(
                                        integration_barrier_full_lifecycle
                                            PROPERTIES LABELS
                                        "integration;barrier;lifecycle;tier8")

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#integration_warp_barrier_memory_visibility was disabled because the test
#used lane_id directly as the offset for.b32 shared - memory accesses.PTX
#[symbol + register] addresses are byte offsets, so element N is at N * 4; the
#mismatch caused byte - replicated writes / reads and failed 236 assertions.
#Fixed by presetting r1 = lane_id * 4 and using r5 for the lane - id predicate.
#Re - enabled below.
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
                                        add_catch_test(
                                            integration_warp_barrier_memory_visibility
                                                barrier /
                                            test_warp_barrier_memory_visibility
                                                .cpp)
                                            set_tests_properties(
                                                integration_warp_barrier_memory_visibility
                                                    PROPERTIES LABELS
                                                "integration;barrier;memory_"
                                                "visibility")

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#integration_cta_barrier_memory_visibility is the CTA - level(S_BAR bar.sync 0)
#sister of integration_warp_barrier_memory_visibility.It was disabled while
#the S_BAR deadlock was under investigation;                                   \
    the byte - offset PTX shared - memory
#pattern is now correct(see test source).Re - enabled for Fix 2 validation.
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
                                                add_catch_test(
                                                    integration_cta_barrier_memory_visibility
                                                        barrier /
                                                    test_cta_barrier_memory_visibility
                                                        .cpp)
                                                    set_tests_properties(
                                                        integration_cta_barrier_memory_visibility
                                                            PROPERTIES LABELS
                                                        "integration;barrier;"
                                                        "memory_visibility")

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#SIMT 集成测试
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#integration_simt_stack_entry moved to unit /                                  \
    (manual SIMT stack push violates Principle 5)

                                                        add_catch_test(
                                                            integration_simt_thread_pc
                                                                simt /
                                                            test_simt_thread_pc_integrated
                                                                .cpp)
                                                            set_tests_properties(
                                                                integration_simt_thread_pc
                                                                    PROPERTIES LABELS
                                                                "integration;"
                                                                "thread_pc")

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#Divergence 集成测试
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
                                                                add_catch_test(
                                                                    integration_divergence_sync_convergence
                                                                        divergence /
                                                                    test_divergence_sync_convergence
                                                                        .cpp)
                                                                    set_tests_properties(
                                                                        integration_divergence_sync_convergence
                                                                            PROPERTIES LABELS
                                                                        "integr"
                                                                        "ation;"
                                                                        "diverg"
                                                                        "ence;"
                                                                        "conver"
                                                                        "gence")

#integration_divergence_sync_standalone moved to unit /                        \
    (advance_all_to_pc violates Principle 5)

                                                                        add_catch_test(
                                                                            integration_nested_divergence
                                                                                divergence /
                                                                            test_nested_divergence
                                                                                .cpp)
                                                                            set_tests_properties(
                                                                                integration_nested_divergence
                                                                                    PROPERTIES LABELS
                                                                                "integration;divergence;nested")

                                                                                add_catch_test(
                                                                                    integration_post_barrier_divergence
                                                                                        divergence /
                                                                                    test_post_barrier_divergence
                                                                                        .cpp)
                                                                                    set_tests_properties(
                                                                                        integration_post_barrier_divergence
                                                                                            PROPERTIES LABELS
                                                                                        "integration;divergence;barrier")

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#Barrier Regression(BUG - CUTE - RMSNORM - BROADCAST - SKIP)
#RED PHASE : Divergent warp pattern from cute_rmsnorm.cu broadcast loop.
#lane 0 takes the @p1 path(writes a value), lanes 1 - 31 fall through
#to the reconvergence label.All 32 lanes must execute the
#broadcast `ld.shared` at reconvergence_pc.
#Without fix : lanes 1 - 31 advance from the barrier PC straight to ret,
#skipping the broadcast read.cute_rmsnorm then writes 0 to output[0].
#See : docs / developer -                                                      \
    guide / KNOWN_ISSUES.md §"cute_rmsnorm — broadcast-after-barrier skipped"
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
                                                                                        add_catch_test(
                                                                                            integration_broadcast_after_barrier
                                                                                                divergence /
                                                                                            test_broadcast_after_barrier
                                                                                                .cpp)
                                                                                            set_tests_properties(
                                                                                                integration_broadcast_after_barrier
                                                                                                    PROPERTIES LABELS
                                                                                                "integration;barrier;divergence;regression;BUG-CUTE-RMSNORM-BROADCAST-SKIP")

#BUG - DISPATCH - GATE - LANE0 - SKIP(2026 - 06 - 16):
#cute_rmsnorm.ptx uses `bar.sync 0` (S_BAR, CTA - level barrier), NOT
# `bar.warp.sync` (S_BAR_WARP_SYNC, warp - level).The existing I - 1 / I - 2
#tests use bar.warp.sync which is a DIFFERENT code path.This test
#reproduces the cute_rmsnorm S_BAR(bar.sync 0) reduction + broadcast
#pattern to verify lane 0's st.shared at the lane - 0 - only path is
#actually dispatched.Without fix : lane 0's st.shared is skipped
#because the dispatch gate sees a stale SIMT stack entry from the
#reduction loop and blocks lane 0 at the reconvergence PC.
                                                                                                add_catch_test(
                                                                                                    integration_cute_rmsnorm_bar_sync_pattern
                                                                                                        divergence /
                                                                                                    test_cute_rmsnorm_bar_sync_pattern
                                                                                                        .cpp)
                                                                                                    set_tests_properties(
                                                                                                        integration_cute_rmsnorm_bar_sync_pattern
                                                                                                            PROPERTIES LABELS
                                                                                                        "integration;barrier;divergence;s_bar;regression;BUG-DISPATCH-GATE-LANE0-SKIP")

#临时禁用(2026 - 06 - 30) : baseline red,                                      \
    待 Fix 2(open - fix - 2 - sbar - deadlock) 解决 S_BAR 路径后重新启用。
#详见 docs / developer - guide / open - fix - 2 - sbar - deadlock.md
#set_tests_properties(                                                         \
    integration_cute_rmsnorm_bar_sync_pattern PROPERTIES DISABLED TRUE)

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#Barrier Regression(BUG - POSTBARRIER - TWOHALVES)
#RED PHASE : Divergent warp splits into two halves that hit the same
#barrier at different times.The second release must NOT lose the first
#half from active_mask.Without fix : warp never finishes(first half
#lanes are stuck, scheduler skips them every step).
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
                                                                                                        add_catch_test(
                                                                                                            integration_post_barrier_two_halves
                                                                                                                divergence /
                                                                                                            test_post_barrier_two_halves
                                                                                                                .cpp)
                                                                                                            set_tests_properties(
                                                                                                                integration_post_barrier_two_halves
                                                                                                                    PROPERTIES LABELS
                                                                                                                "integration;divergence;barrier;regression;BUG-POSTBARRIER-TWOHALVES")

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#Barrier Regression(BUG - RECONVERGENCE - SIMPLEGEMM)
#RED PHASE : simpleGEMM kernel — two divergent halves converge at barrier.
#All 32 lanes must reach the post - barrier main loop.Currently, lanes 0 - 15
#get stuck at the barrier(wbar initialized with mask = 0xFFFF0000 only).
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
                                                                                                                add_catch_test(
                                                                                                                    integration_post_barrier_reconvergence_simplegemm
                                                                                                                        divergence /
                                                                                                                    test_post_barrier_reconvergence_simplegemm
                                                                                                                        .cpp)
                                                                                                                    set_tests_properties(
                                                                                                                        integration_post_barrier_reconvergence_simplegemm PROPERTIES LABELS "integration;divergence;barrier;regression;BUG-RECONVERGENCE-SIMPLEGEMM")

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#Exec 集成测试
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
                                                                                                                        add_catch_test(
                                                                                                                            integration_warp_state
                                                                                                                                exec /
                                                                                                                            test_warp_state_integrated
                                                                                                                                .cpp)
                                                                                                                            set_tests_properties(
                                                                                                                                integration_warp_state
                                                                                                                                    PROPERTIES LABELS
                                                                                                                                "integration;warp_state")

                                                                                                                                add_catch_test(
                                                                                                                                    integration_ptx_lane_verification
                                                                                                                                        exec /
                                                                                                                                    test_ptx_lane_verification
                                                                                                                                        .cpp)
                                                                                                                                    target_compile_definitions(
                                                                                                                                        integration_ptx_lane_verification
                                                                                                                                            PRIVATE TEST_SOURCE_DIR =
                                                                                                                                                "${CMAKE_SOURCE_DIR}")
                                                                                                                                        set_tests_properties(
                                                                                                                                            integration_ptx_lane_verification
                                                                                                                                                PROPERTIES LABELS
                                                                                                                                            "integration;ptx_lane_verification")

#fix - ldglobal - active - count -                                             \
    hang : regression test for ld.global hang(issue fix - ldglobal - active -  \
                                              count - hang)
#Drives a minimal ld.global.u32 + st.global.u32 sequence through step_warp with
#simulated SM ticks, verifying the warp completes within a bounded cycle count.
                                                                                                                                            add_catch_test(
                                                                                                                                                integration_ldglobal_no_hang
                                                                                                                                                    exec /
                                                                                                                                                test_ldglobal_no_hang
                                                                                                                                                    .cpp) set_tests_properties(integration_ldglobal_no_hang PROPERTIES LABELS "integration;exec;ldglobal;hang_regression")

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#PC 集成测试
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
                                                                                                                                                add_catch_test(
                                                                                                                                                    integration_pc_management
                                                                                                                                                        pc /
                                                                                                                                                    test_pc_management_integrated
                                                                                                                                                        .cpp)
                                                                                                                                                    set_tests_properties(
                                                                                                                                                        integration_pc_management
                                                                                                                                                            PROPERTIES LABELS
                                                                                                                                                        "integration;pc")

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#Sync 集成测试
#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
                                                                                                                                                        add_catch_test(
                                                                                                                                                            integration_sync_mechanism
                                                                                                                                                                sync /
                                                                                                                                                            test_sync_mechanism_integrated
                                                                                                                                                                .cpp)
                                                                                                                                                            set_tests_properties(
                                                                                                                                                                integration_sync_mechanism
                                                                                                                                                                    PROPERTIES LABELS
                                                                                                                                                                "integration;sync")

                                                                                                                                                                add_catch_test(
                                                                                                                                                                    integration_syncthreads_full_pipeline
                                                                                                                                                                        sync /
                                                                                                                                                                    test_syncthreads_full_pipeline
                                                                                                                                                                        .cpp) set_tests_properties(integration_syncthreads_full_pipeline
                                                                                                                                                                                                       PROPERTIES LABELS
                                                                                                                                                                                                   "integration;syncthreads")

#== == == == == == == == == == == == == == == == == == == == == == == == == == \
    == == == == == == == == == == == ==
#PTX 指令集成测试 — 驱动模拟器（非真 GPU）
#A2 plan : rewrite tests / unit / ptx/* to drive SMContext+WarpContext+step_warp
# instead of running real GPU. Start with the simplest ld/st round-trip.
# ============================================================================
add_catch_test(integration_ptx_ld_st_shared
    ptx/test_ld_st_shared.cpp
)
set_tests_properties(integration_ptx_ld_st_shared PROPERTIES LABELS "integration;ptx;ld_st;shared;shared_memory")

add_catch_test(integration_ptx_ld_st_latency_table
    ptx/test_ld_st_latency_table.cpp
)
set_tests_properties(integration_ptx_ld_st_latency_table PROPERTIES LABELS "integration;ptx;ld_st;latency")

# BUG-VECOP-STALE: instruction sequence (VEC-source mov.b64 followed by
# ld/st.shared.v4) round-trips correctly. On unpatched code the VEC
# pushed by the mov would not be popped, and the next V4 LD/ST would
# pop that 2-element stale entry, dereferencing past its tail.
add_catch_test(integration_ptx_vec_ld_st_isolation
    ptx/test_vec_ld_st_isolation.cpp
)
set_tests_properties(integration_ptx_vec_ld_st_isolation PROPERTIES LABELS "integration;ptx;ld_st;vec;regression;BUG-VECOP-STALE")

add_catch_test(integration_ptx_integer_arith
    ptx/test_integer_arith.cpp
)
set_tests_properties(integration_ptx_integer_arith PROPERTIES LABELS "integration;ptx;integer;add;sub;mul")

# ============================================================================
# P1-4: Tier 3 simulator-driven equivalent tests
# (added 2026-06-06 per docs/superpowers/specs/2026-06-06-ptx-emu-tier3-ptx-tests-design.md)
# ============================================================================
add_catch_test(integration_ptx_bitwise
    ptx/test_bitwise.cpp
)
set_tests_properties(integration_ptx_bitwise PROPERTIES LABELS "integration;ptx;bitwise")

add_catch_test(integration_ptx_atom_global_add
    ptx/test_atom_add.cpp
)
set_tests_properties(integration_ptx_atom_global_add PROPERTIES LABELS "integration;ptx;atom")

add_catch_test(integration_ptx_atom_global_exch
    ptx/test_atom_exch.cpp
)
set_tests_properties(integration_ptx_atom_global_exch PROPERTIES LABELS "integration;ptx;atom;exch")

# Phase 1 of implement-atomic-cas-and-true-atomicity change
add_catch_test(integration_ptx_atom_global_cas
    ptx/test_atom_global_cas.cpp
)
set_tests_properties(integration_ptx_atom_global_cas PROPERTIES LABELS "integration;ptx;atom;cas")

# Phase 2/3 multi-warp CAS verification
add_catch_test(integration_ptx_atom_global_cas_multiwarp
    ptx/test_atom_global_cas_multiwarp.cpp
)
set_tests_properties(integration_ptx_atom_global_cas_multiwarp PROPERTIES LABELS "integration;ptx;atom;cas;multiwarp")

add_catch_test(integration_ptx_cvt
    ptx/test_cvt.cpp
)
set_tests_properties(integration_ptx_cvt PROPERTIES LABELS "integration;ptx;cvt")

add_catch_test(integration_ptx_cvt_edge_cases
    ptx/test_cvt_edge_cases.cpp
)
set_tests_properties(integration_ptx_cvt_edge_cases PROPERTIES
    LABELS "integration;ptx;cvt;edge_case")

# T2-6 Sub-task 5: 94 new integration tests across 5 files
# (cvt_int_to_int + cvt_int_to_float + cvt_float_to_int + cvt_float_to_float + cvt_saturation)
add_catch_test(integration_ptx_cvt_int_to_int
    ptx/test_cvt_int_to_int.cpp
)
set_tests_properties(integration_ptx_cvt_int_to_int PROPERTIES
    LABELS "integration;ptx;cvt;i2i")

add_catch_test(integration_ptx_cvt_int_to_float
    ptx/test_cvt_int_to_float.cpp
)
set_tests_properties(integration_ptx_cvt_int_to_float PROPERTIES
    LABELS "integration;ptx;cvt;i2f")

add_catch_test(integration_ptx_cvt_float_to_int
    ptx/test_cvt_float_to_int.cpp
)
set_tests_properties(integration_ptx_cvt_float_to_int PROPERTIES
    LABELS "integration;ptx;cvt;f2i")

add_catch_test(integration_ptx_cvt_float_to_float
    ptx/test_cvt_float_to_float.cpp
)
set_tests_properties(integration_ptx_cvt_float_to_float PROPERTIES
    LABELS "integration;ptx;cvt;f2f")

add_catch_test(integration_ptx_cvt_saturation
    ptx/test_cvt_saturation.cpp
)
set_tests_properties(integration_ptx_cvt_saturation PROPERTIES
    LABELS "integration;ptx;cvt;sat")

add_catch_test(integration_ptx_float_arith
    ptx/test_float_arith.cpp
)
set_tests_properties(integration_ptx_float_arith PROPERTIES LABELS "integration;ptx;float;fadd;fsub;fmul;fdiv;ffma")

add_catch_test(integration_ptx_extended
    ptx/test_extended.cpp
)
set_tests_properties(integration_ptx_extended PROPERTIES LABELS "integration;ptx;extended;addc;subc;mad;mul24")

add_catch_test(integration_ptx_cvta
    ptx/test_cvta.cpp
)
set_tests_properties(integration_ptx_cvta PROPERTIES LABELS "integration;ptx;cvta")

# ============================================================================
# tcgen05 parse → IR integration tests (ADR-0016, fix-tcgen05-test-coverage-gaps)
# Verifies makeTcgen05Instr constructs Tcgen05Instr with correct op_kind,
# qualifiers, and operands count for each of 5 core tcgen05 sub-ops.
# ============================================================================
add_catch_test(integration_ptx_tcgen05_mma_parse
    ptx/test_tcgen05_mma_parse.cpp
)
set_tests_properties(integration_ptx_tcgen05_mma_parse PROPERTIES
    LABELS "integration;ptx;tcgen05;parse;mma")

add_catch_test(integration_ptx_tcgen05_ld_parse
    ptx/test_tcgen05_ld_parse.cpp
)
set_tests_properties(integration_ptx_tcgen05_ld_parse PROPERTIES
    LABELS "integration;ptx;tcgen05;parse;ld")

add_catch_test(integration_ptx_tcgen05_st_parse
    ptx/test_tcgen05_st_parse.cpp
)
set_tests_properties(integration_ptx_tcgen05_st_parse PROPERTIES
    LABELS "integration;ptx;tcgen05;parse;st")

add_catch_test(integration_ptx_tcgen05_commit_parse
    ptx/test_tcgen05_commit_parse.cpp
)
set_tests_properties(integration_ptx_tcgen05_commit_parse PROPERTIES
    LABELS "integration;ptx;tcgen05;parse;commit")

add_catch_test(integration_ptx_tcgen05_wait_parse
    ptx/test_tcgen05_wait_parse.cpp
)
set_tests_properties(integration_ptx_tcgen05_wait_parse PROPERTIES
    LABELS "integration;ptx;tcgen05;parse;wait")

# ============================================================================
# tcgen05 dispatch integration tests (ADR-0016, fix-tcgen05-handler-dispatch)
# Verifies InstructionFactory::initialize() registers non-null handlers for
# all 11 S_TCGEN05_* enum values (per X-Macro in instruction_factory.cpp).
# ============================================================================
add_catch_test(integration_tcgen05_dispatch
    tcgen05/test_tcgen05_dispatch.cpp
)
set_tests_properties(integration_tcgen05_dispatch PROPERTIES
    LABELS "integration;tcgen05;dispatch")

# ============================================================================
# Phase 1 of implement-tcgen05-handlers-extended (ADR-0016, Oracle 2026-07-09):
# integration tests for the 3 alloc-family handlers (alloc/dealloc/relinquish).
# 11 TEST_CASEs exercise handler-level state transitions, cta_group::2
# exception path, permit semantics, and a full alloc→dealloc→relinquish
# workflow.
# ============================================================================
add_catch_test(integration_tcgen05_alloc_dealloc_relinquish
    tcgen05/test_alloc_dealloc_relinquish.cpp
)
set_tests_properties(integration_tcgen05_alloc_dealloc_relinquish PROPERTIES
    LABELS "integration;tcgen05;alloc;dealloc;relinquish;handler")

# ============================================================================
# Phase 2 of tcgen05-cp-test-coverage-and-exception-cleanup:
# integration tests for tcgen05.cp SMEM → TMEM copy.
# 3 TEST_CASEs exercise the 128-byte copy happy path (offset 0 + non-zero
# offset) and the out-of-bounds exception path. Helper-level coverage
# lives in tests/unit/tcgen05/test_tcgen05_cp.cpp.
# ============================================================================
add_catch_test(integration_tcgen05_cp
    tcgen05/test_tcgen05_cp.cpp
)
set_tests_properties(integration_tcgen05_cp PROPERTIES
    LABELS "integration;tcgen05;cp;handler")

# ============================================================================
# Phase 3 of implement-tcgen05-handlers-extended (Oracle 2026-07-08 A-path):
# integration tests for tcgen05.mma.ws qualifier-based routing inside
# processTcgen05Mma. 3 TEST_CASEs: ws+f16+cta_group golden C verification,
# dispatch path (op_kind=MMA_WS routes to mma), Q3-A scope violation path.
# Helper-level coverage lives in tests/unit/tcgen05/test_tcgen05_mma_ws.cpp.
# ============================================================================
add_catch_test(integration_tcgen05_mma_ws
    tcgen05/test_tcgen05_mma_ws.cpp
)
set_tests_properties(integration_tcgen05_mma_ws PROPERTIES
    LABELS "integration;tcgen05;mma_ws;golden")

# ============================================================================
# Multi-op TMEM persistence test (Phase A Step 3 of Oracle 2026-07-10
# FlashAttention-readiness audit). Validates Oracle H5 hypothesis:
# TMEM state survives across mma/cp/mma chained operations.
# T1: mma × 2 (verifies H1 overwrite-vs-accumulate)
# T2: mma + cp (slot isolation between helper outputs)
# T3: mma + cp + mma (pipeline chained ops)
# ============================================================================
add_catch_test(integration_tcgen05_mma_persistence
    tcgen05/test_tcgen05_mma_persistence.cpp
)
set_tests_properties(integration_tcgen05_mma_persistence PROPERTIES
    LABELS "integration;tcgen05;mma;cp;persistence;chain")

# ============================================================================
# Phase 1 (H1) B2 hardening: mma → commit → wait → mma sequence.
# Guards against TcQueue pipeline corruption and verifies that the 2nd mma
# after wait produces correct C output (overwrite in Phase 1; accumulate
# path wired via H5 hook in Phases 3-4).
# ============================================================================
add_catch_test(integration_tcgen05_mma_commit_wait_sequence
    tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp
)
set_tests_properties(integration_tcgen05_mma_commit_wait_sequence PROPERTIES
    LABELS "integration;tcgen05;mma;commit;wait;sequence")

# ============================================================================
# FU-1 Oracle C3 fix: commit/wait cta_group routing integration test.
# ============================================================================
add_catch_test(integration_tcgen05_commit_wait_group
    tcgen05/test_tcgen05_commit_wait_group.cpp
)
set_tests_properties(integration_tcgen05_commit_wait_group PROPERTIES
    LABELS "integration;tcgen05;commit;wait;cta_group;FU-1")

# ============================================================================
# Phase 4 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q5-C / Q6-B):
# Integration tests for processTcgen05Fence no-op marker. Forward paths,
# multi-warp independence, alloc/fence/dealloc interleave, cta_group::2 path.
# Unit-level coverage lives in tests/unit/ptx_ir/test_tcgen05_extended_opkind.cpp.
# ============================================================================
add_catch_test(integration_tcgen05_extended_parse
    tcgen05/test_tcgen05_extended_parse.cpp
)
set_tests_properties(integration_tcgen05_extended_parse PROPERTIES
    LABELS "integration;tcgen05;fence;handler;audit")

# ============================================================================
# FU-3 Oracle C2 fix: implicit per-warp slot cursor for ld/st/cp.
# 7 TEST_CASEs: cursor API (3), ld slot routing (1), st readback (1),
# ld→st round-trip (1), cp slot routing (1).
# ============================================================================
add_catch_test(integration_tcgen05_ld_st_slot_routing
    tcgen05/test_tcgen05_ld_st_slot_routing.cpp
)
set_tests_properties(integration_tcgen05_ld_st_slot_routing PROPERTIES
    LABELS "integration;tcgen05;ld;st;cp;slot;FU-3")

# ============================================================================
# FU-4 Oracle C4 fix: multi-warp fragment isolation.
# 5 TEST_CASEs: warp_id=0 backward compat, warp 0, warp 1, 2-warp parallel
# no-conflict, warp_id=-1 exception.
# ============================================================================
add_catch_test(integration_tcgen05_mma_multi_warp
    tcgen05/test_tcgen05_mma_multi_warp.cpp
)
set_tests_properties(integration_tcgen05_mma_multi_warp PROPERTIES
    LABELS "integration;tcgen05;mma;multi_warp;FU-4")

# ============================================================================
# FA-B6 hardener: end-to-end mma → ld → st → mma chain with f32 C persistence.
# 3 TEST_CASEs: full chain (TC1), f32 storage format survives TMEM ops (TC2),
# two consecutive chains remain independent (TC3 regression guard).
# ============================================================================
add_catch_test(integration_tcgen05_mma_ld_st_chain
    tcgen05/test_tcgen05_mma_ld_st_chain.cpp
)
set_tests_properties(integration_tcgen05_mma_ld_st_chain PROPERTIES
    LABELS "integration;tcgen05;mma;chain;ld;st;B6;flashattention")

# =============================================================================
# FU-5: FlashAttention coverage tests (tcgen05-flashattention-coverage)
# =============================================================================

# Phase 1: K=128 mma accumulation test (Oracle B1 K-loop coverage)
add_catch_test(integration_tcgen05_mma_k_loop_128
    tcgen05/test_tcgen05_mma_k_loop_128.cpp
)
set_tests_properties(integration_tcgen05_mma_k_loop_128 PROPERTIES
    LABELS "integration;tcgen05;mma;flashattention;k-loop;FU-5;B1")

# Phase 3: cp→mma data flow test (Oracle B4 coverage)
add_catch_test(integration_tcgen05_mma_cp_data_flow
    tcgen05/test_tcgen05_mma_cp_data_flow.cpp
)
set_tests_properties(integration_tcgen05_mma_cp_data_flow PROPERTIES
    LABELS "integration;tcgen05;cp;mma;flashattention;data-flow;FU-5;B4")

# ============================================================================
# Memory 集成测试 — 驱动模拟器
# B1 plan: shared memory layout, multi-buffer offsets, zero-init verification
# ============================================================================
add_catch_test(integration_shared_memory_layout
    memory/test_shared_memory_layout.cpp
)
set_tests_properties(integration_shared_memory_layout PROPERTIES LABELS "integration;memory;shared;layout;shared_memory")

# Test re-enabled: Q_LOCAL branch restored in thread_context.cpp:518-539.
# Verifies ld.local / st.local round-trip on the simulator.
add_catch_test(integration_local_memory
memory/test_local_memory.cpp
)
set_tests_properties(integration_local_memory
PROPERTIES LABELS "integration;memory;local;ld_st")

# ============================================================================
# Dynamic Shared Memory Integration Test
# ============================================================================
add_catch_test(integration_dynamic_shared_memory
    memory/test_dynamic_shared_memory.cpp
)
set_tests_properties(integration_dynamic_shared_memory PROPERTIES LABELS "integration;memory;shared_memory;dynamic")

# ============================================================================
# Phase 0.5.1 (Fix #9a): TMA descriptor store integration with CTAContext
# ============================================================================
add_catch_test(integration_tma_with_cta_context
    tma/test_tma_with_cta_context.cpp
)
set_tests_properties(integration_tma_with_cta_context PROPERTIES LABELS "integration;tma;cta")

# ============================================================================
# Phase 0.5.2 (Fix #9b): TMEM integration with CTAContext
# ============================================================================
add_catch_test(integration_tmem_with_cta_context
    tmem/test_tmem_with_cta_context.cpp
)
set_tests_properties(integration_tmem_with_cta_context PROPERTIES LABELS "integration;tmem;cta")

# ============================================================================
# Phase 0.5.3 (Fix #9c): Cluster integration with CTAContext
# ============================================================================
add_catch_test(integration_cluster_with_cta_context
    cluster/test_cluster_with_cta_context.cpp
)
set_tests_properties(integration_cluster_with_cta_context PROPERTIES LABELS "integration;cluster;cta")

# ============================================================================
# Phase 0.5.4 (Fix #9d): TcQueue integration with CTAContext
# ============================================================================
add_catch_test(integration_tc_queue_with_cta_context
    async/test_tc_queue_with_cta_context.cpp
)
set_tests_properties(integration_tc_queue_with_cta_context PROPERTIES LABELS "integration;async;tc_queue;cta")

# ============================================================================
# Phase 1.2 (Fix #11): tcgen05.mma integration test
# ============================================================================
# integration_tcgen05_mma_sync — migrated to tests/integration/tcgen05/test_tcgen05_mma_parse.cpp (ADR-0016 Phase 2)
# integration_tcgen05_ld_st_commit — migrated to tests/integration/tcgen05/test_tcgen05_ld_parse.cpp (ADR-0016 Phase 2)
# ============================================================================

# ============================================================================
# CppTLM Bridge 集成测试 (cpptlm-d1-full change, ADR-0021)
# ============================================================================
add_catch_test(integration_cpptlm_singleton_guard
    cpptlm/test_singleton_guard.cpp
)
set_tests_properties(integration_cpptlm_singleton_guard PROPERTIES LABELS "integration;cpptlm;singleton")

add_catch_test(integration_cpptlm_async_launchkernel
    cpptlm/test_async_launchkernel.cpp
)
set_tests_properties(integration_cpptlm_async_launchkernel PROPERTIES LABELS "integration;cpptlm;async")

add_catch_test(integration_cpptlm_ld_st_bridge
    cpptlm/test_ld_st_bridge.cpp
)
set_tests_properties(integration_cpptlm_ld_st_bridge PROPERTIES LABELS "integration;cpptlm;ld_st")

# PTX-7b: Scoreboard allocation integration tests (cpptlm-phase8b-injection-points, ADR-0020)
add_catch_test(integration_scoreboard_allocation
    cpptlm/test_scoreboard_allocation.cpp
)
set_tests_properties(integration_scoreboard_allocation PROPERTIES LABELS "integration;cpptlm;scoreboard")
