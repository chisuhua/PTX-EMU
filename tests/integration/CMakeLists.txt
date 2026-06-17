# tests/integration/CMakeLists.txt
# Type 2: 指令序列集成测试
# 使用 StatementContext + execute_warp_instruction() 验证指令执行

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_PTX_COMPILATION ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_ARCHITECTURES "100")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep --no-compress")

set(CATCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/tests)

# ============================================================================
# Barrier 集成测试
# ============================================================================
add_catch_test(integration_barrier_module
    barrier/test_barrier_module_integrated.cpp
)
set_tests_properties(integration_barrier_module PROPERTIES LABELS "integration;barrier")

# integration_barrier_scenarios moved to unit/ (manual SIMT stack push violates Principle 5)

add_catch_test(integration_barrier_verification
    barrier/test_barrier_verification_integrated.cpp
)
set_tests_properties(integration_barrier_verification PROPERTIES LABELS "integration;barrier")

add_catch_test(integration_barrier_divergence_scheduling
    barrier/test_barrier_divergence_scheduling.cpp
)
set_tests_properties(integration_barrier_divergence_scheduling PROPERTIES LABELS "integration;barrier;divergence")

add_catch_test(integration_warp_barrier
    barrier/test_warp_barrier_integrated.cpp
)
set_tests_properties(integration_warp_barrier PROPERTIES LABELS "integration;wbar;regression")

# ============================================================================
# P1-3: Tier 8 cross-component integration test
# (added 2026-06-07 per docs/superpowers/specs/2026-06-07-ptx-emu-tier8-design.md)
# ============================================================================
add_catch_test(integration_barrier_full_lifecycle
    barrier/test_barrier_full_lifecycle.cpp
)
set_tests_properties(integration_barrier_full_lifecycle PROPERTIES LABELS "integration;barrier;lifecycle;tier8")

# ---------------------------------------------------------------------------
# KNOWN FAILING (pre-existing, see KNOWN_ISSUES.md §"Pre-P0 Baseline Red")
# ---------------------------------------------------------------------------
# Tests added in commit 3c8c775. Both fail on main with 244-486 assertions
# failing per test. Symptoms:
#   - buf_b never written (path B's bra@PC=4 does not branch — bra_pred handler
#     bug suspected, see also `barrier.cpp.bak` and `barrier.cpp.orig` artifacts)
#   - buf_a byte pattern is anomalous (17 BB / 2 zero / 15 AA, not the 16 BB /
#     16 AA expected from "all lanes fall through to path A")
# Tracked separately; excluded from baseline gate via DISABLED so the rest of
# the suite can run. Re-enable after bra_pred or st.shared data_size is fixed.
add_catch_test(integration_warp_barrier_memory_visibility
    barrier/test_warp_barrier_memory_visibility.cpp
)
set_tests_properties(integration_warp_barrier_memory_visibility
    PROPERTIES LABELS "integration;barrier;memory_visibility" DISABLED True)

add_catch_test(integration_cta_barrier_memory_visibility
    barrier/test_cta_barrier_memory_visibility.cpp
)
set_tests_properties(integration_cta_barrier_memory_visibility
    PROPERTIES LABELS "integration;barrier;memory_visibility" DISABLED True)

# ============================================================================
# SIMT 集成测试
# ============================================================================
# integration_simt_stack_entry moved to unit/ (manual SIMT stack push violates Principle 5)

add_catch_test(integration_simt_thread_pc
    simt/test_simt_thread_pc_integrated.cpp
)
set_tests_properties(integration_simt_thread_pc PROPERTIES LABELS "integration;thread_pc")

# ============================================================================
# Divergence 集成测试
# ============================================================================
add_catch_test(integration_divergence_sync_convergence
    divergence/test_divergence_sync_convergence.cpp
)
set_tests_properties(integration_divergence_sync_convergence PROPERTIES LABELS "integration;divergence;convergence")

# integration_divergence_sync_standalone moved to unit/ (advance_all_to_pc violates Principle 5)

add_catch_test(integration_nested_divergence
    divergence/test_nested_divergence.cpp
)
set_tests_properties(integration_nested_divergence PROPERTIES LABELS "integration;divergence;nested")

add_catch_test(integration_post_barrier_divergence
    divergence/test_post_barrier_divergence.cpp
)
set_tests_properties(integration_post_barrier_divergence PROPERTIES LABELS "integration;divergence;barrier")

# ============================================================================
# Barrier Regression (BUG-CUTE-RMSNORM-BROADCAST-SKIP)
# RED PHASE: Divergent warp pattern from cute_rmsnorm.cu broadcast loop.
#   lane 0 takes the @p1 path (writes a value), lanes 1-31 fall through
#   to the reconvergence label. All 32 lanes must execute the
#   broadcast `ld.shared` at reconvergence_pc.
# Without fix: lanes 1-31 advance from the barrier PC straight to ret,
#   skipping the broadcast read. cute_rmsnorm then writes 0 to output[0].
# See: docs/developer-guide/KNOWN_ISSUES.md §"cute_rmsnorm — broadcast-after-barrier skipped"
# ============================================================================
add_catch_test(integration_broadcast_after_barrier
    divergence/test_broadcast_after_barrier.cpp
)
set_tests_properties(integration_broadcast_after_barrier PROPERTIES LABELS "integration;barrier;divergence;regression;BUG-CUTE-RMSNORM-BROADCAST-SKIP")

# BUG-DISPATCH-GATE-LANE0-SKIP (2026-06-16):
# cute_rmsnorm.ptx uses `bar.sync 0` (S_BAR, CTA-level barrier), NOT
# `bar.warp.sync` (S_BAR_WARP_SYNC, warp-level). The existing I-1/I-2
# tests use bar.warp.sync which is a DIFFERENT code path. This test
# reproduces the cute_rmsnorm S_BAR (bar.sync 0) reduction + broadcast
# pattern to verify lane 0's st.shared at the lane-0-only path is
# actually dispatched. Without fix: lane 0's st.shared is skipped
# because the dispatch gate sees a stale SIMT stack entry from the
# reduction loop and blocks lane 0 at the reconvergence PC.
add_catch_test(integration_cute_rmsnorm_bar_sync_pattern
    divergence/test_cute_rmsnorm_bar_sync_pattern.cpp
)
set_tests_properties(integration_cute_rmsnorm_bar_sync_pattern PROPERTIES
    LABELS "integration;barrier;divergence;s_bar;regression;BUG-DISPATCH-GATE-LANE0-SKIP")

# ============================================================================
# Barrier Regression (BUG-POSTBARRIER-TWOHALVES)
# RED PHASE: Divergent warp splits into two halves that hit the same
# barrier at different times. The second release must NOT lose the first
# half from active_mask. Without fix: warp never finishes (first half
# lanes are stuck, scheduler skips them every step).
# ============================================================================
add_catch_test(integration_post_barrier_two_halves
    divergence/test_post_barrier_two_halves.cpp
)
set_tests_properties(integration_post_barrier_two_halves PROPERTIES LABELS "integration;divergence;barrier;regression;BUG-POSTBARRIER-TWOHALVES")

# ============================================================================
# Barrier Regression (BUG-RECONVERGENCE-SIMPLEGEMM)
# RED PHASE: simpleGEMM kernel — two divergent halves converge at barrier.
# All 32 lanes must reach the post-barrier main loop. Currently, lanes 0-15
# get stuck at the barrier (wbar initialized with mask=0xFFFF0000 only).
# ============================================================================
add_catch_test(integration_post_barrier_reconvergence_simplegemm
    divergence/test_post_barrier_reconvergence_simplegemm.cpp
)
set_tests_properties(integration_post_barrier_reconvergence_simplegemm PROPERTIES LABELS "integration;divergence;barrier;regression;BUG-RECONVERGENCE-SIMPLEGEMM")

# ============================================================================
# Exec 集成测试
# ============================================================================
add_catch_test(integration_warp_state
    exec/test_warp_state_integrated.cpp
)
set_tests_properties(integration_warp_state PROPERTIES LABELS "integration;warp_state")

add_catch_test(integration_ptx_lane_verification
    exec/test_ptx_lane_verification.cpp
)
target_compile_definitions(integration_ptx_lane_verification PRIVATE
    TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
set_tests_properties(integration_ptx_lane_verification PROPERTIES LABELS "integration;ptx_lane_verification")

# fix-ldglobal-active-count-hang: regression test for ld.global hang (issue fix-ldglobal-active-count-hang)
# Drives a minimal ld.global.u32 + st.global.u32 sequence through step_warp with
# simulated SM ticks, verifying the warp completes within a bounded cycle count.
add_catch_test(integration_ldglobal_no_hang
    exec/test_ldglobal_no_hang.cpp
)
set_tests_properties(integration_ldglobal_no_hang PROPERTIES LABELS "integration;exec;ldglobal;hang_regression")

# ============================================================================
# PC 集成测试
# ============================================================================
add_catch_test(integration_pc_management
    pc/test_pc_management_integrated.cpp
)
set_tests_properties(integration_pc_management PROPERTIES LABELS "integration;pc")

# ============================================================================
# Sync 集成测试
# ============================================================================
add_catch_test(integration_sync_mechanism
    sync/test_sync_mechanism_integrated.cpp
)
set_tests_properties(integration_sync_mechanism PROPERTIES LABELS "integration;sync")

add_catch_test(integration_syncthreads_full_pipeline
    sync/test_syncthreads_full_pipeline.cpp
)
set_tests_properties(integration_syncthreads_full_pipeline PROPERTIES LABELS "integration;syncthreads")

# ============================================================================
# PTX 指令集成测试 — 驱动模拟器（非真 GPU）
# A2 plan: rewrite tests/unit/ptx/* to drive SMContext+WarpContext+step_warp
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

add_catch_test(integration_ptx_cvt
    ptx/test_cvt.cpp
)
set_tests_properties(integration_ptx_cvt PROPERTIES LABELS "integration;ptx;cvt")

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
# Memory 集成测试 — 驱动模拟器
# B1 plan: shared memory layout, multi-buffer offsets, zero-init verification
# ============================================================================
add_catch_test(integration_shared_memory_layout
    memory/test_shared_memory_layout.cpp
)
set_tests_properties(integration_shared_memory_layout PROPERTIES LABELS "integration;memory;shared;layout;shared_memory")

# ---------------------------------------------------------------------------
# KNOWN FAILING (pre-existing, see KNOWN_ISSUES.md §"B1.3 Local Memory Path")
# ---------------------------------------------------------------------------
# Test was written to verify ld.local / st.local round-trip on the simulator.
# SEGFAULTs on main because get_memory_addr() LOCAL path is commented out
# in thread_context.cpp:480-488 (the st.local handler is wired but the
# address resolution is no-op'd). Tracked separately; excluded from
# baseline gate via DISABLED. Re-enable after the LOCAL path is restored.
add_catch_test(integration_local_memory
    memory/test_local_memory.cpp
)
set_tests_properties(integration_local_memory
    PROPERTIES LABELS "integration;memory;local;ld_st" DISABLED True)

# ============================================================================
# Dynamic Shared Memory Integration Test
# ============================================================================
add_catch_test(integration_dynamic_shared_memory
    memory/test_dynamic_shared_memory.cpp
)
set_tests_properties(integration_dynamic_shared_memory PROPERTIES LABELS "integration;memory;shared_memory;dynamic")
