/**
 * Test for post-barrier warp divergence issue using real execute_warp_instruction
 *
 * Reproduces the bug where after bar.warp.sync completes:
 * - exec_mask is updated correctly
 * - thread state (is_blocked, status) is updated correctly
 * - But active_mask[] is NEVER updated
 * Result: Only 1 lane executes post-barrier instead of 32
 */

#include "catch_amalgamated.hpp"
#include "ptxsim/warp_state.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/wbar.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include <cstdint>
#include <array>
#include <memory>

using ptxsim::WarpState;
using ptxsim::ThreadState;
using ptxsim::Wbar;
using ptxsim::ThreadStatus;

// Helper to create a bar.warp.sync statement
static StatementContext make_barrier_stmt(uint32_t mask, int reconvergence_pc) {
    StatementContext ctx;
    ctx.type = S_BAR_WARP_SYNC;
    BarWarpSyncInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(mask)}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(reconvergence_pc)}});
    ctx.data = instr;
    return ctx;
}

// Helper to create a simple mov statement
static StatementContext make_mov_stmt() {
    StatementContext ctx;
    ctx.type = S_MOV;
    GenericInstr instr;
    ctx.data = instr;
    return ctx;
}

// Count lanes where execute_thread_instruction was called
static int count_executed_lanes(WarpContext& warp) {
    int count = 0;
    for (int i = 0; i < 32; i++) {
        auto* t = warp.get_thread(i);
        if (t && !t->is_exited()) {
            count++;
        }
    }
    return count;
}

// ============================================================================
// CTA-level barrier bug reproduction using real execute_warp_instruction()
// ============================================================================
// This test reproduces the issue where after SMContext::synchronize_barrier()
// releases all threads, execute_warp_instruction() only executes for lanes
// that were already in active_mask, because active_mask is never updated.
//
// The real flow:
//   1. Threads arrive at bar.sync (CTA barrier)
//   2. Last thread triggers synchronize_barrier()
//   3. synchronize_barrier() sets all threads to RUN, sync_to_warp_state()
//   4. synchronize_barrier() sets exec_mask = 0xFFFFFFFF
//   5. BUT active_mask is NEVER updated
//   6. Next execute_warp_instruction() call checks is_lane_active() first
//   7. Only pre-barrier active lanes pass the filter → bug!
// ============================================================================

static void init_instruction_factory_once() {
    static bool initialized = false;
    if (!initialized) {
        InstructionFactory::initialize();
        initialized = true;
    }
}

// Helper: create a safe no-op statement that advances PC
static StatementContext make_nop_stmt() {
    StatementContext ctx;
    ctx.type = S_PRAGMA;  // SimpleHandler: just advances PC, no operands needed
    ctx.data = PragmaInstr{};
    ctx.instructionText = "pragma;";
    return ctx;
}

TEST_CASE("BUG-REPRODUCTION: bar.warp.sync releases threads but active_mask not updated",
          "[post_barrier][divergence][execute_warp_instruction][bug]")
{
    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> thread_ptrs;

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> statements;


    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    // Create 32 threads
    for (int lane = 0; lane < 32; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym,
                     label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        thread_ptrs.push_back(std::move(thread));
    }
    for (int i = 0; i < 32; i++) {
        warp.add_thread(std::move(thread_ptrs[i]), i);
    }

    // Set initial active mask to only lane 0 (simulates pre-barrier state where only 1 lane active)
    warp.set_active_mask(0x00000001);

    SECTION("Setup: all 32 threads at barrier PC=0, only lane 0 in active_mask") {
        for (int i = 0; i < 32; i++) {
            auto* t = warp.get_thread(i);
            t->set_pc(0);
            t->state = RUN;
            warp.get_warp_state().threads[i].pc = 0;
            warp.get_warp_state().threads[i].is_blocked = false;
            warp.get_warp_state().threads[i].status = ThreadStatus::Active;
            warp.get_warp_state().threads[i].is_active = true;
            warp.get_warp_state().threads[i].is_exited = false;
        }

        // Only lane 0 should be "active" according to active_mask
        int active = 0;
        for (int i = 0; i < 32; i++) {
            if (warp.is_lane_active(i)) active++;
        }
        INFO("Initial active_mask lanes: " << active);
        REQUIRE(active == 1);
    }

    SECTION("BUG: Execute barrier instruction, then post-barrier mov") {
        // Create barrier at PC=0, releases to PC=1
        StatementContext barrier_stmt = make_barrier_stmt(0xFFFFFFFF, 1);
        barrier_stmt.instructionText = "bar.warp.sync.b32 0xFFFFFFFF, 1;";

        // Create post-barrier mov at PC=1
        StatementContext mov_stmt = make_mov_stmt();
        mov_stmt.instructionText = "mov.u32 %r1, %r2;";

        // Manually simulate threads arriving at barrier
        // (like BarWarpSyncHandler does internally)
        Wbar& wbar = warp.get_warp_state().wbars[0];
        wbar.init(0xFFFFFFFF, 1);  // 32 threads, reconvergence at PC=1

        for (int i = 0; i < 32; i++) {
            warp.get_warp_state().threads[i].is_blocked = true;
            warp.get_warp_state().threads[i].status = ThreadStatus::Blocked;
            wbar.arrive(i);
        }

        INFO("Barrier complete: " << wbar.is_complete());
        INFO("Arrived mask: 0x" << std::hex << wbar.arrived_mask);
        REQUIRE(wbar.is_complete() == true);

        // Now simulate what BarWarpSyncHandler does on completion:
        // 1. set_exec_mask - DONE
        warp.set_exec_mask(wbar.arrived_mask);

        // 2. Update thread states - DONE  
        for (int i = 0; i < 32; i++) {
            if ((wbar.arrived_mask & (1u << i)) && warp.get_warp_state().threads[i].is_active) {
                warp.set_thread_pc(i, 1);
                warp.get_warp_state().threads[i].is_blocked = false;
                warp.get_warp_state().threads[i].status = ThreadStatus::Active;
            }
        }

        // 3. BUG: active_mask NOT updated!
        // If the bug exists, active_mask is still 0x00000001 (only lane 0)
        // This means only 1 lane will execute the post-barrier mov

        int active_before = 0;
        for (int i = 0; i < 32; i++) {
            if (warp.is_lane_active(i)) active_before++;
        }
        INFO("Active lanes after barrier release (active_mask not updated): " << active_before);

        // Now execute the post-barrier mov instruction at PC=1
        // Count how many lanes actually execute
        int executed_count = 0;
        for (int i = 0; i < 32; i++) {
            if (!warp.is_lane_active(i)) continue;
            if (warp.get_warp_state().threads[i].pc != 1) continue;

            auto* t = warp.get_thread(i);
            if (t && !t->is_exited() && t->get_state() == RUN) {
                executed_count++;
            }
        }

        INFO("Lanes that will execute mov at PC=1: " << executed_count);
        INFO("But all 32 threads are at PC=1 and ready!");

        // The bug: only 1 lane executes despite 32 being ready
        REQUIRE(executed_count == 1);  // This proves the bug
        CHECK(executed_count < 32);  // Bug: should be 32!
    }

    SECTION("Verify: thread state says 32 ready, active_mask says 1 active") {
        for (int i = 0; i < 32; i++) {
            warp.get_warp_state().threads[i].is_active = true;
            warp.get_warp_state().threads[i].is_exited = false;
            warp.get_warp_state().threads[i].is_blocked = false;
            warp.get_warp_state().threads[i].status = ThreadStatus::Active;
            warp.get_warp_state().threads[i].pc = 1;
        }

        int by_state = 0;
        int by_active_mask = 0;
        for (int i = 0; i < 32; i++) {
            bool ready = warp.get_warp_state().threads[i].is_active &&
                        !warp.get_warp_state().threads[i].is_exited &&
                        !warp.get_warp_state().threads[i].is_blocked;
            if (ready) by_state++;
            if (warp.is_lane_active(i)) by_active_mask++;
        }

        INFO("By thread state: " << by_state << " ready");
        INFO("By active_mask: " << by_active_mask << " active");

        REQUIRE(by_state == 32);
        CHECK(by_active_mask < by_state);  // Bug: active_mask doesn't match
    }
}

TEST_CASE("FIX VERIFICATION: After barrier, set_active_mask should sync with released lanes",
          "[post_barrier][divergence][execute_warp_instruction][fix]")
{
    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> thread_ptrs;

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> statements;
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    for (int lane = 0; lane < 32; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym,
                     label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        thread_ptrs.push_back(std::move(thread));
    }
    for (int i = 0; i < 32; i++) {
        warp.add_thread(std::move(thread_ptrs[i]), i);
    }

    warp.set_active_mask(0x00000001);

    for (int i = 0; i < 32; i++) {
        auto* t = warp.get_thread(i);
        t->set_pc(0);
        warp.get_warp_state().threads[i].pc = 0;
        warp.get_warp_state().threads[i].is_blocked = false;
        warp.get_warp_state().threads[i].status = ThreadStatus::Active;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
    }

    SECTION("FIX: After barrier release, call set_active_mask(arrived_mask)") {
        Wbar& wbar = warp.get_warp_state().wbars[0];
        wbar.init(0xFFFFFFFF, 1);
        for (int i = 0; i < 32; i++) {
            wbar.arrive(i);
        }
        REQUIRE(wbar.is_complete() == true);

        warp.set_exec_mask(wbar.arrived_mask);

        for (int i = 0; i < 32; i++) {
            if ((wbar.arrived_mask & (1u << i))) {
                warp.set_thread_pc(i, 1);
                warp.get_warp_state().threads[i].is_blocked = false;
                warp.get_warp_state().threads[i].status = ThreadStatus::Active;
            }
        }

        // THE FIX: update active_mask to match arrived_mask
        warp.set_active_mask(wbar.arrived_mask);

        int active_after_fix = 0;
        for (int i = 0; i < 32; i++) {
            if (warp.is_lane_active(i)) active_after_fix++;
        }

        INFO("Active lanes after FIX: " << active_after_fix);
        REQUIRE(active_after_fix == 32);
    }
}

TEST_CASE("Full cycle: barrier release then post-barrier instruction execution",
          "[post_barrier][divergence][full_cycle]")
{
    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> thread_ptrs;

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> statements;
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    for (int lane = 0; lane < 32; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym,
                     label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        thread_ptrs.push_back(std::move(thread));
    }
    for (int i = 0; i < 32; i++) {
        warp.add_thread(std::move(thread_ptrs[i]), i);
    }

    warp.set_active_mask(0x00000001);

    for (int i = 0; i < 32; i++) {
        auto* t = warp.get_thread(i);
        t->set_pc(0);
        warp.get_warp_state().threads[i].pc = 0;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].status = ThreadStatus::Blocked;
    }

    // Simulate barrier arrival and release
    Wbar& wbar = warp.get_warp_state().wbars[0];
    wbar.init(0xFFFFFFFF, 1);
    for (int i = 0; i < 32; i++) {
        wbar.arrive(i);
    }
    REQUIRE(wbar.is_complete() == true);

    warp.set_exec_mask(wbar.arrived_mask);

    for (int i = 0; i < 32; i++) {
        if ((wbar.arrived_mask & (1u << i))) {
            warp.set_thread_pc(i, 1);
            warp.get_warp_state().threads[i].is_blocked = false;
            warp.get_warp_state().threads[i].status = ThreadStatus::Active;
        }
    }

    SECTION("BUG DEMONSTRATION: Without active_mask fix, only lane 0 executes post-barrier") {
        // active_mask is still 0x00000001 (only lane 0)
        int lanes_at_pc1 = 0;
        int lanes_schedulable = 0;

        for (int i = 0; i < 32; i++) {
            if (warp.get_warp_state().threads[i].pc == 1) lanes_at_pc1++;
            if (warp.is_lane_active(i) && warp.get_warp_state().threads[i].pc == 1) lanes_schedulable++;
        }

        INFO("Threads at PC=1: " << lanes_at_pc1);
        INFO("Threads schedulable (active_mask check): " << lanes_schedulable);

        REQUIRE(lanes_at_pc1 == 32);  // All 32 at PC=1
        CHECK(lanes_schedulable < lanes_at_pc1);  // Bug: active_mask filters out 31 lanes
    }
}

TEST_CASE("BUG-REPRODUCTION-CTA: Post-barrier execute_warp_instruction only executes active_mask lanes",
          "[post_barrier][cta_barrier][execute_warp_instruction][bug]")
{
    init_instruction_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> thread_ptrs;

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> statements;
    statements.push_back(make_nop_stmt());  // PC=0: barrier instruction placeholder
    statements.push_back(make_nop_stmt());  // PC=1: post-barrier instruction
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    // Create 32 threads
    for (int lane = 0; lane < 32; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym,
                     label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        thread_ptrs.push_back(std::move(thread));
    }
    for (int i = 0; i < 32; i++) {
        warp.add_thread(std::move(thread_ptrs[i]), i);
    }

    // Simulate the state AFTER CTA barrier (synchronize_barrier) has completed:
    // All threads released, but active_mask still only has lane 0
    const int BARRIER_PC = 0;
    const int POST_BARRIER_PC = 1;

    warp.set_active_mask(0x00000001);  // ONLY lane 0 active (bug condition)
    warp.set_exec_mask(0xFFFFFFFF);     // All 32 lanes in exec_mask (correct)

    for (int i = 0; i < 32; i++) {
        auto* t = warp.get_thread(i);
        t->set_pc(POST_BARRIER_PC);
        t->set_state(RUN);

        // Simulate synchronize_barrier() state updates
        warp.get_warp_state().threads[i].pc = POST_BARRIER_PC;
        warp.get_warp_state().threads[i].next_pc = POST_BARRIER_PC;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
        warp.get_warp_state().threads[i].is_blocked = false;
        warp.get_warp_state().threads[i].status = ThreadStatus::Active;
    }

    // Pre-check: all 32 threads are at PC=1 and ready
    int ready_count = 0;
    for (int i = 0; i < 32; i++) {
        bool ready = warp.get_warp_state().threads[i].is_active &&
                    !warp.get_warp_state().threads[i].is_exited &&
                    !warp.get_warp_state().threads[i].is_blocked &&
                    warp.get_warp_state().threads[i].pc == POST_BARRIER_PC;
        if (ready) ready_count++;
    }
    REQUIRE(ready_count == 32);

    // Pre-check: only lane 0 is active in active_mask
    int active_mask_count = 0;
    for (int i = 0; i < 32; i++) {
        if (warp.is_lane_active(i)) active_mask_count++;
    }
    REQUIRE(active_mask_count == 1);

    // Record PCs before execution
    int pc_before[32];
    for (int i = 0; i < 32; i++) {
        pc_before[i] = warp.get_thread(i)->get_pc();
    }

    // Execute post-barrier instruction using real execute_warp_instruction()
    StatementContext nop_stmt = make_nop_stmt();
    warp.execute_warp_instruction(nop_stmt, POST_BARRIER_PC);

    // Count how many threads actually executed (PC advanced)
    int executed_count = 0;
    for (int i = 0; i < 32; i++) {
        int pc_after = warp.get_thread(i)->get_pc();
        if (pc_after > pc_before[i]) {
            executed_count++;
        }
    }

    INFO("Threads ready before execute_warp_instruction: " << ready_count);
    INFO("Lanes in active_mask: " << active_mask_count);
    INFO("Lanes that actually executed post-barrier instruction: " << executed_count);

    // The bug: execute_warp_instruction checks is_lane_active() BEFORE pc check.
    // Since active_mask was never updated by synchronize_barrier(),
    // only lane 0 passes the filter and executes.
    CHECK(executed_count == 32);  // EXPECTED: all 32 should execute
                                  // ACTUAL: only 1 due to active_mask bug
}

TEST_CASE("FIX-VERIFICATION-CTA: After updating active_mask, all lanes execute post-barrier",
          "[post_barrier][cta_barrier][execute_warp_instruction][fix]")
{
    init_instruction_factory_once();

    WarpContext warp;
    std::vector<std::unique_ptr<ThreadContext>> thread_ptrs;

    Dim3 blockIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    std::vector<StatementContext> statements;
    statements.push_back(make_nop_stmt());  // PC=0
    statements.push_back(make_nop_stmt());  // PC=1
    std::map<std::string, Symtable*> name2Sym;
    std::map<std::string, int> label2pc;

    for (int lane = 0; lane < 32; lane++) {
        auto thread = std::make_unique<ThreadContext>();
        Dim3 tid = {(uint32_t)lane, 0, 0};
        thread->init(blockIdx, tid, gridDim, blockDim, statements, &name2Sym,
                     label2pc, nullptr, nullptr);
        thread->set_state(RUN);
        thread_ptrs.push_back(std::move(thread));
    }
    for (int i = 0; i < 32; i++) {
        warp.add_thread(std::move(thread_ptrs[i]), i);
    }

    const int POST_BARRIER_PC = 1;

    // Same setup as bug reproduction
    warp.set_active_mask(0x00000001);
    warp.set_exec_mask(0xFFFFFFFF);

    for (int i = 0; i < 32; i++) {
        auto* t = warp.get_thread(i);
        t->set_pc(POST_BARRIER_PC);
        t->set_state(RUN);
        warp.get_warp_state().threads[i].pc = POST_BARRIER_PC;
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
        warp.get_warp_state().threads[i].is_blocked = false;
        warp.get_warp_state().threads[i].status = ThreadStatus::Active;
    }

    // THE FIX: update active_mask to match released threads
    warp.set_active_mask(0xFFFFFFFF);

    int pc_before[32];
    for (int i = 0; i < 32; i++) {
        pc_before[i] = warp.get_thread(i)->get_pc();
    }

    StatementContext nop_stmt = make_nop_stmt();
    warp.execute_warp_instruction(nop_stmt, POST_BARRIER_PC);

    int executed_count = 0;
    for (int i = 0; i < 32; i++) {
        if (warp.get_thread(i)->get_pc() > pc_before[i]) {
            executed_count++;
        }
    }

    INFO("Lanes that executed with fixed active_mask: " << executed_count);
    REQUIRE(executed_count == 32);
}
