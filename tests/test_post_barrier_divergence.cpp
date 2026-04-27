/**
 * Test for post-barrier warp divergence issue
 *
 * Root cause: After warp-level barrier completes, exec_mask and arrived_mask
 * are correctly restored, but active_mask[] (used by scheduler) is not
 * updated to reflect all released lanes.
 */

#include "catch_amalgamated.hpp"
#include "ptxsim/warp_state.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/wbar.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/execution_types.h"
#include <cstdint>
#include <array>
#include <memory>

using ptxsim::WarpState;
using ptxsim::ThreadState;
using ptxsim::Wbar;
using ptxsim::ThreadStatus;

TEST_CASE("active_mask not updated after barrier release causes single-thread execution",
          "[post_barrier][divergence][active_mask][bug]")
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
        thread_ptrs.push_back(std::move(thread));
    }

    for (int i = 0; i < 32; i++) {
        warp.add_thread(std::move(thread_ptrs[i]), i);
    }

    warp.set_active_mask(0xFFFFFFFF);

    SECTION("Initialize: all 32 threads blocked at barrier") {
        for (int i = 0; i < 32; i++) {
            warp.get_warp_state().threads[i].is_active = true;
            warp.get_warp_state().threads[i].is_exited = false;
            warp.get_warp_state().threads[i].is_blocked = true;
            warp.get_warp_state().threads[i].pc = 25;
            warp.get_warp_state().threads[i].status = ThreadStatus::Blocked;
        }

        for (int i = 0; i < 32; i++) {
            REQUIRE(warp.get_warp_state().threads[i].is_blocked == true);
        }
    }

    SECTION("BUG: After barrier release, active_mask decouples from thread state") {
        warp.get_warp_state().threads[0].is_active = true;
        warp.get_warp_state().threads[0].is_exited = false;
        warp.get_warp_state().threads[0].is_blocked = false;
        warp.get_warp_state().threads[0].pc = 26;
        warp.get_warp_state().threads[0].status = ThreadStatus::Active;

        for (int i = 1; i < 32; i++) {
            warp.get_warp_state().threads[i].is_active = true;
            warp.get_warp_state().threads[i].is_exited = false;
            warp.get_warp_state().threads[i].is_blocked = false;
            warp.get_warp_state().threads[i].pc = 26;
            warp.get_warp_state().threads[i].status = ThreadStatus::Active;
        }

        warp.set_exec_mask(0xFFFFFFFF);

        int schedulable_by_state = 0;
        int schedulable_by_active_mask = 0;

        for (int i = 0; i < 32; i++) {
            bool thread_ok = warp.get_warp_state().threads[i].is_active &&
                            !warp.get_warp_state().threads[i].is_exited &&
                            !warp.get_warp_state().threads[i].is_blocked;
            if (thread_ok) schedulable_by_state++;
            if (warp.is_lane_active(i)) schedulable_by_active_mask++;
        }

        INFO("Schedulable by thread state: " << schedulable_by_state);
        INFO("Schedulable by active_mask: " << schedulable_by_active_mask);
        INFO("active_mask value: 0x" << std::hex << warp.get_exec_mask());

        REQUIRE(schedulable_by_state == 32);
        INFO("BUG: active_mask shows fewer lanes than thread state");
    }
}

TEST_CASE("Post-barrier: active_mask must match barrier release state",
          "[post_barrier][divergence][active_mask]")
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
        thread_ptrs.push_back(std::move(thread));
    }

    for (int i = 0; i < 32; i++) {
        warp.add_thread(std::move(thread_ptrs[i]), i);
    }

    SECTION("Verify is_lane_active uses active_mask array") {
        warp.set_active_mask(0xFFFFFFFF);

        int active = 0;
        for (int i = 0; i < 32; i++) {
            if (warp.is_lane_active(i)) active++;
        }

        INFO("active_mask=0xFFFFFFFF, lanes active: " << active);
        REQUIRE(active == 32);
    }

    SECTION("active_mask with partial mask") {
        warp.set_active_mask(0xFFFF0000);

        int active = 0;
        for (int i = 0; i < 32; i++) {
            if (warp.is_lane_active(i)) active++;
        }

        INFO("active_mask=0xFFFF0000, lanes active: " << active);
        REQUIRE(active == 16);
    }

    SECTION("BUG REPRODUCTION: After barrier release, thread state updated but active_mask not") {
        warp.set_active_mask(0x00000001);

        for (int i = 0; i < 32; i++) {
            warp.get_warp_state().threads[i].is_active = true;
            warp.get_warp_state().threads[i].is_exited = false;
            warp.get_warp_state().threads[i].is_blocked = false;
            warp.get_warp_state().threads[i].status = ThreadStatus::Active;
        }

        warp.set_exec_mask(0xFFFFFFFF);

        int by_state = 0;
        int by_active_mask = 0;
        for (int i = 0; i < 32; i++) {
            if (warp.get_warp_state().threads[i].is_active &&
                !warp.get_warp_state().threads[i].is_exited &&
                !warp.get_warp_state().threads[i].is_blocked) by_state++;
            if (warp.is_lane_active(i)) by_active_mask++;
        }

        INFO("By thread state: " << by_state << " schedulable");
        INFO("By active_mask: " << by_active_mask << " schedulable");

        REQUIRE(by_state == 32);
        CHECK(by_active_mask < by_state);
    }
}

TEST_CASE("Full barrier release: verify mismatch between thread state and active_mask",
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
        thread_ptrs.push_back(std::move(thread));
    }

    for (int i = 0; i < 32; i++) {
        warp.add_thread(std::move(thread_ptrs[i]), i);
    }

    warp.set_active_mask(0x00000001);

    Wbar wbar;
    wbar.init(0xFFFFFFFF, 26);

    for (int i = 0; i < 32; i++) {
        warp.get_warp_state().threads[i].is_active = true;
        warp.get_warp_state().threads[i].is_exited = false;
        warp.get_warp_state().threads[i].is_blocked = true;
        warp.get_warp_state().threads[i].pc = 25;
        wbar.arrive(i);
    }

    REQUIRE(wbar.is_complete() == true);

    warp.set_exec_mask(wbar.arrived_mask);

    for (int i = 0; i < 32; i++) {
        if ((wbar.arrived_mask & (1u << i)) && warp.get_warp_state().threads[i].is_active) {
            warp.get_warp_state().threads[i].is_blocked = false;
            warp.get_warp_state().threads[i].status = ThreadStatus::Active;
            warp.get_warp_state().threads[i].pc = 26;
        }
    }

    int ready = 0;
    int active = 0;
    for (int i = 0; i < 32; i++) {
        if (warp.get_warp_state().threads[i].is_active &&
            !warp.get_warp_state().threads[i].is_exited &&
            !warp.get_warp_state().threads[i].is_blocked) ready++;
        if (warp.is_lane_active(i)) active++;
    }

    INFO("Barrier released " << wbar.count_arrived() << " threads to PC=" << 26);
    INFO("Threads ready (by state): " << ready);
    INFO("Threads active (by active_mask): " << active);
    INFO("exec_mask: 0x" << std::hex << warp.get_exec_mask());

    REQUIRE(ready == 32);
    CHECK(active < ready);
}
