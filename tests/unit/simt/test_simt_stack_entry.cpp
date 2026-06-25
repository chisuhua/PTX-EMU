#include "catch_amalgamated.hpp"
#include "ptxsim/simt_stack.h"
#include "ptxsim/thread_state.h"
#include <array>

using namespace ptxsim;

TEST_CASE("B4: exited threads don't block convergence", "[simt_entry][bug][critical]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;

    for (int i = 0; i < 32; i++) {
        threads[i].pc = 0;
        threads[i].is_active = true;
        threads[i].is_exited = false;
    }

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.active_mask = 0xFFFF0000;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 20;

    stack.push(entry);

    for (int i = 0; i < 16; i++) {
        threads[i].is_exited = true;
        threads[i].is_active = false;
        threads[i].pc = 0;
    }

    for (int i = 16; i < 32; i++) {
        threads[i].pc = 20;
    }

    bool converged = stack.check_reconvergence(threads);
    REQUIRE(converged == true);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("B1: all threads at reconvergence PC", "[simt_entry]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    for (int i = 0; i < 32; i++) {
        threads[i].pc = 20;
        threads[i].is_active = true;
        threads[i].is_exited = false;
    }

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 20;
    entry.active_mask = 0x0000FFFF;
    stack.push(entry);

    bool converged = stack.check_reconvergence(threads);
    REQUIRE(converged == true);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("B2: active_mask lanes not all arrived",
          "[simt_entry]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    for (int i = 0; i < 32; i++) {
        threads[i].is_active = true;
        threads[i].is_exited = false;
    }
    for (int i = 0; i < 16; i++) threads[i].pc = 20;
    for (int i = 16; i < 32; i++) threads[i].pc = 15;

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.active_mask = 0x0000FFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 20;
    stack.push(entry);

    bool converged = stack.check_reconvergence(threads);
    REQUIRE(converged == true);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("B3: return_mask excludes unaffected threads", "[simt_entry]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    for (int i = 0; i < 32; i++) {
        threads[i].is_active = true;
        threads[i].is_exited = false;
    }
    for (int i = 0; i < 16; i++) threads[i].pc = 20;
    for (int i = 16; i < 32; i++) threads[i].pc = 99;

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.return_mask = 0x0000FFFF;
    entry.return_pc = 20;
    entry.active_mask = 0;
    stack.push(entry);

    bool converged = stack.check_reconvergence(threads);
    REQUIRE(converged == true);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("B5: empty return_mask converges immediately", "[simt_entry]") {
    SIMTStack stack;
    std::array<ThreadState, 32> threads;

    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.return_mask = 0x00000000;
    entry.return_pc = 20;
    entry.active_mask = 0;
    stack.push(entry);

    bool converged = stack.check_reconvergence(threads);
    REQUIRE(converged == true);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("B6: toString produces expected format", "[simt_entry]") {
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.active_mask = 0xFFFF;
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 20;

    std::string s = entry.toString();
    REQUIRE(s.find("branch_pc=10") != std::string::npos);
    REQUIRE(s.find("reconvergence_pc=20") != std::string::npos);
    REQUIRE(s.find("active_mask=0xffff") != std::string::npos);
}