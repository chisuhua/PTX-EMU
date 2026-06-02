#include "catch_amalgamated.hpp"
#include "ptxsim/simt_stack.h"
#include "ptxsim/thread_state.h"
#include <array>

using namespace ptxsim;

TEST_CASE("A1: push and pop operations", "[simt_stack][basic]") {
    SIMTStack stack;
    REQUIRE(stack.empty() == true);
    REQUIRE(stack.depth() == 0);

    SIMTStackEntry e1;
    e1.branch_pc = 10;
    e1.reconvergence_pc = 20;
    e1.active_mask = 0xFFFF;
    e1.return_mask = 0xFFFFFFFF;
    e1.return_pc = 20;

    stack.push(e1);
    REQUIRE(stack.empty() == false);
    REQUIRE(stack.depth() == 1);
    REQUIRE(stack.top().reconvergence_pc == 20);

    SIMTStackEntry popped = stack.pop();
    REQUIRE(popped.branch_pc == 10);
    REQUIRE(stack.empty() == true);
}

TEST_CASE("A2: empty and depth tracking", "[simt_stack][basic]") {
    SIMTStack stack;
    SIMTStackEntry e;
    e.branch_pc = 0; e.reconvergence_pc = 0; e.return_pc = 0;
    e.active_mask = 0; e.return_mask = 0;

    stack.push(e); stack.push(e); stack.push(e);
    REQUIRE(stack.depth() == 3);
    stack.pop();
    REQUIRE(stack.depth() == 2);
    stack.pop(); stack.pop();
    REQUIRE(stack.empty() == true);
}

TEST_CASE("A3: top returns most recent entry", "[simt_stack][basic]") {
    SIMTStack stack;
    SIMTStackEntry e1, e2;
    e1.branch_pc = 10; e1.reconvergence_pc = 20;
    e2.branch_pc = 30; e2.reconvergence_pc = 40;
    e1.return_pc = 0; e2.return_pc = 0;
    e1.active_mask = 0; e2.active_mask = 0;
    e1.return_mask = 0; e2.return_mask = 0;

    stack.push(e1);
    REQUIRE(stack.top().branch_pc == 10);
    stack.push(e2);
    REQUIRE(stack.top().branch_pc == 30);
    stack.pop();
    REQUIRE(stack.top().branch_pc == 10);
}

TEST_CASE("A4: clear empties the stack", "[simt_stack][basic]") {
    SIMTStack stack;
    SIMTStackEntry e;
    e.branch_pc = 0; e.reconvergence_pc = 0; e.return_pc = 0;
    e.active_mask = 0; e.return_mask = 0;
    for (int i = 0; i < 5; i++) stack.push(e);
    stack.clear();
    REQUIRE(stack.empty() == true);
}

TEST_CASE("A5: pop on empty throws exception", "[simt_stack][exception]") {
    SIMTStack stack;
    REQUIRE_THROWS_AS(stack.pop(), std::runtime_error);
}

TEST_CASE("A6: top on empty throws exception", "[simt_stack][exception]") {
    SIMTStack stack;
    REQUIRE_THROWS_AS(stack.top(), std::runtime_error);
}

TEST_CASE("A7: nested push preserves LIFO order", "[simt_stack][nested]") {
    SIMTStack stack;
    SIMTStackEntry e1, e2, e3;
    e1.branch_pc = 10; e1.reconvergence_pc = 30;
    e2.branch_pc = 15; e2.reconvergence_pc = 25;
    e3.branch_pc = 20; e3.reconvergence_pc = 22;
    e1.return_pc = 0; e2.return_pc = 0; e3.return_pc = 0;
    e1.active_mask = 0; e2.active_mask = 0; e3.active_mask = 0;
    e1.return_mask = 0; e2.return_mask = 0; e3.return_mask = 0;

    stack.push(e1); stack.push(e2); stack.push(e3);
    REQUIRE(stack.top().branch_pc == 20);
    stack.pop();
    REQUIRE(stack.top().branch_pc == 15);
    stack.pop();
    REQUIRE(stack.top().branch_pc == 10);
}

TEST_CASE("A8: maximum depth enforcement", "[simt_stack][limit]") {
    SIMTStack stack;
    SIMTStackEntry e;
    e.branch_pc = 0; e.reconvergence_pc = 0; e.return_pc = 0;
    e.active_mask = 0; e.return_mask = 0;

    for (int i = 0; i < 10; i++) stack.push(e);
    REQUIRE(stack.depth() == 10);
    REQUIRE_THROWS_AS(stack.push(e), std::runtime_error);
}
