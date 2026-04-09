/**
 * SIMT Stack Integration Test
 * Tests SIMT stack push/pop with divergent branches
 */

#include <iostream>
#include <cassert>
#include <array>

#include "ptxsim/simt_stack.h"
#include "ptxsim/thread_state.h"

using namespace ptxsim;

void create_test_threads(std::array<ThreadState, 32>& threads, int reconvergence_pc) {
    for (int i = 0; i < 32; i++) {
        threads[i].is_active = true;
        threads[i].pc = reconvergence_pc;
        threads[i].is_exited = false;
    }
}

void test_simt_stack_basic() {
    std::cout << "Test 1: SIMT Stack Basic Operations... ";
    
    SIMTStack stack;
    assert(stack.empty());
    assert(stack.depth() == 0);
    
    // Push entry
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.active_mask = 0xFFFF;  // lanes 0-15
    entry.return_mask = 0xFFFFFFFF;
    entry.return_pc = 20;
    
    stack.push(entry);
    assert(!stack.empty());
    assert(stack.depth() == 1);
    assert(stack.top().reconvergence_pc == 20);
    
    // Pop entry
    SIMTStackEntry popped = stack.pop();
    assert(popped.reconvergence_pc == 20);
    assert(stack.empty());
    
    std::cout << "PASS ✓" << std::endl;
}

void test_reconvergence_check() {
    std::cout << "Test 2: Reconvergence Check... ";
    
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    
    // Setup: all threads at reconvergence point
    create_test_threads(threads, 20);
    
    // Push entry with reconvergence_pc=20
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.return_mask = 0xFFFFFFFF;
    entry.active_mask = 0xFFFF;
    entry.return_pc = 20;
    
    stack.push(entry);
    
    // Check reconvergence (should succeed)
    bool converged = stack.check_reconvergence(threads);
    assert(converged);
    assert(stack.empty());
    
    std::cout << "PASS ✓" << std::endl;
}

void test_no_reconvergence() {
    std::cout << "Test 3: No Reconvergence Yet... ";
    
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    
    // Setup: threads NOT at reconvergence point
    create_test_threads(threads, 15);  // Not at 20 yet
    
    // Push entry with reconvergence_pc=20
    SIMTStackEntry entry;
    entry.branch_pc = 10;
    entry.reconvergence_pc = 20;
    entry.return_mask = 0xFFFFFFFF;
    
    stack.push(entry);
    
    // Check reconvergence (should fail)
    bool converged = stack.check_reconvergence(threads);
    assert(!converged);
    assert(!stack.empty());  // Still on stack
    
    std::cout << "PASS ✓" << std::endl;
}

void test_nested_stacks() {
    std::cout << "Test 4: Nested SIMT Stacks... ";
    
    SIMTStack stack;
    std::array<ThreadState, 32> threads;
    
    // Push first level
    SIMTStackEntry entry1;
    entry1.branch_pc = 10;
    entry1.reconvergence_pc = 30;
    entry1.active_mask = 0xFFFF;
    entry1.return_mask = 0xFFFFFFFF;
    stack.push(entry1);
    
    // Push second level (nested branch)
    SIMTStackEntry entry2;
    entry2.branch_pc = 15;
    entry2.reconvergence_pc = 25;
    entry2.active_mask = 0xFF;
    entry2.return_mask = 0xFFFF;
    stack.push(entry2);
    
    assert(stack.depth() == 2);
    
    // Create threads at inner reconvergence point (25)
    create_test_threads(threads, 25);
    
    // Check inner reconvergence
    bool converged_inner = stack.check_reconvergence(threads);
    assert(converged_inner);
    assert(stack.depth() == 1);  // Inner popped
    
    // Now at outer reconvergence point (30)
    create_test_threads(threads, 30);
    
    // Check outer reconvergence
    bool converged_outer = stack.check_reconvergence(threads);
    assert(converged_outer);
    assert(stack.empty());
    
    std::cout << "PASS ✓" << std::endl;
}

int main() {
    std::cout << "=== SIMT Stack Integration Tests ===" << std::endl;
    std::cout << std::endl;
    
    test_simt_stack_basic();
    test_reconvergence_check();
    test_no_reconvergence();
    test_nested_stacks();
    
    std::cout << std::endl;
    std::cout << "=== All Tests PASSED ===" << std::endl;
    return 0;
}
