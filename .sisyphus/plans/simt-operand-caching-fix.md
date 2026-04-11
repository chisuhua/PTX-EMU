# Architectural Fix Plan: SIMT Operand Caching Bug

## Executive Summary

**Problem**: In `PipelineHandler::acquireAllOperands` (src/ptxsim/instruction_base.cpp:109), the check `if (!operands[i].operand_phy_addr)` causes only the first thread to resolve operand addresses. Subsequent threads reuse the cached addresses, breaking SIMT execution where each thread must resolve its own register values.

**Impact**: Divergent execution with shared memory stores causes all threads to write to thread 0's address instead of their own addresses.

**Root Cause**: `GenericInstr::operands` lives in `StatementContext`, which is **shared across all threads in a CTA**. Caching `operand_phy_addr` on shared state breaks per-thread address resolution.

**Recommended Fix**: **Approach BModified** - Remove the caching check and force per-thread re-resolution. Always call `acquire_operand()` for every thread, and reset `operand_phy_addr` to nullptr at the end of each thread's execution.

---

## 1. Task Dependency Graph

| Task | Depends On | Reason |
|------|------------|--------|
| Task 1: Analyze current data flow | None | Foundation - understand the bug |
| Task 2: Design thread-local operand collection | Task 1 | Requires understanding of current flow |
| Task 3: Implement fix in instruction_base.cpp | Task 2 | Implementation depends on design |
| Task 4: Update ThreadContext::collect_operands | Task 3 | Must align with new collection strategy |
| Task 5: Add/reset operand_phy_addr lifecycle | Task 3 | Ensures proper cleanup per thread |
| Task 6: Create test case for divergent st.shared | Task 1 | Validates the fix works |
| Task 7: Run full test suite | Task 6 | Verify no regressions |

---

## 2. Parallel Execution Graph

**Wave 1 (Start immediately - no dependencies):**
- Task 1: Analyze current data flow (exploration)
- Task 6: Create test case (can proceed independently with known bug pattern)

**Wave 2 (After Wave 1):**
- Task 2: Design thread-local operand collection
- Task 3: Implement fix in instruction_base.cpp (can start once design is clear)

**Wave 3 (After Wave 2):**
- Task 4: Update ThreadContext::collect_operands
- Task 5: Add/reset operand_phy_addr lifecycle

**Wave 4 (After Wave 3):**
- Task 7: Run full test suite

**Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 7  
**Estimated Parallel Speedup**: ~30% faster than sequential ( Waves 2-3 can overlap)

---

## 3. Tasks

### Task 1: Analyze Current Data Flow

**Description**: Document the exact flow of operand resolution from warp execution to handler usage.

**Delegation Recommendation:**
- Category: `cpp` - General C++ code analysis task
- Skills: [`cpp-architecture`] - For dependency graph analysis

**Skills Evaluation:**
- ✅ INCLUDED `cpp-architecture`: Analyzes data flow across modules
- ❌ OMITTED `cpp-debug`: Not a runtime bug investigation (already diagnosed)
- ❌ OMITTED `planning-with-files`: Planning already done in this document

**Depends On**: None

**Acceptance Criteria:**
- [ ] Data flow diagram showing BEFORE state
- [ ] Identified all files involved in operand resolution
- [ ] Documented exact line numbers where bug manifests

---

### Task 2: Design Thread-Local Operand Collection

**Description**: Design the fix strategy - choose between approaches A-D and specify exact implementation details.

**Delegation Recommendation:**
- Category: `cpp` - C++ architectural design
- Skills: [`cpp-architecture`, `cpp-pro`] - Architecture analysis + modern C++ patterns

**Skills Evaluation:**
- ✅ INCLUDED `cpp-architecture`: Ensures minimal disruption to existing architecture
- ✅ INCLUDED `cpp-pro`: Modern C++ patterns for thread-local storage
- ❌ OMITTED `cuda`: Not CUDA kernel work, this is simulator C++ code

**Depends On**: Task 1

**Acceptance Criteria:**
- [ ] Selected approach with justification
- [ ] Data flow diagram showing AFTER state
- [ ] Identified all modification points with line numbers
- [ ] Assessed performance impact (expected: <5% slowdown)

---

### Task 3: Implement Fix in instruction_base.cpp

**Description**: Modify `PipelineHandler::acquireAllOperands` to remove caching and force per-thread resolution.

**Delegation Recommendation:**
- Category: `cpp` - C++ implementation
- Skills: [`cpp-pro`] - High-quality C++ implementation

**Skills Evaluation:**
- ✅ INCLUDED `cpp-pro`: Ensures idiomatic, efficient implementation
- ❌ OMITTED `cpp-modernize`: Not legacy code modernization
- ❌ OMITTED `cuda`: Simulator C++ code, not CUDA

**Depends On**: Task 2

**Acceptance Criteria:**
- [ ] Removed `if (!operands[i].operand_phy_addr)` check
- [ ] `acquire_operand()` called for every thread
- [ ] Compiles without warnings
- [ ] Passes clang-tidy checks

---

### Task 4: Update ThreadContext::collect_operands

**Description**: Align operand collection with new strategy - ensure per-thread collection uses fresh addresses.

**Delegation Recommendation:**
- Category: `cpp` - C++ implementation
- Skills: [`cpp-pro`] - Thread-safe patterns

**Skills Evaluation:**
- ✅ INCLUDED `cpp-pro`: Thread-local collection patterns
- ❌ OMITTED `cpp-debug`: Not debugging, this is planned modification

**Depends On**: Task 3

**Acceptance Criteria:**
- [ ] `operand_collected` populated per-thread
- [ ] No reliance on cached `operand_phy_addr` from previous threads
- [ ] Compiles without warnings

---

### Task 5: Add/Reset operand_phy_addr Lifecycle

**Description**: Implement proper lifecycle management - reset `operand_phy_addr` to nullptr after each thread completes instruction.

**Delegation Recommendation:**
- Category: `cpp` - C++ lifecycle management
- Skills: [`cpp-pro`] - Resource management patterns

**Skills Evaluation:**
- ✅ INCLUDED `cpp-pro`: Proper lifecycle/resource management
- ❌ OMITTED `cpp-modernize`: Not modernization, this is bug fix

**Depends On**: Task 3

**Acceptance Criteria:**
- [ ] `operand_phy_addr` reset to nullptr in `releaseAllOperands`
- [ ] Optional: Add assertion to verify reset happened
- [ ] No memory leaks introduced

---

### Task 6: Create Test Case for Divergent st.shared

**Description**: Create a minimal PTX test that demonstrates the bug - divergent threads writing to shared memory.

**Delegation Recommendation:**
- Category: `cpp` - Test development (PTX + Catch2)
- Skills: [`cpp-pro`] - Test pattern expertise

**Skills Evaluation:**
- ✅ INCLUDED `cpp-pro`: High-quality test patterns
- ❌ OMITTED `cpp-debug`: Test creation, not debugging

**Depends On**: Task 1

**Acceptance Criteria:**
- [ ] PTX file with divergent execution + `st.shared`
- [ ] Test asserts each thread writes to its own address
- [ ] Test FAILS before fix, PASSES after fix
- [ ] Added to CMakeLists.txt test suite

---

### Task 7: Run Full Test Suite

**Description**: Verify no regressions - all existing tests must pass.

**Delegation Recommendation:**
- Category: `quick` - Simple command execution
- Skills: [] - No special skills needed

**Skills Evaluation:**
- ✅ OMITTED all skills: Simple test execution
- ❌ OMITTED `cpp-debug`: Only if tests fail

**Depends On**: Tasks 3, 4, 5, 6

**Acceptance Criteria:**
- [ ] `cd build && ctest` passes 100%
- [ ] New test (Task 6) passes
- [ ] No new warnings from clang-tidy
- [ ] Performance regression <5% (if measurable)

---

## 4. Data Flow Diagram

### BEFORE (Bug State)

```
Thread 0 executes PC=22 (st.shared.u32 [%r3], %r14):
  acquireAllOperands() 
    → operands[0].operand_phy_addr == nullptr (first thread)
    → acquire_operand() → resolves %r3 → 0xSHARED_BASE+0
    → operands[0].setPhyAddr(0xSHARED_BASE+0)  ← CACHED ON SHARED STATE
    
Thread 1-15 execute SAME PC=22:
  acquireAllOperands()
    → operands[0].operand_phy_addr != nullptr (cached by Thread 0)
    → SKIP acquire_operand()  ← BUG: Reuses Thread 0's address
    → ALL threads write to Thread 0's shared memory address
```

### AFTER (Fixed State)

```
Thread 0 executes PC=22 (st.shared.u32 [%r3], %r14):
  acquireAllOperands()
    → acquire_operand() ALWAYS called
    → resolves %r3 → 0xSHARED_BASE+tid_0*4
    → operands[0].setPhyAddr(0xSHARED_BASE+tid_0*4)
    
Thread 1-15 execute SAME PC=22:
  acquireAllOperands()
    → acquire_operand() ALWAYS called (no cache check)
    → resolves %r3 → 0xSHARED_BASE+tid_1*4
    → operands[0].setPhyAddr(0xSHARED_BASE+tid_1*4)
    
commitResults():
  releaseAllOperands() → operands[0].setPhyAddr(nullptr)
  ← Prepares for next instruction or next thread's execution
```

---

## 5. Risk Analysis

### What Could Break

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance regression | Medium (20%) | Low (expected <5% slowdown) | Benchmark before/after; profile if >10% |
| Other instructions break | Low (10%) | High | Run full test suite; check WMMA/atomic pipelines |
| Operand cache was needed for something | Low (5%) | Medium | Check git history for why caching was added |
| Thread-local buffer overflow | Low (5%) | High | Verify MAX_OPERANDS_PER_INSTR is sufficient |

### Verification Strategy

1. **Immediate**: New test (Task 6) must pass
2. **Short-term**: All existing tests pass
3. **Long-term**: Add performance benchmark to CI

---

## 6. Testing Strategy

### Unit Tests
- [ ] Task 6 test: Divergent st.shared with per-thread verification
- [ ] Add test for divergent ld.shared (read path)
- [ ] Add test for non-divergent case (ensure no regression)

### Integration Tests
- [ ] Existing barrier_sync_test.ptx
- [ ] Existing test_divergence_sync.ptx (if exists, verify it passes)
- [ ] All Catch2 tests in tests/

### Regression Tests
```bash
cd build
ctest -R barrier               # Barrier tests
ctest -R warp                  # Warp-level tests
ctest -R sync                  # Synchronization tests
ctest -L ptx                   # All PTX instruction tests
```

---

## 7. Performance Impact

### Expected Impact
- **Slowdown**: <5% (estimated)
- **Reason**: `acquire_operand()` is already fast (register lookup is O(1))
- **Worst case**: Instructions with many operands (e.g., WMMA with 8+ operands)

### Measurement Plan
```bash
# Before fix
git stash
make -C build RAY
time ./bin/RAY 256 256
# Record: X seconds

# After fix
git stash pop
make -C build RAY
time ./bin/RAY 256 256
# Record: Y seconds

# Regression: (Y-X)/X * 100 < 5%
```

### If Performance >10% Regression
1. Profile to find hotspot in acquire_operand()
2. Consider Approach A (thread-local cache) instead of full re-resolution
3. Add per-instruction caching flag (only cache for non-divergent?)

---

## 8. Alternative Approaches

### If Approach BModified Fails

**Fallback**: **Approach A (Thread-Local Operand Cache)**

**Implementation**:
```cpp
// Add to ThreadContext:
std::vector<void*> thread_local_operand_cache;

// In acquireAllOperands:
if (!thread_local_operand_cache[i]) {
    void *result = context->acquire_operand(operands[i], qualifiers);
    thread_local_operand_cache[i] = result;
}
// No need to reset - it's per-thread
```

**Pros**:
- Maintains caching optimization
- No performance regression
  
**Cons**:
- More invasive (requires ThreadContext changes)
- More code changes

---

## 9. Exact Code Changes

### File 1: src/ptxsim/instruction_base.cpp

**Line 109**: Remove the caching check

```cpp
bool PipelineHandler::acquireAllOperands(ThreadContext *context, 
                                       std::vector<OperandContext> &operands, 
                                       const std::vector<Qualifier> &qualifiers, 
                                       int opCount) {
    for (int i = 0; i < opCount && i < static_cast<int>(operands.size()); i++) {
        // OLD CODE (BUG):
        // if (!operands[i].operand_phy_addr) {
        //     void *result = context->acquire_operand(operands[i], qualifiers);
        //     ...
        //     operands[i].setPhyAddr(result);
        // }
        
        // NEW CODE (FIX):
        void *result = context->acquire_operand(operands[i], qualifiers);
        if (!result) {
            PTX_DEBUG_EMU("Failed to get operand address for op[%d]", i);
            // ... existing error handling ...
            return false;
        }
        operands[i].setPhyAddr(result);
    }
    return true;
}
```

**Line 138-142**: Add reset in releaseAllOperands

```cpp
void PipelineHandler::releaseAllOperands(std::vector<OperandContext> &operands, int opCount) {
    for (int i = 0; i < opCount && i < static_cast<int>(operands.size()); i++) {
        operands[i].setPhyAddr(nullptr);  // Already exists, but now critical
    }
}
```

### File 2: src/ptxsim/core/thread_context.cpp

**No changes expected** - `collect_operands` already copies `operand_phy_addr` to `operand_collected` per-thread. The fix ensures `operand_phy_addr` is fresh for each thread.

---

## 10. Commit Strategy

### Atomic Commit Plan

**Commit 1**: Add test case (Task 6)
```bash
git add tests/ptx/test_divergent_shared_mem.ptx
git add tests/CMakeLists.txt
git commit -m "test: Add divergent shared memory test (fails before fix)"
```

**Commit 2**: Implement fix (Tasks 3-5)
```bash
git add src/ptxsim/instruction_base.cpp
git commit -m "fix: Remove operand caching to enable per-thread address resolution

- Always call acquire_operand() for every thread
- Reset operand_phy_addr to nullptr after each instruction
- Fixes SIMT execution bug where divergent threads reuse first thread's addresses

Fixes: [issue number if exists]"
```

**Commit 3**: Verify all tests pass
```bash
git commit --allow-empty -m "chore: Verify all tests pass after operand caching fix"
```

---

## 11. Success Criteria

1. **New test passes**: Divergent st.shared test from Task 6
2. **All existing tests pass**: `cd build && ctest` shows 100% pass
3. **No new warnings**: Clang-tidy clean
4. **Performance acceptable**: <5% slowdown measured on RAY benchmark
5. **Debug output correct**: Each thread logs its own shared memory address

---

## 12. Files Modified Summary

| File | Lines Changed | Change Type |
|------|---------------|-------------|
| `src/ptxsim/instruction_base.cpp` | ~15 lines | Logic change (remove cache check) |
| `tests/ptx/test_divergent_shared_mem.ptx` | NEW FILE | Test case |
| `tests/CMakeLists.txt` | +1 line | Add test to suite |

**Total**: 2 files modified, 1 file created

---

## 13. Implementation Notes

### Why Approach BModified?

1. **Minimally invasive**: Only changes instruction_base.cpp
2. **Correct**: Guarantees per-thread resolution
3. **Simple**: Removes optimization rather than complicating architecture
4. **Safe**: No new data structures or thread-local storage needed

### Why NOT other approaches?

- **Approach A**: Too invasive (requires ThreadContext changes)
- **Approach C**: Same as B but more complex wording
- **Approach D**: Loses optimization entirely without clear benefit

### Key Insight

The caching was a well-intentioned optimization that assumed "same instruction = same operands". But in SIMT, different threads have different register values even for the same instruction. The fix is to recognize that **operand resolution is inherently per-thread work** and cannot be shared.

---

## Appendix A: Relevant Code Locations

**Bug Location**: `src/ptxsim/instruction_base.cpp:109`
```cpp
if (!operands[i].operand_phy_addr) {  // ← THIS CHECK IS THE BUG
```

**Shared State**: `include/ptx_ir/statement_context.h:137-139`
```cpp
struct GenericInstr {
    std::vector<OperandContext> operands;  // ← SHARED ACROSS THREADS
};
```

**Operand Context**: `include/ptx_ir/operand_context.h:67`
```cpp
mutable void *operand_phy_addr = nullptr;  // ← CACHED ADDRESS
```

**Collection Point**: `src/ptxsim/core/thread_context.cpp:370`
```cpp
operand_collected[i] = operands[i].operand_phy_addr;  // ← COPIED TO THREAD-LOCAL
```

**Execution Point**: `src/ptxsim/core/warp_context.cpp:145-163`
```cpp
for (int i = 0; i < WARP_SIZE; i++) {
    if (!is_lane_active(i) || threads[i] == nullptr) continue;
    thread->execute_thread_instruction();  // ← EACH THREAD EXECUTES
}
```

---

## Appendix B: Debug Evidence

From the user's debug output:
```
CLK:374: Thread 0 at PC=22 writes (first st.shared.data_b, pc=22)
CLK:377: Threads 1-15 RE-EXECUTE PC=10 (st.shared.data_a), proving they got wrong PC/address
```

This proves threads 1-15 didn't resolve their own addresses - they reused thread 0's cached address and executed the wrong memory operation.

---

## Appendix C: Performance Benchmark Script

```bash
#!/bin/bash
# Save as benchmarks/perf_operand_resolution.sh

echo "===Operand Resolution Performance Test==="
echo ""

# Build release
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target RAY

# Run 3 times and average
echo "Running RAY 256x256 benchmark..."
times=()
for i in {1..3}; do
    echo "Run $i..."
    start=$(date +%s.%N)
    ./build/bin/RAY 256 256 > /dev/null 2>&1
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    times+=($elapsed)
done

# Calculate average
avg=$(echo "(${times[0]} + ${times[1]} + ${times[2]}) / 3" | bc -l)
echo "Average: $avg seconds"

# Compare to baseline (run before fix)
baseline=1.234  # Replace with actual baseline
regression=$(echo "($avg - $baseline) / $baseline * 100" | bc -l)
echo "Regression: $regression%"

if (( $(echo "$regression > 5" | bc -l) )); then
    echo "WARNING: Performance regression > 5%"
    exit 1
fi
echo "PASS: Performance within acceptable range"
```
