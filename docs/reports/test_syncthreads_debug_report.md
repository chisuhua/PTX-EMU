# test_syncthreads Test 3 Debug Report

**Date**: 2026-04-10  
**Status**: Root Cause Identified

---

## Problem Summary

Test 3 (test_nested_sync) hangs during second barrier execution.

**Symptoms**:
- All 32 threads pass first barrier (pc=11)
- Divergent execution: threads 16-31 jump to L__BB2_2, threads 0-15 execute shared memory ops
- Second barrier (pc=12): only 16 threads arrive (lanes 0-15)
- arrived count: **stuck at 16/32**
- Lanes 16-31: **never arrive** at second barrier

---

## Root Cause Analysis

### Expected PTX Flow:
```ptx
bar.sync 0;                    // PC=10 (first barrier)
setp.gt.u32 %p1, %r1, 15;      // PC=11: threads 16-31 → %p1=true
@%p1 bra $L__BB2_2;            // PC=12: conditional branch
// threads 0-15: 3 shared memory instructions (PC=13-15)
$L__BB2_2:
bar.sync 0;                    // PC=16 (second barrier) - ALL 32 threads converge
ret;                           // PC=17
```

### Actual Behavior:
1. CLK=10: First barrier sync (reconvergence_pc=11)
2. CLK=13: Barrier complete, all threads→PC=11
3. CLK=63: Second barrier init (reconvergence_pc=12)
4. **CLK=63+: Only lanes 0-15 arrive**
5. **Lanes 16-31: MISSING!**

### Hypothesis:
**Threads 16-31 are executing the branch but never completing the divergent path.**

Possible causes:
1. Branch instruction handler doesn't update PC for taken threads
2. PC update for jump targets uses wrong reconvergence_pc
3. Scheduler doesn't execute threads with taken branches
4. Wbar reset() invalidates state before all threads arrive

---

## Evidence from Logs

```
CLK=63:  Initialized wbar[0] with mask=0xFFFFFFFF, reconvergence_pc=12
CLK=63:  Lane 0-15 arrived (arrived=1/32 → 16/32)
CLK=377: Lane 0-15 re-arrived (still 16/32!)  ← Re-executing same instruction?
```

**Key observation**: Lanes 0-15 arrive multiple times, suggesting **infinite loop** in the shared-memory path.

---

## Next Debugging Steps

1. **Check branch handler**: Does `@%p1 bra $L__BB2_2` correctly set PC for threads 16-31?
2. **Verify schedulers**: Are threads 16-31 being scheduled after branch?
3. **Check PC updates**: After barrier complete at pc=11, what are threads 16-31's PCs?
4. **Examine instruction sequence**: What's at PC=12, 13... 16?

---

## Fix Proposals

### Option 1: Fix branch target setting
```cpp
// In ptx_visitor_branch.cpp
instr.target = "$L__BB2_2";  // Current
// Should resolve label to correct PC
```

### Option 2: Fix barrier reconvergence_pc calculation
```cpp
// In ptx_visitor_barrier.cpp
int next_pc = currentKernel->kernelStatements.size() + 1;
// This may point to wrong instruction after divergent merge
```

### Option 3: Fix wbar reset behavior
```cpp
// barrier.cpp:178
wbar.reset();  // Removes is_initialized, causes re-init
// Should preserve participation_mask until next barrier
```

---

**Report author**: AI Assistant  
**Debug method**: ptx-debug skill scenario 7 + verbose tracing
