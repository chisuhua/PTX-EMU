# SIMT Architecture Fix - Learnings

## Root Cause Analysis

### Problem 1: Operand Caching (FIXED)
- **Location**: `PipelineHandler::acquireAllOperands` in `src/ptxsim/instruction_base.cpp`
- **Issue**: `if (!operands[i].operand_phy_addr)` check caused operand caching on shared `StatementContext::operands`
- **Impact**: First thread resolved addresses (e.g., register %r3), subsequent threads reused them instead of resolving their own register values
- **Fix**: Removed the caching check - always call `acquire_operand` for each thread

### Problem 2: PC Corruption via warp_state (FIXED)
- **Location**: `get_lanes_by_pc` and `execute_warp_instruction` in `src/ptxsim/core/warp_context.cpp`
- **Issue**: Both used `warp_state.threads[i].pc` for lane grouping
- **Impact**: Sequential lane execution corrupted warp_state via `sync_to_warp_state`, causing lanes to execute wrong instructions
- **Fix**: Changed to use `threads[i]->pc` (ThreadContext::pc) directly

### The Key Insight
In SIMT execution, lanes are NOT truly independent. They share `StatementContext` and `warp_state`. When one lane updates warp_state (via `sync_to_warp_state`), it corrupts the state for all other lanes.

**Correct approach**: Use ThreadContext-local state (like `ThreadContext::pc`) for per-thread execution state. Use warp_state only for warp-level coordination (barriers, divergence masks).

### Remaining Issue: reconvergence_pc (Not Fixed)
- **Location**: `ptx_visitor_barrier.cpp` - `VISITOR_BARRIER` macro
- **Issue**: `reconvergence_pc = currentKernel->kernelStatements.size() + 1` gives wrong values
- **Impact**: Threads released to wrong instruction after barrier, causing infinite loops
- **Proper Fix**: `reconvergence_pc` should be the actual index of the next instruction in kernelStatements, not the total count

## Architectural Principles for SIMT

1. **ThreadContext::pc** is the SOURCE OF TRUTH for each thread's program counter
2. **warp_state.threads[i].pc** is a COPY that gets corrupted by sequential execution - DO NOT USE for lane grouping
3. **Operand resolution** must happen per-thread, never cached on shared instruction data
4. **Barrier release** should update ThreadContext::pc for the releasing thread, and warp_state for coordination

## Files Changed
- `src/ptxsim/instruction_base.cpp` - Removed operand caching (line 109)
- `src/ptxsim/core/warp_context.cpp` - Use ThreadContext::pc (lines 286, 139)
- `src/ptxsim/core/thread_context.cpp` - Minor PC sync tweak
- `src/ptxsim/instructions/barrier.cpp` - Only mark active lanes as arrived
