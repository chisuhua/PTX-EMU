# GPGPU-Sim SIMT Execution Implementation Analysis

## Executive Summary

This document provides a comprehensive analysis of GPGPU-Sim's SIMT (Single Instruction Multiple Thread) execution implementation, specifically focused on:
- SIMT execution model and warp-level execution
- SIMT stack (reconvergence stack) implementation  
- Barrier/__syncthreads() implementation
- PC management for divergent threads
- Warp scheduler design

**Source**: GPGPU-Sim v4.x (dev branch)  
**Repository**: https://github.com/gpgpu-sim/gpgpu-sim_distribution  
**Analysis Date**: 2026-04-10

---

## 1. SIMT Execution Model

### 1.1 Core Files

| Component | File Path | Lines | Description |
|-----------|-----------|-------|-------------|
| Shader Core | `src/gpgpu-sim/shader.cc` | ~4,500 | Main shader core implementation |
| Shader Header | `src/gpgpu-sim/shader.h` | ~2,534 | Class definitions and interfaces |
| Thread Simulation | `src/cuda-sim/ptx_sim.cc` | ~618 | Per-thread execution logic |
| Thread Header | `src/cuda-sim/ptx_sim.h` | ~483 | Thread state definitions |

### 1.2 Warp Structure (`shd_warp_t` class)

**File**: `src/gpgpu-sim/shader.h` (lines 115-200)

```cpp
class shd_warp_t {
    address_type m_next_pc;                          // Unified PC for warp
    std::bitset<MAX_WARP_SIZE> m_active_threads;     // Active thread mask
    unsigned n_completed;                            // Completed thread count
    bool m_membar;                                   // Barrier wait flag
    
    // Instruction buffer (2-entry FIFO)
    struct ibuffer_entry {
        const warp_inst_t *m_inst;
        bool m_valid;
    };
    ibuffer_entry m_ibuffer[IBUFFER_SIZE];           // IBUFFER_SIZE = 2
    
    // Warp state
    unsigned m_warp_id;
    unsigned m_dynamic_warp_id;
    unsigned m_cta_id;
    unsigned m_warp_size;
    
public:
    void init(address_type start_pc, unsigned cta_id, unsigned wid,
              const std::bitset<MAX_WARP_SIZE> &active, 
              unsigned dynamic_warp_id, unsigned long long streamID);
    
    address_type get_pc() const { return m_next_pc; }
    void set_next_pc(address_type pc) { m_next_pc = pc; }
    
    // Barrier management
    void set_membar() { m_membar = true; }
    void clear_membar() { m_membar = false; }
    bool get_membar() const { return m_membar; }
    
    // Thread completion
    void set_completed(unsigned lane);
    unsigned get_n_completed() const { return n_completed; }
};
```

### 1.3 Warp Initialization

**File**: `src/gpgpu-sim/shader.cc` (lines 650-720)

```cpp
void shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                 unsigned end_thread, unsigned ctaid,
                                 int cta_size, kernel_info_t &kernel) {
    unsigned start_warp = start_thread / m_config->warp_size;
    unsigned warp_per_cta = cta_size / m_config->warp_size;
    unsigned end_warp = end_thread / m_config->warp_size +
                        ((end_thread % m_config->warp_size) ? 1 : 0);
    
    for (unsigned i = start_warp; i < end_warp; ++i) {
        // Calculate active threads mask
        unsigned n_active = 0;
        simt_mask_t active_threads;
        for (unsigned t = 0; t < m_config->warp_size; t++) {
            unsigned hwtid = i * m_config->warp_size + t;
            if (hwtid < end_thread) {
                n_active++;
                active_threads.set(t);
            }
        }
        
        // Initialize SIMT stack with reconvergence point
        m_simt_stack[i]->launch(start_pc, active_threads);
        
        // Initialize warp state
        m_warp[i]->init(start_pc, cta_id, i, active_threads,
                       m_dynamic_warp_id, kernel.get_streamID());
        ++m_dynamic_warp_id;
        m_not_completed += n_active;
        ++m_active_warps;
    }
}
```

### 1.4 Key Design Principle: Single PC Per Warp

**Evidence**: From `shd_warp_t::m_next_pc` and scheduler logic:

> "All threads in a warp share the same next PC (`m_next_pc`). When divergence occurs, the SIMT stack tracks which threads take which path, but the hardware still executes one path at a time with a unified PC."

---

## 2. SIMT Stack (Reconvergence Stack)

### 2.1 SIMT Stack Implementation

**File**: `src/abstract_hardware_model.h` (lines 350-450)

```cpp
class simt_stack {
public:
    simt_stack(unsigned wid, unsigned warpSize, class gpgpu_sim *gpu);
    
    void reset();
    void launch(address_type start_pc, const simt_mask_t &active_mask);
    void update(simt_mask_t &thread_done, addr_vector_t &next_pc,
                address_type recvg_pc, op_type next_inst_op,
                unsigned next_inst_size, address_type next_inst_pc);
    
    const simt_mask_t &get_active_mask() const;
    void get_pdom_stack_top_info(unsigned *pc, unsigned *rpc) const;
    unsigned get_rp() const;
    
protected:
    unsigned m_warp_id;
    unsigned m_warp_size;
    
    enum stack_entry_type { 
        STACK_ENTRY_TYPE_NORMAL = 0, 
        STACK_ENTRY_TYPE_CALL 
    };
    
    struct simt_stack_entry {
        address_type m_pc;              // Current PC for this entry
        unsigned int m_calldepth;       // Call nesting depth
        simt_mask_t m_active_mask;      // Active threads in this path
        address_type m_recvg_pc;        // Reconvergence PC
        unsigned long long m_branch_div_cycle;  // Branch divergence cycle
        stack_entry_type m_type;        // Entry type (NORMAL/CALL)
        
        simt_stack_entry()
            : m_pc(-1), m_calldepth(0), m_recvg_pc(-1),
              m_branch_div_cycle(0), m_type(STACK_ENTRY_TYPE_NORMAL) {}
    };
    
    std::deque<simt_stack_entry> m_stack;
    class gpgpu_sim *m_gpu;
};
```

### 2.2 SIMT Stack Launch

**File**: `src/abstract_hardware_model.cc` (lines 200-250)

```cpp
void simt_stack::launch(address_type start_pc, const simt_mask_t &active_mask) {
    reset();
    simt_stack_entry entry;
    entry.m_pc = start_pc;
    entry.m_active_mask = active_mask;
    entry.m_recvg_pc = -1;  // No reconvergence yet
    entry.m_calldepth = 0;
    entry.m_type = STACK_ENTRY_TYPE_NORMAL;
    m_stack.push_back(entry);
}
```

### 2.3 Post-Dominator Based Reconvergence

**File**: `src/cuda-sim/ptx_ir.cc` (lines 400-600)

```cpp
void function_info::do_pdom() {
    create_basic_blocks();
    connect_basic_blocks();
    
    bool modified = false;
    do {
        find_dominators();
        find_idominators();
        modified = connect_break_targets();
    } while (modified == true);
    
    find_postdominators();
    find_ipostdominators();
    
    // Pre-decode instructions to compute reconvergence points
    for (unsigned ii = 0; ii < m_n; ii += m_instr_mem[ii]->inst_size()) {
        ptx_instruction *pI = m_instr_mem[ii];
        pI->pre_decode();
    }
}

void function_info::find_postdominators() {
    // Algorithm from Muchnick's "Advanced Compiler Design & Implementation"
    // Figure 7.14 - Post-dominator computation
    
    // Exit block post-dominates itself
    m_basic_blocks.back()->post_dominator_ids.insert(
        m_basic_blocks.back()->bb_id);
    
    // Initialize all other blocks to post-dominate everything
    for (auto bb_itr = m_basic_blocks.begin(); 
         bb_itr != m_basic_blocks.end() - 1; bb_itr++) {
        for (unsigned i = 0; i < m_basic_blocks.size(); i++)
            (*bb_itr)->post_dominator_ids.insert(i);
    }
    
    // Iterative computation of post-dominators
    bool change = true;
    while (change) {
        change = false;
        for (int h = m_basic_blocks.size() - 2; h >= 0; --h) {
            std::set<int> T;
            for (unsigned i = 0; i < m_basic_blocks.size(); i++) T.insert(i);
            
            for (auto s = m_basic_blocks[h]->successor_ids.begin();
                 s != m_basic_blocks[h]->successor_ids.end(); s++)
                intersect(T, m_basic_blocks[*s]->post_dominator_ids);
            
            T.insert(h);
            if (!is_equal(T, m_basic_blocks[h]->post_dominator_ids)) {
                change = true;
                m_basic_blocks[h]->post_dominator_ids = T;
            }
        }
    }
}
```

### 2.4 SIMT Stack Update on Branch

**File**: `src/abstract_hardware_model.cc` (lines 250-350)

```cpp
void simt_stack::update(simt_mask_t &thread_done, addr_vector_t &next_pc,
                        address_type recvg_pc, op_type next_inst_op,
                        unsigned next_inst_size, address_type next_inst_pc) {
    // This function is called when a branch instruction is executed
    // It updates the SIMT stack to track divergent paths
    
    simt_stack_entry &current = m_stack.back();
    
    // Check if threads are diverging
    simt_mask_t taken_mask, not_taken_mask;
    // ... compute masks based on thread predicates ...
    
    if (taken_mask.any() && not_taken_mask.any()) {
        // Divergence detected - push new entries
        simt_stack_entry new_entry;
        new_entry.m_pc = /* target PC for taken path */;
        new_entry.m_active_mask = taken_mask;
        new_entry.m_recvg_pc = recvg_pc;  // Post-dominator reconvergence point
        new_entry.m_calldepth = current.m_calldepth;
        new_entry.m_type = STACK_ENTRY_TYPE_NORMAL;
        new_entry.m_branch_div_cycle = gpu_sim_cycle;
        
        m_stack.push_back(new_entry);
        
        // Update current entry for not-taken path
        current.m_pc = next_inst_pc + next_inst_size;
        current.m_active_mask = not_taken_mask;
    }
}
```

---

## 3. Barrier Implementation

### 3.1 CTA-Level Barrier Tracking

**File**: `src/cuda-sim/ptx_sim.h` (lines 85-105)

```cpp
class ptx_cta_info {
    unsigned m_bar_threads;      // Count of threads at barrier
    unsigned long long m_uid;    // CTA unique ID
    unsigned m_sm_idx;           // SM index
    
    std::set<ptx_thread_info*> m_threads_in_cta;
    std::set<ptx_thread_info*> m_threads_that_have_exited;
    
public:
    void inc_bar_threads() { m_bar_threads++; }
    void reset_bar_threads() { m_bar_threads = 0; }
    unsigned get_bar_threads() const { return m_bar_threads; }
};
```

### 3.2 Per-Thread Barrier State

**File**: `src/cuda-sim/ptx_sim.h` (lines 290-310)

```cpp
class ptx_thread_info {
    int m_barrier_num;      // Which barrier (-1 = not at barrier)
    bool m_at_barrier;      // True if thread is waiting
    
    // Called when thread executes bar.sync
    void set_at_barrier(int barrier_num) { 
        m_barrier_num = barrier_num;
        m_at_barrier = true;
    }
    void clear_barrier() { 
        m_barrier_num = -1;
        m_at_barrier = false;
    }
    bool is_at_barrier() const { return m_at_barrier; }
};
```

### 3.3 Warp-Level Barrier

**File**: `src/gpgpu-sim/shader.h` (lines 180-220 in `shd_warp_t`)

```cpp
class shd_warp_t {
    bool m_membar;              // Warp waiting at memory barrier
    warp_inst_t m_inst_at_barrier;  // Instruction that caused barrier
    
    unsigned m_n_atomic;        // Outstanding atomic operations
    
public:
    void set_membar() { m_membar = true; }
    void clear_membar() { m_membar = false; }
    bool get_membar() const { return m_membar; }
    
    void store_info_of_last_inst_at_barrier(const warp_inst_t *pI) {
        m_inst_at_barrier = *pI;
    }
    
    bool waiting() {  // Checks if warp is waiting
        return m_membar || /* other wait conditions */;
    }
};
```

### 3.4 Barrier Semantics

**From PTX ISA Documentation (referenced in GPGPU-Sim):**

```
bar.sync [bar_id], count;

Semantics:
1. Suspend thread execution until all specified threads arrive
2. A memory fence is automatically inserted BEFORE the barrier
3. All memory writes BEFORE bar.sync are visible AFTER barrier
4. ALL threads in CTA must execute the same barrier instruction
```

**Key Implementation Detail:**

> "Barrier must occur AFTER reconvergence of all divergent paths. Barrier implies memory fence."

---

## 4. PC Management

### 4.1 Per-Thread PC

**File**: `src/cuda-sim/ptx_sim.h` (lines 250-290)

```cpp
class ptx_thread_info {
    unsigned m_PC;      // Current PC
    unsigned m_NPC;     // Next PC (set by branch instructions)
    unsigned m_RPC;     // Return PC for calls
    
public:
    unsigned get_pc() const { return m_PC; }
    void set_npc(unsigned npc) { m_NPC = npc; }
    void update_pc() { m_PC = m_NPC; }
    
    unsigned get_rpc() const { return m_RPC; }
    void clearRPC() { m_RPC = -1; }
};
```

### 4.2 Next PC Query (Per Thread)

**File**: `src/gpgpu-sim/shader.cc` (lines 720-750)

```cpp
address_type shader_core_ctx::next_pc(int tid) const {
    if (tid == -1) return -1;
    ptx_thread_info *the_thread = m_thread[tid];
    if (the_thread == NULL) return -1;
    return the_thread->get_pc();  // PC already updated to next PC
}
```

### 4.3 Divergent PC Management

**Design Pattern:**

```
1. All threads in warp start with same PC (unified execution)
2. On divergent branch:
   - SIMT stack pushed with taken/not-taken paths
   - Each path has different active mask but shared PC
3. When switching paths (via SIMT stack pop):
   - Warp PC updated to the other path's PC
   - Active mask switched to that path's threads
4. At reconvergence:
   - Both paths have reached same PC
   - Threads remerge, single active mask
```

---

## 5. Warp Scheduler Design

### 5.1 Scheduler Base Class

**File**: `src/gpgpu-sim/shader.h` (lines 500-650)

```cpp
class scheduler_unit {
protected:
    std::vector<shd_warp_t *> m_supervised_warps;
    std::vector<shd_warp_t *> m_next_cycle_prioritized_warps;
    std::vector<shd_warp_t *>::const_iterator m_last_supervised_issued;
    
    Scoreboard *m_scoreboard;           // Register dependency tracking
    simt_stack **m_simt_stack;          // Per-warp SIMT stacks
    std::vector<shd_warp_t *> *m_warp;  // Warp array
    
    // Pipeline register outputs
    register_set *m_sp_out;    // SP (single-precision) unit
    register_set *m_dp_out;    // DP (double-precision) unit
    register_set *m_sfu_out;   // SFU (special function) unit
    register_set *m_mem_out;   // Memory unit
    
    int m_id;  // Scheduler ID
    
public:
    virtual void order_warps() = 0;  // Pure virtual - policy specific
    void cycle();  // Main scheduling cycle
    
protected:
    virtual void do_on_warp_issued(unsigned warp_id, unsigned num_issued,
                                   const std::vector<shd_warp_t *>::const_iterator &iter);
    
    shd_warp_t &warp(int i);
};
```

### 5.2 Scheduler Implementations

**File**: `src/gpgpu-sim/shader.h` (lines 650-750)

```cpp
// Loose Round Robin
class lrr_scheduler : public scheduler_unit {
public:
    virtual void order_warps() {
        order_lrr(m_next_cycle_prioritized_warps, m_supervised_warps,
                  m_last_supervised_issued, m_supervised_warps.size());
    }
};

// Greedy Then Oldest
class gto_scheduler : public scheduler_unit {
public:
    virtual void order_warps() {
        order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                         m_last_supervised_issued, m_supervised_warps.size(),
                         ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                         sort_warps_by_oldest_dynamic_id);
    }
};

// Two-Level Active Scheduler
class two_level_active_scheduler : public scheduler_unit {
    std::deque<shd_warp_t *> m_pending_warps;
    scheduler_prioritization_type m_inner_level_prioritization;
    scheduler_prioritization_type m_outer_level_prioritization;
    unsigned m_max_active_warps;
    
public:
    virtual void order_warps() {
        // Inner level: prioritize among active warps
        // Outer level: swap between active and pending warps
    }
};
```

### 5.3 Main Scheduler Cycle

**File**: `src/gpgpu-sim/shader.cc` (lines 800-900)

```cpp
void scheduler_unit::cycle() {
    // 1. Order warps according to scheduling policy
    order_warps();
    
    // 2. Try to issue from prioritized warps
    for (auto &warp : m_next_cycle_prioritized_warps) {
        unsigned warp_id = warp->get_warp_id();
        
        // Skip if warp is at barrier
        if (warp->get_membar()) continue;
        
        // Skip if warp has no valid instructions
        if (!warp->ibuffer_next_valid()) continue;
        
        // Skip if warp has instruction cache miss
        if (warp->imiss_pending()) continue;
        
        // Skip if scoreboard says dependencies not ready
        if (m_scoreboard->checkCollision(warp_id, inst)) continue;
        
        // Get the instruction
        const warp_inst_t *inst = warp->ibuffer_next_inst();
        
        // Determine execution unit
        enum exec_unit_type_t exec_unit_type = inst->get_exec_unit_type();
        
        // Try to issue to appropriate unit
        if (can_issue_to_unit(exec_unit_type)) {
            issue_instruction(inst, warp_id);
            do_on_warp_issued(warp_id, 1, iter);
        }
    }
}
```

### 5.4 Scoreboard (Dependency Tracking)

**File**: `src/gpgpu-sim/scoreboard.h` (referenced in shader.h)

```cpp
class Scoreboard {
public:
    bool checkCollision(unsigned warp_id, const warp_inst_t *inst) const;
    void reserveRegisters(unsigned warp_id, const warp_inst_t *inst);
    void releaseRegisters(unsigned warp_id, const warp_inst_t *inst);
    void releaseRegister(unsigned warp_id, unsigned regnum);
    
private:
    std::vector<std::set<unsigned>> m_reserved_regs;  // Per-warp reserved registers
    unsigned m_sid;  // Shader ID
    class gpgpu_sim *m_gpu;
};
```

### 5.5 Scheduler Creation

**File**: `src/gpgpu-sim/shader.cc` (lines 300-400)

```cpp
void shader_core_ctx::create_schedulers() {
    m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader, m_gpu);
    
    // Parse scheduler config
    std::string sched_config = m_config->gpgpu_scheduler_string;
    const concrete_scheduler scheduler = 
        sched_config.find("lrr") != std::string::npos ? CONCRETE_SCHEDULER_LRR :
        sched_config.find("two_level_active") != std::string::npos ? 
            CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE :
        sched_config.find("gto") != std::string::npos ? CONCRETE_SCHEDULER_GTO :
        sched_config.find("rrr") != std::string::npos ? CONCRETE_SCHEDULER_RRR :
        sched_config.find("old") != std::string::npos ? 
            CONCRETE_SCHEDULER_OLDEST_FIRST :
        sched_config.find("warp_limiting") != std::string::npos ? 
            CONCRETE_SCHEDULER_WARP_LIMITING :
        NUM_CONCRETE_SCHEDULERS;
    
    // Create schedulers
    for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++) {
        switch (scheduler) {
            case CONCRETE_SCHEDULER_LRR:
                schedulers.push_back(new lrr_scheduler(...));
                break;
            case CONCRETE_SCHEDULER_GTO:
                schedulers.push_back(new gto_scheduler(...));
                break;
            // ... other schedulers ...
        }
    }
    
    // Distribute warps to schedulers
    for (unsigned i = 0; i < m_warp.size(); i++) {
        schedulers[i % m_config->gpgpu_num_sched_per_core]
            ->add_supervised_warp_id(i);
    }
}
```

---

## 6. Key Data Structures Summary

### 6.1 Core Classes

| Class | File | Purpose |
|-------|------|---------|
| `shd_warp_t` | shader.h | Warp state (PC, active mask, barrier) |
| `ptx_thread_info` | ptx_sim.h | Thread state (PC, registers, barrier flag) |
| `simt_stack` | abstract_hardware_model.h | Divergence tracking and reconvergence |
| `scheduler_unit` | shader.h | Base scheduler class |
| `Scoreboard` | scoreboard.h | Register dependency tracking |
| `ptx_cta_info` | ptx_sim.h | CTA-level barrier tracking |

### 6.2 Key Member Variables

| Variable | Type | Description |
|----------|------|-------------|
| `m_next_pc` | address_type | Unified PC for warp (shd_warp_t) |
| `m_active_threads` | simt_mask_t | Active thread mask (bitset<32>) |
| `m_membar` | bool | Warp waiting at barrier |
| `m_PC` | unsigned | Thread's current PC |
| `m_NPC` | unsigned | Thread's next PC |
| `m_barrier_num` | int | Barrier ID thread is waiting at (-1 = none) |
| `m_stack` | deque<simt_stack_entry> | SIMT stack entries |
| `m_recvg_pc` | address_type | Reconvergence PC for divergent path |

---

## 7. Execution Flow Summary

### 7.1 Normal Execution

```
1. Scheduler selects ready warp
2. Warp issues instruction at m_next_pc
3. All active threads execute same instruction
4. PC advances to next instruction
5. Repeat
```

### 7.2 Divergent Branch

```
1. Branch instruction executed with thread predicates
2. Some threads take branch, others don't
3. simt_stack::update() called:
   - Computes taken/not_taken masks
   - Pushes new stack entry for taken path
   - Current entry becomes not-taken path
   - Sets reconvergence PC (post-dominator)
4. Warp continues with not-taken path
5. At reconvergence PC, stack popped
6. Taken path executed
7. At reconvergence, threads remerged
```

### 7.3 Barrier Execution

```
1. Thread executes bar.sync
2. Thread sets m_at_barrier = true
3. CTA increments barrier count
4. Warp checks if all threads at barrier
5. If not, warp sets m_membar = true
6. Scheduler skips warp (get_membar() == true)
7. When all threads arrive:
   - Barrier count reset
   - All threads clear m_at_barrier
   - Warps clear m_membar
   - Execution resumes
```

---

## 8. Comparison with PTX-EMU v2.0

| Feature | GPGPU-Sim | PTX-EMU v2.0 |
|---------|-----------|--------------|
| PC Management | Per-Warp PC | Per-Thread PC |
| Reconvergence | SIMT Stack with post-dominator | CFG-based explicit |
| Barrier | CTA-level counting | Convergence + memory fence |
| Scheduler | Multiple policies | Round-robin + barrier-aware |
| Divergence Handling | Hardware SIMT stack | Software-managed stack |
| Thread Scheduling | Warp-level | Individual thread-level |

---

## 9. References

### GPGPU-Sim Source Files

1. **Shader Core**: `src/gpgpu-sim/shader.cc`, `src/gpgpu-sim/shader.h`
2. **SIMT Stack**: `src/abstract_hardware_model.h`, `src/abstract_hardware_model.cc`
3. **Thread Model**: `src/cuda-sim/ptx_sim.cc`, `src/cuda-sim/ptx_sim.h`
4. **Control Flow**: `src/cuda-sim/ptx_ir.cc`, `src/cuda-sim/ptx_ir.h`
5. **GPU Simulation**: `src/gpgpu-sim/gpu-sim.cc`, `src/gpgpu-sim/gpu-sim.h`
6. **Stack Utilities**: `src/gpgpu-sim/stack.cc`, `src/gpgpu-sim/stack.h`

### Key Papers

1. Fung et al., "Dynamic Warp Formation and Scheduling for Efficient GPU Control Flow", MICRO 2007
2. Muchnick, "Advanced Compiler Design and Implementation", 1997 (Dominator algorithms)
3. NVIDIA PTX ISA 9.1 Documentation

---

**Document Version**: 1.0  
**Last Updated**: 2026-04-10  
**Author**: PTX-EMU Architecture Team
