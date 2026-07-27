#ifndef WARP_CONTEXT_H
#define WARP_CONTEXT_H

#include "ptxsim/contexts/backend_links.h"
#include "ptxsim/contexts/lane_mask.h"
#include "ptxsim/contexts/warp_identity.h"
#include "register/register_bank_manager.h"
#include "simt_stack.h"
#include "thread_context.h"
#include "warp_state.h"
#include <array>
#include <map>
#include <memory>
#include <queue>
#include <vector>

// Forward declarations to avoid circular includes
class SMContext;
class CTAContext;
class ThreadContext;
class WarpScheduler;

// Forward declaration of warp_active_mask helper namespace
// (refactor-warp-context C-18 Phase 1 extraction).
namespace warp_active_mask {
    void set_active_mask_lane(WarpContext* w, int lane_id, bool active);
    void update_active_mask(WarpContext* w);
    uint32_t get_active_mask_u32(const WarpContext* w);
    void set_active_mask_u32(WarpContext* w, uint32_t mask);
}

// Forward declaration of warp_simt helper namespace
// (refactor-warp-context C-18 Phase 2 extraction).
namespace warp_simt {
    bool check_reconvergence(WarpContext* w);
}  // namespace warp_simt

// Forward declaration of warp_dispatch helper namespace
// (refactor-warp-context C-18 Phase 3 extraction).
class StatementContext;
namespace warp_dispatch {
    void execute_warp_instruction(WarpContext* w, StatementContext& stmt, int target_pc);
}  // namespace warp_dispatch

class WarpContext {
public:
    static constexpr int WARP_SIZE = 32;

    WarpContext();
    virtual ~WarpContext() = default;

    // 添加线程到 warp
    void add_thread(std::unique_ptr<ThreadContext> thread, int lane_id);

    // 执行 warp 的一条指令
    void execute_warp_instruction(StatementContext &stmt, int target_pc = -1);

    // 【SIMT v2.0】处理分支指令 (warp 级操作)
    void handle_branch(const std::string &predicate, bool predicate_negated,
                       int target_pc, int reconvergence_pc,
                       int current_inst_pc = -1);

    // 获取 warp 中的线程
    ThreadContext *get_thread(int lane_id) const {
        if (lane_id >= 0 && lane_id < threads.size()) {
            return threads[lane_id].get();
        }
        return nullptr;
    }

    // 检查 warp 是否活跃
    bool is_active() const { return active_count > 0; }

    // 【NEW】检查 warp 是否准备好取指 (所有活跃线程的 pc == next_pc)
    bool is_warp_ready_to_fetch() const;

    // 获取活跃线程数量
    int get_active_count() const { return active_count; }

    // Removed 2026-07-XX — dead-code-cleanup (Fix #1)
    // WarpContext::get_pc() and set_pc() removed — zero production refs.
    // Replaced by: warp_state.threads[lane_id].pc + advance_thread_pc()

    // 【NEW】获取每线程 PC
    uint32_t get_thread_pc(int lane_id) const {
        if (lane_id >= 0 && lane_id < WARP_SIZE) {
            return warp_state.threads[lane_id].pc;
        }
        return 0;
    }

    // 【NEW】设置每线程 PC (legacy, does NOT sync ThreadContext)
    // Prefer advance_thread_to() which keeps both sources consistent
    [[deprecated(
        "Use advance_thread_pc() instead — unified PC advancement path")]]
    void set_thread_pc(int lane_id, uint32_t new_pc) {
        advance_thread_pc(lane_id, new_pc);
    }

    // 【UNIFIED PC】 Advance a single thread's PC, updating both warp_state
    // and ThreadContext simultaneously to prevent dual-PC inconsistencies.
    // This is the only approved way to change a thread's PC after
    // initialization.
    void advance_thread_pc(int lane_id, int new_pc);

    // 【NEW】获取执行掩码
    uint32_t get_exec_mask() const { return warp_state.exec_mask; }

    // 【NEW】设置执行掩码
    void set_exec_mask(uint32_t mask) { warp_state.exec_mask = mask; }

    // 【NEW】检查 lane 是否可调度
    bool is_lane_schedulable(int lane_id) const {
        if (lane_id >= 0 && lane_id < WARP_SIZE) {
            return warp_state.threads[lane_id].is_schedulable();
        }
        return false;
    }

    // 【NEW】获取可调度 lane 数量
    int count_schedulable_lanes() const {
        return warp_state.count_schedulable_lanes();
    }

    // Divergent execution support
    std::map<int, std::vector<int>> get_lanes_by_pc() const;
    std::vector<int> get_unique_pcs() const;

    // 【SIMT v2.0】Check if divergent threads have reconverged, pop SIMT stack
    // if so. Returns true if an entry was popped, false otherwise.
    bool check_reconvergence();

    bool
    check_and_block_at_reconvergence_point(int target_pc,
                                           std::vector<int> &blocked_lanes);

    // 更新活跃掩码（例如，遇到分支指令时）
    void update_active_mask();

    // 【NEW】Decrement blocked_cycles_remaining for all threads and unblock
    // when count hits 0. Extracted from sm_context.cpp:180-197 (B4.1 Bug #2 +
    // #3 fix) so it can be unit-tested without depending on sm_context
    // internals. This function does NOT update WarpContext::active_count — the
    // caller must invoke update_active_mask() afterwards if needed.
    static void decrement_blocked_cycles(ptxsim::WarpState &ws);

    // 【NEW】CppTLM D1-Full injection (ADR-0020, Phase 8.B PTX-5a):
    // Set blocked_cycles_remaining + is_blocked for ALL active+non-blocked threads
    // in the warp. Replaces per-thread LD-only path. Calls update_active_mask()
    // to keep active_mask[] / active_count synchronized (T2-1 contract).
    void set_blocked_cycles_for_active(uint32_t cycles);

    // 设置活跃掩码
    void set_active_mask(int lane_id, bool active);

    // 检查特定 lane 是否活跃（委托给 warp_state 权威源，避免 active_mask[]
    // 缓存滞后） 详见 ISSUE-005 — 必须立即反映 warp_state
    // 变更（屏障释放、退出等），不能等下一周期 update_active_mask() 同步。
    bool is_lane_active(int lane_id) const {
        return is_lane_schedulable(lane_id);
    }

    // 获取 warp 内线程 ID
    int get_warp_thread_id(int lane_id) const {
        return lane_id < WARP_SIZE ? warp_thread_ids[lane_id] : -1;
    }

    // 获取 warp 索引
    int get_warp_id() const { return warp_id; }

    // 设置 warp 索引
    void set_warp_id(int id) { warp_id = id; }

    // 重置 warp 状态
    void reset();

    // 检查 warp 是否完成 - 现在检查是否所有线程都已退出
    bool is_finished() const;

    // 检查 warp 是否真正完成（所有线程都已退出），而不是仅活跃计数为 0
    bool is_all_threads_exited() const;

    // 同步 warp 内所有线程
    void sync_threads();

    // 检查是否有分歧
    bool has_divergence() const { return divergence_detected; }

    // 获取活跃掩码（32 位）
    uint32_t get_active_mask() const;

    // 设置活跃掩码（32 位）
    void set_active_mask(uint32_t mask);

    // 设置寄存器银行管理器
    void
    set_register_bank_manager(std::shared_ptr<RegisterBankManager> manager) {
        register_bank_manager_ = manager;
    }

    std::shared_ptr<RegisterBankManager> get_register_bank_manager() const {
        return register_bank_manager_;
    }

    // 获取 warp 中所有线程的引用
    const std::vector<std::unique_ptr<ThreadContext>> &get_threads() const {
        return threads;
    }

    // 获取指定范围内的活跃线程
    std::vector<ThreadContext *> get_active_threads() const {
        std::vector<ThreadContext *> active_threads;
        for (int i = 0; i < threads.size(); ++i) {
            if (is_lane_active(i) && threads[i]) {
                active_threads.push_back(threads[i].get());
            }
        }
        return active_threads;
    }

    // 设置 SM Context
    void set_sm_context(SMContext *sm_ctx) { sm_context_ = sm_ctx; }

    // 获取 SM Context
    SMContext *get_sm_context() const { return sm_context_; }

    // 设置 CTA Context（反向链接，用于 barrier 模块访问）
    void set_cta_context(CTAContext *cta_ctx) { cta_context_ = cta_ctx; }

    // 获取 CTA Context
    CTAContext *get_cta_context() const { return cta_context_; }

    // 【NEW】获取 warp state 引用
    ptxsim::WarpState &get_warp_state() { return warp_state; }
    const ptxsim::WarpState &get_warp_state() const { return warp_state; }

    // 【NEW】SIMT Stack 访问
    ptxsim::SIMTStack &get_simt_stack() { return simt_stack; }
    const ptxsim::SIMTStack &get_simt_stack() const { return simt_stack; }

    // Friend declarations for warp_active_mask helper module (refactor-warp-context
    // C-18 Phase 1 extraction). The helper functions need direct access to
    // active_mask[] / warp_state / active_count for performance — wrapping every
    // access through public getters would add overhead in the per-instruction hot
    // path (update_active_mask is called at the end of every execute_warp_instruction).
    friend void warp_active_mask::set_active_mask_lane(WarpContext*, int, bool);
    friend void warp_active_mask::update_active_mask(WarpContext*);
    friend uint32_t warp_active_mask::get_active_mask_u32(const WarpContext*);
    friend void warp_active_mask::set_active_mask_u32(WarpContext*, uint32_t);
    friend bool warp_simt::check_reconvergence(WarpContext*);
    friend void warp_dispatch::execute_warp_instruction(WarpContext*, StatementContext&, int);

    // 【BARRIER RECONVERGENCE】Force all non-exited threads to reconverge at
    // barrier_pc + 1. This matches hardware behavior per sm90_100.md:294:
    // "bar.sync — 未汇合的 Warp 会在此被强制汇合" For multi-warp CTAs, threads
    // may arrive at barrier in different reconvergence states. This method
    // forces them all to continue from the instruction after the barrier.
    void force_reconvergence_at_barrier(int barrier_pc);

private:
    std::vector<std::unique_ptr<ThreadContext>>
        threads; // warp 中的线程 unique_ptr
    // 活跃掩码 — 与 warp_state.exec_mask 是两个独立机制，共同维护：
    // - exec_mask: WarpState 中的 uint32_t，用于 PTX activemask 指令返回值和
    // SIMT stack 管理
    // - active_mask[]: WarpContext 中的 bool[WARP_SIZE]，用于调度器的
    // is_lane_active() 判断 两者在以下时序点同步：
    //   1. barrier 释放时：barrier.cpp 同时调用 set_exec_mask() 和
    //   set_active_mask()
    //   2. 线程退出/阻塞时：update_active_mask() 根据 is_exited/is_blocked 重建
    //   active_mask[]
    // 注意：handle_branch() 发散时只更新 exec_mask，active_mask[] 在下一
    // execute_warp_instruction()
    //       周期通过 update_active_mask() 重建。这是已知行为，由
    //       test_post_barrier_divergence 测试验证。
    std::array<bool, WARP_SIZE> active_mask;
    std::array<int, WARP_SIZE> warp_thread_ids; // 对应的线程 ID
    int active_count;                           // 活跃线程数量
    int warp_id;                                // warp ID
    int physical_warp_id;                       // 物理 warp ID
    int physical_block_id;                      // 物理 block ID

    bool divergence_detected; // 分歧检测标志

    // 寄存器银行管理器
    std::shared_ptr<RegisterBankManager> register_bank_manager_;

    // 单步执行模式
    bool single_step_mode;

    // 指向 SMContext 的指针
    SMContext *sm_context_ = nullptr;
    CTAContext *cta_context_ = nullptr;

    // 调度状态
    bool is_scheduled_{false}; // 表示 warp 是否被调度执行

    // 【NEW】SIMT 架构升级：每线程状态
    ptxsim::WarpState warp_state;
    ptxsim::SIMTStack simt_stack; // SIMT control flow stack

public:
    // 调度状态相关方法
    void set_scheduled(bool scheduled) { is_scheduled_ = scheduled; }
    bool is_scheduled() const { return is_scheduled_; }

    // 物理 ID 管理方法
    void set_physical_warp_id(int id) { physical_warp_id = id; }

    // Phase 1 of implement-tcgen05-handlers-extended: tcgen05.alloc
    // permit accessors. Backed by `warp_state.allocate_permit`.
    void set_allocate_permit(bool permit) {
        warp_state.allocate_permit = permit;
    }
    bool get_allocate_permit() const {
        return warp_state.allocate_permit;
    }
    int get_physical_warp_id() const { return physical_warp_id; }

    // Phase 4 of implement-tcgen05-handlers-extended (ADR-0016, Oracle Q6-B):
    // tcgen05.fence no-op marker (design D8). Extension point — recorder only,
    // no membar / WarpBarrier interaction. Forward-compatible: kFenceUnknown
    // catches future PTX ISA §9.7.16 scope variants without raising exceptions.
    enum FencePosition : int8_t {
        kFenceNone = 0,        // No fence recorded (initial state)
        kFenceBefore = 1,      // tcgen05.fence::before_thread_sync
        kFenceAfter = 2,       // tcgen05.fence::after_thread_sync
        kFenceUnknown = 3      // Forward-compat bucket for future PTX ISA variants
    };
    // Backed by `warp_state.fence_position`. Single-writer (warp scheduler);
    // no mutex per ptx-lessons-learned §2 recursive-lock audit.
    void record_fence_position(FencePosition position) {
        warp_state.fence_position = static_cast<int8_t>(position);
    }
    FencePosition get_last_fence_position() const {
        return static_cast<FencePosition>(warp_state.fence_position);
    }

    // FU-3 C2 (Oracle Q5): implicit per-warp slot cursor for tcgen05.ld/.st/.cp
    //
    // Per Oracle Q5 verification (bench/cute/include/cute/arch/copy_sm100.hpp):
    // real Blackwell tcgen05.ld/st have NO slot operand in PTX. Slot management
    // is IMPLICIT — the warp scheduler tracks slots via register state.
    // This cursor implements that implicit tracking:
    //   - ld allocates next available slot (0, 1, 2, ...), records as last
    //   - st reads from last_ld_slot_ (the most recent ld target)
    //   - cp uses separate pool (slots 32+) to avoid overlap with ld/st
    //
    // Consumers: processTcgen05Ld (tcgen05.cpp:443), processTcgen05St
    // (tcgen05.cpp:485), processTcgen05Cp (tcgen05_cp.cpp:138).
    // Without this cursor, ld/st/cp write to hardcoded slot 0, breaking
    // FlashAttention QK^T→softmax→PV data flow (mma writes C to slot[64..95]).
    void reset_ld_cursor() { next_ld_slot_ = 0; last_ld_slot_ = 0; }
    size_t allocate_ld_slot() { return next_ld_slot_++; }
    size_t last_ld_slot() const { return last_ld_slot_; }
    void set_last_ld_slot(size_t slot) { last_ld_slot_ = slot; }
    size_t allocate_cp_slot() { return next_cp_slot_++; }
    void reset_cp_cursor() { next_cp_slot_ = 0; }

    void set_physical_block_id(int id) { physical_block_id = id; }
    int get_physical_block_id() const { return physical_block_id; }

    // T2-3 A4a: LaneMaskPod at END of class. Same destruction order
    // rationale as ThreadContext A3a: POD (default-constructed types)
    // destroys first, then legacy fields. Legacy fields above remain
    // canonical source until A4c removes them.
    ptxsim::contexts::LaneMaskPod lane_mask_;

    // FU-3 C2 (Oracle Q5): implicit per-warp ld/st/cp slot cursors.
    // ld allocates from next_ld_slot_ (0,1,2,...), st reads last_ld_slot_.
    // cp uses next_cp_slot_ (0..) offset by +32 to avoid ld/st overlap.
    size_t next_ld_slot_ = 0;
    size_t last_ld_slot_ = 0;
    size_t next_cp_slot_ = 0;
};

#endif // WARP_CONTEXT_H
