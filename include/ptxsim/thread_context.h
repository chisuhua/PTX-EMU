#ifndef THREAD_CONTEXT_H
#define THREAD_CONTEXT_H

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h" // 包含通用类型定义
#include "ptxsim/contexts/exec_state.h"
#include "ptxsim/contexts/memory_ref.h"
#include "ptxsim/contexts/program_ref.h"
#include "ptxsim/contexts/register_predicate.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/ptx_config.h"
#include "ptxsim/warp_state.h"

class PtxInterpreter;
class WarpContext;
class SMContext;
class CTAContext;

#include "register/condition_code_register.h"
#include "register/register_bank_manager.h"
#include "utils/logger.h"
#include <any>
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

class ThreadContext {
public:
    // 资源管理
    std::vector<StatementContext> *statements;
    std::map<std::string, std::unique_ptr<Symtable>> *name2Sym;
    std::map<std::string, std::unique_ptr<Symtable>>
        *name2Share; // 添加共享内存符号表引用

    // 使用寄存器银行管理器或独立寄存器管理器
    std::shared_ptr<RegisterBankManager> register_bank_manager_;
    std::map<std::string, int> label2pc;

    // 线程状态
    Dim3 BlockIdx, ThreadIdx, GridDim, BlockDim;
    int bar_id;
    EXE_STATE state;

    // 条件码寄存器
    ConditionCodeRegister cc_reg;

    // 当前指令执行状态
    // 临时数据存储：VEC 操作数各元素的物理地址，每调用一次 acquire_operand
    // for VEC 就 push 一个新向量（最多 4 个槽位，PTX 最大 V4）。handler 通过
    // op[0]/op[1] 拿到的指针直接指向这里，cast 成 void** 后按 V2/V4 限定符
    // 迭代。
    // BUGFIX: 之前用 std::queue<FIFO>，导致 mov.b64 带 vector 源会 push 不
    // pop， 留下 stale entry，下一条 V4 LD/ST pop 到错误 entry。改成
    // per-ThreadContext 栈式存储（push 必配 pop）后，vector LD/ST 始终读自己刚
    // push 的 buffer。
    std::vector<std::vector<void *>> vecOp_phy_addrs;

    // warp和lane标识
    int warp_id_;
    int lane_id_;

    // 共享内存基地址
    void *shared_mem_space = nullptr;

    // 本地内存基地址
    void *local_mem_space = nullptr;

    // 指向warp的指针，用于访问SMContext
    WarpContext *warp_context_ = nullptr;

    // 函数调用栈
    std::stack<int> call_stack;

    void
    init(Dim3 &blockIdx, Dim3 &threadIdx, Dim3 GridDim, Dim3 BlockDim,
         std::vector<StatementContext> &statements,
         std::map<std::string, std::unique_ptr<Symtable>> *name2Sym,
         std::map<std::string, int> &label2pc,
         std::map<std::string, std::unique_ptr<Symtable>> *name2Share = nullptr,
         CTAContext *cta_ctx = nullptr);

    // EXE_STATE exe_once();
    void clear_temporaries();

    // 通用访问接口
    void *acquire_operand(const OperandContext &op,
                          const std::vector<Qualifier> &qualifiers);
    void collect_operands(StatementContext &stmt,
                          const std::vector<OperandContext> &operands,
                          const std::vector<Qualifier> *qualifier);

    void commit_operand(StatementContext &stmt, const OperandContext &operand,
                        const std::vector<Qualifier> &qualifier);
    // void *get_memory_addr(OperandContext::FA *fa,
    //                       std::vector<Qualifier> &qualifiers);
    // void *acquire_register(OperandContext::REG *reg,
    //                        std::vector<Qualifier> qualifier);

    void *get_memory_addr(const AddrOperand &op,
                          const std::vector<Qualifier> &qualifiers);
    // 寄存器访问接口
    void *acquire_register(const RegOperand &op,
                           std::vector<Qualifier> qualifier);

    // Shared memory初始化
    void initialize_shared_memory(const std::string &name, uint64_t address);

    // 设置本地内存空间
    void set_local_memory_space(void *local_mem_space);

    // 通用操作
    void mov_data(void *src, void *dst, std::vector<Qualifier> &qualifiers);
    void trace_instruction(StatementContext &statement);

    // 辅助函数接口（供指令处理器使用）
    void mov(void *from, void *to, const std::vector<Qualifier> &q);
    bool isIMMorVEC(OperandContext &op);

    // 新增：为断点条件准备上下文
    void prepare_breakpoint_context(
        std::unordered_map<std::string, std::any> &context);

    // 新增：转储线程状态
    void dump_state(std::ostream &os) const;

    // Maximum operands per instruction
    static constexpr size_t MAX_OPERANDS_PER_INSTR = 4;

    std::vector<void *>
        operand_collected; // collect operand addr  from BASE_INSTR operands

    // Track which operands are immediate (vs register/variable)
    // For immediate operands, operand_collected[i] is a pointer to the
    // immediate value For register/variable operands, operand_collected[i] is
    // the actual value/address Usage example:
    //   if (operand_is_immediate_[i]) {
    //       // Direct access: operand_collected[i] points to immediate value
    //       int val = *static_cast<int*>(operand_collected[i]);
    //   } else {
    //       // Register/variable access: operand_collected[i] is the address
    //       int val = *static_cast<int*>(operand_collected[i]);
    //   }
    // Note: Using char instead of bool to avoid std::vector<bool> bit
    // compression
    std::vector<char> operand_is_immediate_;

    // 新增：打印指令状态
    void print_instruction_status(StatementContext &stmt);

    // 新增：模板函数用于打印线程状态信息
    template <typename... Args>
    void trace_status(ptxsim::log_level level, const std::string &component,
                      const char *fmt, Args &&...args) {
        // 检查当前线程的lane是否在trace_lanes掩码中
        if (!ptxsim::DebugConfig::get().is_lane_traced(lane_id_)) {
            // 如果lane不在掩码中，则不输出任何内容
            return;
        }

        // 首先格式化用户提供的消息
        std::string formatted_msg =
            ptxsim::detail::printf_format(fmt, std::forward<Args>(args)...);

        // 然后构建包含线程和块维度信息的消息
        std::string full_msg = ptxsim::detail::printf_format(
            "[%d,%d,%d][%d,%d,%d][%d,%d] %s", BlockIdx.x, BlockIdx.y,
            BlockIdx.z, ThreadIdx.x, ThreadIdx.y, ThreadIdx.z, warp_id_,
            lane_id_, formatted_msg.c_str());

        // 使用logger的printf_to_logger_simple函数打印消息
        ptxsim::printf_to_logger_simple(level, component, "%s",
                                        full_msg.c_str());
    }

    // 新增接口：获取线程状态
    EXE_STATE get_state() const { return state; }

    // 检查是否活跃
    bool is_active() const { return state != EXIT; }

    // 检查是否在屏障等待
    bool is_at_barrier() const { return state == BAR_SYNC; }

    // 检查是否退出
    bool is_exited() const { return state == EXIT; }

    // 设置线程状态
    void set_state(EXE_STATE new_state) { state = new_state; }

    // 获取PC值（通过WarpState）
    int get_pc() const;

    // 设置PC值（通过WarpState）
    void set_pc(int new_pc);

    // 获取下一个PC值（通过WarpState）
    int get_next_pc() const;

    // 设置下一个PC值（通过WarpState）
    void set_next_pc(int new_next_pc);

    // 【单一入口】提交 PC 变更：pc ← next_pc
    // 这是正常执行期间 PC 推进的唯一方法。
    // 原始 set_pc() 仅用于初始化、同步和重置。
    void commit_pc();

    // 【强制写入】仅设置 PC（不修改 next_pc）
    // 用于 warp 级操作（barrier 完成）对非当前线程的直接写入
    // 注意：当前线程应通过 set_next_pc() + commit_pc() 流程推进 PC
    [[deprecated("Use set_pc() instead which writes both pc and next_pc")]]
    void force_set_pc(int new_pc);

    // 获取线程索引
    Dim3 get_thread_idx() const { return ThreadIdx; }

    // 获取块索引
    Dim3 get_block_idx() const { return BlockIdx; }

    // 检查条件码寄存器
    const ConditionCodeRegister &get_condition_codes() const { return cc_reg; }

    // 设置条件码寄存器
    void set_condition_codes(const ConditionCodeRegister &new_cc) {
        cc_reg = new_cc;
    }

    // 检查PC是否有效
    bool is_valid_pc() const {
        int current_pc = get_pc();
        return statements != nullptr && current_pc >= 0 &&
               current_pc < static_cast<int>(statements->size());
    }

    bool is_valid_pc(int p) const {
        return statements != nullptr && p >= 0 &&
               p < static_cast<int>(statements->size());
    }

    int statements_size() const {
        return statements ? static_cast<int>(statements->size()) : 0;
    }

    StatementContext *get_statement_at(int p) {
        if (statements != nullptr && p >= 0 &&
            p < static_cast<int>(statements->size())) {
            return &(*statements)[p];
        }
        return nullptr;
    }

    // 获取当前指令
    StatementContext *get_current_statement() {
        int current_pc = get_pc();
        if (statements != nullptr && current_pc >= 0 &&
            current_pc < static_cast<int>(statements->size())) {
            return &(*statements)[current_pc];
        }
        return nullptr;
    }

    // 执行单条指令（由WarpContext调用）
    EXE_STATE execute_thread_instruction();

    // 重置线程状态
    void reset();
    void
    set_register_bank_manager(std::shared_ptr<RegisterBankManager> manager) {
        register_bank_manager_ = manager;
    }
    std::shared_ptr<RegisterBankManager> get_register_bank_manager() const {
        return register_bank_manager_;
    }
    int get_warp_id() const { return warp_id_; }

    // Set during collect_operands, read by SetpHandler for predicate writes
    const std::string &get_dst_operand_reg_name() const {
        return dst_operand_reg_name_;
    }

    // 设置warp上下文
    void set_warp_context(WarpContext *warp_ctx) { warp_context_ = warp_ctx; }

    // 获取warp上下文
    WarpContext *get_warp_context() const { return warp_context_; }

    // 【Stage 4】从 warp_state 同步 PC 和状态到 ThreadContext
    void sync_from_warp_state();

    // 【Stage 4】将 ThreadContext 的 PC 和状态同步到 warp_state
    void sync_to_warp_state();

private:
    void _execute_once();
    bool is_immediate_or_vector(OperandContext &op);
    // 用于存储已收集的寄存器地址，避免重复分配
    // std::map<std::string, void *> cached_register_addrs;

    // 指向CTAContext的指针，用于访问本地内存符号表
    CTAContext *cta_context_ = nullptr;

    std::string dst_operand_reg_name_;

public:
    // T2-3 A3a: POD-only members at the END of class. Destruction is
    // reverse of declaration order, so PODs (with default-constructed
    // POD types) destroy first, then legacy fields (including
    // shared_ptr<>s and containers). Placed in a re-opened public
    // section so tests can read POD defaults. Legacy fields above
    // remain the canonical source until A3c removes them.
    ptxsim::contexts::ExecStatePod exec_state_;
    ptxsim::contexts::RegisterPredicatePod reg_pred_;
    ptxsim::contexts::MemoryPod memory_;
    ptxsim::contexts::ProgramRefPod program_ref_;
};

#endif // THREAD_CONTEXT_H