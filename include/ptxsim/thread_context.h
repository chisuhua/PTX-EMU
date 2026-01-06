#ifndef THREAD_CONTEXT_H
#define THREAD_CONTEXT_H

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h" // 包含通用类型定义
#include "ptxsim/execution_types.h"
#include "register/condition_code_register.h"
#include "register/register_manager.h"
#include <any>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

class PtxInterpreter; // 前向声明

class ThreadContext {
public:
    // 资源管理
    std::vector<StatementContext> *statements;
    std::map<std::string, Symtable *> *name2Share;
    std::map<std::string, Symtable *> name2Sym;
    RegisterManager register_manager;
    std::map<std::string, int> label2pc;

    // 线程状态
    Dim3 BlockIdx, ThreadIdx, GridDim, BlockDim;
    int pc;
    int next_pc;
    EXE_STATE state;

    // 条件码寄存器
    ConditionCodeRegister cc_reg;

    // 当前指令执行状态
    // 临时数据存储
    std::queue<VEC *> vec;

    void init(Dim3 &blockIdx, Dim3 &threadIdx, Dim3 GridDim, Dim3 BlockDim,
              std::vector<StatementContext> &statements,
              std::map<std::string, Symtable *> &name2Share,
              std::map<std::string, Symtable *> &name2Sym,
              std::map<std::string, int> &label2pc);

    // EXE_STATE exe_once();
    void clear_temporaries();

    // 通用访问接口
    void *acquire_operand(OperandContext &op,
                          std::vector<Qualifier> &qualifiers);
    void collect_operands(StatementContext &stmt,
                          std::vector<OperandContext> &operands,
                          std::vector<Qualifier> *qualifier);

    void commit_operand(StatementContext &stmt, OperandContext &operand,
                        std::vector<Qualifier> &qualifier);
    void *get_memory_addr(OperandContext::FA *fa,
                          std::vector<Qualifier> &qualifiers);

    // 寄存器访问接口
    void *acquire_register(OperandContext::REG *reg,
                           std::vector<Qualifier> qualifier);

    // Shared memory初始化
    void initialize_shared_memory(const std::string &name, uint64_t address);

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

    std::vector<void *>
        operand_collected; // collect operand addr  from BASE_INSTR operands

    // 新增接口：获取线程状态
    EXE_STATE get_state() const { return state; }

    // 检查是否活跃
    bool is_active() const { return state == RUN; }

    // 检查是否在屏障等待
    bool is_at_barrier() const { return state == BAR_SYNC; }

    // 检查是否退出
    bool is_exited() const { return state == EXIT; }

    // 设置线程状态
    void set_state(EXE_STATE new_state) { state = new_state; }

    // 获取PC值
    int get_pc() const { return pc; }

    // 设置PC值
    void set_pc(int new_pc) { pc = new_pc; }

    // 获取下一个PC值
    int get_next_pc() const { return next_pc; }

    // 设置下一个PC值
    void set_next_pc(int new_next_pc) { next_pc = new_next_pc; }

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
        return statements != nullptr && pc >= 0 &&
               pc < static_cast<int>(statements->size());
    }

    // 获取当前指令
    StatementContext *get_current_statement() {
        if (statements != nullptr && pc >= 0 &&
            pc < static_cast<int>(statements->size())) {
            return &(*statements)[pc];
        }
        return nullptr;
    }

    // 执行单条指令（由WarpContext调用）
    EXE_STATE execute_thread_instruction();
    void preallocate_registers(const std::vector<StatementContext> &statements);

    // 重置线程状态
    void reset();

private:
    void _execute_once();
    bool is_immediate_or_vector(OperandContext &op);
    void set_immediate_value(std::string value, Qualifier type);
};

#endif // THREAD_CONTEXT_H