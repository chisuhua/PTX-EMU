#ifndef THREAD_CONTEXT_H
#define THREAD_CONTEXT_H

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/interpreter.h"
#include "ptxsim/ptx_debug.h"
#include "ptxsim/utils/type_utils.h"
#include "register/register_manager.h"
#include <any>
#include <map>
#include <ostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

// 条件码寄存器标志
struct ConditionCodeRegister {
    bool carry : 1;      // 进位标志
    bool overflow : 1;   // 溢出标志
    bool zero : 1;       // 零标志
    bool sign : 1;       // 符号标志
    bool reserved : 4;   // 预留位

    ConditionCodeRegister() : carry(false), overflow(false), zero(false), sign(false), reserved(0) {}
};

class ThreadContext {
public:
    // 资源管理
    std::vector<StatementContext> *statements;
    std::map<std::string, PtxInterpreter::Symtable *> *name2Share;
    std::map<std::string, PtxInterpreter::Symtable *> name2Sym;
    RegisterManager register_manager;
    std::map<std::string, int> label2pc;

    // 线程状态
    Dim3 BlockIdx, ThreadIdx, GridDim, BlockDim;
    int pc;
    EXE_STATE state;

    // 条件码寄存器
    ConditionCodeRegister cc_reg;

    // 当前指令执行状态
    // 临时数据存储
    std::queue<PtxInterpreter::VEC *> vec;

    void init(Dim3 &blockIdx, Dim3 &threadIdx, Dim3 GridDim, Dim3 BlockDim,
              std::vector<StatementContext> &statements,
              std::map<std::string, PtxInterpreter::Symtable *> &name2Share,
              std::map<std::string, PtxInterpreter::Symtable *> &name2Sym,
              std::map<std::string, int> &label2pc);

    EXE_STATE exe_once();
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
private:
    void _execute_once();
    bool is_immediate_or_vector(OperandContext &op);
    void set_immediate_value(std::string value, Qualifier type);
};

#endif // THREAD_CONTEXT_H