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
    void *get_operand_addr(OperandContext &op,
                           std::vector<Qualifier> &qualifiers);
    void *get_register_addr(OperandContext::REG *reg,
                            Qualifier qualifier = Qualifier::Q_U32);
    void *get_memory_addr(OperandContext::FA *fa,
                          std::vector<Qualifier> &qualifiers);

    // Shared memory初始化
    void initialize_shared_memory(const std::string &name, uint64_t address);

    // 通用操作
    void mov_data(void *src, void *dst, std::vector<Qualifier> &qualifiers);
    void handle_statement(StatementContext &statement);

    // 辅助函数接口（供指令处理器使用）
    void mov(void *from, void *to, const std::vector<Qualifier> &q);
    bool isIMMorVEC(OperandContext &op);

    // 新增：为断点条件准备上下文
    void prepare_breakpoint_context(
        std::unordered_map<std::string, std::any> &context);

    // 新增：转储线程状态
    void dump_state(std::ostream &os) const;

private:
    void _execute_once();
    bool is_immediate_or_vector(OperandContext &op);
    void set_immediate_value(std::string value, Qualifier type);
};

#endif // THREAD_CONTEXT_H