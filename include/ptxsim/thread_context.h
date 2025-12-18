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
#include <any>
// include <driver_types.h>
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
    std::map<std::string, PtxInterpreter::Reg *> name2Reg;
    std::map<std::string, int> label2pc;

    // 线程状态
    Dim3 BlockIdx, ThreadIdx, GridDim, BlockDim;
    int pc;
    EXE_STATE state;

    // 临时数据存储
    std::queue<PtxInterpreter::IMM *> imm;
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

    // 统一的寄存器跟踪接口
    void trace_register(OperandContext::REG *reg, void *value,
                        std::vector<Qualifier> &qualifiers, bool is_write);

    // 内存访问跟踪接口
    void memory_access(bool is_write, const std::string &addr_expr, void *addr,
                       size_t size, void *value,
                       std::vector<Qualifier> &qualifiers,
                       void *target = nullptr,
                       OperandContext::REG *reg_operand = nullptr);

    // 辅助函数接口（供指令处理器使用）
    bool QvecHasQ(std::vector<Qualifier> &qvec, Qualifier q);
    int getBytes(std::vector<Qualifier> &q);
    void mov(void *from, void *to, std::vector<Qualifier> &q);
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

    // 保留原有的辅助函数声明
    int getBytes(Qualifier q);
    void setIMM(std::string s, Qualifier q);
    void clearIMM_VEC();
};

#endif // THREAD_CONTEXT_H