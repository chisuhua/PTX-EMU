#ifndef THREAD_CONTEXT_H
#define THREAD_CONTEXT_H

#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/interpreter.h"
#include <driver_types.h>
#include <map>
#include <queue>
#include <string>
#include <vector>

// Forward declarations
// class PtxInterpreter;

class ThreadContext {
public:
    // 添加DTYPE枚举定义
    enum DTYPE { DFLOAT, DINT, DNONE };

    std::vector<StatementContext> *statements;
    std::map<std::string, PtxInterpreter::Symtable *> *name2Share;
    std::map<std::string, PtxInterpreter::Symtable *> name2Sym;
    std::map<std::string, PtxInterpreter::Reg *> name2Reg;
    std::map<std::string, int> label2pc;
    dim3 BlockIdx, ThreadIdx, GridDim, BlockDim;
    std::queue<PtxInterpreter::IMM *> imm; // TODO fix memory leak
    std::queue<PtxInterpreter::VEC *> vec; // TODO fix memory leak
    int pc;
    EXE_STATE state;

    void init(dim3 &blockIdx, dim3 &threadIdx, dim3 GridDim, dim3 BlockDim,
              std::vector<StatementContext> &statements,
              std::map<std::string, PtxInterpreter::Symtable *> &name2Share,
              std::map<std::string, PtxInterpreter::Symtable *> &name2Sym,
              std::map<std::string, int> &label2pc);

    EXE_STATE exe_once();
    void _exe_once();

    // Memory access functions
    void *getOperandAddr(OperandContext &op, std::vector<Qualifier> &q);
    void *getRegAddr(OperandContext::REG *regContext);
    void *getFaAddr(OperandContext::FA *fa, std::vector<Qualifier> &q);

    // Instruction handlers
    void handle_reg(StatementContext::REG *ss);
    void handle_shared(StatementContext::SHARED *ss);
    void handle_local(StatementContext::LOCAL *ss);
    void handle_dollor(StatementContext::DOLLOR *ss);
    void handle_at(StatementContext::AT *ss);
    void handle_pragma(StatementContext::PRAGMA *ss);
    void handle_ret(StatementContext::RET *ss);
    void handle_bar(StatementContext::BAR *ss);
    void handle_bra(StatementContext::BRA *ss);
    void handle_rcp(StatementContext::RCP *ss);
    void handle_ld(StatementContext::LD *ss);
    void handle_mov(StatementContext::MOV *ss);
    void handle_add(StatementContext::ADD *ss);
    void handle_sub(StatementContext::SUB *ss);
    void handle_mul(StatementContext::MUL *ss);
    void handle_div(StatementContext::DIV *ss);
    void handle_and(StatementContext::AND *ss);
    void handle_or(StatementContext::OR *ss);
    void handle_xor(StatementContext::XOR *ss);
    void handle_shl(StatementContext::SHL *ss);
    void handle_shr(StatementContext::SHR *ss);
    void handle_st(StatementContext::ST *ss);
    void handle_cvta(StatementContext::CVTA *ss);
    void handle_setp(StatementContext::SETP *ss);
    void handle_cvt(StatementContext::CVT *ss);
    void handle_max(StatementContext::MAX *ss);
    void handle_min(StatementContext::MIN *ss);
    void handle_mad(StatementContext::MAD *ss);
    void handle_fma(StatementContext::FMA *ss);
    void handle_neg(StatementContext::NEG *ss);
    void handle_sqrt(StatementContext::SQRT *ss);
    void handle_atom(StatementContext::ATOM *ss);
    void handle_abs(StatementContext::ABS *ss);
    void handle_selp(StatementContext::SELP *ss);
    void handle_sin(StatementContext::SIN *ss);
    void handle_cos(StatementContext::COS *ss);
    void handle_lg2(StatementContext::LG2 *ss);
    void handle_ex2(StatementContext::EX2 *ss);
    void handle_rem(StatementContext::REM *ss);
    void handle_rsqrt(StatementContext::RSQRT *ss);
    void handle_not(StatementContext::NOT *ss);
    void handle_wmma(StatementContext::WMMA *ss);

    // 添加辅助函数声明
    void add(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void sub(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void mul(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void div(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void And(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void Or(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void Xor(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void shl(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void shr(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void setp(void *to, void *op1, void *op2, Qualifier cmpOp, std::vector<Qualifier> &q);
    void cvt(void *to, void *from, std::vector<Qualifier> &toQ, std::vector<Qualifier> &fromQ);
    void max(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void min(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void mad(void *to, void *op1, void *op2, void *op3, std::vector<Qualifier> &q);
    void fma(void *to, void *op1, void *op2, void *op3, std::vector<Qualifier> &q);
    void neg(void *to, void *op, std::vector<Qualifier> &q);
    void sqrt(void *to, void *op, std::vector<Qualifier> &q);
    void atom(void *to, void *op, std::vector<Qualifier> &q);
    void abs(void *to, void *op, std::vector<Qualifier> &q);
    void rcp(void *to, void *from, std::vector<Qualifier> &q);
    void selp(void *to, void *op0, void *op1, void *pred, std::vector<Qualifier> &q);
    void m_sin(void *to, void *op, std::vector<Qualifier> &q);
    void m_cos(void *to, void *op, std::vector<Qualifier> &q);
    void m_lg2(void *to, void *op, std::vector<Qualifier> &q);
    void m_ex2(void *to, void *op, std::vector<Qualifier> &q);
    void m_rem(void *to, void *op1, void *op2, std::vector<Qualifier> &q);
    void m_rsqrt(void *to, void *op, std::vector<Qualifier> &q);

    // 添加类型判断辅助函数声明
    Qualifier getCMPOP(std::vector<Qualifier> &q);
    DTYPE getDType(std::vector<Qualifier> &q);
    DTYPE getDType(Qualifier q);
    bool Signed(Qualifier q);

private:
    bool QvecHasQ(std::vector<Qualifier> &qvec, Qualifier q);
    Qualifier getDataType(std::vector<Qualifier> &q);
    int getBytes(std::vector<Qualifier> &q);
    int getBytes(Qualifier q);
    void setIMM(std::string s, Qualifier q);
    void mov(void *from, void *to, std::vector<Qualifier> &q);
    void clearIMM_VEC();
    bool isIMMorVEC(OperandContext &op);
    void handleStatement(StatementContext &statement);
};

#endif // THREAD_CONTEXT_H