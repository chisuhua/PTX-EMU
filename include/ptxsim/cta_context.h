#ifndef CTA_CONTEXT_H
#define CTA_CONTEXT_H

#include "ptxsim/execution_types.h"
#include "ptxsim/interpreter.h"
#include <map>
#include <string>
#include <vector>

class StatementContext; // 前向声明
class ThreadContext;    // 前向声明
// class PtxInterpreter;
// class PtxInterpreter {
// public:
//     class Symtable; // 前向声明
// };

class CTAContext {
public:
    ThreadContext *thread = nullptr;
    bool *exitThread = nullptr;
    bool *barThread = nullptr;
    int threadNum, curExeThreadId, exitThreadNum, barThreadNum;
    Dim3 blockIdx, GridDim, BlockDim;
    std::map<std::string, PtxInterpreter::Symtable *> name2Share;

    void init(Dim3 &GridDim, Dim3 &BlockDim, Dim3 &blockIdx,
              std::vector<StatementContext> &statements,
              std::map<std::string, PtxInterpreter::Symtable *> &name2Sym,
              std::map<std::string, int> &label2pc);

    EXE_STATE exe_once();

    bool allThreadsExited() const { return exitThreadNum == threadNum; }
    bool allThreadsAtBarrier() const { return barThreadNum == threadNum; }

    ~CTAContext();
};

#endif // CTA_CONTEXT_H