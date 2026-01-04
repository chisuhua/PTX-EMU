#ifndef CTA_CONTEXT_H
#define CTA_CONTEXT_H

#include "ptxsim/execution_types.h"
#include "ptxsim/interpreter.h"
#include "ptxsim/warp_context.h"
#include <map>
#include <string>
#include <vector>
#include <memory>

class StatementContext; // 前向声明
class ThreadContext;    // 前向声明

class CTAContext {
public:
    std::vector<std::unique_ptr<WarpContext>> warps;
    int warpNum;
    int curExeWarpId;
    
    int threadNum;
    int curExeThreadId;
    int exitThreadNum;
    int barThreadNum;
    
    size_t sharedMemBytes = 0;
    Dim3 blockIdx, GridDim, BlockDim;
    std::map<std::string, PtxInterpreter::Symtable *> name2Share;

    void init(Dim3 &GridDim, Dim3 &BlockDim, Dim3 &blockIdx,
              std::vector<StatementContext> &statements,
              std::map<std::string, PtxInterpreter::Symtable *> &name2Sym,
              std::map<std::string, int> &label2pc);

    EXE_STATE exe_once();

    bool allThreadsExited() const { return exitThreadNum == threadNum; }
    bool allThreadsAtBarrier() const { return barThreadNum == threadNum; }
    
    // 提供方法来获取warp的所有权
    std::vector<std::unique_ptr<WarpContext>> release_warps();

    ~CTAContext();
};

#endif // CTA_CONTEXT_H