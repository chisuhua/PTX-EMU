#include "ptxsim/cta_context.h"
#include "ptx_ir/kernel_context.h"
#include "ptxsim/thread_context.h"
#include <cassert>
#include <cstring>
#ifdef DEBUGINTE
extern bool sync_thread;
#endif
#ifdef LOGINTE
extern bool IFLOG();
#endif

void CTAContext::init(
    Dim3 &GridDim, Dim3 &BlockDim, Dim3 &blockIdx,
    std::vector<StatementContext> &statements,
    std::map<std::string, PtxInterpreter::Symtable *> &name2Sym,
    std::map<std::string, int> &label2pc) {

    threadNum = BlockDim.x * BlockDim.y * BlockDim.z;
    curExeThreadId = 0;
    exitThreadNum = 0;
    barThreadNum = 0;

    this->GridDim = GridDim;
    this->BlockDim = BlockDim;
    this->blockIdx = blockIdx;

    // init thread
    assert(threadNum > 0 && threadNum <= 2048);
    if (!thread)
        thread = new ThreadContext[threadNum];
    if (!exitThread)
        exitThread = new bool[threadNum];
    if (!barThread)
        barThread = new bool[threadNum];
    memset(exitThread, 0, sizeof(bool) * threadNum);
    memset(barThread, 0, sizeof(bool) * threadNum);
    Dim3 threadIdx;
    for (int i = 0; i < threadNum; i++) {
        threadIdx.z = i / (BlockDim.x * BlockDim.y);
        threadIdx.y = i % (BlockDim.x * BlockDim.y) / (BlockDim.x);
        threadIdx.x = i % (BlockDim.x * BlockDim.y) % (BlockDim.x);
        thread[i].init(blockIdx, threadIdx, GridDim, BlockDim, statements,
                       name2Share, name2Sym, label2pc);
    }
}

EXE_STATE CTAContext::exe_once() {
    if (exitThreadNum == threadNum)
        return EXIT;
    if (barThreadNum == threadNum) {
#ifdef LOGINTE
        if (IFLOG())
            printf("INTE: bar.sync BlockIdx(%d,%d,%d)\n", blockIdx.x,
                   blockIdx.y, blockIdx.z);
#endif
        for (int i = 0; i < threadNum; i++) {
            thread[i].state = RUN;
            barThread[i] = 0;
        }
        barThreadNum = 0;
#ifdef DEBUGINTE
        sync_thread = 0;
#endif
    }
    EXE_STATE state = thread[curExeThreadId].exe_once();
    if (state != RUN) {
        if (state == EXIT && !exitThread[curExeThreadId]) {
            exitThreadNum++;
            exitThread[curExeThreadId] = 1;
        } else if (state == BAR && !barThread[curExeThreadId]) {
            barThreadNum++;
            barThread[curExeThreadId] = 1;
        }
        curExeThreadId++;
        curExeThreadId %= threadNum;
    }
    return RUN;
}

CTAContext::~CTAContext() {
    if (thread)
        delete[] thread;
    if (exitThread)
        delete[] exitThread;
    if (barThread)
        delete[] barThread;
}