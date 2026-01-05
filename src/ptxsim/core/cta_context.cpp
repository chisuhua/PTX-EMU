#include "ptxsim/cta_context.h"
#include "ptx_ir/kernel_context.h"
#include "ptxsim/thread_context.h"
#include <cassert>
#include <cstring>
#include <memory>
#include <algorithm>
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
    curExeWarpId = 0;
    curExeThreadId = 0;
    exitThreadNum = 0;
    barThreadNum = 0;

    this->GridDim = GridDim;
    this->BlockDim = BlockDim;
    this->blockIdx = blockIdx;

    // 计算需要多少个warp
    int numWarpsNeeded = (threadNum + WarpContext::WARP_SIZE - 1) / WarpContext::WARP_SIZE;
    warpNum = numWarpsNeeded;
    
    // 创建warp并分配线程
    warps.clear();
    warps.reserve(warpNum);
    
    for (int w = 0; w < warpNum; w++) {
        auto warp = std::make_unique<WarpContext>();
        warp->set_warp_id(w);
        warps.push_back(std::move(warp));
    }
    
    // 创建线程池，管理线程对象的生命周期
    thread_pool.clear();
    thread_pool.reserve(threadNum);
    
    // 创建所有线程上下文
    for (int i = 0; i < threadNum; i++) {
        Dim3 threadIdx;
        threadIdx.z = i / (BlockDim.x * BlockDim.y);
        threadIdx.y = i % (BlockDim.x * BlockDim.y) / BlockDim.x;
        threadIdx.x = i % (BlockDim.x * BlockDim.y) % BlockDim.x;
        
        auto thread = std::make_unique<ThreadContext>();
        thread->init(blockIdx, threadIdx, GridDim, BlockDim, statements,
                     name2Share, name2Sym, label2pc);
        thread_pool.push_back(std::move(thread));
    }
    
    // 分配线程到warp
    for (int i = 0; i < threadNum; i++) {
        int warpId = i / WarpContext::WARP_SIZE;
        int laneId = i % WarpContext::WARP_SIZE;
        
        // 将线程所有权转移到对应的warp
        warps[warpId]->add_thread(std::move(thread_pool[i]), laneId);
    }
    
    // 清空线程池，因为所有权已转移给warps
    thread_pool.clear();
}

EXE_STATE CTAContext::exe_once() {
    // CTAContext不再执行指令，因为warp已转移给SMContext
    // 此函数现在只返回当前状态

    // 重新计算退出和屏障线程数
    exitThreadNum = 0;
    barThreadNum = 0;
    
    for (auto& warp : warps) {
        for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
            ThreadContext* thread = warp->get_thread(lane);
            if (thread) {
                if (thread->is_exited()) {
                    exitThreadNum++;
                } else if (thread->is_at_barrier()) {
                    barThreadNum++;
                }
            }
        }
    }
    
    if (exitThreadNum == threadNum)
        return EXIT;
    if (barThreadNum == threadNum) {
#ifdef LOGINTE
        if (IFLOG())
            printf("INTE: bar.sync BlockIdx(%d,%d,%d)\n", blockIdx.x,
                   blockIdx.y, blockIdx.z);
#endif
        // 恢复所有线程的运行状态
        for (auto& warp : warps) {
            for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
                ThreadContext* thread = warp->get_thread(lane);
                if (thread != nullptr) {
                    thread->set_state(RUN);
                }
            }
        }
        barThreadNum = 0;
#ifdef DEBUGINTE
        sync_thread = 0;
#endif
        return BAR_SYNC;
    }
    
    return RUN;
}

std::vector<std::unique_ptr<WarpContext>> CTAContext::release_warps() {
    return std::move(warps);
}

CTAContext::~CTAContext() {
    // warps和thread_pool会自动清理
}