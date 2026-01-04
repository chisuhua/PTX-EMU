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
    
    // 创建线程上下文数组，用于内存管理
    std::vector<std::unique_ptr<ThreadContext>> threads;
    threads.reserve(threadNum);
    
    // 创建所有线程上下文
    for (int i = 0; i < threadNum; i++) {
        Dim3 threadIdx;
        threadIdx.z = i / (BlockDim.x * BlockDim.y);
        threadIdx.y = i % (BlockDim.x * BlockDim.y) / BlockDim.x;
        threadIdx.x = i % (BlockDim.x * BlockDim.y) % BlockDim.x;
        
        auto thread = std::make_unique<ThreadContext>();
        thread->init(blockIdx, threadIdx, GridDim, BlockDim, statements,
                     name2Share, name2Sym, label2pc);
        threads.push_back(std::move(thread));
    }
    
    // 分配线程到warp
    for (int i = 0; i < threadNum; i++) {
        int warpId = i / WarpContext::WARP_SIZE;
        int laneId = i % WarpContext::WARP_SIZE;
        
        // 添加到对应的warp（使用raw pointer，warp不负责内存管理）
        warps[warpId]->add_thread(threads[i].get(), laneId);
    }
    
    // 保存线程上下文以便管理内存
    // 这里我们需要一个方式来跟踪所有线程的退出和屏障状态
    // 为简单起见，我们直接在exe_once中计算
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
    // warps会自动清理
    // 线程内存由warp中的unique_ptr管理，会自动释放
}