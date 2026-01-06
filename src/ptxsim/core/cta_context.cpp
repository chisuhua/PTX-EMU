#include "ptxsim/cta_context.h"
#include "ptx_ir/kernel_context.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/thread_context.h"
#include "register/register_bank_manager.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#ifdef DEBUGINTE
extern bool sync_thread;
#endif
#ifdef LOGINTE
extern bool IFLOG();
#endif

void CTAContext::init(Dim3 &GridDim, Dim3 &BlockDim, Dim3 &blockIdx,
                      std::vector<StatementContext> &statements,
                      std::map<std::string, Symtable *> &name2Sym,
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
    int numWarpsNeeded =
        (threadNum + WarpContext::WARP_SIZE - 1) / WarpContext::WARP_SIZE;
    warpNum = numWarpsNeeded;

    // 创建寄存器银行管理器
    auto register_bank_manager = std::make_shared<RegisterBankManager>(
        numWarpsNeeded, WarpContext::WARP_SIZE);

    // 首先分析所有线程需要的寄存器，以CTA为单位进行预分配
    auto registers = RegisterAnalyzer::analyze_registers(statements);

    // 预分配所有寄存器
    register_bank_manager->preallocate_registers(registers);

    // 创建warp并分配线程
    warps.clear();
    warps.reserve(warpNum);

    for (int w = 0; w < warpNum; w++) {

        auto warp = std::make_unique<WarpContext>();
        warp->set_warp_id(w);
        // 设置寄存器银行管理器
        warp->set_register_bank_manager(register_bank_manager);
        warps.push_back(std::move(warp));
    }

    // 直接创建并分配线程到warp，不再使用thread_pool
    for (int i = 0; i < threadNum; i++) {
        Dim3 threadIdx;
        threadIdx.z = i / (BlockDim.x * BlockDim.y);
        threadIdx.y = i % (BlockDim.x * BlockDim.y) / BlockDim.x;
        threadIdx.x = i % (BlockDim.x * BlockDim.y) % BlockDim.x;

        auto thread = std::make_unique<ThreadContext>();
        thread->init(blockIdx, threadIdx, GridDim, BlockDim, statements,
                     name2Share, name2Sym, label2pc);

        // 设置线程使用寄存器银行管理器
        thread->set_register_bank_manager(register_bank_manager);

        // 直接将线程添加到对应的warp
        int warpId = i / WarpContext::WARP_SIZE;
        int laneId = i % WarpContext::WARP_SIZE;
        warps[warpId]->add_thread(std::move(thread), laneId);
    }
}

EXE_STATE CTAContext::exe_once() {
    // CTAContext不再执行指令，因为warp已转移给SMContext
    // 此函数现在只返回当前状态

    // 重新计算退出和屏障线程数
    exitThreadNum = 0;
    barThreadNum = 0;

    for (auto &warp : warps) {
        for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
            ThreadContext *thread = warp->get_thread(lane);
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
        for (auto &warp : warps) {
            for (int lane = 0; lane < WarpContext::WARP_SIZE; lane++) {
                ThreadContext *thread = warp->get_thread(lane);
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
    // 转移warps的所有权，并清空warps向量
    auto result = std::move(warps);
    // 清空原向量，避免重复析构
    warps.clear();
    return result;
}

CTAContext::~CTAContext() {
    // warps向量应该已经被清空，因为所有权已经转移
    // thread_pool也应该被清空
    name2Share.clear();
}