#include "ptxsim/cta_context.h"
#include "memory/memory_manager.h" // 添加MemoryManager头文件
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/thread_context.h"
#include "register/register_bank_manager.h"
#include "utils/logger.h"
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

    // 计算共享内存大小，遍历PTX语句查找.shared声明
    sharedMemBytes = 0;
    for (const auto &stmt : statements) {
        if (stmt.statementType == S_SHARED) {
            auto sharedStmt = (StatementContext::SHARED *)stmt.statement;
            // 计算变量大小
            size_t element_size = Q2bytes(sharedStmt->dataType[0]);
            size_t var_size = element_size * sharedStmt->size;
            sharedMemBytes += var_size;
        }
    }

    // 分配共享内存空间
    if (sharedMemBytes > 0) {
        sharedMemSpace =
            MemoryManager::instance().malloc_shared(sharedMemBytes);
        if (sharedMemSpace == nullptr) {
            PTX_DEBUG_EMU("Failed to allocate SHARED memory of size %zu",
                          sharedMemBytes);
            // 可以考虑抛出异常或返回错误
        } else {
            PTX_DEBUG_EMU("Allocated SHARED memory of size %zu at %p",
                          sharedMemBytes, sharedMemSpace);
            memset(sharedMemSpace, 0, sharedMemBytes);
        }
    } else {
        sharedMemSpace = nullptr;
        PTX_DEBUG_EMU("No SHARED memory needed, sharedMemBytes is 0");
    }

    // 构建共享内存符号表
    size_t shared_offset = 0;
    for (const auto &stmt : statements) {
        if (stmt.statementType == S_SHARED) {
            auto sharedStmt = (StatementContext::SHARED *)stmt.statement;
            // 创建Symtable条目
            auto s = std::make_unique<Symtable>();
            s->name = sharedStmt->name;
            s->elementNum = sharedStmt->size;
            s->symType = sharedStmt->dataType[0]; // 假设dataType[0]是元素类型
            s->byteNum = Q2bytes(sharedStmt->dataType[0]);

            size_t var_size = s->byteNum * sharedStmt->size;

            // 设置符号表中的地址
            s->val = (uint64_t)((char *)sharedMemSpace + shared_offset);

            PTX_DEBUG_EMU(
                "Shared memory variable: name=%s, elementNum=%d, byteNum=%d, "
                "var_size=%zu, shared_mem_offset=%zu, stored_addr=%p",
                s->name.c_str(), s->elementNum, s->byteNum, var_size, shared_offset,
                (void *)s->val);

            name2Share[s->name] = std::move(s);
            shared_offset += var_size;
        }
    }

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

    for (int i = 0; i < threadNum; i++) {
        Dim3 threadIdx;
        threadIdx.z = i / (BlockDim.x * BlockDim.y);
        threadIdx.y = i % (BlockDim.x * BlockDim.y) / BlockDim.x;
        threadIdx.x = i % (BlockDim.x * BlockDim.y) % BlockDim.x;

        auto thread = std::make_unique<ThreadContext>();

        // 将name2Share中的Symtable指针复制到map中传递给线程
        std::map<std::string, Symtable *> thread_name2Share;
        for (const auto &pair : name2Share) {
            thread_name2Share[pair.first] = pair.second.get();
        }

        thread->init(blockIdx, threadIdx, GridDim, BlockDim, statements,
                     name2Sym, label2pc, &thread_name2Share);

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
    // 释放共享内存空间
    if (sharedMemSpace != nullptr) {
        PTX_DEBUG_EMU("Freeing SHARED memory at %p", sharedMemSpace);
        MemoryManager::instance().free_shared(sharedMemSpace);
        sharedMemSpace = nullptr;
    }

    // warps向量应该已经被清空，因为所有权已经转移
    // thread_pool也应该被清空
    name2Share.clear();
}