#include "ptxsim/cta_context.h"
#include "memory/hardware_memory_manager.h" // 添加MemoryManager头文件
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/register_analyzer.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
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
                      std::map<std::string, Symtable *> *name2Sym,
                      std::map<std::string, int> &label2pc,
                      void *local_memory_base, size_t local_mem_per_thread) {

    threadNum = BlockDim.x * BlockDim.y * BlockDim.z;
    curExeWarpId = 0;
    curExeThreadId = 0;
    exitThreadNum = 0;
    barThreadNum = 0;

    this->GridDim = GridDim;
    this->BlockDim = BlockDim;
    this->blockIdx = blockIdx;

    // 保存statements引用，用于后续构建共享内存符号表
    this->init_statements = &statements;

    // 计算共享内存大小，遍历PTX语句查找.shared声明
    sharedMemBytes = 0;

    for (const auto &stmt : statements) {
        if (stmt.type == S_SHARED) {
            const DeclarationInstr &sharedStmt =
                std::get<DeclarationInstr>(stmt.data);
            // 计算变量大小
            size_t element_size = Q2bytes(sharedStmt.dataType);
            size_t var_size = element_size * (*sharedStmt.size);
            sharedMemBytes += var_size;
        }
    }

    // 使用从SMContext传入的本地内存大小信息
    this->localMemBytesPerThread = local_mem_per_thread;

    // 预先创建共享内存符号表的结构，但不分配实际内存空间
    size_t shared_offset = 0;
    for (const auto &stmt : statements) {
        if (stmt.type == S_SHARED) {
            const DeclarationInstr &ss = std::get<DeclarationInstr>(stmt.data);
            // 使用new操作符创建Symtable实例
            Symtable *s = new Symtable();
            s->byteNum = Q2bytes(ss.dataType) * (*ss.size);
            s->elementNum = *ss.size;
            s->name = ss.name;
            s->symType = ss.dataType; // 假设dataType[0]是元素类型

            size_t var_size = s->byteNum;

            // 暂时设置地址为0，等待SMContext分配后填入
            s->val = 0;

            PTX_DEBUG_EMU("Prepared shared memory variable: name=%s, "
                          "elementNum=%d, byteNum=%d, "
                          "var_size=%zu, shared_mem_offset=%zu",
                          s->name.c_str(), s->elementNum, s->byteNum, var_size,
                          shared_offset);

            name2Share[s->name] = s;
            shared_offset += var_size;
        }
    }

    // 预先创建本地内存符号表的结构
    size_t local_offset = 0;
    for (const auto &stmt : statements) {
        if (stmt.type == S_LOCAL) {
            const DeclarationInstr &ls = std::get<DeclarationInstr>(stmt.data);
            // 使用new操作符创建Symtable实例
            Symtable *s = new Symtable();
            s->byteNum = Q2bytes(ls.dataType) * (*ls.size);
            s->elementNum = *ls.size;
            s->name = ls.name;
            s->symType = ls.dataType; // 假设dataType[0]是元素类型

            size_t var_size = s->byteNum;

            // 设置本地内存变量的偏移量
            s->val = local_offset;

            PTX_DEBUG_EMU("Prepared local memory variable: name=%s, "
                          "elementNum=%d, byteNum=%d, "
                          "var_size=%zu, local_mem_offset=%zu",
                          s->name.c_str(), s->elementNum, s->byteNum, var_size,
                          local_offset);

            name2Local[s->name] = s;
            local_offset += var_size;
        }
    }

    // 共享内存分配现在在build_shared_memory_symbol_table中进行，由SMContext提供空间
    sharedMemSpace = nullptr; // 初始化为nullptr
    PTX_DEBUG_EMU("Calculated shared memory size needed: %zu bytes",
                  sharedMemBytes);
    PTX_DEBUG_EMU("Calculated local memory size needed per thread: %zu bytes",
                  localMemBytesPerThread);

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

        // 传递this指针作为CTAContext指针，使ThreadContext可以访问本地内存符号表
        thread->init(blockIdx, threadIdx, GridDim, BlockDim, statements,
                     name2Sym, label2pc, &name2Share, this);

        // 设置线程使用寄存器银行管理器
        thread->set_register_bank_manager(register_bank_manager);

        // 直接将线程添加到对应的warp
        int warpId = i / WarpContext::WARP_SIZE;
        int laneId = i % WarpContext::WARP_SIZE;
        warps[warpId]->add_thread(std::move(thread), laneId);
    }

    // 如果提供了本地内存基础地址，则分配每个线程的本地内存空间
    if (local_memory_base != nullptr && localMemBytesPerThread > 0) {
        // 计算当前CTA在全局本地内存中的偏移量
        size_t block_id =
            blockIdx.x + GridDim.x * (blockIdx.y + GridDim.y * blockIdx.z);
        size_t cta_thread_offset = block_id *
                                   (BlockDim.x * BlockDim.y * BlockDim.z) *
                                   localMemBytesPerThread;

        // 为当前CTA的每个线程分配本地内存空间
        localMemSpaces.resize(threadNum);
        for (int i = 0; i < threadNum; i++) {
            // 计算该线程在全局内存中的位置
            size_t thread_offset =
                cta_thread_offset + i * localMemBytesPerThread;
            localMemSpaces[i] =
                (void *)((uint64_t)local_memory_base + thread_offset);

            if (localMemSpaces[i]) {
                memset(localMemSpaces[i], 0, localMemBytesPerThread);
                PTX_DEBUG_EMU("Assigned local memory space of size %zu for "
                              "thread %d at %p",
                              localMemBytesPerThread, i, localMemSpaces[i]);
            } else {
                PTX_ERROR_EMU(
                    "Failed to assign local memory space for thread %d", i);
            }
        }
    } else {
        // 如果没有提供本地内存基础地址，仍然创建空的容器
        localMemSpaces.resize(threadNum);
        for (int i = 0; i < threadNum; i++) {
            localMemSpaces[i] = nullptr;
        }
    }

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

        // 传递this指针作为CTAContext指针，使ThreadContext可以访问本地内存符号表
        thread->init(blockIdx, threadIdx, GridDim, BlockDim, statements,
                     name2Sym, label2pc, &name2Share, this);

        // 设置线程使用寄存器银行管理器
        thread->set_register_bank_manager(register_bank_manager);

        // 设置线程的本地内存空间
        if (i < static_cast<int>(localMemSpaces.size()) && localMemSpaces[i]) {
            thread->set_local_memory_space(localMemSpaces[i]);
            // thread->set_local_memory_space(local_memory_base);
        }

        // 直接将线程添加到对应的warp
        int warpId = i / WarpContext::WARP_SIZE;
        int laneId = i % WarpContext::WARP_SIZE;
        warps[warpId]->add_thread(std::move(thread), laneId);
    }

    // 构建本地内存符号表（如果尚未分配本地内存空间，则在此处分配）
    // if (local_memory_base == nullptr && localMemBytesPerThread > 0) {
    //     build_local_memory_symbol_table();
    // }
}

// 实现新方法：构建共享内存符号表，现在需要接收分配好的共享内存空间
void CTAContext::build_shared_memory_symbol_table(void *shared_mem_space) {
    if (!init_statements) {
        PTX_DEBUG_EMU("Error: init_statements is null, cannot build shared "
                      "memory symbol table");
        return;
    }

    // 设置共享内存空间
    sharedMemSpace = shared_mem_space;

    if (sharedMemBytes > 0 && sharedMemSpace != nullptr) {
        PTX_DEBUG_EMU("Using provided SHARED memory of size %zu at %p",
                      sharedMemBytes, sharedMemSpace);
        memset(sharedMemSpace, 0, sharedMemBytes);
    }

    // 填充name2Share中的地址信息
    size_t shared_offset = 0;
    for (const auto &stmt : *init_statements) {
        if (stmt.type == S_SHARED) {
            const DeclarationInstr &ss = std::get<DeclarationInstr>(stmt.data);

            // 查找对应的Symtable并设置地址
            auto it = name2Share.find(ss.name);
            if (it != name2Share.end()) {
                Symtable *s = it->second;

                size_t var_size = s->byteNum * s->elementNum;

                // 设置符号表中的地址为相对于共享内存基地址的偏移量
                s->val = shared_offset;

                PTX_DEBUG_EMU(
                    "Updated shared memory variable: name=%s, elementNum=%d, "
                    "byteNum=%d, "
                    "var_size=%zu, shared_mem_offset=%zu, stored_offset=%zu",
                    s->name.c_str(), s->elementNum, s->byteNum, var_size,
                    shared_offset, s->val);

                shared_offset += var_size;
            }
        }
    }

    // 将共享内存基地址传递给所有线程
    for (auto &warp : warps) {
        for (auto &thread : warp->get_threads()) {
            if (thread) {
                thread->shared_mem_space = shared_mem_space;
            }
        }
    }
}

// 实现构建本地内存符号表的方法
// void CTAContext::build_local_memory_symbol_table() {
//     if (!init_statements) {
//         PTX_DEBUG_EMU("Error: init_statements is null, cannot build local "
//                       "memory symbol table");
//         return;
//     }

//     // 检查本地内存是否已经分配（通过检查第一个线程的本地内存空间）
//     if (localMemSpaces.size() > 0 && localMemSpaces[0] != nullptr) {
//         // 本地内存空间已经分配，只需设置线程的本地内存空间
//         PTX_DEBUG_EMU("Local memory spaces already allocated, setting thread
//         local memory pointers");
//     } else {
//         // 本地内存空间尚未分配，需要分配内存
//         localMemSpaces.resize(threadNum);
//         for (int i = 0; i < threadNum; i++) {
//             if (localMemBytesPerThread > 0) {
//                 localMemSpaces[i] = malloc(localMemBytesPerThread);
//                 if (localMemSpaces[i]) {
//                     memset(localMemSpaces[i], 0, localMemBytesPerThread);
//                     PTX_DEBUG_EMU("Allocated local memory space of size %zu
//                     for thread %d at %p",
//                                   localMemBytesPerThread, i,
//                                   localMemSpaces[i]);
//                 } else {
//                     PTX_ERROR_EMU("Failed to allocate local memory space for
//                     thread %d", i);
//                 }
//             } else {
//                 localMemSpaces[i] = nullptr;
//             }
//         }
//     }

//     // 将本地内存基地址传递给所有线程
//     int threadIdx = 0;
//     for (auto &warp : warps) {
//         for (auto &thread : warp->get_threads()) {
//             if (thread) {
//                 if (threadIdx < static_cast<int>(localMemSpaces.size())) {
//                     thread->set_local_memory_space(localMemSpaces[threadIdx]);
//                 }
//                 threadIdx++;
//             }
//         }
//     }
// }

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
    // 注意：共享内存空间不由CTAContext释放，而是由SMContext管理
    // 因此这里不需要释放sharedMemSpace
    // 确保sharedMemSpace被设置为nullptr，防止任何可能的重复释放
    sharedMemSpace = nullptr;

    // warps向量应该已经被清空，因为所有权已经转移
    // thread_pool也应该被清空

    // name2Share中的Symtable内存由build_shared_memory_symbol_table统一管理释放
    // 这里只需要清空map，不要delete指针
    name2Share.clear();
}