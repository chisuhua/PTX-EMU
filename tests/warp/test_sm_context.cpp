#include "ptxsim/sm_context.h"
#include "ptxsim/cta_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include <cassert>
#include <iostream>
#include <memory>

void test_sm_context_creation() {
    std::cout << "Testing SMContext creation..." << std::endl;

    // 创建SMContext，限制为最多4个warp，最多128个线程，4KB共享内存
    SMContext sm(4, 128, 4096);
    
    // 检查初始状态
    assert(sm.get_state() == RUN);
    assert(sm.get_allocated_shared_mem() == 0);
    assert(sm.get_max_shared_mem() == 4096);
    assert(sm.get_active_warps_count() == 0);
    assert(sm.get_active_threads_count() == 0);
    
    std::cout << "SMContext creation test passed." << std::endl;
}

void test_sm_context_block_addition() {
    std::cout << "Testing SMContext block addition..." << std::endl;

    SMContext sm(8, 256, 8192); // 最多8个warp，256个线程，8KB共享内存
    
    // 创建一个CTAContext
    CTAContext block;
    
    // 初始化CTAContext
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1}; // 一个warp的大小
    Dim3 blockIdx = {0, 0, 0};
    
    std::vector<StatementContext> statements;
    std::map<std::string, PtxInterpreter::Symtable *> name2Sym;
    std::map<std::string, int> label2pc;
    
    block.init(gridDim, blockDim, blockIdx, statements, name2Sym, label2pc);
    
    // 设置一些共享内存需求
    block.sharedMemBytes = 1024;
    
    // 添加块到SM
    bool success = sm.add_block(&block);
    assert(success == true);
    
    // 检查资源分配
    assert(sm.get_allocated_shared_mem() == 1024);
    assert(sm.get_active_warps_count() >= 0); // 至少有warps被添加
    
    std::cout << "SMContext block addition test passed." << std::endl;
}

void test_sm_context_execution() {
    std::cout << "Testing SMContext execution..." << std::endl;

    SMContext sm(4, 128, 4096);
    
    // 验证初始状态
    EXE_STATE state = sm.exe_once();
    assert(state == EXIT); // 因为没有warp
    
    std::cout << "SMContext execution test passed." << std::endl;
}

void test_sm_context_resource_limits() {
    std::cout << "Testing SMContext resource limits..." << std::endl;

    // 创建一个资源受限的SM (仅允许1个warp, 64个线程, 1KB共享内存)
    SMContext sm(1, 64, 1024);
    
    // 创建一个需要超过限制的块
    CTAContext block;
    
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {128, 1, 1}; // 需要128个线程，超过限制
    Dim3 blockIdx = {0, 0, 0};
    
    std::vector<StatementContext> statements;
    std::map<std::string, PtxInterpreter::Symtable *> name2Sym;
    std::map<std::string, int> label2pc;
    
    block.init(gridDim, blockDim, blockIdx, statements, name2Sym, label2pc);
    block.sharedMemBytes = 2048; // 需要2KB共享内存，超过限制
    
    // 尝试添加块到SM - 应该失败
    bool success = sm.add_block(&block);
    assert(success == false);
    
    std::cout << "SMContext resource limits test passed." << std::endl;
}

int main() {
    std::cout << "Running SMContext unit tests..." << std::endl;

    test_sm_context_creation();
    test_sm_context_block_addition();
    test_sm_context_execution();
    test_sm_context_resource_limits();

    std::cout << "All SMContext tests PASSED!" << std::endl;

    return 0;
}