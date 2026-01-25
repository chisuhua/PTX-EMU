#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include <cassert>
#include <iostream>

void test_warp_creation() {
    std::cout << "Testing WarpContext creation..." << std::endl;

    WarpContext warp;

    // 检查默认状态
    assert(warp.get_active_count() == 32);
    assert(warp.is_active() == true);
    assert(warp.get_active_mask() == 0xFFFFFFFF); // 所有线程默认活跃

    std::cout << "Warp creation test passed." << std::endl;
}

void test_warp_active_mask_management() {
    std::cout << "Testing WarpContext active mask management..." << std::endl;

    WarpContext warp;

    // 测试设置活跃掩码
    warp.set_active_mask(0x0000000F); // 前4个线程活跃
    assert(warp.get_active_count() == 4);
    assert(warp.get_active_mask() == 0x0000000F);

    // 测试设置单个lane活跃状态
    warp.set_active_mask(5, true); // 激活第5个线程
    assert(warp.is_lane_active(5) == true);
    assert(warp.get_active_count() == 5);

    warp.set_active_mask(0, false); // 去激活第0个线程
    assert(warp.is_lane_active(0) == false);
    assert(warp.get_active_count() == 4);

    std::cout << "Warp active mask management test passed." << std::endl;
}

void test_warp_thread_addition() {
    std::cout << "Testing WarpContext thread addition..." << std::endl;

    WarpContext warp;

    // 创建一个简单的ThreadContext，使用unique_ptr
    Dim3 blockIdx = {0, 0, 0};
    Dim3 threadIdx = {0, 0, 0};
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};

    std::vector<StatementContext> statements;
    std::map<std::string, Symtable *> name2Sym;
    std::map<std::string, int> label2pc;

    auto thread = std::make_unique<ThreadContext>();
    thread->init(blockIdx, threadIdx, gridDim, blockDim, statements, &name2Sym,
                 label2pc, nullptr, nullptr);  // 最后两个参数分别是name2Share和cta_ctx

    // 添加线程到warp，使用std::move传递unique_ptr的所有权
    warp.add_thread(std::move(thread), 0);

    assert(warp.get_warp_thread_id(0) == 0); // 验证线程ID被正确设置

    std::cout << "Warp thread addition test passed." << std::endl;
}

void test_warp_completion() {
    std::cout << "Testing WarpContext completion status..." << std::endl;

    WarpContext warp;

    // 初始时warp不完成
    assert(warp.is_finished() == false);

    // 设置所有线程为非活跃
    warp.set_active_mask(0x0);
    assert(warp.is_finished() == true);

    std::cout << "Warp completion test passed." << std::endl;
}

int main() {
    std::cout << "Running WarpContext unit tests..." << std::endl;

    test_warp_creation();
    test_warp_active_mask_management();
    test_warp_thread_addition();
    test_warp_completion();

    std::cout << "All WarpContext tests PASSED!" << std::endl;

    return 0;
}