#include "ptxsim/warp_scheduler.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/execution_types.h"
#include <cassert>
#include <iostream>
#include <memory>

void test_round_robin_scheduler() {
    std::cout << "Testing RoundRobinWarpScheduler..." << std::endl;

    RoundRobinWarpScheduler scheduler;
    
    // 创建一些warp用于测试
    auto warp1 = std::make_unique<WarpContext>();
    auto warp2 = std::make_unique<WarpContext>();
    auto warp3 = std::make_unique<WarpContext>();
    
    // 设置warp ID
    warp1->set_warp_id(0);
    warp2->set_warp_id(1);
    warp3->set_warp_id(2);
    
    // 添加warp到调度器
    scheduler.add_warp(warp1.get());
    scheduler.add_warp(warp2.get());
    scheduler.add_warp(warp3.get());
    
    // 测试调度顺序
    WarpContext* scheduled_warp = scheduler.schedule_next();
    assert(scheduled_warp != nullptr);
    assert(scheduled_warp->get_warp_id() == 0);
    
    scheduled_warp = scheduler.schedule_next();
    assert(scheduled_warp != nullptr);
    assert(scheduled_warp->get_warp_id() == 1);
    
    scheduled_warp = scheduler.schedule_next();
    assert(scheduled_warp != nullptr);
    assert(scheduled_warp->get_warp_id() == 2);
    
    // 再次调度应返回第一个warp（循环）
    scheduled_warp = scheduler.schedule_next();
    assert(scheduled_warp != nullptr);
    assert(scheduled_warp->get_warp_id() == 0);
    
    std::cout << "RoundRobinWarpScheduler test passed." << std::endl;
}

void test_greedy_scheduler() {
    std::cout << "Testing GreedyWarpScheduler..." << std::endl;

    GreedyWarpScheduler scheduler;
    
    // 创建一些warp用于测试
    auto warp1 = std::make_unique<WarpContext>();
    auto warp2 = std::make_unique<WarpContext>();
    auto warp3 = std::make_unique<WarpContext>();
    
    // 设置warp ID
    warp1->set_warp_id(0);
    warp2->set_warp_id(1);
    warp3->set_warp_id(2);
    
    // 添加warp到调度器
    scheduler.add_warp(warp1.get());
    scheduler.add_warp(warp2.get());
    scheduler.add_warp(warp3.get());
    
    // 测试调度器状态更新
    scheduler.update_state();
    
    // 检查是否所有warp都被认为是活跃的
    WarpContext* scheduled_warp = scheduler.schedule_next();
    assert(scheduled_warp != nullptr);
    
    std::cout << "GreedyWarpScheduler test passed." << std::endl;
}

void test_scheduler_with_inactive_warps() {
    std::cout << "Testing scheduler with inactive warps..." << std::endl;

    RoundRobinWarpScheduler scheduler;
    
    // 创建一些warp用于测试
    auto warp1 = std::make_unique<WarpContext>();
    auto warp2 = std::make_unique<WarpContext>();
    
    // 设置warp ID
    warp1->set_warp_id(0);
    warp2->set_warp_id(1);
    
    // 使warp2完成
    warp2->set_active_mask(0x0); // 使warp2不活跃
    
    // 添加warp到调度器
    scheduler.add_warp(warp1.get());
    scheduler.add_warp(warp2.get());
    
    // 调度应该返回活跃的warp1
    WarpContext* scheduled_warp = scheduler.schedule_next();
    assert(scheduled_warp != nullptr);
    assert(scheduled_warp->get_warp_id() == 0);
    
    // 检查是否所有warp都已完成
    // warp2已完成，但warp1仍然活跃
    assert(!scheduler.all_warps_finished());
    
    std::cout << "Scheduler with inactive warps test passed." << std::endl;
}

int main() {
    std::cout << "Running WarpScheduler unit tests..." << std::endl;

    test_round_robin_scheduler();
    test_greedy_scheduler();
    test_scheduler_with_inactive_warps();

    std::cout << "All WarpScheduler tests PASSED!" << std::endl;

    return 0;
}