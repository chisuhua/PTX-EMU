// ptx_lane_verification.cpp
// 功能: PTX Lane 级别分支决策验证工具实现
// 作者: AI Agent
// 最后修改日期: 2026-05-11

#include "utils/ptx_lane_verification.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptx_ir/statement_factory.h"
#include "memory/resource_manager.h"
#include <memory>

namespace ptxsim::verification {

static void init_instruction_factory_once() {
    static bool initialized = false;
    if (!initialized) {
        InstructionFactory::initialize();
        initialized = true;
    }
}

WarpContext* create_execution_warp(
    SMContext* sm,
    const ExecutionEngineConfig& config
) {
    init_instruction_factory_once();

    ResourceManager::instance().initialize(1, 8192);

    Dim3 gridDim{1, 1, 1};
    Dim3 blockDim{32, 1, 1};
    Dim3 blockIdx{0, 0, 0};

    auto block = std::make_unique<CTAContext>();
    std::vector<ptxemu::ir::StatementContext> statements;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;

    block->init(gridDim, blockDim, blockIdx, statements, &name2Sym, const_cast<std::map<std::string, int>&>(config.label2pc));

    block->sharedMemBytes = 1024;
    bool success = sm->add_block(std::move(block));
    if (!success) {
        return nullptr;
    }

    WarpContext* warp = sm->get_warp(0);
    if (!warp) {
        return nullptr;
    }

    warp->set_exec_mask(0xFFFFFFFF);

    for (int i = 0; i < WarpContext::WARP_SIZE; i++) {
        warp->set_active_mask(i, true);
    }

    return warp;
}

void set_predicate_value(ThreadContext* thread, const std::string& predicate_name, bool value) {
    if (!thread || predicate_name.empty()) {
        return;
    }

    std::string pred_name = predicate_name;
    bool is_negated = false;

    if (!pred_name.empty() && pred_name[0] == '%') {
        pred_name = pred_name.substr(1);
    }
    if (!pred_name.empty() && pred_name[0] == '!') {
        is_negated = true;
        pred_name = pred_name.substr(1);
    }

    if (pred_name.empty()) {
        return;
    }

    bool final_value = is_negated ? !value : value;

    auto reg_manager = thread->get_register_bank_manager();
    if (reg_manager) {
        void* reg_addr = reg_manager->get_register(pred_name, thread->warp_id_, thread->lane_id_);
        if (reg_addr) {
            *static_cast<uint8_t*>(reg_addr) = final_value ? 1 : 0;
        }
    }
}

void reset_path_state(WarpContext* warp) {
    if (!warp) {
        return;
    }

    warp->reset();

    for (int i = 0; i < 32; i++) {
        warp->advance_thread_pc(i, 0);
    }

    warp->set_exec_mask(0xFFFFFFFF);
}

std::map<int, std::vector<PCTraceEntry>> collect_pc_traces() {
    std::map<int, std::vector<PCTraceEntry>> result;

    if (!ptxsim::ExecutionTracer::is_enabled()) {
        return result;
    }

    const auto& trace = ptxsim::ExecutionTracer::get_trace();

    for (int i = 0; i < 32; i++) {
        const auto& thread_trace = trace.threads[i];
        std::vector<PCTraceEntry> lane_trace;

        for (const auto& entry : thread_trace.entries) {
            PCTraceEntry pc_entry;
            pc_entry.pc = static_cast<int>(entry.pc);
            pc_entry.instruction = entry.instruction_text;
            lane_trace.push_back(pc_entry);
        }

        result[i] = lane_trace;
    }

    return result;
}

std::vector<PathVerificationResult> verify_warp_branch_decisions(
    WarpContext* warp,
    const std::vector<PathConfig>& paths,
    const ExecutionEngineConfig& config
) {
    std::vector<PathVerificationResult> results;
    results.reserve(paths.size());

    static constexpr int FINISHED_PC = -1;
    static constexpr int MAX_ITERATIONS = 10000;

    for (const auto& path : paths) {
        PathVerificationResult result;
        result.path_name = path.name;
        result.expected_lanes = path.lane_ids;

        // 1. 重置路径状态
        reset_path_state(warp);

        // 2. 执行 WHILE 循环直到所有 lane 完成
        bool any_lane_active = true;
        int iteration_count = 0;

        while (any_lane_active && iteration_count < MAX_ITERATIONS) {
            iteration_count++;

            for (size_t pc = 0; pc < path.statements.size(); ++pc) {
                ptxemu::ir::StatementContext stmt = path.statements[pc];
                warp->execute_warp_instruction(stmt, static_cast<int>(pc));
            }

            // 检查是否所有 lane 结束
            any_lane_active = false;
            for (int i = 0; i < 32; ++i) {
                int lane_pc = warp->get_thread(i)->get_pc();
                if (lane_pc != FINISHED_PC) {
                    any_lane_active = true;
                    break;
                }
            }
        }

        if (iteration_count >= MAX_ITERATIONS) {
            result.passed = false;
            result.error_msg = "Infinite loop detected (MAX_ITERATIONS exceeded)";
            results.push_back(result);
            continue;
        }

        // 3. 收集实际 lane 分组（根据最终 PC）
        std::vector<int> actual_lanes;
        for (int i = 0; i < 32; ++i) {
            int lane_pc = warp->get_thread(i)->get_pc();
            if (lane_pc == FINISHED_PC) {
                actual_lanes.push_back(i);
            }
        }
        result.actual_lanes = actual_lanes;

        // 4. 简单验证：检查 lane 分组是否符合预期
        bool lanes_match = (result.expected_lanes.size() == result.actual_lanes.size());
        if (lanes_match) {
            for (size_t i = 0; i < result.expected_lanes.size(); ++i) {
                if (result.expected_lanes[i] != result.actual_lanes[i]) {
                    lanes_match = false;
                    break;
                }
            }
        }

        if (lanes_match) {
            result.passed = true;
            result.error_msg = "";
        } else {
            result.passed = false;
            result.error_msg = "Lane分组不匹配: expected " +
                std::to_string(result.expected_lanes.size()) + " lanes, got " +
                std::to_string(result.actual_lanes.size()) + " lanes";
        }

        results.push_back(result);
    }

    return results;
}

} // namespace ptxsim::verification
