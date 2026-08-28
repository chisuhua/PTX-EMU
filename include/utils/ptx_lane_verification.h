// ptx_lane_verification.h
// 功能: PTX Lane 级别分支决策验证工具
// 作者: AI Agent
// 最后修改日期: 2026-05-11

#ifndef PTX_LANE_VERIFICATION_H
#define PTX_LANE_VERIFICATION_H

#include "ptx_ir/statement_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/execution_trace.h"
#include "utils/logger.h"
#include <map>
#include <vector>
#include <string>

namespace ptxsim::verification {

// PC trace 条目（预留接口）
struct PCTraceEntry {
    int pc;
    int line;
    std::string instruction;
};

// 单条路径验证结果
struct PathVerificationResult {
    std::string path_name;
    bool passed;
    std::string error_msg;
    std::vector<int> expected_lanes;
    std::vector<int> actual_lanes;
};

// 分支决策（每条分支指令的期望行为）
struct BranchDecision {
    int pc;
    std::string predicate;
    bool expect_taken;
    int fallback_pc;
    int target_pc;
};

// 路径配置
struct PathConfig {
    std::string name;
    std::vector<int> lane_ids;
    std::vector<ptxemu::ir::StatementContext> statements;
    std::vector<BranchDecision> expected_decisions;
};

// 执行引擎配置
struct ExecutionEngineConfig {
    std::map<std::string, int> label2pc;
    std::map<std::string, int> name2RegIndex;
    bool enable_pc_trace = false;
};

// 辅助函数：构建完整执行环境
WarpContext* create_execution_warp(
    SMContext* sm,
    const ExecutionEngineConfig& config
);

// 辅助函数：重置 warp 状态
void reset_warp_state(
    WarpContext* warp,
    const std::vector<int>& lane_ids,
    const std::map<std::string, int>& register_values
);

// 辅助函数：设置线程的谓词值
void set_predicate_value(
    ThreadContext* thread,
    const std::string& predicate_name,
    bool value
);

// 辅助函数：重置执行路径状态（SIMT stack、exec_mask、PC）
void reset_path_state(WarpContext* warp);

// 辅助函数：收集 PC traces
std::map<int, std::vector<PCTraceEntry>> collect_pc_traces();

// 辅助函数：执行单条语句
void execute_single_statement(
    WarpContext* warp,
    ptxemu::ir::StatementContext& stmt,
    int target_pc = -1
);

// 批量验证 warp 分支决策
std::vector<PathVerificationResult> verify_warp_branch_decisions(
    WarpContext* warp,
    const std::vector<PathConfig>& paths,
    const ExecutionEngineConfig& config
);

// 辅助函数：检查分支决策是否符合预期
bool check_branch_decision(
    WarpContext* warp,
    const BranchDecision& decision,
    const ExecutionEngineConfig& config
);

} // namespace ptxsim::verification

#endif // PTX_LANE_VERIFICATION_H