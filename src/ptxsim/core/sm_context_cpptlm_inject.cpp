#include "sm_context_cpptlm_inject.h"

#include "ptxsim/warp_context.h"
#include "ptxsim/sm_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/instruction_latency_table.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/tensor_core_interface.h"
#include <cmath>
#include <cstdint>

namespace sm_cpptlm_inject {

void step_b_set_blocked_cycles(IPipelineLatencyProvider* pipeline,
                               ITensorCoreTiming* tc,
                               WarpContext* warp,
                               const StatementContext& stmt) {
    if (!pipeline && !tc)
        return; // both nullptr = no-op (lessons-learned §14 byte-identical fallback)
    uint32_t instr_latency = 0;
    if (pipeline) {
        double frac = pipeline->get_fractional_cycles_by_type(
            static_cast<int>(stmt.type),
            SMContext::map_instruction_to_pipeline(stmt));
        if (frac > 0.0)
            instr_latency = static_cast<uint32_t>(std::ceil(frac));
    }
    if (instr_latency == 0 && tc &&
        SMContext::is_tensor_core_instruction(stmt)) {
        instr_latency =
            tc->get_latency(SMContext::map_instruction_to_tc_precision(stmt));
    }
    if (instr_latency == 0) {
        instr_latency = ptxsim::getLatency(stmt.type).cycles;
    }
    if (instr_latency > 0)
        warp->set_blocked_cycles_for_active(instr_latency);
}

}  // namespace sm_cpptlm_inject