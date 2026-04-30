#include "ptxsim/execution_trace.h"

namespace ptxsim {

bool ExecutionTracer::enabled_ = false;
ExecutionTrace ExecutionTracer::trace_;

} // namespace ptxsim