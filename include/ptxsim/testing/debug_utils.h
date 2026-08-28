#ifndef PTXSIM_TESTING_DEBUG_UTILS_H
#define PTXSIM_TESTING_DEBUG_UTILS_H

#include "ptx_ir/statement_context.h"

#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace ptxsim::testing {

// ============================================================================
// Statement Sequence Printing (debugging)
// ============================================================================

inline void print_stmts(std::ostream& os, const std::vector<ptxemu::ir::StatementContext>& stmts, const char* label = "") {
    os << "=== Statements " << (label ? label : "") << " (count=" << stmts.size() << ") ===" << std::endl;
    for (size_t i = 0; i < stmts.size() && i < 30; i++) {
        os << "  [" << std::setw(3) << i << "] " << stmts[i].instructionText << std::endl;
    }
    if (stmts.size() > 30)
        os << "  ... (" << (stmts.size() - 30) << " more)" << std::endl;
}

}  // namespace ptxsim::testing

#endif  // PTXSIM_TESTING_DEBUG_UTILS_H