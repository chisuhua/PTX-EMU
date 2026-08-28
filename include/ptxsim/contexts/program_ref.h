#ifndef PTXSIM_CONTEXTS_PROGRAM_REF_H
#define PTXSIM_CONTEXTS_PROGRAM_REF_H

#include "ptx_ir/statement_context.h"
#include "ptxsim/common_types.h"
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <vector>

namespace ptxsim {
namespace contexts {

/**
 * @brief Program reference POD: per-thread references to the parsed PTX
 *        program (statements, symbol tables, labels) plus call stack.
 *
 * @details Groups all fields that reference the parsed PTX program state.
 *          statements is the canonical IR vector for the current kernel;
 *          name2Sym / name2Share are the symbol tables; label2pc is the
 *          label-to-PC lookup; call_stack tracks function-call depth.
 *
 * @author PTX-EMU Team (T2-3 god-class split)
 * @date 2026-06-24
 */
struct ProgramRefPod {
    // Current kernel IR statements
    std::vector<ptxemu::ir::StatementContext> *statements = nullptr;

    // Symbol tables
    std::map<std::string, std::unique_ptr<Symtable>> *name2Sym = nullptr;
    std::map<std::string, std::unique_ptr<Symtable>> *name2Share = nullptr;

    // Label → program-counter lookup
    std::map<std::string, int> label2pc;

    // Function-call return PC stack
    std::stack<int> call_stack;
};

}  // namespace contexts
}  // namespace ptxsim

#endif  // PTXSIM_CONTEXTS_PROGRAM_REF_H