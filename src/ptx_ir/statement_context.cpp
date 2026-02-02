// statement_context.cpp
#include "ptx_ir/statement_context.h"
#include "ptx_ir/ptx_types.h"
#include <algorithm>
#include <cassert>
#include <cctype> // for tolower (optional)
#include <sstream>

std::string S2s(StatementType s) {
    switch (s) {
#define X(stype, opkind, opname, count, struct_kind)                           \
    case stype:                                                                \
        return #opname;
#include "ptx_ir/ptx_op.def"
#undef X
    case S_UNKNOWN:
        return "unknown";
    default:
        assert(false && "Unknown StatementType");
        return "invalid";
    }
}
std::string qualifiersToString(const std::vector<Qualifier> &quals) {
    std::string s;
    for (const auto &q : quals) {
        s += Q2s(q);
    }
    return s;
}

std::string operandsToString(const std::vector<OperandContext> &ops) {
    std::ostringstream oss;
    for (size_t i = 0; i < ops.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << ops[i].toString(); // Assumes OperandContext::toString() exists
    }
    return oss.str();
}

// ------------------------------------------------------------
// StatementContext::toString()
// ------------------------------------------------------------
std::string StatementContext::toString() const {
    // Prefer original source if available
    if (!instructionText.empty()) {
        return instructionText;
    }

    std::ostringstream oss;

    // Special: Predicate prefix (@%p1)
    if (type == S_AT) {
        const auto &pred = get<PredicatePrefix>();
        oss << "@";
        if (!pred.operands.empty()) {
            oss << "(" << operandsToString(pred.operands) << ")";
        }
        oss << pred.target;

        return oss.str();
    }

    // Special: Dollar name ( $ r1)
    if (type == S_DOLLOR) {
        const auto &d = get<DollarNameInstr>();
        oss << d.name;
        return oss.str();
    }

    // Special: Pragma
    if (type == S_PRAGMA) {
        const auto &p = get<PragmaInstr>();
        oss << p.content;
        return oss.str();
    }

    // Declaration directives: .reg, .const, .shared, etc.
    if (type == S_REG || type == S_CONST || type == S_SHARED ||
        type == S_LOCAL || type == S_GLOBAL || type == S_PARAM) {
        const auto &decl = get<DeclarationInstr>();
        std::string directive;
        if (type == S_REG)
            directive = ".reg";
        else if (type == S_CONST)
            directive = ".const";
        else if (type == S_SHARED)
            directive = ".shared";
        else if (type == S_LOCAL)
            directive = ".local";
        else if (type == S_GLOBAL)
            directive = ".global";
        else if (type == S_PARAM)
            directive = ".param";

        oss << directive << " " << Q2s(decl.dataType) << " " << decl.name;

        if (decl.size && *decl.size > 1) {
            oss << "[" << *decl.size << "]";
        }

        if (decl.alignment) {
            oss << " <" << *decl.alignment << ">";
        }

        if (!decl.initValues.empty()) {
            oss << " {";
            for (size_t i = 0; i < decl.initValues.size(); ++i) {
                if (i > 0)
                    oss << ", ";
                oss << decl.initValues[i];
            }
            oss << "}";
        }

        oss << ";";
        return oss.str();
    }

    // Void instructions (e.g., ret)
    if (type == S_RET) {
        oss << "ret;";
        return oss.str();
    }

    // Branch
    if (type == S_BRA) {
        const auto &b = get<BranchInstr>();
        oss << "bra" << qualifiersToString(b.qualifiers) << " " << b.target
            << ";";
        return oss.str();
    }

    // Barrier
    if (type == S_BAR) {
        const auto &bar = get<BarrierInstr>();
        oss << "bar.sync" << qualifiersToString(bar.qualifiers);
        if (bar.barId) {
            oss << " " << bar.type << ", " << *bar.barId;
        } else if (!bar.type.empty()) {
            oss << " " << bar.type;
        }
        oss << ";";
        return oss.str();
    }

    // Call
    if (type == S_CALL) {
        const auto &call = get<CallInstr>();
        oss << "call" << qualifiersToString(call.qualifiers) << " "
            << call.funcName;
        if (!call.operands.empty()) {
            oss << "(" << operandsToString(call.operands) << ")";
        }
        oss << ";";
        return oss.str();
    }

    // Generic, WMMA, Atom — all use operand list + qualifiers
    if (holds_alternative<GenericInstr>(data)) {
        const auto &g = get<GenericInstr>();
        oss << S2s(type) << qualifiersToString(g.qualifiers) << " "
            << operandsToString(g.operands) << ";";
        return oss.str();
    }

    if (type == S_WMMA) {
        const auto &w = get<WmmaInstr>();
        oss << "wmma.mma" << qualifiersToString(w.qualifiers) << " "
            << operandsToString(w.operands) << ";";
        return oss.str();
    }

    if (type == S_ATOM) {
        const auto &a = get<AtomInstr>();
        oss << "atom" << qualifiersToString(a.qualifiers) << " "
            << operandsToString(a.operands) << ";";
        return oss.str();
    }

    // Fallback: unknown or unhandled type
    oss << "// unknown stmt: " << S2s(type);
    return oss.str();
}