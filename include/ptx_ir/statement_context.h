// statement_context.h
#ifndef STATEMENT_CONTEXT_H
#define STATEMENT_CONTEXT_H

#include "operand_context.h"
#include "ptx_types.h"
#include "ptxsim/execution_types.h"
#include <optional>
#include <string>
#include <variant>
#include <vector>

// -----------------------------------------------------------------------------
// Declaration-like statements (.reg, .const, .shared, etc.)
// Used for S_REG, S_CONST, S_SHARED, S_LOCAL, S_GLOBAL, S_PARAM
// -----------------------------------------------------------------------------
struct DeclarationInstr {
    enum class Kind { REG, CONST, SHARED, LOCAL, GLOBAL, PARAM };

    Kind kind;
    std::string name;             // e.g., "%r1", "buf"
    Qualifier dataType;           // e.g., .b32, .u64
    std::optional<int> alignment; // from .align N
    std::optional<int> size;      // FIXME total size in bytes
    int array_size;
    std::vector<int> initValues; // for .const {1,2,3}
};

// -----------------------------------------------------------------------------
//  $ r1 style inline register reference (S_DOLLOR → SIMPLE_NAME)
// -----------------------------------------------------------------------------
struct DollarNameInstr {
    std::string name; // e.g., " $ r1"
};

// -----------------------------------------------------------------------------
// Pragma or directive with string content (S_PRAGMA → SIMPLE_STRING)
// -----------------------------------------------------------------------------
struct PragmaInstr {
    std::string content; // e.g., "#pragma unroll"
};

// -----------------------------------------------------------------------------
// Label definition (S_LABEL → LABEL_INSTR)
// -----------------------------------------------------------------------------
struct LabelInstr {
    std::string labelName;
};

// -----------------------------------------------------------------------------
// Void instructions: no operands (S_RET, etc. → VOID_INSTR)
// -----------------------------------------------------------------------------
struct VoidInstr {
    // intentionally empty
};

// -----------------------------------------------------------------------------
// Branch instruction (S_BRA → BRANCH)
// -----------------------------------------------------------------------------
struct BranchInstr {
    std::vector<Qualifier> qualifiers;
    std::string target; // label name, e.g., "L_123"
};

// -----------------------------------------------------------------------------
// Barrier/sync instruction (S_BAR → BARRIER)
// -----------------------------------------------------------------------------
struct BarrierInstr {
    std::vector<Qualifier> qualifiers;
    std::string type;         // e.g., "cta", "gl"
    std::optional<int> barId; // optional barrier ID
};

// -----------------------------------------------------------------------------
// Membar instruction (S_MEMBAR → MEMBAR_INSTR)
// -----------------------------------------------------------------------------
struct MembarInstr {
    std::vector<Qualifier> qualifiers;
    std::string scope; // e.g., "ctx", "sys", "coherent"
};

// -----------------------------------------------------------------------------
// Fence instruction (S_FENCE → FENCE_INSTR)
// -----------------------------------------------------------------------------
struct FenceInstr {
    std::vector<Qualifier> qualifiers;
    std::string memoryOrder; // e.g., "acquire", "release"
    std::string scope;       // e.g., "gpu", "sys"
};

// -----------------------------------------------------------------------------
// Redux sync instruction (S_REDUX_SYNC → REDUX_INSTR)
// -----------------------------------------------------------------------------
struct ReduxSyncInstr {
    std::vector<Qualifier> qualifiers;
    std::string operation; // e.g., "add", "min", "max"
    std::vector<OperandContext> operands;
};

// -----------------------------------------------------------------------------
// Mbarrier instructions (S_MBARRIER_* → MBARRIER_INSTR)
// -----------------------------------------------------------------------------
struct MbarrierInstr {
    std::vector<Qualifier> qualifiers;
    std::string operation; // "init", "arrive", "try_wait"
    std::vector<OperandContext> operands;
};

// -----------------------------------------------------------------------------
// Function call (S_CALL → CALL_INSTR)
// -----------------------------------------------------------------------------
struct CallInstr {
    std::string funcName;
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands; // args + return address (if any)
};

// -----------------------------------------------------------------------------
// Predicate prefix (@%p1, @!%p2) — treated as standalone for IR clarity
// (S_AT → PREDICATE_PREFIX)
// Note: In PTX, it modifies the next instruction, but we represent it as a
// stmt.
// -----------------------------------------------------------------------------
struct PredicatePrefix {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands; // predicate register
    std::string target;                   // e.g., "%p1"
};

// -----------------------------------------------------------------------------
// Generic arithmetic/memory instructions (most ops → GENERIC_INSTR)
// e.g., add, ld, st, mov, cvt, setp, etc.
// -----------------------------------------------------------------------------
struct GenericInstr {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands;
};

// -----------------------------------------------------------------------------
// WMMA instruction (S_WMMA → WMMA_INSTR)
// -----------------------------------------------------------------------------
struct WmmaInstr {
    WmmaType wmmaType;
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands;
};

// -----------------------------------------------------------------------------
// Atomic instruction (S_ATOM → ATOM_INSTR)
// -----------------------------------------------------------------------------
struct AtomInstr {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands;
    int operandNum = 0; // metadata (e.g., number of effective operands)
};

// -----------------------------------------------------------------------------
// Warp-level instructions (S_VOTE, S_SHFL → VOTE_INSTR, SHFL_INSTR)
// -----------------------------------------------------------------------------
struct VoteInstr {
    std::vector<Qualifier> qualifiers;
    std::string mode; // "ballot", "any", "all", etc.
    std::vector<OperandContext> operands;
};

struct ShflInstr {
    std::vector<Qualifier> qualifiers;
    std::string mode; // "up", "down", "bfly", "idx"
    std::vector<OperandContext> operands;
};

// -----------------------------------------------------------------------------
// Texture/Surface instructions (S_TEX, S_SURF → TEXTURE_INSTR, SURFACE_INSTR)
// -----------------------------------------------------------------------------
struct TextureInstr {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands;
};

struct SurfaceInstr {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands;
};

// -----------------------------------------------------------------------------
// Reduction/Prefetch instructions (S_RED, S_PREFETCH → REDUCTION_INSTR, PREFETCH_INSTR)
// -----------------------------------------------------------------------------
struct ReductionInstr {
    std::vector<Qualifier> qualifiers;
    std::string operation; // "add", "min", "max"
    std::vector<OperandContext> operands;
};

struct PrefetchInstr {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands;
};

// -----------------------------------------------------------------------------
// Async Store Instruction (S_ST_ASYNC → ASYNC_STORE)
// -----------------------------------------------------------------------------
struct AsyncStoreInstr {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands; // [addr, value]
};

// -----------------------------------------------------------------------------
// Async Reduction Instruction (S_RED_ASYNC → ASYNC_REDUCE)
// -----------------------------------------------------------------------------
struct AsyncReduceInstr {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands; // [addr, value, (optional) old]
};

// -----------------------------------------------------------------------------
// Tensor Core Generator Instructions (S_TCGEN_* → TCGEN_INSTR)
// -----------------------------------------------------------------------------
struct TcgenInstr {
    std::string opName; // "alloc", "mma", etc.
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands;
};

// -----------------------------------------------------------------------------
// Tensor Map Replace (S_TENSORMAP_REPLACE → TENSORMAP_INSTR)
// -----------------------------------------------------------------------------
struct TensormapInstr {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands; // e.g., [handle, desc_ptr, layout]
};

// -----------------------------------------------------------------------------
// ABI Preserve Directive (S_ABI_PRESERVE → ABI_DIRECTIVE)
// Operand is register number (e.g., 15 → %r15)
// -----------------------------------------------------------------------------
struct AbiDirective {
    int regNumber; // from .abi_preserve N
};

// Add this to statement_context.h
struct CpAsyncInstr {
    std::vector<Qualifier> qualifiers;
    std::vector<OperandContext> operands; // [dst, src, size]
};

// =============================================================================
// 2. Unified variant type for all instruction kinds
// Manually listed based on struct_kind in ptx_op.def (no macro magic)
// =============================================================================
using InstrVariant =
    std::variant<DeclarationInstr,     // OPERAND_REG / OPERAND_CONST / OPERAND_MEMORY
                 DollarNameInstr,      // SIMPLE_NAME
                 PragmaInstr,          // SIMPLE_STRING
                 LabelInstr,           // LABEL_INSTR
                 VoidInstr,            // VOID_INSTR
                 BranchInstr,          // BRANCH
                 BarrierInstr,         // BARRIER
                 MembarInstr,          // MEMBAR_INSTR
                 FenceInstr,           // FENCE_INSTR
                 ReduxSyncInstr,       // REDUX_INSTR
                 MbarrierInstr,        // MBARRIER_INSTR
                 CallInstr,            // CALL_INSTR
                 PredicatePrefix,      // PREDICATE_PREFIX
                 GenericInstr,         // GENERIC_INSTR
                 WmmaInstr,            // WMMA_INSTR
                 AtomInstr,            // ATOM_INSTR
                 VoteInstr,            // VOTE_INSTR
                 ShflInstr,            // SHFL_INSTR
                 TextureInstr,         // TEXTURE_INSTR
                 SurfaceInstr,         // SURFACE_INSTR
                 ReductionInstr,       // REDUCTION_INSTR
                 PrefetchInstr,        // PREFETCH_INSTR
                 CpAsyncInstr,         // CP_ASYNC_INSTR  // Add this!
                 AsyncStoreInstr, 
                 AsyncReduceInstr, 
                 TcgenInstr, 
                 TensormapInstr,
                 AbiDirective>;

class StatementContext {
public:
    StatementType type;
    std::vector<Qualifier> qualifier; // e.g., {.u32, .sat, .rn}
    InstrVariant data;
    InstructionState state = InstructionState::READY;

    // Original PTX source line (for debugging/printing)
    std::string instructionText;

    // Constructor: initialize with specific instruction type
    template <typename T>
    StatementContext(StatementType t, T &&instr)
        : type(t), data(std::forward<T>(instr)) {}

    // Default constructor (needed for containers)
    StatementContext() = default;

    // Accessors (safe)
    template <typename T> T &get() { return std::get<T>(data); }

    template <typename T> const T &get() const { return std::get<T>(data); }

    // Optional: visit support
    template <typename Visitor> auto visit(Visitor &&vis) {
        return std::visit(std::forward<Visitor>(vis), data);
    }

    template <typename Visitor> auto visit(Visitor &&vis) const {
        return std::visit(std::forward<Visitor>(vis), data);
    }
    [[nodiscard]] std::string toString() const;
};

#endif