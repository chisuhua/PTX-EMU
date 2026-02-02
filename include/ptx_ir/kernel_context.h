#ifndef KERNEL_CONTEXT_H
#define KERNEL_CONTEXT_H

#include "param_context.h"
#include "statement_context.h"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declare for ABI register tracking
struct RegisterState;

class KernelContext {
public:
    // --- Basic identity ---
    bool ifVisibleKernel;
    bool ifEntryKernel; // true for .entry, false for .func
    std::string kernelName;

    // --- Launch configuration constraints ---
    struct Maxntid {
        int x = 0, y = 0, z = 0;
        Maxntid() = default;
        Maxntid(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    };
    Maxntid maxntid;
    int minnctapersm = 0;
    Maxntid reqntid; // .reqntid x,y,z (new in PTX)

    // --- Resource limits ---
    std::optional<int> maxRegisters; // .maxnreg N
    size_t sharedMemorySize = 0;     // Accumulated from .shared declarations
    std::optional<size_t>
        constMemorySize; // Optional: track .const size if needed

    // --- Target & ABI info ---
    std::optional<std::string> target; // e.g., "sm_110", "sm_120a"
    std::optional<int> addressSize;    // 32 or 64

    // --- Parameters ---
    std::vector<ParamContext> kernelParams;

    // --- Statements (IR for simulation) ---
    std::vector<StatementContext> kernelStatements;

    // === NEW: PTX 9.0+ ABI Preserve Support ===
    // Registers that must be preserved across function calls (for
    // .abi_preserve) Format: { register_name → bit_width } Example: {"%r15",
    // 32}, {"%rd7", 64}
    std::unordered_map<std::string, int> abiPreservedRegisters;

    // === NEW: Cluster-scoped resources (Hopper+) ===
    // For kernels using cluster grid (sm_90+), used in async/atom/red.scope
    bool usesClusterScope = false;
    std::optional<int> clusterDimX; // From launch config or inferred

    // === NEW: Tensor Core Generator State (tcgen05.*) ===
    // For simulation of Hopper tensor core allocation
    struct TCGenState {
        bool active = false;
        int allocId = -1; // tcgen05.alloc %r1 => allocId = value of %r1
        std::string kind; // e.g., "mxf4nvf4"
        int scaleVecSize = 0;
        bool blockScale = false;
        // Add more as needed for your simulator
    };
    std::optional<TCGenState> tcgenState;

    // === Utility: Query if kernel uses async memory ops ===
    bool usesAsyncStore() const {
        for (const auto &stmt : kernelStatements) {
            if (stmt.type == S_ST_ASYNC)
                return true;
        }
        return false;
    }

    bool usesRedAsync() const {
        for (const auto &stmt : kernelStatements) {
            if (stmt.type == S_RED_ASYNC)
                return true;
        }
        return false;
    }

    // Constructor
    KernelContext()
        : ifVisibleKernel(false), ifEntryKernel(false), minnctapersm(0),
          reqntid(0, 0, 0), sharedMemorySize(0) {}

    // Helper: Add shared memory allocation (from .shared decl)
    void addSharedMemory(size_t bytes) { sharedMemorySize += bytes; }

    // Helper: Record ABI-preserved register
    void addAbiPreservedRegister(const std::string &regName, int bitWidth) {
        abiPreservedRegisters[regName] = bitWidth;
    }

    // Helper: Mark use of cluster scope
    void setUsesClusterScope(bool uses = true) { usesClusterScope = uses; }

    // Helper: Initialize TCGen state
    void initTcgenState(int allocId, const std::string &kind = "",
                        int scaleVecSize = 0, bool blockScale = false) {
        tcgenState = TCGenState{true, allocId, kind, scaleVecSize, blockScale};
    }
};

#endif // KERNEL_CONTEXT_H