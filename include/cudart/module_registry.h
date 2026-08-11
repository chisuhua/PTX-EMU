#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "cudart/cudart_intrinsics.h"  // CUmodule/CUfunction/CUresult
#include "ptx_ir/statement_context.h"  // StatementContext (complete type for std::vector)

namespace ptxemu::cudart {

// Local CUDA Driver API error codes (no CUDA_ERROR_* exist in this codebase).
// Reuses CUresult return type; success code stays CUDA_SUCCESS = 0.
static constexpr CUresult CUDA_ERROR_INVALID_VALUE  = static_cast<CUresult>(1);
static constexpr CUresult CUDA_ERROR_INVALID_HANDLE = static_cast<CUresult>(2);
static constexpr CUresult CUDA_ERROR_NOT_FOUND      = static_cast<CUresult>(3);
static constexpr CUresult CUDA_ERROR_INVALID_IMAGE  = static_cast<CUresult>(4);
static constexpr CUresult CUDA_ERROR_INVALID_PTX    = static_cast<CUresult>(5);

// Lock order (Oracle C1, ptx-lessons-learned §1):
//   ModuleRegistry::mutex  →  per-PtxContext lock   (NEVER reverse)
// Holders of ModuleRegistry::mutex MUST NOT call other public ModuleRegistry
// methods that take the same lock.
class ModuleRecord {
public:
    ModuleRecord(const uint8_t* bytes, size_t size);
    std::unique_ptr<uint8_t[]> image_bytes;
    size_t image_size = 0;
    std::vector<StatementContext> parsed_statements;
};

class FunctionRecord {
public:
    CUmodule parent = nullptr;
    std::string name;
};

class ModuleRegistry {
public:
    CUresult insert(const uint8_t* bytes, size_t size, CUmodule* out);
    ModuleRecord* lookup(CUmodule mod);
    void remove(CUmodule mod);
    CUresult insert_function(CUmodule parent, const char* name, CUfunction* out);
    FunctionRecord* lookup_function(CUfunction func);
    // Snapshot child functions of a module under registry lock (safe for cleanup).
    std::vector<std::pair<CUfunction, FunctionRecord*>>
    snapshot_functions_for(CUmodule parent);
private:
    void invalidate_functions_of(CUmodule parent);  // private, caller holds mutex
    std::mutex mutex_;
    uint64_t next_module_id_ = 1;
    uint64_t next_function_id_ = 1;
    std::unordered_map<CUmodule, std::unique_ptr<ModuleRecord>> modules_;
    std::unordered_map<CUfunction, std::unique_ptr<FunctionRecord>> funcs_;
};

ModuleRegistry& global_registry();
}  // namespace ptxemu::cudart
