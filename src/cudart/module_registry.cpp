#include "cudart/module_registry.h"
#include <cstring>

namespace ptxemu::cudart {

// -----------------------------------------------------------------------------
// ModuleRecord
// -----------------------------------------------------------------------------
ModuleRecord::ModuleRecord(const uint8_t* bytes, size_t size)
    : image_size(size) {
    if (bytes && size > 0) {
        image_bytes = std::make_unique<uint8_t[]>(size);
        std::memcpy(image_bytes.get(), bytes, size);
    }
}

// -----------------------------------------------------------------------------
// ModuleRegistry
// -----------------------------------------------------------------------------
CUresult ModuleRegistry::insert(const uint8_t* bytes, size_t size, CUmodule* out) {
    if (!bytes || size == 0 || out == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard lock(mutex_);
    CUmodule handle = reinterpret_cast<CUmodule>(next_module_id_++);
    auto record = std::make_unique<ModuleRecord>(bytes, size);
    modules_.emplace(handle, std::move(record));
    *out = handle;
    return CUDA_SUCCESS;
}

ModuleRecord* ModuleRegistry::lookup(CUmodule mod) {
    std::lock_guard lock(mutex_);
    auto it = modules_.find(mod);
    if (it != modules_.end()) {
        return it->second.get();
    }
    return nullptr;
}

void ModuleRegistry::remove(CUmodule mod) {
    std::lock_guard lock(mutex_);
    invalidate_functions_of(mod);
    modules_.erase(mod);
}

CUresult ModuleRegistry::insert_function(CUmodule parent, const char* name, CUfunction* out) {
    if (parent == nullptr || name == nullptr || out == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard lock(mutex_);
    // Verify parent module exists
    if (modules_.find(parent) == modules_.end()) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    CUfunction handle = reinterpret_cast<CUfunction>(next_function_id_++);
    auto record = std::make_unique<FunctionRecord>();
    record->parent = parent;
    record->name = name;
    funcs_.emplace(handle, std::move(record));
    *out = handle;
    return CUDA_SUCCESS;
}

FunctionRecord* ModuleRegistry::lookup_function(CUfunction func) {
    std::lock_guard lock(mutex_);
    auto it = funcs_.find(func);
    if (it != funcs_.end()) {
        return it->second.get();
    }
    return nullptr;
}

void ModuleRegistry::invalidate_functions_of(CUmodule parent) {
    for (auto it = funcs_.begin(); it != funcs_.end(); ) {
        if (it->second->parent == parent) {
            it = funcs_.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<std::pair<CUfunction, FunctionRecord*>>
ModuleRegistry::snapshot_functions_for(CUfunction parent) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<CUfunction, FunctionRecord*>> out;
    for (auto& kv : funcs_) {
        if (kv.second->parent == parent) {
            out.emplace_back(kv.first, kv.second.get());
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// global_registry
// -----------------------------------------------------------------------------
ModuleRegistry& global_registry() {
    static ModuleRegistry instance;
    return instance;
}

}  // namespace ptxemu::cudart
