#include "cudart/module_registry.h"
#include "cudart/ptxir_loader.h"
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
    auto mod_it = modules_.find(parent);
    if (mod_it == modules_.end()) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    ModuleRecord* mod = mod_it->second.get();
    // SC-8: first-match-wins — check cache first
    auto cache_it = mod->name_to_function.find(name);
    if (cache_it != mod->name_to_function.end()) {
        *out = cache_it->second;
        return CUDA_SUCCESS;
    }
    if (!mod->manifest.kernels.empty()) {
        for (const auto& kernel : mod->manifest.kernels) {
            if (kernel.name == name) {
                CUfunction handle = reinterpret_cast<CUfunction>(next_function_id_++);
                auto record = std::make_unique<FunctionRecord>();
                record->parent = parent;
                record->name = name;
                funcs_.emplace(handle, std::move(record));
                mod->name_to_function[name] = handle;
                *out = handle;
                return CUDA_SUCCESS;
            }
        }
        return CUDA_ERROR_NOT_FOUND;
    }
    CUfunction handle = reinterpret_cast<CUfunction>(next_function_id_++);
    auto record = std::make_unique<FunctionRecord>();
    record->parent = parent;
    record->name = name;
    funcs_.emplace(handle, std::move(record));
    mod->name_to_function[name] = handle;
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
    auto mod_it = modules_.find(parent);
    if (mod_it != modules_.end()) {
        mod_it->second->name_to_function.clear();
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
