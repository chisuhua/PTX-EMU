#include "memory/simple_memory_allocator.h"

namespace ptxsim {

static SimpleMemoryAllocator* global_allocator_instance = nullptr;

SimpleMemoryAllocator& get_global_allocator() {
    if (!global_allocator_instance) {
        static SimpleMemoryAllocator allocator;
        global_allocator_instance = &allocator;
    }
    return *global_allocator_instance;
}

} // namespace ptxsim