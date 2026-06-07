// test_memory_manager.cpp
// =============================================================================
// Unit test (类型一) — SimpleMemoryAllocator basic operations
//
// Historical note: this file originally also tested a MemoryManager class,
// but that class does not exist in the current codebase (see
// docs/developer-guide/KNOWN_ISSUES.md §"Pre-P0 Baseline Red" for the legacy
// CMake bug that prevented this file from ever compiling). The MemoryManager
// TEST_CASEs are removed here — the class they reference is a no-op. The
// SimpleMemoryAllocator tests are kept because SimpleMemoryAllocator is the
// real class used by the cudart simulator and is worth validating.
//
// Scope: allocation, no-overlap, deallocation, reuse.
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/cuda_driver.h"
#include "cudart/simple_memory_allocator.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

TEST_CASE("SimpleMemoryAllocator basic operations", "[unit][memory][allocator]") {
    // init(size) must be called before allocate; constructor does not auto-create
    // the pool (see simple_memory_allocator.cpp:4).
    constexpr size_t POOL = 1ULL << 20;
    SimpleMemoryAllocator allocator;
    allocator.init(POOL);

    SECTION("Basic allocation") {
        size_t offset1 = allocator.allocate(1024);
        REQUIRE(offset1 != static_cast<size_t>(-1));
        INFO("Allocated 1024 bytes at offset: " << offset1);

        size_t offset2 = allocator.allocate(2048);
        REQUIRE(offset2 != static_cast<size_t>(-1));
        REQUIRE(offset2 >= offset1 + 1024); // ensure no overlap
        INFO("Allocated 2048 bytes at offset: " << offset2);

        allocator.deallocate(offset1);
        allocator.deallocate(offset2);
    }

    SECTION("Allocation and deallocation") {
        size_t offset1 = allocator.allocate(1024);
        size_t offset2 = allocator.allocate(2048);

        REQUIRE(offset1 != static_cast<size_t>(-1));
        REQUIRE(offset2 != static_cast<size_t>(-1));

        allocator.deallocate(offset1);

        size_t offset3 = allocator.allocate(512); // should reuse freed space
        REQUIRE(offset3 != static_cast<size_t>(-1));
        INFO("Reused after free: offset3=" << offset3);

        allocator.deallocate(offset2);
        allocator.deallocate(offset3);
    }

    SECTION("Pool exhaustion returns -1") {
        SimpleMemoryAllocator small;
        small.init(1024);
        size_t a = small.allocate(1024);
        REQUIRE(a != static_cast<size_t>(-1));
        size_t b = small.allocate(1);
        REQUIRE(b == static_cast<size_t>(-1));
    }

    SECTION("Deallocate unknown offset is a no-op") {
        allocator.deallocate(999999);
        SUCCEED();
    }
}
