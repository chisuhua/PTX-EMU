#include "catch_amalgamated.hpp"
#include "memory/simple_memory.h"
#include "memory/hardware_memory_manager.h"
#include "ptxsim/ptx_exceptions.h"

TEST_CASE("SimpleMemory bounds checking", "[memory]") {
    SimpleMemory mem(1024);

    SECTION("Valid access at offset 0") {
        uint8_t data[16] = {0};
        REQUIRE_NOTHROW(mem.direct_access(0, data, 16, false));
    }

    SECTION("Valid access at last valid offset") {
        uint8_t data[16] = {0};
        REQUIRE_NOTHROW(mem.direct_access(1008, data, 16, false));
    }

    SECTION("Out of bounds: offset beyond total size") {
        uint8_t data[16] = {0};
        REQUIRE_THROWS_AS(mem.direct_access(2000, data, 16, false),
                         InvalidMemoryAccessException);
    }

    SECTION("Out of bounds: offset + size exceeds total") {
        uint8_t data[16] = {0};
        REQUIRE_THROWS_AS(mem.direct_access(1020, data, 16, false),
                         InvalidMemoryAccessException);
    }

    SECTION("validate_offset: edge cases") {
        REQUIRE(mem.validate_offset(0, 1) == true);
        REQUIRE(mem.validate_offset(0, 1024) == true);
        REQUIRE(mem.validate_offset(1023, 1) == true);
        REQUIRE(mem.validate_offset(1023, 2) == false);
        REQUIRE(mem.validate_offset(1024, 1) == false);
        REQUIRE(mem.validate_offset(500, 525) == false);
        REQUIRE(mem.validate_offset(500, 524) == true);
    }

    SECTION("Valid write access") {
        uint8_t data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        REQUIRE_NOTHROW(mem.direct_access(0, data, 16, true));
    }

    SECTION("Out of bounds write throws") {
        uint8_t data[16] = {0};
        REQUIRE_THROWS_AS(mem.direct_access(2000, data, 16, true),
                         InvalidMemoryAccessException);
    }
}

TEST_CASE("HardwareMemoryManager region bounds checking", "[memory]") {
    HardwareMemoryManager& mgr = HardwareMemoryManager::instance();
    SimpleMemory mem(4096);
    mgr.set_simple_memory(&mem);

    uint8_t* global_base = mem.get_global_pool();

    mgr.register_region({"global",
        reinterpret_cast<uint64_t>(global_base),
        mem.get_global_size(), true, true});
    mgr.register_region({"constant", 0x2000, 0x800, false, true});

    SECTION("Access within registered region succeeds") {
        uint8_t data[8] = {0};
        uint8_t* valid_addr = global_base + 16;
        REQUIRE_NOTHROW(mgr.access(valid_addr, data, 8, false, MemorySpace::GLOBAL));
    }

    SECTION("Out of bounds access throws") {
        uint8_t data[8] = {0};
        uint8_t* out_of_bounds_addr = global_base + mem.get_global_size() + 100;
        REQUIRE_THROWS_AS(
            mgr.access(out_of_bounds_addr, data, 8, false, MemorySpace::GLOBAL),
            InvalidMemoryAccessException);
    }

    SECTION("Write to read-only region throws") {
        uint8_t data[8] = {0};
        uint8_t* const_addr = reinterpret_cast<uint8_t*>(0x2000);
        REQUIRE_THROWS_AS(
            mgr.access(const_addr, data, 8, true, MemorySpace::CONST),
            InvalidMemoryAccessException);
    }

    SECTION("Read from read-only region succeeds") {
        uint8_t data[8] = {0};
        uint8_t* const_addr = reinterpret_cast<uint8_t*>(0x2000);
        REQUIRE_THROWS_AS(
            mgr.access(const_addr, data, 8, false, MemorySpace::CONST),
            InvalidMemoryAccessException);
    }

    mgr.unregister_region("global");
    mgr.unregister_region("constant");
    mgr.set_simple_memory(nullptr);
}