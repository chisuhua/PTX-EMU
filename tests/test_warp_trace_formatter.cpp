#include "catch_amalgamated.hpp"
#include "ptxsim/warp_trace_formatter.h"
#include "ptxsim/simt_stack.h"
#include <map>
#include <vector>

using namespace ptxsim;

TEST_CASE("WTF1: format_lane_ranges with full mask", "[warp_trace_formatter]") {
    REQUIRE(WarpTraceFormatter::format_lane_ranges(0xFFFFFFFFu) == "[FFFFFFFF]");
}

TEST_CASE("WTF2: format_lane_ranges with zero mask", "[warp_trace_formatter]") {
    REQUIRE(WarpTraceFormatter::format_lane_ranges(0) == "no_active_lanes");
}

TEST_CASE("WTF3: format_lane_ranges with single lane", "[warp_trace_formatter]") {
    REQUIRE(WarpTraceFormatter::format_lane_ranges(1u << 5) == "[00000020]");
}

TEST_CASE("WTF4: format_lane_ranges with contiguous range", "[warp_trace_formatter]") {
    uint32_t mask = 0x0000FFFFu; // lanes 0-15
    REQUIRE(WarpTraceFormatter::format_lane_ranges(mask) == "[0000FFFF]");
}

TEST_CASE("WTF5: format_lane_ranges with multiple ranges", "[warp_trace_formatter]") {
    uint32_t mask = 0xFFFEFFFEu; // lanes 1-15 and 17-31
    std::string result = WarpTraceFormatter::format_lane_ranges(mask);
    REQUIRE(result == "[FFFEFFFE]");
}

TEST_CASE("WTF6: format_lane_ranges with scattered lanes", "[warp_trace_formatter]") {
    uint32_t mask = (1u << 0) | (1u << 2) | (1u << 4);
    std::string result = WarpTraceFormatter::format_lane_ranges(mask);
    REQUIRE(result == "[00000015]");
}

TEST_CASE("WTF7: format_instruction basic", "[warp_trace_formatter]") {
    std::string result = WarpTraceFormatter::format_instruction(
        42, 0, 3, 16, "ld.param", 0xFFFFFFFFu);
    REQUIRE(result == "Cycle 42: SM 0 Warp 3 PC=16  [FFFFFFFF] ld.param");
}

TEST_CASE("WTF8: format_instruction with partial mask", "[warp_trace_formatter]") {
    std::string result = WarpTraceFormatter::format_instruction(
        1, 1, 0, 4, "@%p1 bra", 0xFFFFFFFEu);
    REQUIRE(result == "Cycle 1: SM 1 Warp 0 PC=4  [FFFFFFFE] @%p1 bra");
}

TEST_CASE("WTF9: format_simt_push", "[warp_trace_formatter]") {
    SIMTStackEntry entry;
    entry.branch_pc = 4;
    entry.reconvergence_pc = 12;
    entry.active_mask = 0xFFFFFFFEu;
    entry.return_mask = 0xFFFFFFFFu;
    entry.return_pc = 12;

    SIMTStack stack;
    stack.push(entry);

    std::string result = WarpTraceFormatter::format_simt_push(
        5, 0, 0, entry, 0xFFFFFFFEu, stack);

    REQUIRE(result.find("SIMT Stack push") != std::string::npos);
    REQUIRE(result.find("branch_pc=4") != std::string::npos);
    REQUIRE(result.find("reconvergence_pc=12") != std::string::npos);
    REQUIRE(result.find("taken_mask=[FFFFFFFE]") != std::string::npos);
}

TEST_CASE("WTF10: format_simt_pop", "[warp_trace_formatter]") {
    SIMTStackEntry entry;
    entry.branch_pc = 4;
    entry.reconvergence_pc = 12;
    entry.active_mask = 0xFFFFFFFEu;
    entry.return_mask = 0xFFFFFFFFu;
    entry.return_pc = 12;

    std::string result = WarpTraceFormatter::format_simt_pop(
        15, 0, 0, entry);
    
    REQUIRE(result.find("SIMT Stack pop") != std::string::npos);
    REQUIRE(result.find("reconvergence_pc=12") != std::string::npos);
}

TEST_CASE("WTF11: format_divergence with multiple PC groups", "[warp_trace_formatter]") {
    std::map<int, std::vector<int>> pc_to_lanes;
    pc_to_lanes[5] = {0};
    pc_to_lanes[8] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

    std::string result = WarpTraceFormatter::format_divergence(pc_to_lanes);
    REQUIRE(result.find("divergence") != std::string::npos);
    REQUIRE(result.find("PC=5 [00000001]") != std::string::npos);
    REQUIRE(result.find("PC=8 [FFFFFFFE]") != std::string::npos);
}

TEST_CASE("WTF12: format_divergence with single PC group returns empty", "[warp_trace_formatter]") {
    std::map<int, std::vector<int>> pc_to_lanes;
    pc_to_lanes[10] = {0, 1, 2, 3};

    std::string result = WarpTraceFormatter::format_divergence(pc_to_lanes);
    REQUIRE(result == "");
}
