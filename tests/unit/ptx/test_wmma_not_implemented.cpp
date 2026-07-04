// test_wmma_not_implemented.cpp
// Type 1 unit test for c5 (replace-silent-stub-failures, Fix #1).
//
// WmmaHandler::processWmmaOperation is a stub. Per
// stub-explicit-failure spec (requirement WMMA-Stub-Throws-Exception):
//   - MUST throw UnsupportedInstructionException
//   - MUST call PTX_ERROR_EMU
//   - exception message MUST start with "wmma."
//   - exception error_code MUST be UNSUPPORTED_INSTRUCTION
//
// Previously this handler was a silent no-op (dst register retained
// uninitialized value), which masked downstream bugs. This test
// locks in the new behavior so any future regression to silent
// no-op is caught at unit level.

#include "catch_amalgamated.hpp"

#include "ptxsim/instruction_handlers.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"

#include <string>
#include <vector>

TEST_CASE("WmmaHandler throws UnsupportedInstructionException",
          "[unit][ptx][wmma][stub][c5]") {
    ThreadContext ctx;
    WmmaHandler handler;
    void *ops[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<Qualifier> quals;
    quals.push_back(Qualifier::Q_F16);
    quals.push_back(Qualifier::Q_F16);
    quals.push_back(Qualifier::Q_F16);
    quals.push_back(Qualifier::Q_F16);

    REQUIRE_THROWS_AS(
        handler.processWmmaOperation(&ctx, ops, quals),
        UnsupportedInstructionException);
}

TEST_CASE("WmmaHandler exception identifies wmma instruction",
          "[unit][ptx][wmma][stub][c5]") {
    // Do not change to what().rfind("wmma.", 0) == 0: PtxEmuException
    // prepends "Unsupported PTX instruction: " to what(). The prefix lives
    // in get_instruction_name(); what() only contains it as substring.
    ThreadContext ctx;
    WmmaHandler handler;
    void *ops[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<Qualifier> quals;
    quals.push_back(Qualifier::Q_F16);

    try {
        handler.processWmmaOperation(&ctx, ops, quals);
        FAIL("expected UnsupportedInstructionException");
    } catch (const UnsupportedInstructionException &e) {
        REQUIRE(e.get_instruction_name() == "wmma.*");
        std::string msg(e.what());
        REQUIRE(msg.find("wmma.*") != std::string::npos);
    }
}

TEST_CASE("WmmaHandler exception error_code is UNSUPPORTED_INSTRUCTION",
          "[unit][ptx][wmma][stub][c5]") {
    ThreadContext ctx;
    WmmaHandler handler;
    void *ops[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<Qualifier> quals;

    try {
        handler.processWmmaOperation(&ctx, ops, quals);
        FAIL("expected UnsupportedInstructionException");
    } catch (const UnsupportedInstructionException &e) {
        REQUIRE(e.get_error_code() ==
                PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION);
        REQUIRE(e.get_instruction_name() == "wmma.*");
    }
}

TEST_CASE("WmmaHandler accepts empty qualifier list and still throws",
          "[unit][ptx][wmma][stub][c5]") {
    ThreadContext ctx;
    WmmaHandler handler;
    void *ops[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<Qualifier> quals; // empty

    REQUIRE_THROWS_AS(
        handler.processWmmaOperation(&ctx, ops, quals),
        UnsupportedInstructionException);
}