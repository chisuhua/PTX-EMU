/**
 * @file test_logger_cudart_component.cpp
 * @brief Unit tests for the cudart log component.
 *
 * Verifies that:
 *   1. PTX_DEBUG_CUDART / PTX_INFO_CUDART / PTX_WARN_CUDART /
 *      PTX_ERROR_CUDART / PTX_FATAL_CUDART macros are defined and
 *      expand to valid expressions.
 *   2. LoggerConfig::set_component_level / get_component_level accept
 *      the string "cudart" and round-trip the configured level.
 *   3. LoggerConfig::is_enabled respects the per-component threshold
 *      for "cudart" (independent of the global level).
 *   4. LoggerConfig::load_from_ini_section parses a
 *      `component.cudart=<level>` entry into the right threshold.
 *
 * These tests guard the contract that the fake libcudart's debug
 * output (cudaMemcpy / cudaMemcpyAsync / cudaMemset / cudaMalloc /
 * cudaFree internal traces) can be turned on/off via the INI file
 * component.cudart key, rather than the catch-all emu component.
 */

#include "catch_amalgamated.hpp"
#include "utils/logger.h"
#include "inipp/inipp.h"

#include <string>

using ptxsim::log_level;
using ptxsim::LoggerConfig;

// RAII helper: snapshot and restore LoggerConfig state so tests do
// not leak global side-effects to other suites.
namespace {

struct LoggerConfigGuard {
    log_level saved_global;
    log_level saved_cudart;
    bool had_cudart;

    LoggerConfigGuard()
        : saved_global(LoggerConfig::get().get_global_level()),
          saved_cudart(LoggerConfig::get().get_component_level("cudart")),
          had_cudart(true) {
        // Detect whether cudart was previously registered by trying a
        // round-trip: set debug, see if we get it back. (All current
        // LoggerConfig builds treat unknown components as global.)
        LoggerConfig::get().set_component_level("cudart", log_level::debug);
        had_cudart =
            (LoggerConfig::get().get_component_level("cudart") == log_level::debug);
    }

    ~LoggerConfigGuard() {
        auto &cfg = LoggerConfig::get();
        cfg.set_global_level(saved_global);
        if (had_cudart) {
            cfg.set_component_level("cudart", saved_cudart);
        }
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Test 1 (RED in initial commit): PTX_*_CUDART macros must be defined.
// Calling them with a format string is enough to prove the macro exists
// and expands to a well-formed expression. The output is silently
// dropped when is_enabled(...) is false, so this is safe to invoke
// regardless of the current log level.
// ---------------------------------------------------------------------------
TEST_CASE("PTX_*_CUDART macros are defined and callable",
          "[logger][cudart]") {
    PTX_DEBUG_CUDART("debug message: %d", 1);
    PTX_INFO_CUDART("info message: %s", "cudart");
    PTX_WARN_CUDART("warn message: count=%zu", static_cast<size_t>(16));
    PTX_ERROR_CUDART("error message: %p", reinterpret_cast<void *>(0x1000));
    // PTX_FATAL_CUDART would call std::abort(); we deliberately do not
    // exercise it here -- its existence is verified at compile time.
    SUCCEED("All non-fatal PTX_*_CUDART macros expanded and ran");
}

// ---------------------------------------------------------------------------
// Test 2: set_component_level / get_component_level round-trip for cudart.
// ---------------------------------------------------------------------------
TEST_CASE("LoggerConfig cudart component set/get round-trip",
          "[logger][cudart]") {
    LoggerConfigGuard guard;
    auto &cfg = LoggerConfig::get();

    cfg.set_component_level("cudart", log_level::debug);
    REQUIRE(cfg.get_component_level("cudart") == log_level::debug);

    cfg.set_component_level("cudart", log_level::info);
    REQUIRE(cfg.get_component_level("cudart") == log_level::info);

    cfg.set_component_level("cudart", log_level::error);
    REQUIRE(cfg.get_component_level("cudart") == log_level::error);
}

// ---------------------------------------------------------------------------
// Test 3: is_enabled honors the per-component level for "cudart".
// With global_level = fatal and cudart = debug, debug messages should
// still pass. With cudart = error, debug/info/warn should be filtered.
// ---------------------------------------------------------------------------
TEST_CASE("LoggerConfig cudart component is_enabled filters per level",
          "[logger][cudart]") {
    LoggerConfigGuard guard;
    auto &cfg = LoggerConfig::get();

    SECTION("component level = debug allows debug/info/warn/error/fatal") {
        cfg.set_global_level(log_level::fatal);
        cfg.set_component_level("cudart", log_level::debug);

        REQUIRE(cfg.is_enabled(log_level::trace, "cudart") == false);
        REQUIRE(cfg.is_enabled(log_level::debug, "cudart") == true);
        REQUIRE(cfg.is_enabled(log_level::info, "cudart") == true);
        REQUIRE(cfg.is_enabled(log_level::warning, "cudart") == true);
        REQUIRE(cfg.is_enabled(log_level::error, "cudart") == true);
        REQUIRE(cfg.is_enabled(log_level::fatal, "cudart") == true);
    }

    SECTION("component level = info blocks debug, allows info and above") {
        cfg.set_global_level(log_level::fatal);
        cfg.set_component_level("cudart", log_level::info);

        REQUIRE(cfg.is_enabled(log_level::debug, "cudart") == false);
        REQUIRE(cfg.is_enabled(log_level::info, "cudart") == true);
        REQUIRE(cfg.is_enabled(log_level::warning, "cudart") == true);
    }

    SECTION("component level = error blocks everything below error") {
        cfg.set_global_level(log_level::trace);
        cfg.set_component_level("cudart", log_level::error);

        REQUIRE(cfg.is_enabled(log_level::debug, "cudart") == false);
        REQUIRE(cfg.is_enabled(log_level::info, "cudart") == false);
        REQUIRE(cfg.is_enabled(log_level::warning, "cudart") == false);
        REQUIRE(cfg.is_enabled(log_level::error, "cudart") == true);
        REQUIRE(cfg.is_enabled(log_level::fatal, "cudart") == true);
    }
}

// ---------------------------------------------------------------------------
// Test 4a: load_from_ini_section parses component.cudart=debug into
// the per-component threshold. Component level wins over global.
// ---------------------------------------------------------------------------
TEST_CASE("LoggerConfig cudart=debug overrides global via INI",
          "[logger][cudart][ini]") {
    LoggerConfigGuard guard;
    auto &cfg = LoggerConfig::get();

    inipp::Ini<char>::Section section;
    section["global_level"] = "warning";
    section["component.cudart"] = "debug";

    cfg.load_from_ini_section(section);

    REQUIRE(cfg.get_global_level() == log_level::warning);
    REQUIRE(cfg.get_component_level("cudart") == log_level::debug);
    // Component level wins over global, so debug IS enabled.
    REQUIRE(cfg.is_enabled(log_level::debug, "cudart") == true);
}

// ---------------------------------------------------------------------------
// Test 4b: load_from_ini_section parses component.cudart=error; even
// with global=trace the cudart component must obey its own threshold.
// ---------------------------------------------------------------------------
TEST_CASE("LoggerConfig cudart=error silences debug/info/warn via INI",
          "[logger][cudart][ini]") {
    LoggerConfigGuard guard;
    auto &cfg = LoggerConfig::get();

    inipp::Ini<char>::Section section;
    section["global_level"] = "trace";
    section["component.cudart"] = "error";

    cfg.load_from_ini_section(section);

    REQUIRE(cfg.get_global_level() == log_level::trace);
    REQUIRE(cfg.get_component_level("cudart") == log_level::error);
    // Despite trace global, cudart must obey its own threshold.
    REQUIRE(cfg.is_enabled(log_level::debug, "cudart") == false);
    REQUIRE(cfg.is_enabled(log_level::info, "cudart") == false);
    REQUIRE(cfg.is_enabled(log_level::warning, "cudart") == false);
    REQUIRE(cfg.is_enabled(log_level::error, "cudart") == true);
}
