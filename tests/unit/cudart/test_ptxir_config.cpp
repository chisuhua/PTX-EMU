#include "catch_amalgamated.hpp"
#include "cudart/ptxir_config.h"
#include <cstdlib>

using namespace config;

TEST_CASE("isPTXIRModeEnabled_unset_returnsFalse", "[ptxir_config]") {
    unsetenv("PTXIR_MODE");
    setPTXIRModeFromIni(false);
    REQUIRE(isPTXIRModeEnabled() == false);
}

TEST_CASE("isPTXIRModeEnabled_PTXIR_MODE_off_returnsFalse", "[ptxir_config]") {
    setenv("PTXIR_MODE", "off", 1);
    setPTXIRModeFromIni(true);
    REQUIRE(isPTXIRModeEnabled() == false);
}

TEST_CASE("isPTXIRModeEnabled_PTXIR_MODE_auto_returnsTrue", "[ptxir_config]") {
    setenv("PTXIR_MODE", "auto", 1);
    setPTXIRModeFromIni(false);
    REQUIRE(isPTXIRModeEnabled() == true);
}

TEST_CASE("isPTXIRModeEnabled_envOverridesIni_returnsTrue", "[ptxir_config]") {
    setPTXIRModeFromIni(false);
    setenv("PTXIR_MODE", "auto", 1);
    REQUIRE(isPTXIRModeEnabled() == true);
}
