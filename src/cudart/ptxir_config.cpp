#include "cudart/ptxir_config.h"
#include <cstdlib>
#include <cstring>

namespace config {

namespace {
constexpr int kEnvUnset = -2;
int g_ini_mode = -1;
int g_env_cached = kEnvUnset;
bool g_env_evaluated = false;

void evaluate_env_once() {
    if (!g_env_evaluated) {
        const char* env = std::getenv("PTXIR_MODE");
        if (!env) {
            g_env_cached = kEnvUnset;
        } else if (std::strcmp(env, "auto") == 0) {
            g_env_cached = 1;
        } else {
            g_env_cached = 0;
        }
        g_env_evaluated = true;
    }
}
}

void setPTXIRModeFromIni(bool enabled) {
    g_ini_mode = enabled ? 1 : 0;
    g_env_evaluated = false;
}

bool isPTXIRModeEnabled() {
    evaluate_env_once();
    if (g_env_cached != kEnvUnset) {
        return g_env_cached == 1;
    }
    return g_ini_mode == 1;
}

}  // namespace config
