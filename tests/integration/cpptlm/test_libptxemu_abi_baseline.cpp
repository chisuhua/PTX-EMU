#include <catch_amalgamated.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

static fs::path baseline_path() {
    fs::path p = fs::current_path();
    while (!p.empty()) {
        if (fs::exists(p / "tests" / "integration" / "cpptlm" / "baselines" / "libptxemu_abi_baseline.txt")) {
            return p / "tests" / "integration" / "cpptlm" / "baselines" / "libptxemu_abi_baseline.txt";
        }
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::path("tests/integration/cpptlm/baselines/libptxemu_abi_baseline.txt");
}

static fs::path lib_path() {
    fs::path p = fs::current_path();
    while (!p.empty()) {
        if (fs::exists(p / "build" / "lib" / "libptxemu_device.so")) {
            return p / "build" / "lib" / "libptxemu_device.so";
        }
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::path("build/lib/libptxemu_device.so");
}

namespace {
std::string run_nm(const fs::path& lib) {
    std::string cmd = "nm -D " + lib.string() +
                      " 2>/dev/null | awk '{print $2, $3}' | grep ptxemu_ | sort -u";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buf[4096];
    std::string out;
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}
}  // namespace

TEST_CASE("ABI baseline: libptxemu_device.so 5 symbols byte-identical",
          "[integration][cpptlm][abi-baseline]") {
    fs::path baseline_file = baseline_path();
    std::ifstream f(baseline_file);
    REQUIRE(f.good());
    std::stringstream ss; ss << f.rdbuf();
    std::string baseline = ss.str();

    std::string current = run_nm(lib_path());
    REQUIRE_FALSE(current.empty());
    REQUIRE(current == baseline);
}
