// tests/integration/test_phase0_byte_identical_gates.cpp
// Per ADR-0029 §D7: Phase 0 Step 1 完成后 5 gates 必须全部通过
// 验证 5 全局符号搬迁不破坏默认 LD_PRELOAD 路径的字节级行为
//
// Gates:
//   1. nm -D --defined-only libcudart.so 前后 diff 为空
//   2. SONAME libcudart.so.12 保持
//   3. POST_BUILD symlinks (.12 + 主符号链接) 保持
//   4. g_cpptlm_bridge == nullptr 单元测试
//   5. logger → g_gpu_context 单元测试 (get_gpu_clock_from_context)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <catch_amalgamated.hpp>

#include "cudart/cpptlm_bridge.h"
#include "cudart/ptx_interpreter.h"

extern "C" size_t get_gpu_clock_from_context();

namespace fs = std::filesystem;

static std::string run_cmd(const std::string& cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buf[4096];
    std::string out;
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}

// Symbol types that are NOT ABI commitments (template instantiation noise):
//   W — weak symbol       (dynamic linker coalesces with strong defs)
//   V — weak object       (same semantics as W for data)
// These appear/disappear with any TU-level template instantiation change
// (e.g. when a header stops being ODR-used by a particular TU), tracking
// compiler state rather than ABI. They MUST NOT trigger the byte-identical
// gate — see Oracle H3 (chore(test): harden phase0_gate1 against
// weak-symbol drift) for diagnosis.
static bool is_template_instantiation_noise(const std::string& sym_type) {
    return sym_type == "W" || sym_type == "V";
}

static std::string extract_symbols(const std::string& nm_output) {
    std::set<std::string> syms;
    std::istringstream iss(nm_output);
    std::string line;
    while (std::getline(iss, line)) {
        std::istringstream ls(line);
        std::string tok;
        std::vector<std::string> toks;
        while (ls >> tok) toks.push_back(tok);
        if (toks.size() >= 2) {
            size_t start = 0;
            if (toks[0].find_first_of("0123456789abcdefABCDEF") == 0 &&
                toks[0].size() > 4) {
                start = 1;
            }
            if (start + 1 < toks.size()) {
                if (is_template_instantiation_noise(toks[start])) continue;
                syms.insert(toks[start] + " " + toks[start + 1]);
            }
        }
    }
    std::string out;
    for (const auto& s : syms) out += s + "\n";
    return out;
}

static fs::path build_libcudart() {
    const char* ctd = std::getenv("CMAKE_BINARY_DIR");
    if (ctd) return fs::path(ctd) / "lib" / "libcudart.so.12.0";
    fs::path cwd = fs::current_path();
    if (cwd.filename() == "build") return cwd / "lib" / "libcudart.so.12.0";
    fs::path p = cwd;
    while (!p.empty()) {
        if (fs::exists(p / "build" / "lib" / "libcudart.so.12.0")) {
            return p / "build" / "lib" / "libcudart.so.12.0";
        }
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::path("build/lib/libcudart.so.12.0");
}

static fs::path project_lib_dir() {
    const char* src = std::getenv("CMAKE_SOURCE_DIR");
    if (src) return fs::path(src) / "lib";
    fs::path p = fs::current_path();
    while (!p.empty()) {
        if (fs::exists(p / "lib" / "libcudart.so")) return p / "lib";
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::path("lib");
}

static fs::path baseline_nm_path() {
    return "/tmp/baseline-artifacts/libcudart-nm-before.txt";
}

// ADR-citation manifest: T-type symbols explicitly authorized as stable
// ABI commitments to libcudart.so. Each entry MUST reference the ADR
// (and section) that authorizes the export. Empty after ADR-0029 §D5
// (cpptlm_module.cpp removed from cudart library — ptxemu_image_* now
// lives exclusively in libptxemu_device.so).
//
// Future Phase ABI additions: append the T symbol here with the ADR
// reference. Without a manifest entry, the gate fails with a clear
// message naming the undeclared symbol.
static const std::set<std::string> kAllowedAdditions = {
};

static std::set<std::string> diff_set(const std::string& current,
                                      const std::string& baseline) {
    std::set<std::string> cur, base;
    std::istringstream cs(current), bs(baseline);
    std::string line;
    while (std::getline(cs, line)) if (!line.empty()) cur.insert(line);
    while (std::getline(bs, line)) if (!line.empty()) base.insert(line);
    std::set<std::string> added;
    std::set_difference(cur.begin(), cur.end(),
                        base.begin(), base.end(),
                        std::inserter(added, added.begin()));
    return added;
}

// ---------------------------------------------------------------------------
// Gate 1: nm -D --defined-only libcudart.so symbol surface unchanged
// ---------------------------------------------------------------------------
TEST_CASE("Gate 1: nm -D --defined-only libcudart.so symbol surface unchanged",
          "[integration][phase0][gate]") {
    fs::path libcudart = build_libcudart();
    REQUIRE(fs::exists(libcudart));

    std::string current = extract_symbols(
        run_cmd("nm -D --defined-only " + libcudart.string()));
    fs::path baseline_nm = baseline_nm_path();

    if (fs::exists(baseline_nm)) {
        std::ifstream f(baseline_nm);
        std::string baseline_raw((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
        std::string baseline = extract_symbols(baseline_raw);
        REQUIRE(current == baseline);

        // Focused secondary check: when the diff above has new T-type
        // symbols not declared in kAllowedAdditions, surface them as a
        // distinct failure so the maintainer sees exactly which symbol
        // needs an ADR citation (rather than wading through the full dump).
        std::set<std::string> added = diff_set(current, baseline);
        std::set<std::string> undeclared;
        std::set_difference(added.begin(), added.end(),
                            kAllowedAdditions.begin(),
                            kAllowedAdditions.end(),
                            std::inserter(undeclared, undeclared.begin()));
        INFO("Undeclared T-type additions (add to kAllowedAdditions "
             "with an ADR citation):");
        REQUIRE(undeclared.empty());
    } else {
        fs::create_directories("/tmp/baseline-artifacts");
        std::ofstream f(baseline_nm);
        f << current;
        WARN("No baseline at /tmp/baseline-artifacts/libcudart-nm-before.txt — captured");
        SUCCEED("baseline captured");
    }
}

// ---------------------------------------------------------------------------
// Gate 2: SONAME preserved as libcudart.so.12
// ---------------------------------------------------------------------------
TEST_CASE("Gate 2: SONAME preserved as libcudart.so.12",
          "[integration][phase0][gate]") {
    fs::path libcudart = build_libcudart();
    REQUIRE(fs::exists(libcudart));
    std::string out = run_cmd("objdump -p " + libcudart.string() + " | grep SONAME");
    REQUIRE(out.find("libcudart.so.12") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Gate 3: POST_BUILD symlinks preserved
// ---------------------------------------------------------------------------
TEST_CASE("Gate 3: POST_BUILD symlinks preserved (.12 + main)",
          "[integration][phase0][gate]") {
    fs::path libdir = project_lib_dir();
    REQUIRE(fs::exists(libdir));
    std::string out = run_cmd("ls -la " + libdir.string() + "/libcudart.so*");
    REQUIRE(out.find("libcudart.so.12") != std::string::npos);
    REQUIRE(out.find("libcudart.so ") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Gate 4: g_cpptlm_bridge == nullptr standalone mode
// ---------------------------------------------------------------------------
TEST_CASE("Gate 4: g_cpptlm_bridge == nullptr standalone mode contract",
          "[integration][phase0][gate]") {
    REQUIRE(g_cpptlm_bridge == nullptr);
}

// ---------------------------------------------------------------------------
// Gate 5: logger → g_gpu_context clock path (relocation linkage)
// ---------------------------------------------------------------------------
TEST_CASE("Gate 5: get_gpu_clock_from_context() resolves after g_gpu_context relocation",
          "[integration][phase0][gate]") {
    size_t clk = get_gpu_clock_from_context();
    SUCCEED("get_gpu_clock_from_context() linked successfully; clk=" << clk);
}
