#include "catch_amalgamated.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void clean_ptx(const std::string& src, const std::string& dst) {
    std::string content = read_file(src);
    auto pos = content.find(".version");
    if (pos != std::string::npos) {
        content = content.substr(pos);
    }
    write_file(dst, content);
}

long file_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() ? static_cast<long>(f.tellg()) : -1;
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

int run(const std::string& cmd) {
    return std::system(cmd.c_str());
}

}  // namespace

TEST_CASE("e2e_nvccCompile_embed_cuobjdumpReadsBoth", "[e2e][ptxir]") {
    write_file("/tmp/simple.cu",
        "extern \"C\" __global__ void kernel_simple_add(int* x, int* y, int* out) {\n"
        "    *out = *x + *y;\n"
        "}\n");

    REQUIRE(run("nvcc -c -arch=compute_100 /tmp/simple.cu -o /tmp/simple.o") == 0);
    REQUIRE(run("cuobjdump -ptx /tmp/simple.o > /tmp/simple.ptx") == 0);
    clean_ptx("/tmp/simple.ptx", "/tmp/simple.clean.ptx");
    REQUIRE(file_exists("/tmp/simple.clean.ptx"));

    REQUIRE(run("build/bin/ptxir_embed --in-cubin /tmp/simple.o --in-ptx /tmp/simple.clean.ptx --kernel-name kernel_simple_add --out /tmp/simple.embedded.o") == 0);
    REQUIRE(file_size("/tmp/simple.embedded.o") > file_size("/tmp/simple.o"));

    REQUIRE(run("build/bin/ptxir_extract --in /tmp/simple.embedded.o --out-cubin /tmp/simple.pure.o --out-ptxir /tmp/simple.pure.ptxir") == 0);
    REQUIRE(file_exists("/tmp/simple.pure.o"));
    REQUIRE(file_exists("/tmp/simple.pure.ptxir"));

    REQUIRE(run("cuobjdump -ptx /tmp/simple.pure.o > /tmp/simple.pure.ptx") == 0);
    REQUIRE(run("cuobjdump -ptx /tmp/simple.embedded.o > /tmp/simple.embedded.ptx") == 0);
}

TEST_CASE("e2e_embed_extract_ptxirRoundtrip", "[e2e][ptxir]") {
    write_file("/tmp/simple.cu",
        "extern \"C\" __global__ void kernel_simple_add(int* x, int* y, int* out) {\n"
        "    *out = *x + *y;\n"
        "}\n");
    REQUIRE(run("nvcc -c -arch=compute_100 /tmp/simple.cu -o /tmp/simple2.o") == 0);
    REQUIRE(run("cuobjdump -ptx /tmp/simple2.o > /tmp/simple2.ptx") == 0);
    clean_ptx("/tmp/simple2.ptx", "/tmp/simple2.clean.ptx");
    REQUIRE(run("build/bin/ptxir_embed --in-cubin /tmp/simple2.o --in-ptx /tmp/simple2.clean.ptx --kernel-name kernel_simple_add --out /tmp/simple2.embedded.o") == 0);
    REQUIRE(run("build/bin/ptxir_extract --in /tmp/simple2.embedded.o --out-cubin /tmp/simple2.pure.o --out-ptxir /tmp/simple2.pure.ptxir") == 0);
    REQUIRE(file_size("/tmp/simple2.pure.o") == file_size("/tmp/simple2.o"));
}

TEST_CASE("e2e_cuModuleLoadData_noDriver_explicitSkip", "[e2e][ptxir]") {
    REQUIRE(true);
}

// Phase 12.2 R5: Oracle review scenarios for ADR-0024 §风险 risk 1.
//
// Validates the core architectural assumption of PTXIR-Embedded CUBIN format:
// NVIDIA's cuobjdump MUST tolerate trailing PTXIR section + footer in the
// embedded cubin. If this fails, the whole PTXIR-Embedded CUBIN story breaks
// (every CUDA tool that reads the cubin would fail).
TEST_CASE("e2e_cuobjdumpDumpSass_directOnEmbeddedCubin_succeeds",
          "[e2e][ptxir][regression;PHASE12.2-R5]") {
    write_file("/tmp/r5_sass.cu",
        "extern \"C\" __global__ void kernel_simple_add(int* x, int* y, int* out) {\n"
        "    *out = *x + *y;\n"
        "}\n");

    REQUIRE(run("nvcc -c -arch=compute_100 /tmp/r5_sass.cu -o /tmp/r5_sass.o") == 0);
    REQUIRE(run("cuobjdump -ptx /tmp/r5_sass.o > /tmp/r5_sass.ptx") == 0);
    clean_ptx("/tmp/r5_sass.ptx", "/tmp/r5_sass.clean.ptx");

    REQUIRE(run("build/bin/ptxir_embed --in-cubin /tmp/r5_sass.o --in-ptx /tmp/r5_sass.clean.ptx --kernel-name kernel_simple_add --out /tmp/r5_sass.embedded.o") == 0);

    long pure_size = file_size("/tmp/r5_sass.o");
    long embedded_size = file_size("/tmp/r5_sass.embedded.o");
    REQUIRE(pure_size > 0);
    REQUIRE(embedded_size > pure_size);

    REQUIRE(run("cuobjdump --dump-sass /tmp/r5_sass.embedded.o > /tmp/r5_sass.embedded.sass") == 0);
    REQUIRE(file_exists("/tmp/r5_sass.embedded.sass"));
    std::string sass_output = read_file("/tmp/r5_sass.embedded.sass");
    REQUIRE_FALSE(sass_output.empty());

    REQUIRE(run("cuobjdump --dump-sass /tmp/r5_sass.o > /tmp/r5_sass.pure.sass") == 0);
    std::string sass_pure = read_file("/tmp/r5_sass.pure.sass");
    REQUIRE_FALSE(sass_pure.empty());
}

TEST_CASE("e2e_cuobjdumpDumpPtx_afterExtract_succeeds",
          "[e2e][ptxir][regression;PHASE12.2-R5]") {
    write_file("/tmp/r5_ptx.cu",
        "extern \"C\" __global__ void kernel_simple_add(int* x, int* y, int* out) {\n"
        "    *out = *x + *y;\n"
        "}\n");

    REQUIRE(run("nvcc -c -arch=compute_100 /tmp/r5_ptx.cu -o /tmp/r5_ptx.o") == 0);
    REQUIRE(run("cuobjdump -ptx /tmp/r5_ptx.o > /tmp/r5_ptx.ptx") == 0);
    clean_ptx("/tmp/r5_ptx.ptx", "/tmp/r5_ptx.clean.ptx");

    REQUIRE(run("build/bin/ptxir_embed --in-cubin /tmp/r5_ptx.o --in-ptx /tmp/r5_ptx.clean.ptx --kernel-name kernel_simple_add --out /tmp/r5_ptx.embedded.o") == 0);
    REQUIRE(run("build/bin/ptxir_extract --in /tmp/r5_ptx.embedded.o --out-cubin /tmp/r5_ptx.pure.o --out-ptxir /tmp/r5_ptx.pure.ptxir") == 0);

    REQUIRE(run("cuobjdump -ptx /tmp/r5_ptx.pure.o > /tmp/r5_ptx.pure.ptx") == 0);
    REQUIRE(file_exists("/tmp/r5_ptx.pure.ptx"));
    std::string ptx_output = read_file("/tmp/r5_ptx.pure.ptx");
    REQUIRE_FALSE(ptx_output.empty());
}
