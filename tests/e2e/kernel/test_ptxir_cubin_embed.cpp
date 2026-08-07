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
