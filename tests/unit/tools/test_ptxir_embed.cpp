#include "catch_amalgamated.hpp"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_writer.h"
#include <cstdlib>
#include <fstream>
#include <vector>

namespace {

std::vector<uint8_t> make_manifest_ptxir(const std::string& kernel_name) {
    StatementContext stmt;
    stmt.type = S_LABEL;
    stmt.data = LabelInstr{"L0"};
    std::ostringstream oss(std::ios::binary);
    PtxirWriter writer(oss);
    ManifestSection manifest;
    manifest.kernel_name = kernel_name;
    manifest.ptx_address_size = 64;
    writer.set_manifest(manifest);
    writer.write({stmt});
    std::string s = oss.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}

void write_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
}

}  // namespace

TEST_CASE("embed_legitimateExe_producesEmbeddedExe", "[ptxir_embed]") {
    std::ofstream exe("/tmp/in_exe", std::ios::binary);
    exe << "FAKE_EXE_PREFIX";
    exe.close();
    write_file("/tmp/in.ptxir", make_manifest_ptxir("vecAdd"));
    int rc = std::system("build/bin/ptxir_embed --in-exe /tmp/in_exe --in-ptxir /tmp/in.ptxir --kernel-name vecAdd --out /tmp/out_embedded");
    REQUIRE(rc == 0);
    std::ifstream out("/tmp/out_embedded", std::ios::binary | std::ios::ate);
    auto sz = out.tellg();
    REQUIRE(sz > 12);
}

TEST_CASE("embed_legitimateCubin_producesEmbeddedCubin", "[ptxir_embed]") {
    std::ofstream cubin("/tmp/in.cubin", std::ios::binary);
    cubin << "FAKE_CUBIN_PREFIX";
    cubin.close();
    write_file("/tmp/in.ptxir", make_manifest_ptxir("k"));
    int rc = std::system("build/bin/ptxir_embed --in-cubin /tmp/in.cubin --in-ptxir /tmp/in.ptxir --kernel-name k --out /tmp/out.cubin.embedded");
    REQUIRE(rc == 0);
}

TEST_CASE("embed_missingKernelName_exitsWithError", "[ptxir_embed]") {
    std::ofstream exe("/tmp/in_exe", std::ios::binary);
    exe << "x";
    exe.close();
    write_file("/tmp/in.ptxir", make_manifest_ptxir("k"));
    int rc = std::system("build/bin/ptxir_embed --in-exe /tmp/in_exe --in-ptxir /tmp/in.ptxir --out /tmp/out 2>/dev/null");
    REQUIRE(rc != 0);
}

TEST_CASE("embed_missingInputFile_exitsWithError", "[ptxir_embed]") {
    write_file("/tmp/in.ptxir", make_manifest_ptxir("k"));
    int rc = std::system("build/bin/ptxir_embed --in-exe /nonexistent --in-ptxir /tmp/in.ptxir --kernel-name k --out /tmp/out 2>/dev/null");
    REQUIRE(rc != 0);
}

TEST_CASE("embed_help_printsUsage", "[ptxir_embed]") {
    int rc = std::system("build/bin/ptxir_embed --help > /dev/null");
    REQUIRE(rc == 0);
}

TEST_CASE("embed_version_printsVersion", "[ptxir_embed]") {
    int rc = std::system("build/bin/ptxir_embed --version > /dev/null");
    REQUIRE(rc == 0);
}
