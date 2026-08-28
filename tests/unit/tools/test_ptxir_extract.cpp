#include "catch_amalgamated.hpp"
#include "cudart/ptxir_loader.h"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_writer.h"
#include <cstdlib>
#include <fstream>
#include <openssl/sha.h>
#include <vector>

namespace {

std::vector<uint8_t> make_embedded(const std::vector<uint8_t>& prefix,
                                   const std::vector<uint8_t>& section) {
    std::vector<uint8_t> out = prefix;
    out.insert(out.end(), section.begin(), section.end());
    uint32_t size_le = static_cast<uint32_t>(section.size());
    out.push_back(static_cast<uint8_t>(size_le & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 24) & 0xFF));
    out.insert(out.end(), cudart::PTXIR_EMBED_MAGIC, cudart::PTXIR_EMBED_MAGIC + 8);
    return out;
}

std::vector<uint8_t> sha256(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> hash(32);
    SHA256(data.data(), data.size(), hash.data());
    return hash;
}

std::vector<uint8_t> make_manifest_ptxir(const std::string& kernel_name,
                                           const std::vector<uint8_t>& prefix) {
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_LABEL;
    stmt.data = LabelInstr{"L0"};
    std::ostringstream oss(std::ios::binary);
    PtxirWriter writer(oss);
    ManifestSection manifest;
    manifest.kernel_name = kernel_name;
    manifest.ptx_address_size = 64;
    manifest.cubin_hash = sha256(prefix);
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

TEST_CASE("extract_legitimateEmbedded_producesPurePrefixAndPTXIR", "[ptxir_extract]") {
    std::vector<uint8_t> prefix = {'P', 'R', 'E'};
    auto section = make_manifest_ptxir("k", prefix);
    auto embedded = make_embedded(prefix, section);
    write_file("/tmp/embedded", embedded);
    int rc = std::system("build/bin/ptxir_extract --in /tmp/embedded --out-cubin /tmp/pure.cubin --out-ptxir /tmp/pure.ptxir");
    REQUIRE(rc == 0);
    std::ifstream cubin("/tmp/pure.cubin", std::ios::binary | std::ios::ate);
    auto sz = cubin.tellg();
    REQUIRE(sz == 3);
    std::ifstream ptxir("/tmp/pure.ptxir", std::ios::binary | std::ios::ate);
    auto psz = ptxir.tellg();
    REQUIRE(psz > 0);
}

TEST_CASE("extract_plainCubin_passthrough", "[ptxir_extract]") {
    std::vector<uint8_t> plain = {'P', 'L', 'A', 'I', 'N'};
    write_file("/tmp/plain.cubin", plain);
    int rc = std::system("build/bin/ptxir_extract --in /tmp/plain.cubin --out-cubin /tmp/plain.out.cubin");
    REQUIRE(rc == 0);
    std::ifstream out("/tmp/plain.out.cubin", std::ios::binary | std::ios::ate);
    REQUIRE(out.tellg() == 5);
}

TEST_CASE("extract_hashMismatch_exitsWithError", "[ptxir_extract]") {
    std::vector<uint8_t> prefix = {'P'};
    auto section = make_manifest_ptxir("k", prefix);
    prefix[0] = 'X';
    auto embedded = make_embedded(prefix, section);
    write_file("/tmp/corrupted", embedded);
    int rc = std::system("build/bin/ptxir_extract --in /tmp/corrupted --out-cubin /tmp/x 2>/dev/null");
    REQUIRE(rc != 0);
}

TEST_CASE("extract_help_printsUsage", "[ptxir_extract]") {
    int rc = std::system("build/bin/ptxir_extract --help > /dev/null");
    REQUIRE(rc == 0);
}
