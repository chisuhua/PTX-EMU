#include "catch_amalgamated.hpp"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_reader.h"
#include "ptx_ir/ptxir_writer.h"
#include <vector>

TEST_CASE("MANIFEST section round-trips through reader", "[ptxir][manifest]") {
    ManifestSection original;
    original.cubin_hash = std::vector<uint8_t>(32, 0xAB);
    original.kernel_name = "vector_add";
    original.ptx_address_size = 64;
    original.params = {{"x", 8, ParamKind::U64}, {"y", 8, ParamKind::U64}};

    std::vector<uint8_t> buffer;
    write_manifest_section(buffer, original);

    ManifestSection recovered = read_manifest_section(buffer);

    REQUIRE(recovered.cubin_hash == original.cubin_hash);
    REQUIRE(recovered.kernel_name == original.kernel_name);
    REQUIRE(recovered.ptx_address_size == original.ptx_address_size);
    REQUIRE(recovered.params.size() == 2);
    REQUIRE(recovered.params[0].name == "x");
    REQUIRE(recovered.params[0].size == 8);
    REQUIRE(recovered.params[0].kind == ParamKind::U64);
}

TEST_CASE("MANIFEST section default on empty buffer", "[ptxir][manifest]") {
    std::vector<uint8_t> empty;
    ManifestSection recovered = read_manifest_section(empty);
    REQUIRE(recovered.cubin_hash.empty());
    REQUIRE(recovered.kernel_name.empty());
    REQUIRE(recovered.ptx_address_size == 64);
    REQUIRE(recovered.params.empty());
}

TEST_CASE("Old PtxirReader skips MANIFEST section without throwing", "[ptxir][manifest][compat]") {
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_LABEL;
    stmt.data = LabelInstr{"L0"};

    std::ostringstream oss(std::ios::binary);
    PtxirWriter writer(oss);
    ManifestSection manifest;
    manifest.kernel_name = "k";
    manifest.ptx_address_size = 64;
    writer.set_manifest(manifest);
    writer.write({stmt});

    std::string data = oss.str();
    std::istringstream iss(data, std::ios::binary);
    PtxirReader reader(iss);
    auto ctx = reader.read();
    REQUIRE(ctx.size() == 1);
    REQUIRE(ctx[0].type == S_LABEL);
    REQUIRE(reader.get_manifest().kernel_name == "k");
}
