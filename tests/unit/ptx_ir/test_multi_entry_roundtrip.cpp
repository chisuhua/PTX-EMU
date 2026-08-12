// test_multi_entry_roundtrip.cpp
// Phase C1: v2 PTXIR writer multi-entry round-trip tests
#include "catch_amalgamated.hpp"
#include "ptx_ir/ptxir_writer.h"
#include "ptx_ir/ptxir_reader.h"
#include "ptx_ir/ptxir_format.h"
#include <sstream>

TEST_CASE("v2 writer: single KernelEntry round-trip preserves name", "[unit][ptxir]") {
    ManifestSection m;
    m.kernel_name = "kernel_a";
    KernelEntry ke;
    ke.name = "kernel_a";
    ke.arg_count = 0;
    ke.arg_byte_size = 0;
    m.kernels.push_back(ke);
    m.ptx_address_size = 64;

    std::ostringstream oss(std::ios::binary);
    PtxirWriter w(oss);
    w.set_manifest(m);
    w.write({});
    auto str = oss.str();

    std::istringstream iss(str, std::ios::binary);
    PtxirReader reader(iss);
    reader.read();
    const auto& manifest_read = reader.get_manifest();
    REQUIRE(manifest_read.kernels.size() == 1);
    REQUIRE(manifest_read.kernels[0].name == "kernel_a");
    REQUIRE(manifest_read.kernel_name == "kernel_a");  // backward-compat
}

TEST_CASE("v2 writer: multi KernelEntry round-trip preserves all names", "[unit][ptxir]") {
    ManifestSection m;
    KernelEntry ke1; ke1.name = "kernel_a"; ke1.arg_count = 0;
    KernelEntry ke2; ke2.name = "kernel_b"; ke2.arg_count = 1; ke2.arg_byte_size = 4;
    KernelEntry ke3; ke3.name = "kernel_c"; ke3.arg_count = 2;
    m.kernels = {ke1, ke2, ke3};
    m.kernel_name = "kernel_a";  // backward-compat: first entry
    m.ptx_address_size = 64;

    std::ostringstream oss(std::ios::binary);
    PtxirWriter w(oss);
    w.set_manifest(m);
    w.write({});
    auto str = oss.str();

    std::istringstream iss(str, std::ios::binary);
    PtxirReader reader(iss);
    reader.read();
    const auto& r = reader.get_manifest();
    REQUIRE(r.kernels.size() == 3);
    REQUIRE(r.kernels[0].name == "kernel_a");
    REQUIRE(r.kernels[1].name == "kernel_b");
    REQUIRE(r.kernels[2].name == "kernel_c");
    REQUIRE(r.kernel_name == "kernel_a");  // backward-compat preserved
}

TEST_CASE("v2 writer: empty kernels vector with empty kernel_name is allowed (dispatch validates)", "[unit][ptxir]") {
    // Writer allows v1-style empty both (backward compat); dispatch layer rejects it.
    // Validation is in the dispatch/reader path, not the writer.
    ManifestSection m;
    m.kernels.clear();
    m.kernel_name.clear();
    m.ptx_address_size = 64;

    std::ostringstream oss(std::ios::binary);
    PtxirWriter w(oss);
    w.set_manifest(m);
    REQUIRE_NOTHROW(w.write({}));  // writer allows v1-style
    auto str = oss.str();

    std::istringstream iss(str, std::ios::binary);
    PtxirReader reader(iss);
    reader.read();
    const auto& r = reader.get_manifest();
    // Backward-compat: kernels empty, kernel_name empty -> no synthesis
    REQUIRE(r.kernels.empty());
    REQUIRE(r.kernel_name.empty());
}

TEST_CASE("v2 writer: kernel_name auto-syncs from kernels[0]", "[unit][ptxir]") {
    ManifestSection m;
    KernelEntry ke; ke.name = "auto_synced_kernel";
    m.kernels.push_back(ke);
    // kernel_name intentionally left empty
    m.ptx_address_size = 64;

    std::ostringstream oss(std::ios::binary);
    PtxirWriter w(oss);
    w.set_manifest(m);
    REQUIRE_NOTHROW(w.write({}));
    auto str = oss.str();

    std::istringstream iss(str, std::ios::binary);
    PtxirReader reader(iss);
    reader.read();
    const auto& r = reader.get_manifest();
    REQUIRE(r.kernel_name == "auto_synced_kernel");
}

TEST_CASE("v2 writer: big-endian serialization produces deterministic bytes", "[unit][ptxir]") {
    ManifestSection m;
    KernelEntry ke; ke.name = "kernel_be";
    m.kernels.push_back(ke);
    m.kernel_name = "kernel_be";
    m.ptx_address_size = 64;

    std::ostringstream oss1(std::ios::binary), oss2(std::ios::binary);
    PtxirWriter w1(oss1), w2(oss2);
    w1.set_manifest(m);
    w2.set_manifest(m);
    w1.write({});
    w2.write({});
    REQUIRE(oss1.str() == oss2.str());  // deterministic
}

TEST_CASE("v2 writer: 3 kernels round-trip", "[unit][ptxir]") {
    // Test 6: 3 entries can be serialized/deserialized
    ManifestSection m;
    for (auto& name : {"vec_add", "mat_mul", "reduce_sum"}) {
        KernelEntry ke;
        ke.name = name;
        ke.arg_count = 0;
        m.kernels.push_back(ke);
    }
    m.kernel_name = "vec_add";
    m.ptx_address_size = 64;

    std::ostringstream oss(std::ios::binary);
    PtxirWriter w(oss);
    w.set_manifest(m);
    w.write({});
    auto str = oss.str();

    std::istringstream iss(str, std::ios::binary);
    PtxirReader reader(iss);
    reader.read();
    const auto& r = reader.get_manifest();
    REQUIRE(r.kernels.size() == 3);
}
