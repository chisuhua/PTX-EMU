#include "catch_amalgamated.hpp"
#include "cudart/ptx_context_adapter.h"
#include "ptx_ir/statement_context.h"

using namespace cudart;

TEST_CASE("fromEmbedded_emptyManifest_populatesDefaults", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.kernelName = "";
    auto ctx = PtxContextAdapter::fromEmbedded({}, m);
    REQUIRE(ctx.ptxKernels.size() == 1);
    REQUIRE(ctx.ptxKernels[0].kernelName == "");
    REQUIRE(ctx.ptxAddressSize == 64);
}

TEST_CASE("fromEmbedded_withKernelName_setsKernelName", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.kernelName = "myKernel";
    auto ctx = PtxContextAdapter::fromEmbedded({}, m);
    REQUIRE(ctx.ptxKernels[0].kernelName == "myKernel");
}

TEST_CASE("fromEmbedded_withParams_populatesKernelParams", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.kernelName = "k";
    m.params.push_back({"x", 8, ParamKind::U64});
    m.params.push_back({"y", 4, ParamKind::U32});
    auto ctx = PtxContextAdapter::fromEmbedded({}, m);
    REQUIRE(ctx.ptxKernels[0].kernelParams.size() == 2);
    REQUIRE(ctx.ptxKernels[0].kernelParams[0].paramName == "x");
    REQUIRE(ctx.ptxKernels[0].kernelParams[0].byteSize == 8);
    REQUIRE(ctx.ptxKernels[0].kernelParams[1].paramName == "y");
}

TEST_CASE("fromEmbedded_withAddressSize_setsPtxAddressSize", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.ptxAddressSize = 32;
    auto ctx = PtxContextAdapter::fromEmbedded({}, m);
    REQUIRE(ctx.ptxAddressSize == 32);
}

TEST_CASE("fromEmbedded_stmtsBecomeKernelStatements", "[ptx_context_adapter]") {
    EmbeddedKernelManifest m;
    m.kernelName = "k";
    std::vector<StatementContext> stmts(5);
    auto ctx = PtxContextAdapter::fromEmbedded(stmts, m);
    REQUIRE(ctx.ptxKernels[0].kernelStatements.size() == 5);
}
