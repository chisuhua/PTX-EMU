#include "catch_amalgamated.hpp"
#include "ptxir/ptxir_serialization.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_reader.h"
#include "ptx_ir/statement_context.h"
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

StatementContext make_stmt(StatementType type, InstrVariant &&instr) {
    StatementContext stmt;
    stmt.type = type;
    stmt.data = std::move(instr);
    return stmt;
}

}  // namespace

TEST_CASE("Roundtrip: BranchInstr") {
    StatementContext stmt =
        make_stmt(S_BRA, BranchInstr{{}, "L1", "%p1", true, 42});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_BRA);
    const auto &out = std::get<BranchInstr>(result[0].data);
    CHECK(out.predicate_negated == true);
    CHECK(out.reconvergence_pc == 42);
}

TEST_CASE("Roundtrip: LabelInstr") {
    StatementContext stmt = make_stmt(S_LABEL, LabelInstr{"L1"});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_LABEL);
    REQUIRE(std::holds_alternative<LabelInstr>(result[0].data));
}

TEST_CASE("Roundtrip: VoidInstr (S_EXIT)") {
    StatementContext stmt = make_stmt(S_EXIT, VoidInstr{});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_EXIT);
    REQUIRE(std::holds_alternative<VoidInstr>(result[0].data));
}

TEST_CASE("Roundtrip: VoidInstr (S_RET)") {
    StatementContext stmt = make_stmt(S_RET, VoidInstr{});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_RET);
    REQUIRE(std::holds_alternative<VoidInstr>(result[0].data));
}

TEST_CASE("Roundtrip: BarrierInstr") {
    StatementContext stmt =
        make_stmt(S_BAR, BarrierInstr{{Qualifier::Q_CTA}, "cta", 0, -1});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_BAR);
    const auto &out = std::get<BarrierInstr>(result[0].data);
    CHECK(out.barId.value_or(-1) == 0);
}

TEST_CASE("Roundtrip: GenericInstr (S_MOV)") {
    // Note: the current writer stores register operands as placeholder IDs,
    // so we verify the type and qualifier roundtrip here.
    StatementContext stmt =
        make_stmt(S_MOV, GenericInstr{{Qualifier::Q_U32}, {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_MOV);
    const auto &out = std::get<GenericInstr>(result[0].data);
    CHECK(out.qualifiers == std::vector<Qualifier>{Qualifier::Q_U32});
}

TEST_CASE("Roundtrip: DeclarationInstr (S_REG)") {
    StatementContext stmt =
        make_stmt(S_REG, DeclarationInstr{DeclarationInstr::Kind::REG,
                                          "%r1",
                                          Qualifier::Q_U32,
                                          std::nullopt,
                                          std::nullopt,
                                          1,
                                          {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_REG);
    const auto &out = std::get<DeclarationInstr>(result[0].data);
    CHECK(out.kind == DeclarationInstr::Kind::REG);
    CHECK(out.dataType == Qualifier::Q_U32);
    CHECK(out.array_size == 1);
}

TEST_CASE("Roundtrip: PragmaInstr") {
    StatementContext stmt =
        make_stmt(S_PRAGMA, PragmaInstr{"#pragma unroll"});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_PRAGMA);
    REQUIRE(std::holds_alternative<PragmaInstr>(result[0].data));
}

TEST_CASE("Roundtrip: DollarNameInstr") {
    StatementContext stmt = make_stmt(S_DOLLOR, DollarNameInstr{"$r1"});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_DOLLOR);
    REQUIRE(std::holds_alternative<DollarNameInstr>(result[0].data));
}

TEST_CASE("Roundtrip: MembarInstr") {
    StatementContext stmt =
        make_stmt(S_MEMBAR, MembarInstr{{Qualifier::Q_CTA}, "cta"});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_MEMBAR);
    CHECK(std::get<MembarInstr>(result[0].data).qualifiers ==
          std::vector<Qualifier>{Qualifier::Q_CTA});
}

TEST_CASE("Roundtrip: FenceInstr") {
    StatementContext stmt = make_stmt(
        S_FENCE, FenceInstr{{Qualifier::Q_GPU}, "acquire", "gpu"});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_FENCE);
    CHECK(std::get<FenceInstr>(result[0].data).qualifiers ==
          std::vector<Qualifier>{Qualifier::Q_GPU});
}

TEST_CASE("Roundtrip: ReduxSyncInstr") {
    StatementContext stmt = make_stmt(
        S_REDUX_SYNC,
        ReduxSyncInstr{{Qualifier::Q_ADD_ATOM, Qualifier::Q_S32}, "add", {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_REDUX_SYNC);
    CHECK(std::get<ReduxSyncInstr>(result[0].data).qualifiers ==
          std::vector<Qualifier>{Qualifier::Q_ADD_ATOM, Qualifier::Q_S32});
}

TEST_CASE("Roundtrip: MbarrierInstr") {
    StatementContext stmt = make_stmt(
        S_MBARRIER_INIT, MbarrierInstr{{Qualifier::Q_CTA}, "init", {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_MBARRIER_INIT);
    CHECK(std::get<MbarrierInstr>(result[0].data).qualifiers ==
          std::vector<Qualifier>{Qualifier::Q_CTA});
}

TEST_CASE("Roundtrip: CallInstr") {
    StatementContext stmt = make_stmt(
        S_CALL, CallInstr{"foo", "call.uni foo", {Qualifier::Q_UNI}, {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_CALL);
    CHECK(std::get<CallInstr>(result[0].data).qualifiers ==
          std::vector<Qualifier>{Qualifier::Q_UNI});
}

// S_PREDICATE_PREFIX is not yet defined in ptx_op.def (the PredicatePrefix
// variant exists but has no StatementType enum value). Enable this test once the
// enum is added and the reader has a matching case.
#ifdef S_PREDICATE_PREFIX
TEST_CASE("Roundtrip: PredicatePrefix") {
    StatementContext stmt =
        make_stmt(S_PREDICATE_PREFIX, PredicatePrefix{{}, {}, "%p1"});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_PREDICATE_PREFIX);
    REQUIRE(std::holds_alternative<PredicatePrefix>(result[0].data));
}
#endif

TEST_CASE("Roundtrip: BarWarpSyncInstr") {
    StatementContext stmt =
        make_stmt(S_BAR_WARP_SYNC, BarWarpSyncInstr{{}, {}, ""});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_BAR_WARP_SYNC);
    REQUIRE(std::holds_alternative<BarWarpSyncInstr>(result[0].data));
}

TEST_CASE("Roundtrip: VoteInstr") {
    StatementContext stmt = make_stmt(
        S_VOTE, VoteInstr{{Qualifier::Q_U32}, "ballot", {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_VOTE);
    CHECK(std::get<VoteInstr>(result[0].data).qualifiers ==
          std::vector<Qualifier>{Qualifier::Q_U32});
}

TEST_CASE("Roundtrip: ShflInstr") {
    StatementContext stmt =
        make_stmt(S_SHFL, ShflInstr{{Qualifier::Q_U32}, "up", {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_SHFL);
    CHECK(std::get<ShflInstr>(result[0].data).qualifiers ==
          std::vector<Qualifier>{Qualifier::Q_U32});
}

TEST_CASE("Roundtrip: AtomInstr") {
    StatementContext stmt = make_stmt(
        S_ATOM, AtomInstr{{Qualifier::Q_U32, Qualifier::Q_GLOBAL},
                          {OperandContext{ImmOperand{"42"}}},
                          0});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_ATOM);
    CHECK(std::get<AtomInstr>(result[0].data).qualifiers ==
          std::vector<Qualifier>{Qualifier::Q_U32, Qualifier::Q_GLOBAL});
}

TEST_CASE("Roundtrip: TextureInstr") {
    StatementContext stmt = make_stmt(S_TEX, TextureInstr{{}, {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_TEX);
    REQUIRE(std::holds_alternative<TextureInstr>(result[0].data));
}

TEST_CASE("Roundtrip: SurfaceInstr") {
    StatementContext stmt = make_stmt(S_SURF, SurfaceInstr{{}, {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_SURF);
    REQUIRE(std::holds_alternative<SurfaceInstr>(result[0].data));
}

TEST_CASE("Roundtrip: ReductionInstr") {
    StatementContext stmt =
        make_stmt(S_RED, ReductionInstr{{}, "add", {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_RED);
    REQUIRE(std::holds_alternative<ReductionInstr>(result[0].data));
}

TEST_CASE("Roundtrip: PrefetchInstr") {
    StatementContext stmt = make_stmt(S_PREFETCH, PrefetchInstr{{}, {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_PREFETCH);
    REQUIRE(std::holds_alternative<PrefetchInstr>(result[0].data));
}

TEST_CASE("Roundtrip: CpAsyncInstr") {
    StatementContext stmt = make_stmt(S_CP_ASYNC, CpAsyncInstr{{}, {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_CP_ASYNC);
    REQUIRE(std::holds_alternative<CpAsyncInstr>(result[0].data));
}

TEST_CASE("Roundtrip: AbiDirective") {
    StatementContext stmt = make_stmt(S_ABI_PRESERVE, AbiDirective{15});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_ABI_PRESERVE);
    REQUIRE(std::holds_alternative<AbiDirective>(result[0].data));
}

TEST_CASE("Roundtrip: mixed 100+ statements") {
    std::vector<StatementContext> stmts;
    stmts.reserve(121);

    for (int i = 0; i < 50; ++i) {
        stmts.push_back(
            make_stmt(S_MOV, GenericInstr{{Qualifier::Q_U32}, {}}));
    }
    for (int i = 0; i < 50; ++i) {
        stmts.push_back(make_stmt(
            S_BRA, BranchInstr{{}, "L" + std::to_string(i), "", false, -1}));
    }

    stmts.push_back(make_stmt(S_EXIT, VoidInstr{}));
    stmts.push_back(make_stmt(S_RET, VoidInstr{}));
    stmts.push_back(make_stmt(S_LABEL, LabelInstr{"L99"}));
    stmts.push_back(
        make_stmt(S_BAR, BarrierInstr{{Qualifier::Q_CTA}, "cta", 0, -1}));
    stmts.push_back(
        make_stmt(S_REG, DeclarationInstr{DeclarationInstr::Kind::REG,
                                          "%r1",
                                          Qualifier::Q_U32,
                                          std::nullopt,
                                          std::nullopt,
                                          1,
                                          {}}));
    stmts.push_back(make_stmt(S_PRAGMA, PragmaInstr{"#pragma unroll"}));
    stmts.push_back(make_stmt(S_DOLLOR, DollarNameInstr{"$r1"}));
    stmts.push_back(
        make_stmt(S_MEMBAR, MembarInstr{{Qualifier::Q_CTA}, "cta"}));
    stmts.push_back(
        make_stmt(S_FENCE, FenceInstr{{Qualifier::Q_GPU}, "acquire", "gpu"}));
    stmts.push_back(make_stmt(
        S_REDUX_SYNC,
        ReduxSyncInstr{{Qualifier::Q_ADD_ATOM, Qualifier::Q_S32}, "add", {}}));
    stmts.push_back(
        make_stmt(S_MBARRIER_INIT, MbarrierInstr{{Qualifier::Q_CTA}, "init", {}}));
    stmts.push_back(make_stmt(
        S_CALL, CallInstr{"foo", "call.uni foo", {Qualifier::Q_UNI}, {}}));
    stmts.push_back(make_stmt(S_BAR_WARP_SYNC, BarWarpSyncInstr{{}, {}, ""}));
    stmts.push_back(
        make_stmt(S_VOTE, VoteInstr{{Qualifier::Q_U32}, "ballot", {}}));
    stmts.push_back(
        make_stmt(S_SHFL, ShflInstr{{Qualifier::Q_U32}, "up", {}}));
    stmts.push_back(make_stmt(S_TEX, TextureInstr{{}, {}}));
    stmts.push_back(make_stmt(S_SURF, SurfaceInstr{{}, {}}));
    stmts.push_back(make_stmt(S_RED, ReductionInstr{{}, "add", {}}));
    stmts.push_back(make_stmt(S_PREFETCH, PrefetchInstr{{}, {}}));
    stmts.push_back(make_stmt(S_CP_ASYNC, CpAsyncInstr{{}, {}}));
    stmts.push_back(make_stmt(S_ABI_PRESERVE, AbiDirective{15}));

    auto data = serialize_to_string(stmts);
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == stmts.size());

    for (size_t i = 0; i < 50; ++i) {
        CHECK(result[i].type == S_MOV);
        CHECK(std::get<GenericInstr>(result[i].data).qualifiers ==
              std::vector<Qualifier>{Qualifier::Q_U32});
    }
    for (size_t i = 50; i < 100; ++i) {
        CHECK(result[i].type == S_BRA);
    }
    for (size_t i = 100; i < result.size(); ++i) {
        CHECK(result[i].type == stmts[i].type);
    }
}

TEST_CASE("Error: invalid magic") {
    std::string bad_data = "XXXX";
    bad_data.resize(100, '\0');
    std::istringstream iss(bad_data, std::ios::binary);
    PtxirReader reader(iss);
    CHECK_THROWS_AS(reader.read(), std::runtime_error);
}

TEST_CASE("Error: unsupported version") {
    PtxirHeader hdr{};
    std::memcpy(hdr.magic, PTXIR_MAGIC, 4);
    hdr.version = 99;
    hdr.section_count = 0;
    hdr.header_size = sizeof(PtxirHeader);

    std::ostringstream oss(std::ios::binary);
    oss.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));

    std::istringstream iss(oss.str(), std::ios::binary);
    PtxirReader reader(iss);
    CHECK_THROWS_AS(reader.read(), std::runtime_error);
}
