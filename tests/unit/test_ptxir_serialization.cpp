#include "catch_amalgamated.hpp"
#include "ptxir/ptxir_serialization.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/ptxir_reader.h"
#include "ptx_ir/statement_context.h"
#include <cstdio>
#include <cstring>
#include <fstream>
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

// Minimal PTX with one branch and one barrier for generate_ptxir() roundtrip.
const char *kPtxForCfgEmbed =
    ".version 8.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry test_cfg_embed() {\n"
    "    .reg .pred %p;\n"
    "    .reg .b32 %r;\n"
    "    mov.u32 %r, 1;\n"
    "    setp.ne.u32 %p, %r, 0;\n"
    "    @%p bra $L_skip;\n"
    "    bar.sync 0, 32;\n"
    "$L_skip:\n"
    "    ret;\n"
    "}\n";

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

TEST_CASE("Roundtrip: BarrierInstr reconvergence_pc") {
    // T1: non-default reconvergence_pc must survive roundtrip (v3 format)
    StatementContext stmt =
        make_stmt(S_BAR, BarrierInstr{{Qualifier::Q_CTA}, "cta", 0, 42});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_BAR);
    const auto &out = std::get<BarrierInstr>(result[0].data);
    CHECK(out.barId.value_or(-1) == 0);
    CHECK(out.reconvergence_pc == 42);  // FAILS on v2 (reader never reads it)
}

TEST_CASE("Roundtrip: BarrierInstr barId=nullopt") {
    // T2: sentinel -1 path - barId nullopt survives roundtrip
    StatementContext stmt =
        make_stmt(S_BAR, BarrierInstr{{}, "", std::nullopt, -1});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    const auto &out = std::get<BarrierInstr>(result[0].data);
    CHECK(!out.barId.has_value());       // FAILS on v2 (writer writes -1, reader sets nullopt only if >= 0)
    CHECK(out.reconvergence_pc == -1);
}

TEST_CASE("PTXIR header version=3 accepted") {
    // T11: read_header() must accept version 3
    PtxirHeader hdr{};
    std::memcpy(hdr.magic, PTXIR_MAGIC, 4);
    hdr.version = 3;
    hdr.section_count = 0;
    hdr.header_size = sizeof(PtxirHeader);

    std::ostringstream oss(std::ios::binary);
    oss.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));

    std::istringstream iss(oss.str(), std::ios::binary);
    PtxirReader reader(iss);
    CHECK_NOTHROW(reader.read());        // FAILS: "Unsupported PTXIR version"
}

TEST_CASE("BARRIER_ENCODED_SIZE includes reconvergence_pc") {
    // T12: constant matches v3 S_BAR encoding
    CHECK(ptxir_encoding::BARRIER_ENCODED_SIZE ==
          sizeof(uint16_t) + sizeof(int32_t) + sizeof(int32_t));  // FAILS on v2
}

TEST_CASE("generate_ptxir embeds reconvergence_pc (v3 file)") {
    // T4: generate -> load with apply_cfg=false must return non-default values
    std::string ptx_path = "test_cfg_embed.ptx";
    std::string ptxir_path = "test_cfg_embed.ptxir";
    {
        std::ofstream f(ptx_path);
        f << kPtxForCfgEmbed;
    }

    REQUIRE(generate_ptxir(ptx_path, ptxir_path));
    auto stmts = load_ptxir(ptxir_path, false);   // apply_cfg=false -> use embedded

    REQUIRE_FALSE(stmts.empty());
    bool saw_bra = false, saw_bar = false;
    for (const auto& s : stmts) {
        if (s.type == S_BRA) {
            saw_bra = true;
            CHECK(std::get<BranchInstr>(s.data).reconvergence_pc != -1);   // FAILS pre-fix (never filled)
        } else if (s.type == S_BAR) {
            saw_bar = true;
            CHECK(std::get<BarrierInstr>(s.data).reconvergence_pc != -1);  // FAILS pre-fix (never filled)
        }
    }
    CHECK(saw_bra);
    CHECK(saw_bar);
    std::remove(ptx_path.c_str());
    std::remove(ptxir_path.c_str());
}

TEST_CASE("load_ptxir v3: apply_cfg=true == apply_cfg=false") {
    // T5: embedded values must equal recomputed CFG values
    std::string ptx_path = "test_cfg_embed2.ptx";
    std::string ptxir_path = "test_cfg_embed2.ptxir";
    {
        std::ofstream f(ptx_path);
        f << kPtxForCfgEmbed;
    }

    REQUIRE(generate_ptxir(ptx_path, ptxir_path));
    auto embedded = load_ptxir(ptxir_path, false);
    auto recomputed = load_ptxir(ptxir_path, true);

    REQUIRE(embedded.size() == recomputed.size());
    for (size_t i = 0; i < embedded.size(); ++i) {
        if (embedded[i].type == S_BRA) {
            CHECK(std::get<BranchInstr>(embedded[i].data).reconvergence_pc ==
                  std::get<BranchInstr>(recomputed[i].data).reconvergence_pc);
        } else if (embedded[i].type == S_BAR) {
            CHECK(std::get<BarrierInstr>(embedded[i].data).reconvergence_pc ==
                  std::get<BarrierInstr>(recomputed[i].data).reconvergence_pc);
        }
    }
    std::remove(ptx_path.c_str());
    std::remove(ptxir_path.c_str());
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

TEST_CASE("Roundtrip: GenericInstr (S_CVTA)") {
    // Previously threw "Unknown StatementType: 28" on deserialize
    StatementContext stmt =
        make_stmt(S_CVTA, GenericInstr{{Qualifier::Q_U64}, {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_CVTA);   // FAILS pre-fix (reader throws)
    const auto &out = std::get<GenericInstr>(result[0].data);
    CHECK(out.qualifiers == std::vector<Qualifier>{Qualifier::Q_U64});
}

TEST_CASE("Roundtrip: GenericInstr (S_FMA)") {
    StatementContext stmt =
        make_stmt(S_FMA, GenericInstr{{Qualifier::Q_F32}, {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_FMA);    // FAILS pre-fix (reader throws)
}

TEST_CASE("Roundtrip: GenericInstr (S_POPC)") {
    StatementContext stmt =
        make_stmt(S_POPC, GenericInstr{{Qualifier::Q_B32}, {}});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_POPC);   // FAILS pre-fix (reader throws)
}

TEST_CASE("Roundtrip: BranchInstr (S_BRX)") {
    StatementContext stmt =
        make_stmt(S_BRX, BranchInstr{{}, "L1", "%p1", false, -1});

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_BRX);    // FAILS pre-fix (reader throws)
}

TEST_CASE("Roundtrip: VoidInstr (S_TRAP / S_BRK / S_BRKPT)") {
    for (StatementType t : {S_TRAP, S_BRK, S_BRKPT}) {
        auto data = serialize_to_string({make_stmt(t, VoidInstr{})});
        auto result = deserialize_from_string(data);
        REQUIRE(result.size() == 1);
        CHECK(result[0].type == t);    // FAILS pre-fix (reader throws)
    }
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
