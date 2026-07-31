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

TEST_CASE("Roundtrip: Tcgen05Instr (S_TCGEN05_MMA)") {
    // T2.3: previously silently dropped by writer (no dispatch branch)
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::MMA;
    instr.qualifiers = {Qualifier::Q_F16};
    StatementContext stmt = make_stmt(S_TCGEN05_MMA, instr);

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_TCGEN05_MMA);              // FAILS pre-fix
    const auto &out = std::get<Tcgen05Instr>(result[0].data);
    CHECK(out.op_kind == Tcgen05OpKind::MMA);            // FAILS pre-fix
    CHECK(out.qualifiers == std::vector<Qualifier>{Qualifier::Q_F16});
}

TEST_CASE("Roundtrip: Tcgen05Instr (S_TCGEN05_MMA_WS) op_kind derivation") {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::MMA_WS;
    StatementContext stmt = make_stmt(S_TCGEN05_MMA_WS, instr);

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_TCGEN05_MMA_WS);
    const auto &out = std::get<Tcgen05Instr>(result[0].data);
    CHECK(out.op_kind == Tcgen05OpKind::MMA_WS);         // FAILS pre-fix
}

TEST_CASE("Roundtrip: Tcgen05Instr (S_TCGEN05_ALLOC)") {
    Tcgen05Instr instr;
    instr.op_kind = Tcgen05OpKind::ALLOC;
    StatementContext stmt = make_stmt(S_TCGEN05_ALLOC, instr);

    auto data = serialize_to_string({stmt});
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == 1);
    CHECK(result[0].type == S_TCGEN05_ALLOC);
    const auto &out = std::get<Tcgen05Instr>(result[0].data);
    CHECK(out.op_kind == Tcgen05OpKind::ALLOC);          // FAILS pre-fix
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

// Includes required for this test: <cstdio> (std::remove), <fstream>
// (already included at top of file)

namespace {
// Representative constructor per struct_kind (minimal valid payload).
StatementContext make_representative(StatementType t) {
    switch (t) {
        case S_LABEL: return make_stmt(t, LabelInstr{"L0"});
        case S_BRA: case S_BRX:
            return make_stmt(t, BranchInstr{{}, "L0", "", false, -1});
        case S_EXIT: case S_RET: case S_TRAP: case S_BRK: case S_BRKPT:
            return make_stmt(t, VoidInstr{});
        case S_BAR: return make_stmt(t, BarrierInstr{{Qualifier::Q_CTA}, "cta", 0, -1});
        case S_REG: case S_CONST: case S_SHARED: case S_LOCAL: case S_GLOBAL: case S_PARAM:
            return make_stmt(t, DeclarationInstr{DeclarationInstr::Kind::REG,
                                                 "%r1", Qualifier::Q_U32,
                                                 std::nullopt, std::nullopt, 1, {}});
        case S_PRAGMA: return make_stmt(t, PragmaInstr{"p"});
        case S_DOLLOR: return make_stmt(t, DollarNameInstr{"$r1"});
        case S_MEMBAR: return make_stmt(t, MembarInstr{{Qualifier::Q_CTA}, "cta"});
        case S_FENCE: return make_stmt(t, FenceInstr{{Qualifier::Q_GPU}, "acquire", "gpu"});
        case S_REDUX_SYNC: return make_stmt(t, ReduxSyncInstr{{}, "add", {}});
        case S_MBARRIER_INIT: case S_MBARRIER_ARRIVE: case S_MBARRIER_TRY_WAIT:
            return make_stmt(t, MbarrierInstr{{Qualifier::Q_CTA}, "init", {}});
        case S_CALL: return make_stmt(t, CallInstr{"foo", "call foo", {Qualifier::Q_UNI}, {}});
        case S_BAR_WARP_SYNC: return make_stmt(t, BarWarpSyncInstr{{}, {}, ""});
        case S_VOTE: return make_stmt(t, VoteInstr{{Qualifier::Q_U32}, "ballot", {}});
        case S_SHFL: return make_stmt(t, ShflInstr{{Qualifier::Q_U32}, "up", {}});
        case S_ATOM: return make_stmt(t, AtomInstr{{Qualifier::Q_GLOBAL, Qualifier::Q_ADD_ATOM}, {}});
        case S_TEX: case S_TEX_LDG: case S_TEX_GRAD: case S_TEX_LOD: case S_TXQ:
            return make_stmt(t, TextureInstr{{}, {}});
        case S_SURF: case S_SULD: case S_SUST: case S_SUQ:
            return make_stmt(t, SurfaceInstr{{}, {}});
        case S_RED: return make_stmt(t, ReductionInstr{{}, "add", {}});
        case S_PREFETCH: case S_PREFETCHU: return make_stmt(t, PrefetchInstr{{}, {}});
        case S_CP_ASYNC: return make_stmt(t, CpAsyncInstr{{}, {}});
        case S_ABI_PRESERVE: return make_stmt(t, AbiDirective{15});
        case S_TCGEN05_ALLOC: case S_TCGEN05_DEALLOC: case S_TCGEN05_RELINQUISH:
        case S_TCGEN05_LD: case S_TCGEN05_ST: case S_TCGEN05_CP:
        case S_TCGEN05_MMA: case S_TCGEN05_MMA_WS: case S_TCGEN05_COMMIT:
        case S_TCGEN05_WAIT: case S_TCGEN05_FENCE: {
            Tcgen05Instr instr;
            instr.qualifiers = {Qualifier::Q_F16};
            return make_stmt(t, instr);
        }
        default:
            // All remaining GENERIC_INSTR enums
            return make_stmt(t, GenericInstr{{Qualifier::Q_U32}, {}});
    }
}
}  // namespace

TEST_CASE("Roundtrip: all 106 StatementType enums") {
    std::vector<StatementContext> stmts;
    for (int t = 0; t < S_UNKNOWN; ++t) {
        stmts.push_back(make_representative(static_cast<StatementType>(t)));
    }
    REQUIRE(stmts.size() == 106);

    auto data = serialize_to_string(stmts);
    auto result = deserialize_from_string(data);

    REQUIRE(result.size() == stmts.size());
    for (size_t i = 0; i < stmts.size(); ++i) {
        CAPTURE(i, static_cast<int>(stmts[i].type));
        CHECK(result[i].type == stmts[i].type);   // FAILS for any uncovered enum
    }
}

TEST_CASE("generate_ptxir + load_ptxir: real kernel with cvta roundtrips") {
    // T3.2: previously threw "Unknown StatementType: 28" (S_CVTA)
    std::string ptx_path = "test_real_kernel.ptx";
    std::string ptxir_path = "test_real_kernel.ptxir";
    {
        std::ifstream in(TEST_SOURCE_DIR "/tests/ptx/test_divergence_sync_standalone.ptx");
        std::ofstream out(ptx_path);
        out << in.rdbuf();
    }

    REQUIRE(generate_ptxir(ptx_path, ptxir_path));
    auto stmts = load_ptxir(ptxir_path, false);   // FAILS pre-fix (S_CVTA)
    REQUIRE_FALSE(stmts.empty());
    std::remove(ptx_path.c_str());
    std::remove(ptxir_path.c_str());
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
