// ptxir_reader.cpp
#include "ptx_ir/ptxir_reader.h"
#include "ptx_ir/operand_context.h"
#include <bit>
#include <cstring>
#include <stdexcept>

static uint16_t read_u16(std::istream& in) {
    uint16_t v;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (std::endian::native == std::endian::big) {
        v = __builtin_bswap16(v);
    }
    return v;
}

static uint32_t read_u32(std::istream& in) {
    uint32_t v;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (std::endian::native == std::endian::big) {
        v = __builtin_bswap32(v);
    }
    return v;
}

static int32_t read_i32(std::istream& in) {
    uint32_t uv;
    in.read(reinterpret_cast<char*>(&uv), sizeof(uv));
    if (std::endian::native == std::endian::big) {
        uv = __builtin_bswap32(uv);
    }
    return static_cast<int32_t>(uv);
}

static uint8_t read_u8(std::istream& in) {
    uint8_t v;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

PtxirReader::PtxirReader(std::istream& in) : in_(in) {}

std::vector<StatementContext> PtxirReader::read() {
    read_header();
    read_string_table();
    return read_kernel_section();
}

void PtxirReader::read_header() {
    PtxirHeader hdr;
    in_.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));

    if (std::memcmp(hdr.magic, PTXIR_MAGIC, 4) != 0) {
        throw std::runtime_error("Invalid PTXIR magic");
    }
    if (hdr.version != PTXIR_VERSION) {
        throw std::runtime_error("Unsupported PTXIR version");
    }
}

void PtxirReader::read_string_table() {
    in_.seekg(static_cast<std::streamoff>(sizeof(PtxirHeader)));
    uint32_t count = read_u32(in_);

    std::streamoff base = in_.tellg();
    for (uint32_t i = 0; i < count; i++) {
        uint16_t len = read_u16(in_);
        std::string s(len, '\0');
        in_.read(s.data(), static_cast<std::streamsize>(len));
        string_table_.push_back(s);
    }
    (void)base;
}

std::vector<StatementContext> PtxirReader::read_kernel_section() {
    std::vector<StatementContext> result;
    statement_count_ = read_u32(in_);

    for (uint32_t i = 0; i < statement_count_; i++) {
        result.push_back(read_instruction());
    }
    return result;
}

StatementContext PtxirReader::read_instruction() {
    uint16_t type_raw = read_u16(in_);
    StatementType type = static_cast<StatementType>(type_raw);

    StatementContext stmt;
    stmt.type = type;

    switch (type) {
        case S_BRA: {
            BranchInstr instr;
            uint32_t pred_id = read_u32(in_);
            if (pred_id < string_table_.size()) {
                instr.predicate = string_table_[pred_id];
            }
            uint32_t target_id = read_u32(in_);
            if (target_id < string_table_.size()) {
                instr.target = string_table_[target_id];
            }
            instr.predicate_negated = read_u8(in_) != 0;
            instr.reconvergence_pc = read_i32(in_);
            instr.qualifiers = {};
            stmt.data = instr;
            break;
        }
        case S_LABEL: {
            LabelInstr instr;
            uint32_t id = read_u32(in_);
            if (id < string_table_.size()) {
                instr.labelName = string_table_[id];
            }
            stmt.data = instr;
            break;
        }
        case S_EXIT:
        case S_RET: {
            VoidInstr instr;
            stmt.data = instr;
            break;
        }
        case S_BAR: {
            BarrierInstr instr;
            int32_t bar_id = read_i32(in_);
            if (bar_id >= 0) {
                instr.barId = bar_id;
            }
            instr.qualifiers = {};
            stmt.data = instr;
            break;
        }
        case S_MOV:
        case S_ADD:
        case S_SUB:
        case S_MUL:
        case S_LD:
        case S_ST:
        case S_SETP:
        case S_CVT: {
            GenericInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            read_u32(in_);  // dst_reg_id (unused for now)
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id != 0xFFFFFFFF) {
                    std::string reg_name;
                    if (id < string_table_.size()) {
                        reg_name = string_table_[id];
                    }
                    instr.operands.emplace_back(RegOperand{reg_name, -1});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_PRAGMA: {
            PragmaInstr instr;
            uint32_t id = read_u32(in_);
            if (id < string_table_.size()) {
                instr.content = string_table_[id];
            }
            stmt.data = instr;
            break;
        }
        case S_DOLLOR: {
            DollarNameInstr instr;
            uint32_t id = read_u32(in_);
            if (id < string_table_.size()) {
                instr.name = string_table_[id];
            }
            stmt.data = instr;
            break;
        }
        case S_REG:
        case S_CONST:
        case S_SHARED:
        case S_LOCAL:
        case S_GLOBAL:
        case S_PARAM: {
            DeclarationInstr instr;
            instr.kind = static_cast<DeclarationInstr::Kind>(read_u8(in_));
            instr.dataType = static_cast<Qualifier>(read_u16(in_));
            uint32_t id = read_u32(in_);
            if (id < string_table_.size()) {
                instr.name = string_table_[id];
            }
            instr.array_size = read_i32(in_);
            stmt.data = instr;
            break;
        }
        case S_BAR_WARP_SYNC: {
            BarWarpSyncInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(ImmOperand{string_table_[id]});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_MEMBAR: {
            MembarInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            stmt.data = instr;
            break;
        }
        case S_FENCE: {
            FenceInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            stmt.data = instr;
            break;
        }
        case S_REDUX_SYNC: {
            ReduxSyncInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(ImmOperand{string_table_[id]});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_MBARRIER_INIT:
        case S_MBARRIER_ARRIVE:
        case S_MBARRIER_TRY_WAIT: {
            MbarrierInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(RegOperand{string_table_[id], -1});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_CALL: {
            CallInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(ImmOperand{string_table_[id]});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_VOTE: {
            VoteInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(RegOperand{string_table_[id], -1});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_SHFL: {
            ShflInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(RegOperand{string_table_[id], -1});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_ATOM: {
            AtomInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            read_u32(in_);  // dst_reg_id (unused, skip 0xFFFFFFFF padding)
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(ImmOperand{string_table_[id]});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_TEX:
        case S_TEX_LDG:
        case S_TEX_GRAD:
        case S_TEX_LOD:
        case S_TXQ: {
            TextureInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(RegOperand{string_table_[id], -1});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_SURF:
        case S_SULD:
        case S_SUST:
        case S_SUQ: {
            SurfaceInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(RegOperand{string_table_[id], -1});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_RED: {
            ReductionInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(RegOperand{string_table_[id], -1});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_PREFETCH:
        case S_PREFETCHU: {
            PrefetchInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(RegOperand{string_table_[id], -1});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_CP_ASYNC: {
            CpAsyncInstr instr;
            uint8_t qcount = read_u8(in_);
            for (uint8_t i = 0; i < qcount; i++) {
                instr.qualifiers.push_back(static_cast<Qualifier>(read_u16(in_)));
            }
            uint8_t ocount = read_u8(in_);
            for (uint8_t i = 0; i < ocount; i++) {
                uint32_t id = read_u32(in_);
                if (id < string_table_.size()) {
                    instr.operands.emplace_back(RegOperand{string_table_[id], -1});
                }
            }
            stmt.data = instr;
            break;
        }
        case S_ABI_PRESERVE: {
            AbiDirective instr;
            stmt.data = instr;
            break;
        }
        // S_PREDICATE_PREFIX: enum not yet defined; writer uses type 0
        // (S_REG) via makePredicatePrefix. Will be added in Phase 2.
        default:
            throw std::runtime_error("Unknown StatementType: " +
                                    std::to_string(type));
    }
    return stmt;
}
