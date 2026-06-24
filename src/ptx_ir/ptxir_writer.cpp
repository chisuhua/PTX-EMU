// ptxir_writer.cpp
#include "ptx_ir/ptxir_writer.h"
#include "ptx_ir/operand_context.h"
#include <bit>
#include <cstring>

static void write_u16(std::ostream& out, uint16_t v) {
    if (std::endian::native == std::endian::big) {
        v = __builtin_bswap16(v);
    }
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

static void write_u32(std::ostream& out, uint32_t v) {
    if (std::endian::native == std::endian::big) {
        v = __builtin_bswap32(v);
    }
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

static void write_i32(std::ostream& out, int32_t v) {
    uint32_t uv = static_cast<uint32_t>(v);
    if (std::endian::native == std::endian::big) {
        uv = __builtin_bswap32(uv);
    }
    out.write(reinterpret_cast<const char*>(&uv), sizeof(uv));
}

static void write_u8(std::ostream& out, uint8_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

PtxirWriter::PtxirWriter(std::ostream& out) : out_(out) {}

void PtxirWriter::write(const std::vector<StatementContext>& statements) {
    stmts_ = statements;
    pre_pass(statements);
    write_header();
    write_string_table();
    write_kernel_section();
}

void PtxirWriter::pre_pass(const std::vector<StatementContext>& statements) {
    for (const auto& stmt : statements) {
        stmt.visit([this](const auto& instr) {
            using T = std::decay_t<decltype(instr)>;
            if constexpr (std::is_same_v<T, GenericInstr>) {
                for (const auto& op : instr.operands) {
                    if (op.kind() == OperandKind::REG) {
                        const auto& reg = std::get<RegOperand>(op.data);
                        if (reg2id_.find(reg.fullName()) == reg2id_.end()) {
                            reg2id_[reg.fullName()] = static_cast<uint32_t>(reg2id_.size());
                        }
                    } else if (op.kind() == OperandKind::IMM) {
                        get_string_id(std::get<ImmOperand>(op.data).value);
                    }
                }
            } else if constexpr (std::is_same_v<T, BranchInstr>) {
                if (!instr.target.empty()) get_string_id(instr.target);
                if (!instr.predicate.empty()) get_string_id(instr.predicate);
            } else if constexpr (std::is_same_v<T, DeclarationInstr>) {
                get_string_id(instr.name);
            } else if constexpr (std::is_same_v<T, LabelInstr>) {
                get_string_id(instr.labelName);
            } else if constexpr (std::is_same_v<T, PragmaInstr>) {
                get_string_id(instr.content);
            } else if constexpr (std::is_same_v<T, DollarNameInstr>) {
                get_string_id(instr.name);
            } else if constexpr (std::is_same_v<T, BarWarpSyncInstr>) {
                for (const auto& op : instr.operands) {
                    if (op.kind() == OperandKind::IMM) {
                        get_string_id(std::get<ImmOperand>(op.data).value);
                    }
                }
            }
        });
    }
}

uint32_t PtxirWriter::get_reg_id(const std::string& name) {
    // Currently returns 0xFFFFFFFF placeholder (dst_reg_id unused in serialization).
    // Future: map register names to compact IDs in string table for roundtrip.
    auto it = reg2id_.find(name);
    return it != reg2id_.end() ? it->second : 0xFFFFFFFF;
}

uint32_t PtxirWriter::get_string_id(const std::string& str) {
    auto it = str2id_.find(str);
    if (it != str2id_.end()) return it->second;
    uint32_t id = static_cast<uint32_t>(strings_.size());
    str2id_[str] = id;
    strings_.push_back(str);
    return id;
}

void PtxirWriter::write_header() {
    PtxirHeader hdr{};
    std::memcpy(hdr.magic, PTXIR_MAGIC, 4);
    hdr.version = PTXIR_VERSION;
    hdr.flags = 0;
    hdr.section_count = 2;
    hdr.reserved = 0;
    hdr.string_table_offset = 0;
    hdr.string_table_size = 0;
    hdr.header_size = sizeof(PtxirHeader);
    out_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
}

void PtxirWriter::write_kernel_section() {
    kernel_section_offset_ = static_cast<uint32_t>(out_.tellp());
    write_u32(out_, static_cast<uint32_t>(stmts_.size()));
    for (const auto& stmt : stmts_) {
        write_instruction(stmt);
    }
}

void PtxirWriter::write_string_table() {
    string_table_offset_ = static_cast<uint32_t>(out_.tellp());
    string_table_size_ = 0;
    write_u32(out_, static_cast<uint32_t>(strings_.size()));
    for (const auto& s : strings_) {
        uint16_t len = static_cast<uint16_t>(s.size());
        write_u16(out_, len);
        out_.write(s.data(), static_cast<std::streamsize>(s.size()));
        string_table_size_ += sizeof(len) + s.size();
    }
}

void PtxirWriter::write_instruction(const StatementContext& stmt) {
    write_u16(out_, static_cast<uint16_t>(stmt.type));
    stmt.visit([this](const auto& instr) {
        using T = std::decay_t<decltype(instr)>;
        if constexpr (std::is_same_v<T, BranchInstr>) {
            write_u32(out_, get_string_id(instr.predicate));
            write_u32(out_, get_string_id(instr.target));
            write_u8(out_, instr.predicate_negated ? 1 : 0);
            write_i32(out_, instr.reconvergence_pc);
        } else if constexpr (std::is_same_v<T, LabelInstr>) {
            write_u32(out_, get_string_id(instr.labelName));
        } else if constexpr (std::is_same_v<T, VoidInstr>) {
        } else if constexpr (std::is_same_v<T, BarrierInstr>) {
            write_i32(out_, instr.barId.value_or(-1));
        } else if constexpr (std::is_same_v<T, GenericInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u32(out_, 0xFFFFFFFF);
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else if (op.kind() == OperandKind::IMM) {
                    write_u32(out_, get_string_id(std::get<ImmOperand>(op.data).value));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, DeclarationInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.kind));
            write_u16(out_, static_cast<uint16_t>(instr.dataType));
            write_u32(out_, get_string_id(instr.name));
            write_i32(out_, instr.array_size);
        } else if constexpr (std::is_same_v<T, BarWarpSyncInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::IMM) {
                    write_u32(out_, get_string_id(std::get<ImmOperand>(op.data).value));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, PragmaInstr>) {
            write_u32(out_, get_string_id(instr.content));
        } else if constexpr (std::is_same_v<T, DollarNameInstr>) {
            write_u32(out_, get_string_id(instr.name));
        } else if constexpr (std::is_same_v<T, MembarInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
        } else if constexpr (std::is_same_v<T, FenceInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
        } else if constexpr (std::is_same_v<T, ReduxSyncInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else if (op.kind() == OperandKind::IMM) {
                    write_u32(out_, get_string_id(std::get<ImmOperand>(op.data).value));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, MbarrierInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, CallInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else if (op.kind() == OperandKind::IMM) {
                    write_u32(out_, get_string_id(std::get<ImmOperand>(op.data).value));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, PredicatePrefix>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
        } else if constexpr (std::is_same_v<T, VoteInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, ShflInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, AtomInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else if (op.kind() == OperandKind::IMM) {
                    write_u32(out_, get_string_id(std::get<ImmOperand>(op.data).value));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, WmmaInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, TextureInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, SurfaceInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, ReductionInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, PrefetchInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, CpAsyncInstr>) {
            write_u8(out_, static_cast<uint8_t>(instr.qualifiers.size()));
            for (const auto& q : instr.qualifiers) {
                write_u16(out_, static_cast<uint16_t>(q));
            }
            write_u8(out_, static_cast<uint8_t>(instr.operands.size()));
            for (const auto& op : instr.operands) {
                if (op.kind() == OperandKind::REG) {
                    const auto& reg = std::get<RegOperand>(op.data);
                    write_u32(out_, get_reg_id(reg.fullName()));
                } else {
                    write_u32(out_, 0xFFFFFFFF);
                }
            }
        } else if constexpr (std::is_same_v<T, AbiDirective>) {
        }
    });
}
