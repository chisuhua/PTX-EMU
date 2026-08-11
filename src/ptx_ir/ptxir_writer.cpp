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

void write_manifest_section(std::vector<uint8_t>& buf, const ManifestSection& m) {
    // Auto-sync: if kernel_name empty but kernels non-empty,
    // set kernel_name = kernels[0].name (backward-compat field)
    ManifestSection normalized = m;
    if (normalized.kernel_name.empty() && !normalized.kernels.empty()) {
        normalized.kernel_name = normalized.kernels[0].name;
    }

    // cubin_hash (32 bytes, zero-padded)
    buf.insert(buf.end(), normalized.cubin_hash.begin(), normalized.cubin_hash.end());
    if (normalized.cubin_hash.size() < 32) {
        buf.insert(buf.end(), 32 - normalized.cubin_hash.size(), 0);
    }

    // kernel_name (NUL-terminated, backward-compat field)
    buf.insert(buf.end(), normalized.kernel_name.begin(), normalized.kernel_name.end());
    buf.push_back(0);

    // ptx_address_size
    buf.push_back(normalized.ptx_address_size);

    // params (backward-compat)
    uint16_t param_count = static_cast<uint16_t>(normalized.params.size());
    buf.push_back(static_cast<uint8_t>(param_count & 0xFF));
    buf.push_back(static_cast<uint8_t>((param_count >> 8) & 0xFF));
    for (const auto& p : normalized.params) {
        buf.insert(buf.end(), p.name.begin(), p.name.end());
        buf.push_back(0);
        buf.push_back(static_cast<uint8_t>(p.size & 0xFF));
        buf.push_back(static_cast<uint8_t>((p.size >> 8) & 0xFF));
        buf.push_back(static_cast<uint8_t>(p.kind));
    }

    // v2 kernels vector (uint16 count, extend-only)
    uint16_t kernel_count = static_cast<uint16_t>(normalized.kernels.size());
    buf.push_back(static_cast<uint8_t>(kernel_count & 0xFF));
    buf.push_back(static_cast<uint8_t>((kernel_count >> 8) & 0xFF));
    for (const auto& ke : normalized.kernels) {
        buf.insert(buf.end(), ke.name.begin(), ke.name.end());
        buf.push_back(0);
        buf.push_back(static_cast<uint8_t>(ke.arg_count & 0xFF));
        buf.push_back(static_cast<uint8_t>((ke.arg_count >> 8) & 0xFF));
        buf.push_back(static_cast<uint8_t>((ke.arg_count >> 16) & 0xFF));
        buf.push_back(static_cast<uint8_t>((ke.arg_count >> 24) & 0xFF));
        buf.push_back(static_cast<uint8_t>(ke.arg_byte_size & 0xFF));
        buf.push_back(static_cast<uint8_t>((ke.arg_byte_size >> 8) & 0xFF));
        buf.push_back(static_cast<uint8_t>((ke.arg_byte_size >> 16) & 0xFF));
        buf.push_back(static_cast<uint8_t>((ke.arg_byte_size >> 24) & 0xFF));
    }
}

PtxirWriter::PtxirWriter(std::ostream& out) : out_(out) {}

void PtxirWriter::write(const std::vector<StatementContext>& statements) {
    stmts_ = statements;
    pre_pass(statements);
    write_header();

    toc_entries_.clear();
    toc_entries_.push_back({static_cast<uint8_t>(PtxirSectionType::REGDECL), 0, 0});
    toc_entries_.push_back({static_cast<uint8_t>(PtxirSectionType::KERNEL), 0, 0});
    if (manifest_.has_value()) {
        toc_entries_.push_back({static_cast<uint8_t>(PtxirSectionType::MANIFEST), 0, 0});
    }
    toc_entries_.push_back({static_cast<uint8_t>(PtxirSectionType::STRING_TABLE), 0, 0});

    toc_entries_[0].offset = static_cast<uint32_t>(out_.tellp());
    write_regdecl_section();

    toc_entries_[1].offset = static_cast<uint32_t>(out_.tellp());
    write_kernel_section();

    if (manifest_.has_value()) {
        toc_entries_[2].offset = static_cast<uint32_t>(out_.tellp());
        write_manifest_section(manifest_buffer_, manifest_.value());
        out_.write(reinterpret_cast<const char*>(manifest_buffer_.data()),
                   static_cast<std::streamsize>(manifest_buffer_.size()));
    }

    toc_entries_.back().offset = static_cast<uint32_t>(out_.tellp());
    write_string_table();

    write_toc_entries();
    backfill_header_offsets();
}

void PtxirWriter::set_manifest(const ManifestSection& manifest) { manifest_ = manifest; }

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
    hdr.section_count = manifest_.has_value() ? 4 : 3;
    hdr.reserved = 0;
    hdr.string_table_offset = 0;
    hdr.string_table_size = 0;
    hdr.header_size = sizeof(PtxirHeader);
    out_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    for (int i = 0; i < hdr.section_count; i++) {
        uint8_t zero[6] = {};
        out_.write(reinterpret_cast<const char*>(zero), 6);
    }
}

void PtxirWriter::write_toc_entries() {
    auto end_pos = out_.tellp();
    out_.seekp(sizeof(PtxirHeader), std::ios::beg);
    for (const auto& entry : toc_entries_) {
        write_u8(out_, entry.type);
        write_u8(out_, entry.reserved);
        write_u32(out_, entry.offset);
    }
    out_.seekp(end_pos, std::ios::beg);
}

void PtxirWriter::write_regdecl_section() {
    write_u32(out_, static_cast<uint32_t>(reg2id_.size()));
    for (const auto& [name, id] : reg2id_) {
        write_u32(out_, get_string_id(name));
    }
}

void PtxirWriter::backfill_header_offsets() {
    auto saved_pos = out_.tellp();
    out_.seekp(12, std::ios::beg);
    write_u32(out_, string_table_offset_);
    write_u32(out_, string_table_size_);
    out_.seekp(saved_pos, std::ios::beg);
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
        if constexpr (std::is_same_v<T, BranchInstr>) { write_branch(instr); }
        else if constexpr (std::is_same_v<T, LabelInstr>) { write_label(instr); }
        else if constexpr (std::is_same_v<T, VoidInstr>) { write_void(instr); }
        else if constexpr (std::is_same_v<T, BarrierInstr>) { write_barrier(instr); }
        else if constexpr (std::is_same_v<T, GenericInstr>) { write_generic(instr); }
        else if constexpr (std::is_same_v<T, DeclarationInstr>) { write_declaration(instr); }
        else if constexpr (std::is_same_v<T, BarWarpSyncInstr>) { write_bar_warp_sync(instr); }
        else if constexpr (std::is_same_v<T, PragmaInstr>) { write_pragma(instr); }
        else if constexpr (std::is_same_v<T, DollarNameInstr>) { write_dollar_name(instr); }
        else if constexpr (std::is_same_v<T, MembarInstr>) { write_membar(instr); }
        else if constexpr (std::is_same_v<T, FenceInstr>) { write_fence(instr); }
        else if constexpr (std::is_same_v<T, ReduxSyncInstr>) { write_redux_sync(instr); }
        else if constexpr (std::is_same_v<T, MbarrierInstr>) { write_mbarrier(instr); }
        else if constexpr (std::is_same_v<T, CallInstr>) { write_call(instr); }
        else if constexpr (std::is_same_v<T, PredicatePrefix>) { write_predicate_prefix(instr); }
        else if constexpr (std::is_same_v<T, VoteInstr>) { write_vote(instr); }
        else if constexpr (std::is_same_v<T, ShflInstr>) { write_shfl(instr); }
        else if constexpr (std::is_same_v<T, AtomInstr>) { write_atom(instr); }
        else if constexpr (std::is_same_v<T, TextureInstr>) { write_texture(instr); }
        else if constexpr (std::is_same_v<T, SurfaceInstr>) { write_surface(instr); }
        else if constexpr (std::is_same_v<T, ReductionInstr>) { write_reduction(instr); }
        else if constexpr (std::is_same_v<T, PrefetchInstr>) { write_prefetch(instr); }
        else if constexpr (std::is_same_v<T, CpAsyncInstr>) { write_cp_async(instr); }
        else if constexpr (std::is_same_v<T, Tcgen05Instr>) { write_tcgen05(instr); }
        else if constexpr (std::is_same_v<T, AbiDirective>) { write_abi_directive(instr); }
    });
}

void PtxirWriter::write_qualifiers(const std::vector<Qualifier>& qualifiers) {
    write_u8(out_, static_cast<uint8_t>(qualifiers.size()));
    for (const auto& q : qualifiers) {
        write_u16(out_, static_cast<uint16_t>(q));
    }
}

void PtxirWriter::write_operand(const OperandContext& op, bool with_imm) {
    if (op.kind() == OperandKind::REG) {
        const auto& reg = std::get<RegOperand>(op.data);
        write_u32(out_, get_reg_id(reg.fullName()));
    } else if (with_imm && op.kind() == OperandKind::IMM) {
        write_u32(out_, get_string_id(std::get<ImmOperand>(op.data).value));
    } else {
        write_u32(out_, 0xFFFFFFFF);
    }
}

void PtxirWriter::write_operands(const std::vector<OperandContext>& operands,
                                 bool with_imm) {
    write_u8(out_, static_cast<uint8_t>(operands.size()));
    for (const auto& op : operands) {
        write_operand(op, with_imm);
    }
}

void PtxirWriter::write_branch(const BranchInstr& instr) {
    write_u32(out_, get_string_id(instr.predicate));
    write_u32(out_, get_string_id(instr.target));
    write_u8(out_, instr.predicate_negated ? 1 : 0);
    write_i32(out_, instr.reconvergence_pc);
}

void PtxirWriter::write_label(const LabelInstr& instr) {
    write_u32(out_, get_string_id(instr.labelName));
}

void PtxirWriter::write_void(const VoidInstr&) {}

void PtxirWriter::write_barrier(const BarrierInstr& instr) {
    write_i32(out_, instr.barId.value_or(-1));
    write_i32(out_, instr.reconvergence_pc);
}

void PtxirWriter::write_generic(const GenericInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_u32(out_, 0xFFFFFFFF);
    write_operands(instr.operands, true);
}

void PtxirWriter::write_declaration(const DeclarationInstr& instr) {
    write_u8(out_, static_cast<uint8_t>(instr.kind));
    write_u16(out_, static_cast<uint16_t>(instr.dataType));
    write_u32(out_, get_string_id(instr.name));
    write_i32(out_, instr.array_size);
}

void PtxirWriter::write_bar_warp_sync(const BarWarpSyncInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, true);
}

void PtxirWriter::write_pragma(const PragmaInstr& instr) {
    write_u32(out_, get_string_id(instr.content));
}

void PtxirWriter::write_dollar_name(const DollarNameInstr& instr) {
    write_u32(out_, get_string_id(instr.name));
}

void PtxirWriter::write_membar(const MembarInstr& instr) {
    write_qualifiers(instr.qualifiers);
}

void PtxirWriter::write_fence(const FenceInstr& instr) {
    write_qualifiers(instr.qualifiers);
}

void PtxirWriter::write_redux_sync(const ReduxSyncInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, true);
}

void PtxirWriter::write_mbarrier(const MbarrierInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, false);
}

void PtxirWriter::write_call(const CallInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, true);
}

void PtxirWriter::write_tcgen05(const Tcgen05Instr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, true);
}

void PtxirWriter::write_predicate_prefix(const PredicatePrefix& instr) {
    write_qualifiers(instr.qualifiers);
}

void PtxirWriter::write_vote(const VoteInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, false);
}

void PtxirWriter::write_shfl(const ShflInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, false);
}

void PtxirWriter::write_atom(const AtomInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, true);
}

void PtxirWriter::write_texture(const TextureInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, false);
}

void PtxirWriter::write_surface(const SurfaceInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, false);
}

void PtxirWriter::write_reduction(const ReductionInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, false);
}

void PtxirWriter::write_prefetch(const PrefetchInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, false);
}

void PtxirWriter::write_cp_async(const CpAsyncInstr& instr) {
    write_qualifiers(instr.qualifiers);
    write_operands(instr.operands, false);
}

void PtxirWriter::write_abi_directive(const AbiDirective&) {}
