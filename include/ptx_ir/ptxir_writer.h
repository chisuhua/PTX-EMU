// ptxir_writer.h
#ifndef PTXIR_WRITER_H
#define PTXIR_WRITER_H

#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/statement_context.h"
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <vector>

void write_manifest_section(std::vector<uint8_t>& buf, const ManifestSection& m);

// ============================================================================
// PtxirWriter - Serialize StatementContext vector → PTXIR binary format
//
// Usage:
//   std::ofstream out("kernel.ptxir", std::ios::binary);
//   PtxirWriter writer(out);
//   writer.write(statements);
// ============================================================================
class PtxirWriter {
public:
    explicit PtxirWriter(std::ostream& out);

    void write(const std::vector<ptxemu::ir::StatementContext>& statements);
    void set_manifest(const ManifestSection& manifest);

private:
    void pre_pass(const std::vector<ptxemu::ir::StatementContext>& statements);
    void write_header();
    void write_toc_entries();
    void write_regdecl_section();
    void write_kernel_section();
    void write_string_table();
    void backfill_header_offsets();
    void write_instruction(const ptxemu::ir::StatementContext& stmt);

    void write_qualifiers(const std::vector<ptxemu::ir::Qualifier>& qualifiers);
    void write_operands(const std::vector<ptxemu::ir::OperandContext>& operands,
                        bool with_imm);
    void write_operand(const ptxemu::ir::OperandContext& op, bool with_imm);

    void write_branch(const BranchInstr& instr);
    void write_label(const LabelInstr& instr);
    void write_void(const VoidInstr&);
    void write_barrier(const BarrierInstr& instr);
    void write_generic(const GenericInstr& instr);
    void write_declaration(const DeclarationInstr& instr);
    void write_bar_warp_sync(const BarWarpSyncInstr& instr);
    void write_pragma(const PragmaInstr& instr);
    void write_dollar_name(const DollarNameInstr& instr);
    void write_membar(const MembarInstr& instr);
    void write_fence(const FenceInstr& instr);
    void write_redux_sync(const ReduxSyncInstr& instr);
    void write_mbarrier(const MbarrierInstr& instr);
    void write_call(const CallInstr& instr);
    void write_predicate_prefix(const PredicatePrefix& instr);
    void write_vote(const VoteInstr& instr);
    void write_shfl(const ShflInstr& instr);
    void write_atom(const AtomInstr& instr);
    void write_texture(const TextureInstr& instr);
    void write_surface(const SurfaceInstr& instr);
    void write_reduction(const ReductionInstr& instr);
    void write_prefetch(const PrefetchInstr& instr);
    void write_cp_async(const CpAsyncInstr& instr);
    void write_tcgen05(const ptxemu::ir::Tcgen05Instr& instr);
    void write_abi_directive(const AbiDirective&);

    uint32_t get_reg_id(const std::string& name);
    uint32_t get_string_id(const std::string& str);
    void flush_string_table();

    std::ostream& out_;

    std::vector<ptxemu::ir::StatementContext> stmts_;
    std::map<std::string, uint32_t> reg2id_;
    std::map<std::string, uint32_t> str2id_;
    std::vector<std::string> strings_;
    std::vector<PtxirSectionTOC> toc_entries_;
    std::optional<ManifestSection> manifest_;
    std::vector<uint8_t> manifest_buffer_;
    uint32_t kernel_section_offset_ = 0;
    uint32_t string_table_offset_ = 0;
    uint32_t string_table_size_ = 0;
};

#endif  // PTXIR_WRITER_H
