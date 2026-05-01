// ptxir_writer.h
#ifndef PTXIR_WRITER_H
#define PTXIR_WRITER_H

#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/statement_context.h"
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

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

    void write(const std::vector<StatementContext>& statements);

private:
    void pre_pass(const std::vector<StatementContext>& statements);
    void write_header();
    void write_toc();
    void write_kernel_section();
    void write_string_table();
    void write_instruction(const StatementContext& stmt);

    uint32_t get_reg_id(const std::string& name);
    uint32_t get_string_id(const std::string& str);
    void flush_string_table();

    std::ostream& out_;

    std::vector<StatementContext> stmts_;
    std::map<std::string, uint32_t> reg2id_;
    std::map<std::string, uint32_t> str2id_;
    std::vector<std::string> strings_;
    std::vector<PtxirSectionTOC> toc_entries_;
    uint32_t kernel_section_offset_ = 0;
    uint32_t string_table_offset_ = 0;
    uint32_t string_table_size_ = 0;
};

#endif  // PTXIR_WRITER_H
