// ptxir_reader.h
#ifndef PTXIR_READER_H
#define PTXIR_READER_H

#include "ptx_ir/ptxir_format.h"
#include "ptx_ir/statement_context.h"
#include <iosfwd>
#include <istream>
#include <map>
#include <string>
#include <vector>

ManifestSection read_manifest_section(const std::vector<uint8_t>& buf);

class PtxirReader {
public:
    explicit PtxirReader(std::istream& in);

    std::vector<ptxemu::ir::StatementContext> read();
    const ManifestSection& get_manifest() const;

private:
    void read_header();
    void read_string_table();
    void read_string_table_v2();
    void read_regdecl_section();
    std::vector<ptxemu::ir::StatementContext> read_kernel_section();
    ptxemu::ir::StatementContext read_instruction();
    std::vector<ptxemu::ir::StatementContext> read_legacy_v1();
    std::vector<ptxemu::ir::StatementContext> read_v2();

    std::istream& in_;
    std::vector<std::string> string_table_;
    uint32_t statement_count_ = 0;
    uint16_t version_ = 0;
    PtxirHeader header_{};
    ManifestSection manifest_;
};

#endif  // PTXIR_READER_H
