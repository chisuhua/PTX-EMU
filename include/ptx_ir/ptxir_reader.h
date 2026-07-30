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

class PtxirReader {
public:
    explicit PtxirReader(std::istream& in);

    std::vector<StatementContext> read();

private:
    void read_header();
    void read_string_table();
    void read_string_table_v2();
    void read_regdecl_section();
    std::vector<StatementContext> read_kernel_section();
    StatementContext read_instruction();
    std::vector<StatementContext> read_legacy_v1();
    std::vector<StatementContext> read_v2();

    std::istream& in_;
    std::vector<std::string> string_table_;
    uint32_t statement_count_ = 0;
    uint16_t version_ = 0;
    PtxirHeader header_{};
};

#endif  // PTXIR_READER_H
