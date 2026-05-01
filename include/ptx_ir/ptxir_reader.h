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
    std::vector<StatementContext> read_kernel_section();
    StatementContext read_instruction();

    std::istream& in_;
    std::vector<std::string> string_table_;
    uint32_t statement_count_ = 0;
};

#endif  // PTXIR_READER_H
