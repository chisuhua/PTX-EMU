#include "ptxir/ptxir_serialization.h"
#include "ptx_ir/ptxir_writer.h"
#include "ptx_ir/ptxir_reader.h"
#include <fstream>
#include <sstream>

std::string serialize_to_string(const std::vector<struct StatementContext>& stmts) {
    std::ostringstream oss(std::ios::binary);
    ::PtxirWriter writer(oss);
    writer.write(stmts);
    return oss.str();
}

std::vector<struct StatementContext> deserialize_from_string(const std::string& data) {
    std::istringstream iss(data, std::ios::binary);
    ::PtxirReader reader(iss);
    return reader.read();
}

bool serialize_statements(const std::vector<struct StatementContext>& stmts, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    ::PtxirWriter writer(out);
    writer.write(stmts);
    return out.good();
}

std::vector<struct StatementContext> deserialize_statements(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("PTXIR file not found: " + path);
    }
    ::PtxirReader reader(in);
    return reader.read();
}
