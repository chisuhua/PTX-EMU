/**
 * @file ptxir_serialization.h
 * @brief PTXIR Serialization/Deserialization API
 * @author PTX-EMU Team
 * @date 2026-05-02
 */

#ifndef PTXIR_SERIALIZATION_H
#define PTXIR_SERIALIZATION_H

#include <string>
#include <vector>

std::string serialize_to_string(const std::vector<struct StatementContext>& stmts);

std::vector<struct StatementContext> deserialize_from_string(const std::string& data);

bool serialize_statements(const std::vector<struct StatementContext>& stmts, const std::string& path);

std::vector<struct StatementContext> deserialize_statements(const std::string& path);

#endif // PTXIR_SERIALIZATION_H
