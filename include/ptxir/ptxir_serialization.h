/**
 * @file ptxir_serialization.h
 * @brief PTXIR Serialization/Deserialization API
 * @author PTX-EMU Team
 * @date 2026-05-02
 */

#ifndef PTXIR_SERIALIZATION_H
#define PTXIR_SERIALIZATION_H

// Phase 1.5c+d: use canonical ptxemu::ir::StatementContext. The old
// `struct StatementContext` elaborated-type-specifier resolved to a
// global ::StatementContext that no longer has a definition after the
// forwarding shim swap; the incomplete type caused stl_vector
// instantiation errors and undefined-symbol link failures at
// consumers. Drag canonical via include (not the shim) to avoid
// pulling the entire ptx_ir bridge into this public-ish header.

#include <ptxemu/ir/statement.h>
#include <string>
#include <vector>

std::string serialize_to_string(const std::vector<ptxemu::ir::StatementContext>& stmts);

std::vector<ptxemu::ir::StatementContext> deserialize_from_string(const std::string& data);

bool serialize_statements(const std::vector<ptxemu::ir::StatementContext>& stmts, const std::string& path);

std::vector<ptxemu::ir::StatementContext> deserialize_statements(const std::string& path);

bool generate_ptxir(const std::string& ptx_path,
                    const std::string& ptxir_path,
                    const std::string& kernel_name = "");

std::vector<ptxemu::ir::StatementContext> load_ptxir(const std::string& ptxir_path,
                                                bool apply_cfg = false);

#endif // PTXIR_SERIALIZATION_H