#ifndef QUALIFIER_UTILS_H
#define QUALIFIER_UTILS_H

#include "memory/memory_interface.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/utils/type_utils.h"
#include <vector>

// Q2bytes removed from header (Phase 1.5c+d): the canonical
// ptxemu::ir::Q2bytes is selected by ADL on the Qualifier argument.
// Keeping a duplicate global declaration caused an ambiguous call
// overload between the canonical strict implementation (asserts on
// unknown qualifiers) and the ptxsim legacy fallback (returns 0 for
// non-data qualifiers like Q_LT). The fallback now lives only in
// src/ptxsim/utils/qualifier_utils.cpp as ptxsim::Q2bytes, reached
// by qualifying the call site when needed.
namespace ptxsim {
int Q2bytes(ptxemu::ir::Qualifier q);
}
bool Signed(ptxemu::ir::Qualifier q);

int getBytes(const std::vector<ptxemu::ir::Qualifier> &q);

DTYPE getDType(std::vector<ptxemu::ir::Qualifier> &q);
DTYPE getDType(ptxemu::ir::Qualifier q);
void splitCvtQualifiers(const std::vector<ptxemu::ir::Qualifier> &qualifiers,
                        std::vector<ptxemu::ir::Qualifier> &dst_qualifiers,
                        std::vector<ptxemu::ir::Qualifier> &src_qualifiers);

ptxemu::ir::Qualifier getDataQualifier(const std::vector<ptxemu::ir::Qualifier> &qualifiers);
ptxemu::ir::Qualifier getCmpOpQualifier(const std::vector<ptxemu::ir::Qualifier> &qualifiers);

void splitDstSrcQualifiers(const std::vector<ptxemu::ir::Qualifier> &qualifiers,
                           std::vector<ptxemu::ir::Qualifier> &dst_qualifiers,
                           std::vector<ptxemu::ir::Qualifier> &src1_qualifiers,
                           std::vector<ptxemu::ir::Qualifier> &src2_qualifiers);

void splitDstSrcQualifiers(const std::vector<ptxemu::ir::Qualifier> &qualifiers,
                           std::vector<ptxemu::ir::Qualifier> &dst_qualifiers,
                           std::vector<ptxemu::ir::Qualifier> &src2_qualifiers);

// 添加获取地址空间的辅助函数
MemorySpace getAddressSpace(const std::vector<ptxemu::ir::Qualifier> &qualifiers);

// 解析立即数到缓冲区
void parseImmediate(const std::string &s, ptxemu::ir::Qualifier q, void *out);

bool QvecHasQ(const std::vector<ptxemu::ir::Qualifier> &qvec, ptxemu::ir::Qualifier q);

// 检查修饰符中是否包含.cc修饰符
bool hasCCQualifier(const std::vector<ptxemu::ir::Qualifier> &qualifiers);

// 获取每个操作数的字节大小
std::vector<int>
getOperandBytes(const std::vector<ptxemu::ir::Qualifier> &operand_qualifiers);

#endif // QUALIFIER_UTILS_H