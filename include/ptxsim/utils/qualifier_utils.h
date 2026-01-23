#ifndef QUALIFIER_UTILS_H
#define QUALIFIER_UTILS_H

#include "memory/memory_interface.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/utils/type_utils.h"
#include <vector>

int Q2bytes(Qualifier q);
bool Signed(Qualifier q);

int getBytes(const std::vector<Qualifier> &q);

DTYPE getDType(std::vector<Qualifier> &q);
DTYPE getDType(Qualifier q);
void splitCvtQualifiers(const std::vector<Qualifier> &qualifiers,
                        std::vector<Qualifier> &dst_qualifiers,
                        std::vector<Qualifier> &src_qualifiers);

Qualifier getDataQualifier(const std::vector<Qualifier> &qualifiers);
Qualifier getCmpOpQualifier(const std::vector<Qualifier> &qualifiers);

void splitDstSrcQualifiers(const std::vector<Qualifier> &qualifiers,
                           std::vector<Qualifier> &dst_qualifiers,
                           std::vector<Qualifier> &src1_qualifiers,
                           std::vector<Qualifier> &src2_qualifiers);

void splitDstSrcQualifiers(const std::vector<Qualifier> &qualifiers,
                           std::vector<Qualifier> &dst_qualifiers,
                           std::vector<Qualifier> &src2_qualifiers);

// 添加获取地址空间的辅助函数
MemorySpace getAddressSpace(const std::vector<Qualifier> &qualifiers);

// 解析立即数到缓冲区
void parseImmediate(const std::string &s, Qualifier q, void *out);

bool QvecHasQ(const std::vector<Qualifier> &qvec, Qualifier q);

// 检查修饰符中是否包含.cc修饰符
bool hasCCQualifier(const std::vector<Qualifier> &qualifiers);

// 获取每个操作数的字节大小
std::vector<int>
getOperandBytes(const std::vector<Qualifier> &operand_qualifiers);

#endif // QUALIFIER_UTILS_H