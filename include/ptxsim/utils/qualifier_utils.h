#ifndef QUALIFIER_UTILS_H
#define QUALIFIER_UTILS_H

#include "memory/memory_interface.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/utils/type_utils.h"
#include <vector>

int Q2bytes(Qualifier q);
bool Signed(Qualifier q);
int getBytes(std::vector<Qualifier> &q);
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
MemorySpace getAddressSpace(std::vector<Qualifier> &qualifiers);

// 解析立即数到缓冲区
void parseImmediate(const std::string& s, Qualifier q, void* out);

#endif // QUALIFIER_UTILS_H