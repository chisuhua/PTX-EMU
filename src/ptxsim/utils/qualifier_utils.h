#ifndef QUALIFIER_UTILS_H
#define QUALIFIER_UTILS_H

#include "ptx_ir/ptx_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/utils/type_utils.h"
#include <vector>

int Q2bytes(Qualifier q);
bool Signed(Qualifier q);
int getBytes(std::vector<Qualifier> &q);
DTYPE getDType(std::vector<Qualifier> &q);
DTYPE getDType(Qualifier q);
Qualifier getDataQualifier(const std::vector<Qualifier> &qualifier);
void splitDstSrcQualifiers(const std::vector<Qualifier> &qualifiers,
                           std::vector<Qualifier> &dst_qualifiers,
                           std::vector<Qualifier> &src_qualifiers);

#endif // QUALIFIER_UTILS_H