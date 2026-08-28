#include "ptxsim/utils/type_utils.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/utils/qualifier_utils.h"

bool TypeUtils::is_float_type(const std::vector<ptxemu::ir::Qualifier> &qualifiers) {
    // 判断是否为浮点类型
    if (qualifiers.empty())
        return false;

    for (const auto &q : qualifiers) {
        if (q == ptxemu::ir::Qualifier::Q_F32 || q == ptxemu::ir::Qualifier::Q_F64 ||
            q == ptxemu::ir::Qualifier::Q_F16 || q == ptxemu::ir::Qualifier::Q_BF16)
            return true;
    }
    return false;
}

ptxemu::ir::Qualifier
TypeUtils::get_comparison_op(const std::vector<ptxemu::ir::Qualifier> &qualifiers) {
    // 查找比较操作符限定符
    for (const auto &q : qualifiers) {
        if (q >= ptxemu::ir::Qualifier::Q_EQ && q <= ptxemu::ir::Qualifier::Q_GE) {
            return q;
        }
    }
    return ptxemu::ir::Qualifier::Q_UNKNOWN; // 如果没有找到比较操作符
}

bool TypeUtils::is_signed_type(const std::vector<ptxemu::ir::Qualifier> &qualifiers) {
    // 判断类型是否有符号
    if (qualifiers.empty())
        return false;

    return Signed(qualifiers.back());
}