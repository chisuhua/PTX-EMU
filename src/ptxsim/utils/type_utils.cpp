#include "ptxsim/utils/type_utils.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/utils/qualifier_utils.h"

bool TypeUtils::is_float_type(const std::vector<Qualifier> &qualifiers) {
    // 判断是否为浮点类型
    if (qualifiers.empty())
        return false;

    Qualifier type = qualifiers.back();
    return (type == Qualifier::Q_F32 || type == Qualifier::Q_F64);
}

Qualifier
TypeUtils::get_comparison_op(const std::vector<Qualifier> &qualifiers) {
    // 查找比较操作符限定符
    for (const auto &q : qualifiers) {
        if (q >= Qualifier::Q_EQ && q <= Qualifier::Q_GE) {
            return q;
        }
    }
    return Qualifier::S_UNKNOWN; // 如果没有找到比较操作符
}

bool TypeUtils::is_signed_type(const std::vector<Qualifier> &qualifiers) {
    // 判断类型是否有符号
    if (qualifiers.empty())
        return false;

    return Signed(qualifiers.back());
}