#include "ptxsim/utils/type_utils.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptx_ir/ptx_types.h"

int TypeUtils::get_bytes(std::vector<Qualifier>& qualifiers) {
    // 获取最后一个限定符作为类型信息
    if (!qualifiers.empty()) {
        return Q2bytes(qualifiers.back());
    }
    return 0;
}

bool TypeUtils::is_float_type(std::vector<Qualifier>& qualifiers) {
    // 判断是否为浮点类型
    if (qualifiers.empty()) return false;
    
    Qualifier type = qualifiers.back();
    return (type == Qualifier::Q_F32 || type == Qualifier::Q_F64);
}

Qualifier TypeUtils::get_comparison_op(std::vector<Qualifier>& qualifiers) {
    // 获取比较操作符，通常在限定符的第一个位置
    if (!qualifiers.empty()) {
        return qualifiers.front();
    }
    return Qualifier::S_UNKNOWN;
}