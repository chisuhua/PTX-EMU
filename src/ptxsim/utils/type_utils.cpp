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

bool TypeUtils::is_signed_type(std::vector<Qualifier>& qualifiers) {
    // 判断类型是否有符号
    if (qualifiers.empty()) return false;
    
    return Signed(qualifiers.back());
}

template<typename T>
bool TypeUtils::apply_comparison(T val1, T val2, Qualifier cmpOp) {
    switch (cmpOp) {
    case Qualifier::Q_EQ:
        return val1 == val2;
    case Qualifier::Q_NE:
        return val1 != val2;
    case Qualifier::Q_LT:
        return val1 < val2;
    case Qualifier::Q_LE:
        return val1 <= val2;
    case Qualifier::Q_GT:
        return val1 > val2;
    case Qualifier::Q_GE:
        return val1 >= val2;
    default:
        // 对于NaN处理等情况，可以根据需要添加
        return false;
    }
}

// 显式实例化模板函数
template bool TypeUtils::apply_comparison<int8_t>(int8_t, int8_t, Qualifier);
template bool TypeUtils::apply_comparison<uint8_t>(uint8_t, uint8_t, Qualifier);
template bool TypeUtils::apply_comparison<int16_t>(int16_t, int16_t, Qualifier);
template bool TypeUtils::apply_comparison<uint16_t>(uint16_t, uint16_t, Qualifier);
template bool TypeUtils::apply_comparison<int32_t>(int32_t, int32_t, Qualifier);
template bool TypeUtils::apply_comparison<uint32_t>(uint32_t, uint32_t, Qualifier);
template bool TypeUtils::apply_comparison<int64_t>(int64_t, int64_t, Qualifier);
template bool TypeUtils::apply_comparison<uint64_t>(uint64_t, uint64_t, Qualifier);
template bool TypeUtils::apply_comparison<float>(float, float, Qualifier);
template bool TypeUtils::apply_comparison<double>(double, double, Qualifier);