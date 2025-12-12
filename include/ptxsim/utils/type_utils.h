#ifndef TYPE_UTILS_H
#define TYPE_UTILS_H

#include "ptx_ir/operand_context.h"
#include <vector>
#include <functional>

class ThreadContext;

namespace TypeUtils {
    int get_bytes(std::vector<Qualifier>& qualifiers);
    bool is_float_type(std::vector<Qualifier>& qualifiers);
    Qualifier get_comparison_op(std::vector<Qualifier>& qualifiers);
    bool is_signed_type(std::vector<Qualifier>& qualifiers);
    
    template<typename T>
    bool apply_comparison(T val1, T val2, Qualifier cmpOp);
    
    template<typename T>
    void apply_binary_op(void* dst, void* src1, void* src2, 
                        std::function<T(T, T)> op) {
        *(T*)dst = op(*(T*)src1, *(T*)src2);
    }
}

#endif // TYPE_UTILS_H