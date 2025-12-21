#include "ptxsim/instruction_handlers_decl.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void SQRT::process_operation(ThreadContext *context, void *op[2],
                             std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    // 执行SQRT操作
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    if (is_float) {
        if (bytes == 4) {
            *(float *)dst = std::sqrt(*(float *)src);
        } else if (bytes == 8) {
            *(double *)dst = std::sqrt(*(double *)src);
        }
    } else {
        // 整数开方
        switch (bytes) {
        case 1:
            *(uint8_t *)dst = (uint8_t)std::sqrt(*(uint8_t *)src);
            break;
        case 2:
            *(uint16_t *)dst = (uint16_t)std::sqrt(*(uint16_t *)src);
            break;
        case 4:
            *(uint32_t *)dst = (uint32_t)std::sqrt(*(uint32_t *)src);
            break;
        case 8:
            *(uint64_t *)dst = (uint64_t)std::sqrt(*(uint64_t *)src);
            break;
        default:
            assert(0 && "Unsupported data size for SQRT srceration");
        }
    }
}

void SIN::process_operation(ThreadContext *context, void *op[2],
                            std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    if (is_float) {
        if (bytes == 4) {
            *(float *)dst = std::sin(*(float *)src);
        } else if (bytes == 8) {
            *(double *)dst = std::sin(*(double *)src);
        }
    } else {
        // 整数正弦
        switch (bytes) {
        case 1:
            *(uint8_t *)dst = (uint8_t)std::sin(*(uint8_t *)src);
            break;
        case 2:
            *(uint16_t *)dst = (uint16_t)std::sin(*(uint16_t *)src);
            break;
        case 4:
            *(uint32_t *)dst = (uint32_t)std::sin(*(uint32_t *)src);
            break;
        case 8:
            *(uint64_t *)dst = (uint64_t)std::sin(*(uint64_t *)src);
            break;
        default:
            assert(0 && "Unsupported data size for SIN srceration");
        }
    }
}

void COS::process_operation(ThreadContext *context, void *op[2],
                            std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    if (is_float) {
        if (bytes == 4) {
            *(float *)dst = std::cos(*(float *)src);
        } else if (bytes == 8) {
            *(double *)dst = std::cos(*(double *)src);
        }
    } else {
        // 整数余弦
        switch (bytes) {
        case 1:
            *(uint8_t *)dst = (uint8_t)std::cos(*(uint8_t *)src);
            break;
        case 2:
            *(uint16_t *)dst = (uint16_t)std::cos(*(uint16_t *)src);
            break;
        case 4:
            *(uint32_t *)dst = (uint32_t)std::cos(*(uint32_t *)src);
            break;
        case 8:
            *(uint64_t *)dst = (uint64_t)std::cos(*(uint64_t *)src);
            break;
        default:
            assert(0 && "Unsupported data size for COS srceration");
        }
    }
}

void RCP::process_operation(ThreadContext *context, void *op[2],
                            std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
    // 获取数据类型信息
    int bytes = TypeUtils::get_bytes(qualifiers);
    bool is_float = TypeUtils::is_float_type(qualifiers);

    // RCP指令只支持浮点类型
    assert(is_float && "RCP instruction only supports floating point types");

    // 根据数据类型执行倒数操作
    switch (bytes) {
    case 4: {
        float val = *(float *)src;
        *(float *)dst = 1.0f / val;
        break;
    }
    case 8: {
        double val = *(double *)src;
        *(double *)dst = 1.0 / val;
        break;
    }
    default:
        assert(0 && "Unsupported data size for RCP instruction");
    }
}

// TODO
void LG2::process_operation(ThreadContext *context, void *op[2],
                            std::vector<Qualifier> &qualifiers) {}

void EX2::process_operation(ThreadContext *context, void *op[2],
                            std::vector<Qualifier> &qualifiers) {}

void RSQRT::process_operation(ThreadContext *context, void *op[2],
                              std::vector<Qualifier> &qualifiers) {}