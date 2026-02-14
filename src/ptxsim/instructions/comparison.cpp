#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

// 半精度浮点数转换函数声明
inline float f16_to_f32(uint16_t h);
inline uint16_t f32_to_f16(float f);

// 通用模板函数，用于处理比较操作
template<typename OpFunc>
void process_comparison(void *dst, void *src1, void *src2, int bytes, 
                        bool is_float, bool is_signed, OpFunc op) {
    if (is_float) {
        // 浮点比较
        switch (bytes) {
        case 2: {
            // 需要 f16 支持（简化：转 f32 计算）
            uint16_t h1, h2;
            std::memcpy(&h1, src1, 2);
            std::memcpy(&h2, src2, 2);

            float f1 = f16_to_f32(h1);
            float f2 = f16_to_f32(h2);
            bool result = op(f1, f2);
            *(uint8_t *)dst = result ? 1 : 0;
            break;
        }
        case 4: {
            float a, b;
            std::memcpy(&a, src1, 4);
            std::memcpy(&b, src2, 4);
            bool result = op(a, b);
            *(uint8_t *)dst = result ? 1 : 0;
            break;
        }
        case 8: {
            double a, b;
            std::memcpy(&a, src1, 8);
            std::memcpy(&b, src2, 8);
            bool result = op(a, b);
            *(uint8_t *)dst = result ? 1 : 0;
            break;
        }
        default:
            assert(0 && "Unsupported data size for floating point comparison");
        }
    } else {
        // 整数比较
        if (is_signed) {
            // 有符号整数
            switch (bytes) {
            case 1: {
                int8_t a, b;
                std::memcpy(&a, src1, 1);
                std::memcpy(&b, src2, 1);
                bool result = op(a, b);
                *(uint8_t *)dst = result ? 1 : 0;
                break;
            }
            case 2: {
                int16_t a, b;
                std::memcpy(&a, src1, 2);
                std::memcpy(&b, src2, 2);
                bool result = op(a, b);
                *(uint8_t *)dst = result ? 1 : 0;
                break;
            }
            case 4: {
                int32_t a, b;
                std::memcpy(&a, src1, 4);
                std::memcpy(&b, src2, 4);
                bool result = op(a, b);
                *(uint8_t *)dst = result ? 1 : 0;
                break;
            }
            case 8: {
                int64_t a, b;
                std::memcpy(&a, src1, 8);
                std::memcpy(&b, src2, 8);
                bool result = op(a, b);
                *(uint8_t *)dst = result ? 1 : 0;
                break;
            }
            default:
                assert(0 && "Unsupported data size for signed integer comparison");
            }
        } else {
            // 无符号整数
            switch (bytes) {
            case 1: {
                uint8_t a, b;
                std::memcpy(&a, src1, 1);
                std::memcpy(&b, src2, 1);
                bool result = op(a, b);
                *(uint8_t *)dst = result ? 1 : 0;
                break;
            }
            case 2: {
                uint16_t a, b;
                std::memcpy(&a, src1, 2);
                std::memcpy(&b, src2, 2);
                bool result = op(a, b);
                *(uint8_t *)dst = result ? 1 : 0;
                break;
            }
            case 4: {
                uint32_t a, b;
                std::memcpy(&a, src1, 4);
                std::memcpy(&b, src2, 4);
                bool result = op(a, b);
                *(uint8_t *)dst = result ? 1 : 0;
                break;
            }
            case 8: {
                uint64_t a, b;
                std::memcpy(&a, src1, 8);
                std::memcpy(&b, src2, 8);
                bool result = op(a, b);
                *(uint8_t *)dst = result ? 1 : 0;
                break;
            }
            default:
                assert(0 && "Unsupported data size for unsigned integer comparison");
            }
        }
    }
}

void SetpHandler::processOperation(ThreadContext *context, void *op[3],
                             const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    Qualifier cmpOp = getCmpOpQualifier(qualifiers);
    Qualifier dtype = getDataQualifier(qualifiers);
    uint8_t result;

    SET_P_COMPARE(cmpOp, dtype, &result, src1, src2);

    *static_cast<uint8_t *>(dst) = result;
}

void SelpHandler::processOperation(ThreadContext *context, void *op[4],
                             const std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src1 = op[1];
    void *src2 = op[2];
    void *pred = op[3];
    int bytes = getBytes(qualifiers);

    // 根据用户建议的实现方式
    const void *selected = *(uint8_t *)pred ? src1 : src2;
    std::memcpy(dst, selected, bytes);
}
