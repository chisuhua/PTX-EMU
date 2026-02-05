#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <algorithm>
#include <cmath>
#include <limits>

// 自定义half到float的转换函数
inline float half_to_float(uint16_t h) {
    uint32_t sign = ((h >> 15) & 0x1);
    uint32_t exp = ((h >> 10) & 0x1f);
    uint32_t mantissa = (h & 0x3ff);
    uint32_t f;

    if (exp == 0) {
        if (mantissa == 0) {
            // ±0
            f = sign << 31;
        } else {
            // Subnormal numbers
            exp = 127 - 15;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3ff;
            f = (sign << 31) | ((exp + 127) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        if (mantissa == 0) {
            // ±infinity
            f = (sign << 31) | (0xFF << 23);
        } else {
            // NaN
            f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        }
    } else {
        // Normalized number
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }

    return *reinterpret_cast<float *>(&f);
}

// 自定义float到half的转换函数
inline uint16_t float_to_half(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t *>(&f);
    uint16_t sign = (bits >> 16) & 0x8000;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;

    uint16_t result;

    if (exp == 0) {
        // Zero or subnormal
        if (mantissa == 0) {
            result = sign; // +0 or -0
        } else {
            // Float subnormal -> half might be normal or subnormal
            // Need to normalize the mantissa and calculate the new exponent
            int shift = 0;
            while ((mantissa & 0x800000) == 0) {
                mantissa <<= 1;
                shift++;
            }
            exp = 127 - shift;    // original exp was 0, so real exp is -126
            exp = exp - 127 + 15; // Convert to half exponent
            if (exp <= 0) {
                // Result is subnormal in half
                mantissa = (mantissa & 0x7FFFFF) >> 13;
                if (exp == 0) {
                    // Check if we need to shift right based on exponent
                    // difference
                    mantissa |= 0x400; // Add the implicit bit
                    mantissa >>= 1;
                } else {
                    mantissa >>= (1 - exp);
                }
                result = sign | mantissa;
            } else {
                // Normal half number
                mantissa >>= 13;
                result = sign | (exp << 10) | (mantissa & 0x3FF);
            }
        }
    } else if (exp == 0xFF) {
        // infinity or NaN
        result = sign | (0x1F << 10) | (mantissa != 0 ? 0x200 : 0);
    } else {
        // Normal float number
        exp = exp - 127 + 15; // Convert to half exponent
        if (exp >= 0x1F) {
            // Overflow - infinity
            result = sign | (0x1F << 10);
        } else if (exp <= 0) {
            // Underflow - subnormal or zero
            if (exp <= -10) {
                // Rounds to zero
                result = sign;
            } else {
                // Convert to subnormal
                mantissa = (mantissa | 0x800000) >>
                           (12 - exp); // Add implicit bit and shift
                result = sign | (mantissa >> 13);
            }
        } else {
            // Normal half number
            result = sign | (exp << 10) | (mantissa >> 13);
        }
    }

    return result;
}

void CVT_Handler::processOperation(ThreadContext *context, void **operands,
                                   const std::vector<Qualifier> &qualifiers) {
    void *dst = operands[0];
    void *src = operands[1];
    std::vector<Qualifier> dst_qualifiers, src_qualifiers;
    splitDstSrcQualifiers(qualifiers, dst_qualifiers, src_qualifiers);

    // 使用TypeUtils函数获取目标和源的字节大小以及是否为浮点类型
    int dst_bytes = getBytes(dst_qualifiers);
    int src_bytes = getBytes(src_qualifiers);
    bool dst_is_float = TypeUtils::is_float_type(dst_qualifiers);
    bool src_is_float = TypeUtils::is_float_type(src_qualifiers);

    // 检查是否是half类型（16位浮点数）
    bool dst_is_half = false, src_is_half = false;
    for (const auto &q : dst_qualifiers) {
        if (q == Qualifier::Q_F16) {
            dst_is_half = true;
            dst_is_float = true; // 将half视为浮点类型
            dst_bytes = 2;       // half类型是16位（2字节）
            break;
        }
    }
    for (const auto &q : src_qualifiers) {
        if (q == Qualifier::Q_F16) {
            src_is_half = true;
            src_is_float = true; // 将half视为浮点类型
            src_bytes = 2;       // half类型是16位（2字节）
            break;
        }
    }

    // 如果没有正确识别出类型，使用默认方法
    if (dst_bytes == 0) {
        dst_bytes = getBytes(qualifiers);
    }

    if (src_bytes == 0 && !src_is_half) {
        src_bytes = getBytes(qualifiers);
    }

    // 确保源字节大小至少为1
    if (src_bytes == 0 && !src_is_half) {
        src_bytes = 1;
    }

    bool has_sat = QvecHasQ(qualifiers, Qualifier::Q_SAT);
    // 检查是否有rn(round to nearest even)修饰符 - 用于浮点转换
    bool has_rn = QvecHasQ(qualifiers, Qualifier::Q_RN);
    // 检查是否有rni(round to nearest integer)修饰符 - 用于整型转换
    bool has_rni = QvecHasQ(qualifiers, Qualifier::Q_RNI);
    // 检查是否有rz(round toward zero)修饰符 - 用于浮点转换
    bool has_rz = QvecHasQ(qualifiers, Qualifier::Q_RZ);
    // 检查是否有rzi(round toward zero integer)修饰符 - 用于整型转换
    bool has_rzi = QvecHasQ(qualifiers, Qualifier::Q_RZI);
    // 检查是否有rm(round toward negative infinity)修饰符 - 用于浮点转换
    bool has_rm = QvecHasQ(qualifiers, Qualifier::Q_RM);
    // 检查是否有rmi(round toward negative infinity integer)修饰符 -
    // 用于整型转换
    bool has_rmi = QvecHasQ(qualifiers, Qualifier::Q_RMI);
    // 检查是否有rp(round toward positive infinity)修饰符 - 用于浮点转换
    bool has_rp = QvecHasQ(qualifiers, Qualifier::Q_RP);
    // 检查是否有rpi(round toward positive infinity integer)修饰符 -
    // 用于整型转换
    bool has_rpi = QvecHasQ(qualifiers, Qualifier::Q_RPI);
    // 检查是否有rna(round to nearest, ties away from zero)修饰符
    bool has_rna = QvecHasQ(qualifiers, Qualifier::Q_RNA);
    // 检查是否有rs(stochastic rounding)修饰符
    bool has_rs = QvecHasQ(qualifiers, Qualifier::Q_RS);

    // 检查目标和源的符号类型
    bool dst_is_signed = TypeUtils::is_signed_type(dst_qualifiers);
    bool src_is_signed = TypeUtils::is_signed_type(src_qualifiers);

    // 根据规范，禁止同时使用饱和与舍入修饰符，当使用.sat时应忽略舍入模式
    // 但在我们的实现中，当目标为整数时，我们需要使用适当的舍入模式
    // 为了兼容性，我们定义一个函数来判断是否需要使用舍入模式
    auto useRoundingMode = [&](bool roundingEnabled) -> bool {
        return roundingEnabled && !has_sat;
    };

    // 根据目标数据大小执行转换
    switch (dst_bytes) {
    case 1: { // 8-bit
        if (dst_is_float) {
            // 目标是浮点型
            float temp;
            if (src_is_float) {
                if (src_is_half) {
                    // 源是half类型，转换为float后再处理
                    uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                    temp = half_to_float(h_temp);
                } else if (src_bytes == 4) {
                    temp = *(float *)src;
                } else {
                    temp = (float)*(double *)src;
                }
            } else {
                // 源是整型 - 根据推断出的字节大小进行转换
                if (src_bytes == 1) {
                    if (src_is_signed) {
                        temp = (float)*(int8_t *)src;
                    } else {
                        temp = (float)*(uint8_t *)src;
                    }
                } else if (src_bytes == 2) {
                    if (src_is_signed) {
                        temp = (float)*(int16_t *)src;
                    } else {
                        temp = (float)*(uint16_t *)src;
                    }
                } else if (src_bytes == 4) {
                    if (src_is_signed) {
                        temp = (float)*(int32_t *)src;
                    } else {
                        temp = (float)*(uint32_t *)src;
                    }
                } else {
                    if (src_is_signed) {
                        temp = (float)*(int64_t *)src;
                    } else {
                        temp = (float)*(uint64_t *)src;
                    }
                }
            }

            if (has_sat) {
                if (std::isnan(temp)) {
                    *(float *)dst = 0.0f;
                } else {
                    // 不需要饱和，因为目标是浮点数
                    *(float *)dst = temp;
                }
            } else {
                *(float *)dst = temp;
            }
        } else {
            // 目标是整型
            if (src_is_float) {
                float temp;
                if (src_is_half) {
                    // 源是half类型，转换为float后再处理
                    uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                    temp = half_to_float(h_temp);
                } else if (src_bytes == 4) {
                    temp = *(float *)src;
                } else {
                    temp = (float)*(double *)src;
                }

                if (has_sat) {
                    if (std::isnan(temp)) {
                        *(uint8_t *)dst = 0;
                    } else if (temp <= 0.0f) {
                        *(uint8_t *)dst = 0;
                    } else if (temp >= 255.0f) {
                        *(uint8_t *)dst = 255;
                    } else {
                        // 当使用.sat时，不应用舍入，直接转换
                        *(uint8_t *)dst = static_cast<uint8_t>(temp);
                    }
                } else {
                    // 不使用sat时，应用舍入模式
                    if (has_rni ||
                        has_rn) { // 使用RNI或RN进行四舍五入（用于整数转换）
                        *(uint8_t *)dst =
                            static_cast<uint8_t>(std::round(temp));
                    } else if (
                        has_rzi ||
                        has_rz) { // 使用RZI或RZ进行向零舍入（用于整数转换）
                        *(uint8_t *)dst =
                            static_cast<uint8_t>(std::trunc(temp));
                    } else if (
                        has_rmi ||
                        has_rm) { // 使用RMI或RM进行向下舍入（用于整数转换）
                        *(uint8_t *)dst =
                            static_cast<uint8_t>(std::floor(temp));
                    } else if (
                        has_rpi ||
                        has_rp) { // 使用RPI或RP进行向上舍入（用于整数转换）
                        *(uint8_t *)dst = static_cast<uint8_t>(std::ceil(temp));
                    } else if (
                        has_rna) { // 使用RNA进行向远离零舍入（用于整数转换）
                        float rounded = (temp >= 0.0f) ? std::floor(temp + 0.5f)
                                                       : std::ceil(temp - 0.5f);
                        if (rounded < 0.0f) {
                            *(uint8_t *)dst = 0;
                        } else {
                            *(uint8_t *)dst = static_cast<uint8_t>(rounded);
                        }
                    } else {
                        *(uint8_t *)dst = static_cast<uint8_t>(temp);
                    }
                }
            } else {
                // 整数到整数转换
                if (src_bytes == 1) {
                    if (src_is_signed) {
                        // 有符号源
                        int8_t src_val = *(int8_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，直接转换
                            *(int8_t *)dst = src_val;
                        } else {
                            // 有符号到无符号，直接转换
                            *(uint8_t *)dst = static_cast<uint8_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint8_t src_val = *(uint8_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号，直接转换
                            *(int8_t *)dst = static_cast<int8_t>(src_val);
                        } else {
                            // 无符号到无符号，直接转换
                            *(uint8_t *)dst = src_val;
                        }
                    }
                } else if (src_bytes == 2) {
                    if (src_is_signed) {
                        // 有符号源
                        int16_t src_val = *(int16_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 有符号饱和转换
                                if (src_val > 127) {
                                    *(int8_t *)dst = 127;
                                } else if (src_val < -128) {
                                    *(int8_t *)dst = -128;
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else if (has_rni || has_rn) {
                                *(int8_t *)dst = static_cast<int8_t>(
                                    std::round(static_cast<float>(src_val)));
                            } else if (has_rzi || has_rz) {
                                *(int8_t *)dst = static_cast<int8_t>(
                                    std::trunc(static_cast<float>(src_val)));
                            } else if (has_rmi || has_rm) {
                                *(int8_t *)dst = static_cast<int8_t>(
                                    std::floor(static_cast<float>(src_val)));
                            } else if (has_rpi || has_rp) {
                                *(int8_t *)dst = static_cast<int8_t>(
                                    std::ceil(static_cast<float>(src_val)));
                            } else {
                                *(int8_t *)dst = static_cast<int8_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 有符号到无符号的饱和转换
                                if (src_val < 0) {
                                    *(uint8_t *)dst = 0;
                                } else if (src_val > 255) {
                                    *(uint8_t *)dst = 255;
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            } else {
                                *(uint8_t *)dst = static_cast<uint8_t>(src_val);
                            }
                        }
                    } else {
                        // 无符号源
                        uint16_t src_val = *(uint16_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 无符号到有符号的饱和转换
                                if (src_val > 127) {
                                    *(int8_t *)dst = 127;
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                *(int8_t *)dst = static_cast<int8_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 无符号到无符号的饱和转换
                                if (src_val > 255) {
                                    *(uint8_t *)dst = 255;
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            } else {
                                *(uint8_t *)dst = static_cast<uint8_t>(src_val);
                            }
                        }
                    }
                } else if (src_bytes == 4) {
                    if (src_is_signed) {
                        // 有符号源
                        int32_t src_val = *(int32_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 有符号饱和转换
                                if (src_val > 127) {
                                    *(int8_t *)dst = 127;
                                } else if (src_val < -128) {
                                    *(int8_t *)dst = -128;
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                *(int8_t *)dst = static_cast<int8_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 有符号到无符号的饱和转换
                                if (src_val < 0) {
                                    *(uint8_t *)dst = 0;
                                } else if (src_val > 255) {
                                    *(uint8_t *)dst = 255;
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            } else {
                                *(uint8_t *)dst = static_cast<uint8_t>(src_val);
                            }
                        }
                    } else {
                        // 无符号源
                        uint32_t src_val = *(uint32_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 无符号到有符号的饱和转换
                                if (src_val > 127) {
                                    *(int8_t *)dst = 127;
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                *(int8_t *)dst = static_cast<int8_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 无符号到无符号的饱和转换
                                if (src_val > 255) {
                                    *(uint8_t *)dst = 255;
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            } else {
                                *(uint8_t *)dst = static_cast<uint8_t>(src_val);
                            }
                        }
                    }
                } else {
                    if (src_is_signed) {
                        // 有符号源
                        int64_t src_val = *(int64_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 有符号饱和转换
                                if (src_val > 127) {
                                    *(int8_t *)dst = 127;
                                } else if (src_val < -128) {
                                    *(int8_t *)dst = -128;
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                *(int8_t *)dst = static_cast<int8_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 有符号到无符号的饱和转换
                                if (src_val < 0) {
                                    *(uint8_t *)dst = 0;
                                } else if (src_val > 255) {
                                    *(uint8_t *)dst = 255;
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            } else {
                                *(uint8_t *)dst = static_cast<uint8_t>(src_val);
                            }
                        }
                    } else {
                        // 无符号源
                        uint64_t src_val = *(uint64_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 无符号到有符号的饱和转换
                                if (src_val > 127) {
                                    *(int8_t *)dst = 127;
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                *(int8_t *)dst = static_cast<int8_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 无符号到无符号的饱和转换
                                if (src_val > 255) {
                                    *(uint8_t *)dst = 255;
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            } else {
                                *(uint8_t *)dst = static_cast<uint8_t>(src_val);
                            }
                        }
                    }
                }
            }
        }
        break;
    }
    case 2: { // 16-bit
        if (dst_is_float) {
            // 目标是浮点型
            if (dst_is_half) {
                // 目标是half类型
                float temp;
                if (src_is_float) {
                    if (src_is_half) {
                        // 源是half类型，直接复制
                        *(uint16_t *)dst = *(uint16_t *)src;
                    } else if (src_bytes == 4) {
                        // 源是float类型，转换为half
                        float src_val = *(float *)src;
                        *(uint16_t *)dst = float_to_half(src_val);
                    } else {
                        // 源是double类型，先转为float再转为half
                        float src_val = (float)*(double *)src;
                        *(uint16_t *)dst = float_to_half(src_val);
                    }
                } else {
                    // 源是整型 - 根据推断出的字节大小进行转换
                    if (src_bytes == 1) {
                        float src_val = (float)*(int8_t *)src;
                        *(uint16_t *)dst = float_to_half(src_val);
                    } else if (src_bytes == 2) {
                        float src_val = (float)*(int16_t *)src;
                        *(uint16_t *)dst = float_to_half(src_val);
                    } else if (src_bytes == 4) {
                        float src_val = (float)*(int32_t *)src;
                        *(uint16_t *)dst = float_to_half(src_val);
                    } else {
                        float src_val = (float)*(int64_t *)src;
                        *(uint16_t *)dst = float_to_half(src_val);
                    }
                }
            } else {
                // 目标是float类型（32位）
                float temp;
                if (src_is_float) {
                    if (src_is_half) {
                        // 源是half类型，转换为float
                        uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                        *(float *)dst = half_to_float(h_temp);
                    } else if (src_bytes == 4) {
                        *(float *)dst = *(float *)src;
                    } else {
                        *(float *)dst = (float)*(double *)src;
                    }
                } else {
                    // 源是整型 - 根据推断出的字节大小进行转换
                    if (src_bytes == 1) {
                        *(float *)dst = (float)*(int8_t *)src;
                    } else if (src_bytes == 2) {
                        *(float *)dst = (float)*(int16_t *)src;
                    } else if (src_bytes == 4) {
                        *(float *)dst = (float)*(int32_t *)src;
                    } else {
                        *(float *)dst = (float)*(int64_t *)src;
                    }
                }
            }
        } else {
            // 目标是整型
            if (src_is_float) {
                float temp;
                if (src_is_half) {
                    // 源是half类型，转换为float后再处理
                    uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                    temp = half_to_float(h_temp);
                } else if (src_bytes == 4) {
                    temp = *(float *)src;
                } else {
                    temp = (float)*(double *)src;
                }

                if (has_sat) {
                    if (std::isnan(temp)) {
                        *(uint16_t *)dst = 0;
                    } else if (temp <= 0.0f) {
                        *(uint16_t *)dst = 0;
                    } else if (temp > 65535.0f) { // 使用 > 而不是
                                                  // >=，因为65535本身在范围内
                        *(uint16_t *)dst = 65535;
                    } else {
                        // 当使用.sat时，不应用舍入，直接转换
                        *(uint16_t *)dst = static_cast<uint16_t>(temp);
                    }
                } else {
                    if (has_rni ||
                        has_rn) { // 使用RNI或RN进行四舍五入（用于整数转换）
                        *(uint16_t *)dst =
                            static_cast<uint16_t>(std::round(temp));
                    } else if (
                        has_rzi ||
                        has_rz) { // 使用RZI或RZ进行向零舍入（用于整数转换）
                        *(uint16_t *)dst =
                            static_cast<uint16_t>(std::trunc(temp));
                    } else if (
                        has_rmi ||
                        has_rm) { // 使用RMI或RM进行向下舍入（用于整数转换）
                        *(uint16_t *)dst =
                            static_cast<uint16_t>(std::floor(temp));
                    } else if (
                        has_rpi ||
                        has_rp) { // 使用RPI或RP进行向上舍入（用于整数转换）
                        *(uint16_t *)dst =
                            static_cast<uint16_t>(std::ceil(temp));
                    } else if (
                        has_rna) { // 使用RNA进行向远离零舍入（用于整数转换）
                        float rounded = (temp >= 0.0f) ? std::floor(temp + 0.5f)
                                                       : std::ceil(temp - 0.5f);
                        if (rounded < 0.0f) {
                            *(uint16_t *)dst = 0;
                        } else {
                            *(uint16_t *)dst = static_cast<uint16_t>(rounded);
                        }
                    } else {
                        *(uint16_t *)dst = static_cast<uint16_t>(temp);
                    }
                }
            } else {
                // 整数到整数转换
                if (src_bytes == 1) {
                    if (src_is_signed) {
                        // 有符号源
                        int8_t src_val = *(int8_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，符号扩展
                            *(int16_t *)dst = src_val;
                        } else {
                            // 有符号到无符号
                            *(uint16_t *)dst = static_cast<uint16_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint8_t src_val = *(uint8_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号
                            *(int16_t *)dst = static_cast<int16_t>(src_val);
                        } else {
                            // 无符号到无符号，零扩展
                            *(uint16_t *)dst = src_val;
                        }
                    }
                } else if (src_bytes == 2) {
                    if (src_is_signed) {
                        // 有符号源
                        int16_t src_val = *(int16_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，直接转换
                            *(int16_t *)dst = src_val;
                        } else {
                            // 有符号到无符号，直接转换
                            *(uint16_t *)dst = static_cast<uint16_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint16_t src_val = *(uint16_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号
                            *(int16_t *)dst = static_cast<int16_t>(src_val);
                        } else {
                            // 无符号到无符号，直接转换
                            *(uint16_t *)dst = src_val;
                        }
                    }
                } else if (src_bytes == 4) {
                    if (src_is_signed) {
                        // 有符号源
                        int32_t src_val = *(int32_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 有符号饱和转换
                                if (src_val > 32767) {
                                    *(int16_t *)dst = 32767;
                                } else if (src_val < -32768) {
                                    *(int16_t *)dst = -32768;
                                } else {
                                    *(int16_t *)dst =
                                        static_cast<int16_t>(src_val);
                                }
                            } else {
                                *(int16_t *)dst = static_cast<int16_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 有符号到无符号的饱和转换
                                if (src_val < 0) {
                                    *(uint16_t *)dst = 0;
                                } else if (src_val > 65535) {
                                    *(uint16_t *)dst = 65535;
                                } else {
                                    *(uint16_t *)dst =
                                        static_cast<uint16_t>(src_val);
                                }
                            } else {
                                *(uint16_t *)dst =
                                    static_cast<uint16_t>(src_val);
                            }
                        }
                    } else {
                        // 无符号源
                        uint32_t src_val = *(uint32_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 无符号到有符号的饱和转换
                                if (src_val > 32767) {
                                    *(int16_t *)dst = 32767;
                                } else {
                                    *(int16_t *)dst =
                                        static_cast<int16_t>(src_val);
                                }
                            } else {
                                *(int16_t *)dst = static_cast<int16_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 无符号到无符号的饱和转换
                                if (src_val > 65535) {
                                    *(uint16_t *)dst = 65535;
                                } else {
                                    *(uint16_t *)dst =
                                        static_cast<uint16_t>(src_val);
                                }
                            } else {
                                *(uint16_t *)dst =
                                    static_cast<uint16_t>(src_val);
                            }
                        }
                    }
                } else {
                    if (src_is_signed) {
                        // 有符号源
                        int64_t src_val = *(int64_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 有符号饱和转换
                                if (src_val > 32767) {
                                    *(int16_t *)dst = 32767;
                                } else if (src_val < -32768) {
                                    *(int16_t *)dst = -32768;
                                } else {
                                    *(int16_t *)dst =
                                        static_cast<int16_t>(src_val);
                                }
                            } else {
                                *(int16_t *)dst = static_cast<int16_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 有符号到无符号的饱和转换
                                if (src_val < 0) {
                                    *(uint16_t *)dst = 0;
                                } else if (src_val > 65535) {
                                    *(uint16_t *)dst = 65535;
                                } else {
                                    *(uint16_t *)dst =
                                        static_cast<uint16_t>(src_val);
                                }
                            } else {
                                *(uint16_t *)dst =
                                    static_cast<uint16_t>(src_val);
                            }
                        }
                    } else {
                        // 无符号源
                        uint64_t src_val = *(uint64_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 无符号到有符号的饱和转换
                                if (src_val > 32767) {
                                    *(int16_t *)dst = 32767;
                                } else {
                                    *(int16_t *)dst =
                                        static_cast<int16_t>(src_val);
                                }
                            } else {
                                *(int16_t *)dst = static_cast<int16_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 无符号到无符号的饱和转换
                                if (src_val > 65535) {
                                    *(uint16_t *)dst = 65535;
                                } else {
                                    *(uint16_t *)dst =
                                        static_cast<uint16_t>(src_val);
                                }
                            } else {
                                *(uint16_t *)dst =
                                    static_cast<uint16_t>(src_val);
                            }
                        }
                    }
                }
            }
        }
        break;
    }
    case 4: { // 32-bit
        if (dst_is_float) {
            // 目标是浮点型 (float)
            if (src_is_float) {
                if (src_is_half) {
                    // 源是half类型，转换为float
                    uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                    *(float *)dst = half_to_float(h_temp);
                } else if (src_bytes == 4) {
                    *(float *)dst = *(float *)src;
                } else {
                    *(float *)dst = (float)*(double *)src;
                }
            } else {
                // 源是整型 - 根据推断出的字节大小进行转换
                if (src_bytes == 1) {
                    *(float *)dst = (float)*(int8_t *)src;
                } else if (src_bytes == 2) {
                    *(float *)dst = (float)*(int16_t *)src;
                } else if (src_bytes == 4) {
                    *(float *)dst = (float)*(int32_t *)src;
                } else {
                    *(float *)dst = (float)*(int64_t *)src;
                }
            }
        } else {
            // 目标是整型
            if (src_is_float) {
                float temp;
                if (src_is_half) {
                    // 源是half类型，转换为float
                    uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                    temp = half_to_float(h_temp);
                } else if (src_bytes == 4) {
                    temp = *(float *)src;
                } else {
                    temp = (float)*(double *)src;
                }

                if (has_sat) {
                    if (std::isnan(temp)) {
                        *(uint32_t *)dst = 0;
                    } else if (temp <= 0.0f) {
                        *(uint32_t *)dst = 0;
                    } else if (temp > 4294967295.0f) { // 使用 > 而不是 >=
                        *(uint32_t *)dst = 4294967295U;
                    } else {
                        // 当使用.sat时，不应用舍入，直接转换
                        *(uint32_t *)dst = static_cast<uint32_t>(temp);
                    }
                } else {
                    if (has_rni ||
                        has_rn) { // 使用RNI或RN进行四舍五入（用于整数转换）
                        *(uint32_t *)dst =
                            static_cast<uint32_t>(std::round(temp));
                    } else if (
                        has_rzi ||
                        has_rz) { // 使用RZI或RZ进行向零舍入（用于整数转换）
                        *(uint32_t *)dst =
                            static_cast<uint32_t>(std::trunc(temp));
                    } else if (
                        has_rmi ||
                        has_rm) { // 使用RMI或RM进行向下舍入（用于整数转换）
                        *(uint32_t *)dst =
                            static_cast<uint32_t>(std::floor(temp));
                    } else if (
                        has_rpi ||
                        has_rp) { // 使用RPI或RP进行向上舍入（用于整数转换）
                        *(uint32_t *)dst =
                            static_cast<uint32_t>(std::ceil(temp));
                    } else if (
                        has_rna) { // 使用RNA进行向远离零舍入（用于整数转换）
                        float rounded = (temp >= 0.0f) ? std::floor(temp + 0.5f)
                                                       : std::ceil(temp - 0.5f);
                        if (rounded < 0.0f) {
                            *(uint32_t *)dst = 0;
                        } else {
                            *(uint32_t *)dst = static_cast<uint32_t>(rounded);
                        }
                    } else {
                        *(uint32_t *)dst = static_cast<uint32_t>(temp);
                    }
                }
            } else {
                // 整数到整数转换
                if (src_bytes == 1) {
                    if (src_is_signed) {
                        // 有符号源
                        int8_t src_val = *(int8_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，符号扩展
                            *(int32_t *)dst = src_val;
                        } else {
                            // 有符号到无符号
                            *(uint32_t *)dst = static_cast<uint32_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint8_t src_val = *(uint8_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号
                            *(int32_t *)dst = static_cast<int32_t>(src_val);
                        } else {
                            // 无符号到无符号，零扩展
                            *(uint32_t *)dst = src_val;
                        }
                    }
                } else if (src_bytes == 2) {
                    if (src_is_signed) {
                        // 有符号源
                        int16_t src_val = *(int16_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，符号扩展
                            *(int32_t *)dst = src_val;
                        } else {
                            // 有符号到无符号
                            *(uint32_t *)dst = static_cast<uint32_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint16_t src_val = *(uint16_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号
                            *(int32_t *)dst = static_cast<int32_t>(src_val);
                        } else {
                            // 无符号到无符号，零扩展
                            *(uint32_t *)dst = src_val;
                        }
                    }
                } else if (src_bytes == 4) {
                    if (src_is_signed) {
                        // 有符号源
                        int32_t src_val = *(int32_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，直接转换
                            *(int32_t *)dst = src_val;
                        } else {
                            // 有符号到无符号，直接转换
                            *(uint32_t *)dst = static_cast<uint32_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint32_t src_val = *(uint32_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号，直接转换
                            *(int32_t *)dst = static_cast<int32_t>(src_val);
                        } else {
                            // 无符号到无符号，直接转换
                            *(uint32_t *)dst = src_val;
                        }
                    }
                } else {
                    if (src_is_signed) {
                        // 有符号源
                        int64_t src_val = *(int64_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 有符号饱和转换
                                if (src_val > 2147483647LL) {
                                    *(int32_t *)dst = 2147483647;
                                } else if (src_val < -2147483647LL - 1) {
                                    *(int32_t *)dst = -2147483647 - 1;
                                } else {
                                    *(int32_t *)dst =
                                        static_cast<int32_t>(src_val);
                                }
                            } else {
                                *(int32_t *)dst = static_cast<int32_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 有符号到无符号的饱和转换
                                if (src_val < 0) {
                                    *(uint32_t *)dst = 0;
                                } else if (src_val > 4294967295ULL) {
                                    *(uint32_t *)dst = 4294967295U;
                                } else {
                                    *(uint32_t *)dst =
                                        static_cast<uint32_t>(src_val);
                                }
                            } else {
                                *(uint32_t *)dst =
                                    static_cast<uint32_t>(src_val);
                            }
                        }
                    } else {
                        // 无符号源
                        uint64_t src_val = *(uint64_t *)src;
                        if (dst_is_signed) {
                            if (has_sat) {
                                // 无符号到有符号的饱和转换
                                if (src_val > 2147483647U) {
                                    *(int32_t *)dst = 2147483647;
                                } else {
                                    *(int32_t *)dst =
                                        static_cast<int32_t>(src_val);
                                }
                            } else {
                                *(int32_t *)dst = static_cast<int32_t>(src_val);
                            }
                        } else {
                            if (has_sat) {
                                // 无符号到无符号的饱和转换
                                if (src_val > 4294967295ULL) {
                                    *(uint32_t *)dst = 4294967295U;
                                } else {
                                    *(uint32_t *)dst =
                                        static_cast<uint32_t>(src_val);
                                }
                            } else {
                                *(uint32_t *)dst =
                                    static_cast<uint32_t>(src_val);
                            }
                        }
                    }
                }
            }
        }
        break;
    }
    case 8: { // 64-bit
        if (dst_is_float) {
            // 目标是双精度浮点型 (double)
            if (src_is_float) {
                if (src_is_half) {
                    // 源是half类型，先转为float再转为double
                    uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                    *(double *)dst = (double)half_to_float(h_temp);
                } else if (src_bytes == 4) {
                    *(double *)dst = (double)*(float *)src;
                } else {
                    *(double *)dst = *(double *)src;
                }
            } else {
                // 源是整型 - 根据推断出的字节大小进行转换
                if (src_bytes == 1) {
                    *(double *)dst = (double)*(int8_t *)src;
                } else if (src_bytes == 2) {
                    *(double *)dst = (double)*(int16_t *)src;
                } else if (src_bytes == 4) {
                    *(double *)dst = (double)*(int32_t *)src;
                } else {
                    *(double *)dst = (double)*(int64_t *)src;
                }
            }
        } else {
            // 目标是整型
            if (src_is_float) {
                double temp;
                if (src_is_half) {
                    // 源是half类型，先转为float再转为double
                    uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                    temp = (double)half_to_float(h_temp);
                } else if (src_bytes == 4) {
                    temp = (double)*(float *)src;
                } else {
                    temp = *(double *)src;
                }

                if (has_sat) {
                    if (std::isnan(temp)) {
                        *(uint64_t *)dst = 0;
                    } else if (temp <= 0.0) {
                        *(uint64_t *)dst = 0;
                    } else if (temp >
                               18446744073709551615.0) { // 使用 > 而不是 >=
                        *(uint64_t *)dst = 18446744073709551615ULL;
                    } else {
                        // 当使用.sat时，不应用舍入，直接转换
                        *(uint64_t *)dst = static_cast<uint64_t>(temp);
                    }
                } else {
                    if (has_rni ||
                        has_rn) { // 使用RNI或RN进行四舍五入（用于整数转换）
                        *(uint64_t *)dst =
                            static_cast<uint64_t>(std::round(temp));
                    } else if (
                        has_rzi ||
                        has_rz) { // 使用RZI或RZ进行向零舍入（用于整数转换）
                        *(uint64_t *)dst =
                            static_cast<uint64_t>(std::trunc(temp));
                    } else if (
                        has_rmi ||
                        has_rm) { // 使用RMI或RM进行向下舍入（用于整数转换）
                        *(uint64_t *)dst =
                            static_cast<uint64_t>(std::floor(temp));
                    } else if (
                        has_rpi ||
                        has_rp) { // 使用RPI或RP进行向上舍入（用于整数转换）
                        *(uint64_t *)dst =
                            static_cast<uint64_t>(std::ceil(temp));
                    } else if (
                        has_rna) { // 使用RNA进行向远离零舍入（用于整数转换）
                        double rounded = (temp >= 0.0) ? std::floor(temp + 0.5)
                                                       : std::ceil(temp - 0.5);
                        if (rounded < 0.0) {
                            *(uint64_t *)dst = 0;
                        } else {
                            *(uint64_t *)dst = static_cast<uint64_t>(rounded);
                        }
                    } else {
                        *(uint64_t *)dst = static_cast<uint64_t>(temp);
                    }
                }
            } else {
                // 整数到整数转换
                if (src_bytes == 1) {
                    if (src_is_signed) {
                        // 有符号源
                        int8_t src_val = *(int8_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，符号扩展
                            *(int64_t *)dst = src_val;
                        } else {
                            // 有符号到无符号
                            *(uint64_t *)dst = static_cast<uint64_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint8_t src_val = *(uint8_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号
                            *(int64_t *)dst = static_cast<int64_t>(src_val);
                        } else {
                            // 无符号到无符号，零扩展
                            *(uint64_t *)dst = src_val;
                        }
                    }
                } else if (src_bytes == 2) {
                    if (src_is_signed) {
                        // 有符号源
                        int16_t src_val = *(int16_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，符号扩展
                            *(int64_t *)dst = src_val;
                        } else {
                            // 有符号到无符号
                            *(uint64_t *)dst = static_cast<uint64_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint16_t src_val = *(uint16_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号
                            *(int64_t *)dst = static_cast<int64_t>(src_val);
                        } else {
                            // 无符号到无符号，零扩展
                            *(uint64_t *)dst = src_val;
                        }
                    }
                } else if (src_bytes == 4) {
                    if (src_is_signed) {
                        // 有符号源
                        int32_t src_val = *(int32_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，符号扩展
                            *(int64_t *)dst = src_val;
                        } else {
                            // 有符号到无符号
                            *(uint64_t *)dst = static_cast<uint64_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint32_t src_val = *(uint32_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号
                            *(int64_t *)dst = static_cast<int64_t>(src_val);
                        } else {
                            // 无符号到无符号，零扩展
                            *(uint64_t *)dst = src_val;
                        }
                    }
                } else {
                    if (src_is_signed) {
                        // 有符号源
                        int64_t src_val = *(int64_t *)src;
                        if (dst_is_signed) {
                            // 有符号到有符号，直接转换
                            *(int64_t *)dst = src_val;
                        } else {
                            // 有符号到无符号，直接转换
                            *(uint64_t *)dst = static_cast<uint64_t>(src_val);
                        }
                    } else {
                        // 无符号源
                        uint64_t src_val = *(uint64_t *)src;
                        if (dst_is_signed) {
                            // 无符号到有符号，直接转换
                            *(int64_t *)dst = static_cast<int64_t>(src_val);
                        } else {
                            // 无符号到无符号，直接转换
                            *(uint64_t *)dst = src_val;
                        }
                    }
                }
            }
        }
        break;
    }
    default:
        assert(0 && "Unsupported destination size for CVT instruction");
    }
}
