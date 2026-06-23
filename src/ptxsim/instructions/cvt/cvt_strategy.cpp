// cvt_strategy.cpp
// =============================================================================
// CVT 策略模式实现 (T2-6 Sub-task 3 — skeleton)
//
// 状态:
//   - build_context():  从 Qualifier 列表构造强类型 CvtContext
//   - select_strategy(): 返回单一 GeneralCvtStrategy 实例 (Sub-task 4 拆分)
//   - GeneralCvtStrategy::convert(): 完整保留 arithmetic_conversion.cpp
//     原 switch 逻辑 (line 95-1174)，改用 ctx.xxx 字段代替局部变量。
//     **零行为变更** — Sub-task 3 目标。
//
// Sub-task 4 将 GeneralCvtStrategy::convert() 拆为 5 个具体策略:
//   FloatToFloatStrategy / IntToFloatStrategy / FloatToIntStrategy /
//   IntToIntStrategy + Rounding helpers。
// Sub-task 6 删除 arithmetic_conversion.cpp 的旧 switch。
// =============================================================================

#include "ptxsim/instructions/cvt/cvt_strategy.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/instructions/cvt/cvt_float_to_float.h"
#include "ptxsim/instructions/cvt/cvt_float_to_int.h"
#include "ptxsim/instructions/cvt/cvt_helpers.h"
#include "ptxsim/instructions/cvt/cvt_int_to_float.h"
#include "ptxsim/instructions/cvt/cvt_int_to_int.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ptxsim {
namespace cvt_strategy {

// ---------------------------------------------------------------------------
// build_context: 从 Qualifier 列表提取 CvtContext
//
// 来源: arithmetic_conversion.cpp:17-90 (旧 processOperation 第一段)
// 抽取后: arithmetic_conversion.cpp::processOperation 直接构造 ctx，
//        避免每次重新解析 Qualifier。
// ---------------------------------------------------------------------------
CvtContext build_context(const std::vector<Qualifier> &qualifiers) {
    CvtContext ctx;

    std::vector<Qualifier> dst_qualifiers, src_qualifiers;
    splitDstSrcQualifiers(qualifiers, dst_qualifiers, src_qualifiers);

    ctx.dst_bytes = getBytes(dst_qualifiers);
    ctx.src_bytes = getBytes(src_qualifiers);
    ctx.dst_is_float = TypeUtils::is_float_type(dst_qualifiers);
    ctx.src_is_float = TypeUtils::is_float_type(src_qualifiers);

    // f16 (half) 强制 2 字节 + 视为 float
    for (const auto &q : dst_qualifiers) {
        if (q == Qualifier::Q_F16) {
            ctx.dst_is_half = true;
            ctx.dst_is_float = true;
            ctx.dst_bytes = 2;
            break;
        }
    }
    for (const auto &q : src_qualifiers) {
        if (q == Qualifier::Q_F16) {
            ctx.src_is_half = true;
            ctx.src_is_float = true;
            ctx.src_bytes = 2;
            break;
        }
    }

    // bytes 默认值兜底 (与原逻辑保持一致)
    if (ctx.dst_bytes == 0) {
        ctx.dst_bytes = getBytes(qualifiers);
    }
    if (ctx.src_bytes == 0 && !ctx.src_is_half) {
        ctx.src_bytes = getBytes(qualifiers);
    }
    if (ctx.src_bytes == 0 && !ctx.src_is_half) {
        ctx.src_bytes = 1;
    }

    // 修饰符
    ctx.has_sat = QvecHasQ(qualifiers, Qualifier::Q_SAT);
    ctx.has_rn = QvecHasQ(qualifiers, Qualifier::Q_RN);
    ctx.has_rni = QvecHasQ(qualifiers, Qualifier::Q_RNI);
    ctx.has_rz = QvecHasQ(qualifiers, Qualifier::Q_RZ);
    ctx.has_rzi = QvecHasQ(qualifiers, Qualifier::Q_RZI);
    ctx.has_rm = QvecHasQ(qualifiers, Qualifier::Q_RM);
    ctx.has_rmi = QvecHasQ(qualifiers, Qualifier::Q_RMI);
    ctx.has_rp = QvecHasQ(qualifiers, Qualifier::Q_RP);
    ctx.has_rpi = QvecHasQ(qualifiers, Qualifier::Q_RPI);
    ctx.has_rna = QvecHasQ(qualifiers, Qualifier::Q_RNA);
    ctx.has_rs = QvecHasQ(qualifiers, Qualifier::Q_RS);

    // 符号性
    ctx.dst_is_signed = TypeUtils::is_signed_type(dst_qualifiers);
    ctx.src_is_signed = TypeUtils::is_signed_type(src_qualifiers);

    return ctx;
}

// ---------------------------------------------------------------------------
// GeneralCvtStrategy: 暂存原 arithmetic_conversion.cpp 整个 switch
//
// 注: 这是过渡策略，Sub-task 4 将拆为 5 个具体策略。
// 所有访问 dst/src 的代码使用 ctx.* 字段。
// ---------------------------------------------------------------------------
class GeneralCvtStrategy : public ConversionStrategy {
public:
    void convert(void *dst, void *src, const CvtContext &ctx) const override {
        switch (ctx.dst_bytes) {
        case 1: { // 8-bit
            if (ctx.dst_is_float) {
                float temp;
                if (ctx.src_is_float) {
                    if (ctx.src_is_half) {
                        uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                        temp = cvt_helpers::half_to_float(h_temp);
                    } else if (ctx.src_bytes == 4) {
                        temp = *(float *)src;
                    } else {
                        temp = (float)*(double *)src;
                    }
                } else {
                    if (ctx.src_bytes == 1) {
                        temp = ctx.src_is_signed ? (float)*(int8_t *)src
                                                 : (float)*(uint8_t *)src;
                    } else if (ctx.src_bytes == 2) {
                        temp = ctx.src_is_signed ? (float)*(int16_t *)src
                                                 : (float)*(uint16_t *)src;
                    } else if (ctx.src_bytes == 4) {
                        temp = ctx.src_is_signed ? (float)*(int32_t *)src
                                                 : (float)*(uint32_t *)src;
                    } else {
                        temp = ctx.src_is_signed ? (float)*(int64_t *)src
                                                 : (float)*(uint64_t *)src;
                    }
                }
                if (ctx.has_sat) {
                    if (std::isnan(temp)) {
                        *(float *)dst = 0.0f;
                    } else {
                        *(float *)dst = temp;
                    }
                } else {
                    *(float *)dst = temp;
                }
            } else {
                if (ctx.src_is_float) {
                    float temp;
                    if (ctx.src_is_half) {
                        uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                        temp = cvt_helpers::half_to_float(h_temp);
                    } else if (ctx.src_bytes == 4) {
                        temp = *(float *)src;
                    } else {
                        temp = (float)*(double *)src;
                    }
                    if (ctx.has_sat) {
                        if (std::isnan(temp)) {
                            *(uint8_t *)dst = 0;
                        } else if (temp <= 0.0f) {
                            *(uint8_t *)dst = 0;
                        } else if (temp >= 255.0f) {
                            *(uint8_t *)dst = 255;
                        } else {
                            *(uint8_t *)dst = static_cast<uint8_t>(temp);
                        }
                    } else {
                        if (ctx.has_rni || ctx.has_rn) {
                            *(uint8_t *)dst = static_cast<uint8_t>(
                                cvt_helpers::round_half_to_even(temp));
                        } else if (ctx.has_rzi || ctx.has_rz) {
                            *(uint8_t *)dst =
                                static_cast<uint8_t>(std::trunc(temp));
                        } else if (ctx.has_rmi || ctx.has_rm) {
                            *(uint8_t *)dst =
                                static_cast<uint8_t>(std::floor(temp));
                        } else if (ctx.has_rpi || ctx.has_rp) {
                            *(uint8_t *)dst =
                                static_cast<uint8_t>(std::ceil(temp));
                        } else if (ctx.has_rna) {
                            float rounded = (temp >= 0.0f)
                                                ? std::floor(temp + 0.5f)
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
                    if (ctx.src_bytes == 1) {
                        if (ctx.src_is_signed) {
                            int8_t src_val = *(int8_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int8_t *)dst = src_val;
                            } else {
                                *(uint8_t *)dst = static_cast<uint8_t>(src_val);
                            }
                        } else {
                            uint8_t src_val = *(uint8_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int8_t *)dst = static_cast<int8_t>(src_val);
                            } else {
                                *(uint8_t *)dst = src_val;
                            }
                        }
                    } else if (ctx.src_bytes == 2) {
                        if (ctx.src_is_signed) {
                            int16_t src_val = *(int16_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 127) {
                                        *(int8_t *)dst = 127;
                                    } else if (src_val < -128) {
                                        *(int8_t *)dst = -128;
                                    } else {
                                        *(int8_t *)dst =
                                            static_cast<int8_t>(src_val);
                                    }
                                } else if (ctx.has_rni || ctx.has_rn) {
                                    *(int8_t *)dst = static_cast<int8_t>(
                                        cvt_helpers::round_half_to_even(
                                            static_cast<float>(src_val)));
                                } else if (ctx.has_rzi || ctx.has_rz) {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(std::trunc(
                                            static_cast<float>(src_val)));
                                } else if (ctx.has_rmi || ctx.has_rm) {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(std::floor(
                                            static_cast<float>(src_val)));
                                } else if (ctx.has_rpi || ctx.has_rp) {
                                    *(int8_t *)dst = static_cast<int8_t>(
                                        std::ceil(static_cast<float>(src_val)));
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
                                    if (src_val < 0) {
                                        *(uint8_t *)dst = 0;
                                    } else if (src_val > 255) {
                                        *(uint8_t *)dst = 255;
                                    } else {
                                        *(uint8_t *)dst =
                                            static_cast<uint8_t>(src_val);
                                    }
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            }
                        } else {
                            uint16_t src_val = *(uint16_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 127) {
                                        *(int8_t *)dst = 127;
                                    } else {
                                        *(int8_t *)dst =
                                            static_cast<int8_t>(src_val);
                                    }
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
                                    if (src_val > 255) {
                                        *(uint8_t *)dst = 255;
                                    } else {
                                        *(uint8_t *)dst =
                                            static_cast<uint8_t>(src_val);
                                    }
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            }
                        }
                    } else if (ctx.src_bytes == 4) {
                        if (ctx.src_is_signed) {
                            int32_t src_val = *(int32_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 127) {
                                        *(int8_t *)dst = 127;
                                    } else if (src_val < -128) {
                                        *(int8_t *)dst = -128;
                                    } else {
                                        *(int8_t *)dst =
                                            static_cast<int8_t>(src_val);
                                    }
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
                                    if (src_val < 0) {
                                        *(uint8_t *)dst = 0;
                                    } else if (src_val > 255) {
                                        *(uint8_t *)dst = 255;
                                    } else {
                                        *(uint8_t *)dst =
                                            static_cast<uint8_t>(src_val);
                                    }
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            }
                        } else {
                            uint32_t src_val = *(uint32_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 127) {
                                        *(int8_t *)dst = 127;
                                    } else {
                                        *(int8_t *)dst =
                                            static_cast<int8_t>(src_val);
                                    }
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
                                    if (src_val > 255) {
                                        *(uint8_t *)dst = 255;
                                    } else {
                                        *(uint8_t *)dst =
                                            static_cast<uint8_t>(src_val);
                                    }
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            }
                        }
                    } else {
                        if (ctx.src_is_signed) {
                            int64_t src_val = *(int64_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 127) {
                                        *(int8_t *)dst = 127;
                                    } else if (src_val < -128) {
                                        *(int8_t *)dst = -128;
                                    } else {
                                        *(int8_t *)dst =
                                            static_cast<int8_t>(src_val);
                                    }
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
                                    if (src_val < 0) {
                                        *(uint8_t *)dst = 0;
                                    } else if (src_val > 255) {
                                        *(uint8_t *)dst = 255;
                                    } else {
                                        *(uint8_t *)dst =
                                            static_cast<uint8_t>(src_val);
                                    }
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            }
                        } else {
                            uint64_t src_val = *(uint64_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 127) {
                                        *(int8_t *)dst = 127;
                                    } else {
                                        *(int8_t *)dst =
                                            static_cast<int8_t>(src_val);
                                    }
                                } else {
                                    *(int8_t *)dst =
                                        static_cast<int8_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
                                    if (src_val > 255) {
                                        *(uint8_t *)dst = 255;
                                    } else {
                                        *(uint8_t *)dst =
                                            static_cast<uint8_t>(src_val);
                                    }
                                } else {
                                    *(uint8_t *)dst =
                                        static_cast<uint8_t>(src_val);
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
        case 2: { // 16-bit
            if (ctx.dst_is_float) {
                if (ctx.dst_is_half) {
                    if (ctx.src_is_float) {
                        if (ctx.src_is_half) {
                            *(uint16_t *)dst = *(uint16_t *)src;
                        } else if (ctx.src_bytes == 4) {
                            float src_val = *(float *)src;
                            *(uint16_t *)dst =
                                cvt_helpers::float_to_half(src_val);
                        } else {
                            float src_val = (float)*(double *)src;
                            *(uint16_t *)dst =
                                cvt_helpers::float_to_half(src_val);
                        }
                    } else {
                        float src_val;
                        if (ctx.src_bytes == 1) {
                            src_val = (float)*(int8_t *)src;
                        } else if (ctx.src_bytes == 2) {
                            src_val = (float)*(int16_t *)src;
                        } else if (ctx.src_bytes == 4) {
                            src_val = (float)*(int32_t *)src;
                        } else {
                            src_val = (float)*(int64_t *)src;
                        }
                        *(uint16_t *)dst = cvt_helpers::float_to_half(src_val);
                    }
                } else {
                    if (ctx.src_is_float) {
                        if (ctx.src_is_half) {
                            uint16_t h_temp =
                                *reinterpret_cast<uint16_t *>(src);
                            *(float *)dst = cvt_helpers::half_to_float(h_temp);
                        } else if (ctx.src_bytes == 4) {
                            *(float *)dst = *(float *)src;
                        } else {
                            *(float *)dst = (float)*(double *)src;
                        }
                    } else {
                        if (ctx.src_bytes == 1) {
                            *(float *)dst = (float)*(int8_t *)src;
                        } else if (ctx.src_bytes == 2) {
                            *(float *)dst = (float)*(int16_t *)src;
                        } else if (ctx.src_bytes == 4) {
                            *(float *)dst = (float)*(int32_t *)src;
                        } else {
                            *(float *)dst = (float)*(int64_t *)src;
                        }
                    }
                }
            } else {
                if (ctx.src_is_float) {
                    float temp;
                    if (ctx.src_is_half) {
                        uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                        temp = cvt_helpers::half_to_float(h_temp);
                    } else if (ctx.src_bytes == 4) {
                        temp = *(float *)src;
                    } else {
                        temp = (float)*(double *)src;
                    }
                    if (ctx.has_sat) {
                        if (std::isnan(temp)) {
                            *(uint16_t *)dst = 0;
                        } else if (temp <= 0.0f) {
                            *(uint16_t *)dst = 0;
                        } else if (temp > 65535.0f) {
                            *(uint16_t *)dst = 65535;
                        } else {
                            *(uint16_t *)dst = static_cast<uint16_t>(temp);
                        }
                    } else {
                        if (ctx.has_rni || ctx.has_rn) {
                            *(uint16_t *)dst = static_cast<uint16_t>(
                                cvt_helpers::round_half_to_even(temp));
                        } else if (ctx.has_rzi || ctx.has_rz) {
                            *(uint16_t *)dst =
                                static_cast<uint16_t>(std::trunc(temp));
                        } else if (ctx.has_rmi || ctx.has_rm) {
                            *(uint16_t *)dst =
                                static_cast<uint16_t>(std::floor(temp));
                        } else if (ctx.has_rpi || ctx.has_rp) {
                            *(uint16_t *)dst =
                                static_cast<uint16_t>(std::ceil(temp));
                        } else if (ctx.has_rna) {
                            float rounded = (temp >= 0.0f)
                                                ? std::floor(temp + 0.5f)
                                                : std::ceil(temp - 0.5f);
                            if (rounded < 0.0f) {
                                *(uint16_t *)dst = 0;
                            } else {
                                *(uint16_t *)dst =
                                    static_cast<uint16_t>(rounded);
                            }
                        } else {
                            *(uint16_t *)dst = static_cast<uint16_t>(temp);
                        }
                    }
                } else {
                    if (ctx.src_bytes == 1) {
                        if (ctx.src_is_signed) {
                            int8_t src_val = *(int8_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int16_t *)dst = src_val;
                            } else {
                                *(uint16_t *)dst =
                                    static_cast<uint16_t>(src_val);
                            }
                        } else {
                            uint8_t src_val = *(uint8_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int16_t *)dst = static_cast<int16_t>(src_val);
                            } else {
                                *(uint16_t *)dst = src_val;
                            }
                        }
                    } else if (ctx.src_bytes == 2) {
                        if (ctx.src_is_signed) {
                            int16_t src_val = *(int16_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int16_t *)dst = src_val;
                            } else {
                                *(uint16_t *)dst =
                                    static_cast<uint16_t>(src_val);
                            }
                        } else {
                            uint16_t src_val = *(uint16_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int16_t *)dst = static_cast<int16_t>(src_val);
                            } else {
                                *(uint16_t *)dst = src_val;
                            }
                        }
                    } else if (ctx.src_bytes == 4) {
                        if (ctx.src_is_signed) {
                            int32_t src_val = *(int32_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 32767) {
                                        *(int16_t *)dst = 32767;
                                    } else if (src_val < -32768) {
                                        *(int16_t *)dst = -32768;
                                    } else {
                                        *(int16_t *)dst =
                                            static_cast<int16_t>(src_val);
                                    }
                                } else {
                                    *(int16_t *)dst =
                                        static_cast<int16_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
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
                            uint32_t src_val = *(uint32_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 32767) {
                                        *(int16_t *)dst = 32767;
                                    } else {
                                        *(int16_t *)dst =
                                            static_cast<int16_t>(src_val);
                                    }
                                } else {
                                    *(int16_t *)dst =
                                        static_cast<int16_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
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
                        if (ctx.src_is_signed) {
                            int64_t src_val = *(int64_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 32767) {
                                        *(int16_t *)dst = 32767;
                                    } else if (src_val < -32768) {
                                        *(int16_t *)dst = -32768;
                                    } else {
                                        *(int16_t *)dst =
                                            static_cast<int16_t>(src_val);
                                    }
                                } else {
                                    *(int16_t *)dst =
                                        static_cast<int16_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
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
                            uint64_t src_val = *(uint64_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 32767) {
                                        *(int16_t *)dst = 32767;
                                    } else {
                                        *(int16_t *)dst =
                                            static_cast<int16_t>(src_val);
                                    }
                                } else {
                                    *(int16_t *)dst =
                                        static_cast<int16_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
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
            if (ctx.dst_is_float) {
                if (ctx.src_is_float) {
                    if (ctx.src_is_half) {
                        uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                        *(float *)dst = cvt_helpers::half_to_float(h_temp);
                    } else if (ctx.src_bytes == 4) {
                        *(float *)dst = *(float *)src;
                    } else {
                        *(float *)dst = (float)*(double *)src;
                    }
                } else {
                    if (ctx.src_bytes == 1) {
                        *(float *)dst = (float)*(int8_t *)src;
                    } else if (ctx.src_bytes == 2) {
                        *(float *)dst = (float)*(int16_t *)src;
                    } else if (ctx.src_bytes == 4) {
                        *(float *)dst = (float)*(int32_t *)src;
                    } else {
                        *(float *)dst = (float)*(int64_t *)src;
                    }
                }
            } else {
                if (ctx.src_is_float) {
                    float temp;
                    if (ctx.src_is_half) {
                        uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                        temp = cvt_helpers::half_to_float(h_temp);
                    } else if (ctx.src_bytes == 4) {
                        temp = *(float *)src;
                    } else {
                        temp = (float)*(double *)src;
                    }
                    if (ctx.has_sat) {
                        if (std::isnan(temp)) {
                            *(uint32_t *)dst = 0;
                        } else if (temp <= 0.0f) {
                            *(uint32_t *)dst = 0;
                        } else if (temp > 4294967295.0f) {
                            *(uint32_t *)dst = 4294967295U;
                        } else {
                            *(uint32_t *)dst = static_cast<uint32_t>(temp);
                        }
                    } else {
                        if (ctx.has_rni || ctx.has_rn) {
                            if (cvt_helpers::should_saturate_uint32(
                                    temp, 4294967295.5f)) {
                                *(uint32_t *)dst = 4294967295U;
                            } else {
                                *(uint32_t *)dst = static_cast<uint32_t>(
                                    cvt_helpers::round_half_to_even(
                                        static_cast<double>(temp)));
                            }
                        } else if (ctx.has_rzi || ctx.has_rz) {
                            if (cvt_helpers::should_saturate_uint32(
                                    temp, 4294967296.0f)) {
                                *(uint32_t *)dst = 4294967295U;
                            } else {
                                *(uint32_t *)dst = static_cast<uint32_t>(
                                    std::trunc(static_cast<double>(temp)));
                            }
                        } else if (ctx.has_rmi || ctx.has_rm) {
                            if (cvt_helpers::should_saturate_uint32(
                                    temp, 4294967296.0f)) {
                                *(uint32_t *)dst = 4294967295U;
                            } else {
                                *(uint32_t *)dst = static_cast<uint32_t>(
                                    std::floor(static_cast<double>(temp)));
                            }
                        } else if (ctx.has_rpi || ctx.has_rp) {
                            if (cvt_helpers::should_saturate_uint32(
                                    temp, 4294967295.0f)) {
                                *(uint32_t *)dst = 4294967295U;
                            } else {
                                *(uint32_t *)dst = static_cast<uint32_t>(
                                    std::ceil(static_cast<double>(temp)));
                            }
                        } else if (ctx.has_rna) {
                            float rounded = (temp >= 0.0f)
                                                ? std::floor(temp + 0.5f)
                                                : std::ceil(temp - 0.5f);
                            if (rounded < 0.0f) {
                                *(uint32_t *)dst = 0;
                            } else {
                                *(uint32_t *)dst =
                                    static_cast<uint32_t>(rounded);
                            }
                        } else {
                            *(uint32_t *)dst = static_cast<uint32_t>(temp);
                        }
                    }
                } else {
                    if (ctx.src_bytes == 1) {
                        if (ctx.src_is_signed) {
                            int8_t src_val = *(int8_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int32_t *)dst = src_val;
                            } else {
                                *(uint32_t *)dst =
                                    static_cast<uint32_t>(src_val);
                            }
                        } else {
                            uint8_t src_val = *(uint8_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int32_t *)dst = static_cast<int32_t>(src_val);
                            } else {
                                *(uint32_t *)dst = src_val;
                            }
                        }
                    } else if (ctx.src_bytes == 2) {
                        if (ctx.src_is_signed) {
                            int16_t src_val = *(int16_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int32_t *)dst = src_val;
                            } else {
                                *(uint32_t *)dst =
                                    static_cast<uint32_t>(src_val);
                            }
                        } else {
                            uint16_t src_val = *(uint16_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int32_t *)dst = static_cast<int32_t>(src_val);
                            } else {
                                *(uint32_t *)dst = src_val;
                            }
                        }
                    } else if (ctx.src_bytes == 4) {
                        if (ctx.src_is_signed) {
                            int32_t src_val = *(int32_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int32_t *)dst = src_val;
                            } else {
                                *(uint32_t *)dst =
                                    static_cast<uint32_t>(src_val);
                            }
                        } else {
                            uint32_t src_val = *(uint32_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int32_t *)dst = static_cast<int32_t>(src_val);
                            } else {
                                *(uint32_t *)dst = src_val;
                            }
                        }
                    } else {
                        if (ctx.src_is_signed) {
                            int64_t src_val = *(int64_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 2147483647LL) {
                                        *(int32_t *)dst = 2147483647;
                                    } else if (src_val < -2147483647LL - 1) {
                                        *(int32_t *)dst = -2147483647 - 1;
                                    } else {
                                        *(int32_t *)dst =
                                            static_cast<int32_t>(src_val);
                                    }
                                } else {
                                    *(int32_t *)dst =
                                        static_cast<int32_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
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
                            uint64_t src_val = *(uint64_t *)src;
                            if (ctx.dst_is_signed) {
                                if (ctx.has_sat) {
                                    if (src_val > 2147483647U) {
                                        *(int32_t *)dst = 2147483647;
                                    } else {
                                        *(int32_t *)dst =
                                            static_cast<int32_t>(src_val);
                                    }
                                } else {
                                    *(int32_t *)dst =
                                        static_cast<int32_t>(src_val);
                                }
                            } else {
                                if (ctx.has_sat) {
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
            if (ctx.dst_is_float) {
                if (ctx.src_is_float) {
                    if (ctx.src_is_half) {
                        uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                        *(double *)dst =
                            (double)cvt_helpers::half_to_float(h_temp);
                    } else if (ctx.src_bytes == 4) {
                        *(double *)dst = (double)*(float *)src;
                    } else {
                        *(double *)dst = *(double *)src;
                    }
                } else {
                    if (ctx.src_bytes == 1) {
                        *(double *)dst = (double)*(int8_t *)src;
                    } else if (ctx.src_bytes == 2) {
                        *(double *)dst = (double)*(int16_t *)src;
                    } else if (ctx.src_bytes == 4) {
                        *(double *)dst = (double)*(int32_t *)src;
                    } else {
                        *(double *)dst = (double)*(int64_t *)src;
                    }
                }
            } else {
                if (ctx.src_is_float) {
                    double temp;
                    if (ctx.src_is_half) {
                        uint16_t h_temp = *reinterpret_cast<uint16_t *>(src);
                        temp = (double)cvt_helpers::half_to_float(h_temp);
                    } else if (ctx.src_bytes == 4) {
                        temp = (double)*(float *)src;
                    } else {
                        temp = *(double *)src;
                    }
                    if (ctx.has_sat) {
                        if (std::isnan(temp)) {
                            *(uint64_t *)dst = 0;
                        } else if (temp <= 0.0) {
                            *(uint64_t *)dst = 0;
                        } else if (temp > 18446744073709551615.0) {
                            *(uint64_t *)dst = 18446744073709551615ULL;
                        } else {
                            *(uint64_t *)dst = static_cast<uint64_t>(temp);
                        }
                    } else {
                        if (ctx.has_rni || ctx.has_rn) {
                            *(uint64_t *)dst = static_cast<uint64_t>(
                                cvt_helpers::round_half_to_even(temp));
                        } else if (ctx.has_rzi || ctx.has_rz) {
                            *(uint64_t *)dst =
                                static_cast<uint64_t>(std::trunc(temp));
                        } else if (ctx.has_rmi || ctx.has_rm) {
                            *(uint64_t *)dst =
                                static_cast<uint64_t>(std::floor(temp));
                        } else if (ctx.has_rpi || ctx.has_rp) {
                            *(uint64_t *)dst =
                                static_cast<uint64_t>(std::ceil(temp));
                        } else if (ctx.has_rna) {
                            double rounded = (temp >= 0.0)
                                                 ? std::floor(temp + 0.5)
                                                 : std::ceil(temp - 0.5);
                            if (rounded < 0.0) {
                                *(uint64_t *)dst = 0;
                            } else {
                                *(uint64_t *)dst =
                                    static_cast<uint64_t>(rounded);
                            }
                        } else {
                            *(uint64_t *)dst = static_cast<uint64_t>(temp);
                        }
                    }
                } else {
                    if (ctx.src_bytes == 1) {
                        if (ctx.src_is_signed) {
                            int8_t src_val = *(int8_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int64_t *)dst = src_val;
                            } else {
                                *(uint64_t *)dst =
                                    static_cast<uint64_t>(src_val);
                            }
                        } else {
                            uint8_t src_val = *(uint8_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int64_t *)dst = static_cast<int64_t>(src_val);
                            } else {
                                *(uint64_t *)dst = src_val;
                            }
                        }
                    } else if (ctx.src_bytes == 2) {
                        if (ctx.src_is_signed) {
                            int16_t src_val = *(int16_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int64_t *)dst = src_val;
                            } else {
                                *(uint64_t *)dst =
                                    static_cast<uint64_t>(src_val);
                            }
                        } else {
                            uint16_t src_val = *(uint16_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int64_t *)dst = static_cast<int64_t>(src_val);
                            } else {
                                *(uint64_t *)dst = src_val;
                            }
                        }
                    } else if (ctx.src_bytes == 4) {
                        if (ctx.src_is_signed) {
                            int32_t src_val = *(int32_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int64_t *)dst = src_val;
                            } else {
                                *(uint64_t *)dst =
                                    static_cast<uint64_t>(src_val);
                            }
                        } else {
                            uint32_t src_val = *(uint32_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int64_t *)dst = static_cast<int64_t>(src_val);
                            } else {
                                *(uint64_t *)dst = src_val;
                            }
                        }
                    } else {
                        if (ctx.src_is_signed) {
                            int64_t src_val = *(int64_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int64_t *)dst = src_val;
                            } else {
                                *(uint64_t *)dst =
                                    static_cast<uint64_t>(src_val);
                            }
                        } else {
                            uint64_t src_val = *(uint64_t *)src;
                            if (ctx.dst_is_signed) {
                                *(int64_t *)dst = static_cast<int64_t>(src_val);
                            } else {
                                *(uint64_t *)dst = src_val;
                            }
                        }
                    }
                }
            }
            break;
        }
        default:
            throw UnsupportedInstructionException(
                "cvt", "unsupported destination size for CVT instruction");
        }
    }

    const char *name() const override { return "GeneralCvtStrategy"; }
};

const ConversionStrategy &select_strategy(const CvtContext &ctx) {
    static const FloatToFloatStrategy f2f;
    static const FloatToIntStrategy f2i;
    static const IntToFloatStrategy i2f;
    static const IntToIntStrategy i2i;

    if (ctx.dst_is_float) {
        return ctx.src_is_float ? static_cast<const ConversionStrategy &>(f2f)
                                : static_cast<const ConversionStrategy &>(i2f);
    }
    return ctx.src_is_float ? static_cast<const ConversionStrategy &>(f2i)
                            : static_cast<const ConversionStrategy &>(i2i);
}

} // namespace cvt_strategy
} // namespace ptxsim

void CvtHandler::processOperation(
    ThreadContext * /*context*/, void **operands,
    const std::vector<Qualifier> &qualifiers,
    const std::vector<char> * /*operand_is_immediate*/) {
    void *dst = operands[0];
    void *src = operands[1];

    auto ctx = ptxsim::cvt_strategy::build_context(qualifiers);
    const auto &strategy = ptxsim::cvt_strategy::select_strategy(ctx);
    strategy.convert(dst, src, ctx);
}
