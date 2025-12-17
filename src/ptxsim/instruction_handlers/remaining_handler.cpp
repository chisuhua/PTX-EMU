#include "ptxsim/instruction_handlers/remaining_handler.h"
#include "ptxsim/instruction_processor_utils.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

// 定义四个操作数的处理宏
#define PROCESS_OPERATION_4(context, dst, src1, src2, pred, qualifiers, reg)   \
    do {                                                                       \
        process_operation(context, dst, src1, src2, pred, qualifiers);         \
        if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,    \
                                                   "reg")) {                   \
            if (reg)                                                           \
                context->trace_register(reg, dst, qualifiers, true);           \
        }                                                                      \
    } while (0)

void CvtHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::CVT *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->cvtOp[0], ss->cvtQualifier);
    void *from = context->get_operand_addr(ss->cvtOp[1], ss->cvtQualifier);

    // 执行CVT操作并根据日志级别决定是否跟踪寄存器更新
    PROCESS_OPERATION_2(context, to, from, ss->cvtQualifier,
                        (OperandContext::REG *)ss->cvtOp[0].operand);
}

void CvtHandler::process_operation(ThreadContext *context, void *dst, void *src,
                                   std::vector<Qualifier> &qualifiers) {
    // 分离目标和源限定符
    std::vector<Qualifier> dst_qualifiers, src_qualifiers;
    splitDstSrcQualifiers(qualifiers, dst_qualifiers, src_qualifiers);

    // 使用TypeUtils函数获取目标和源的字节大小以及是否为浮点类型
    int dst_bytes = TypeUtils::get_bytes(dst_qualifiers);
    int src_bytes = TypeUtils::get_bytes(src_qualifiers);
    bool dst_is_float = TypeUtils::is_float_type(dst_qualifiers);
    bool src_is_float = TypeUtils::is_float_type(src_qualifiers);

    // 如果没有正确识别出类型，使用默认方法
    if (dst_bytes == 0) {
        dst_bytes = TypeUtils::get_bytes(qualifiers);
    }

    if (src_bytes == 0) {
        src_bytes = context->getBytes(qualifiers);
    }

    bool has_sat = context->QvecHasQ(qualifiers, Qualifier::Q_SAT);

    // 根据目标数据大小执行转换
    switch (dst_bytes) {
    case 1: { // 8-bit
        if (dst_is_float) {
            // 目标是浮点型
            float temp;
            if (src_is_float) {
                if (src_bytes == 4) {
                    temp = *(float *)src;
                } else {
                    temp = (float)*(double *)src;
                }
            } else {
                // 源是整型
                if (src_bytes == 1) {
                    temp = (float)*(uint8_t *)src;
                } else if (src_bytes == 2) {
                    temp = (float)*(uint16_t *)src;
                } else if (src_bytes == 4) {
                    temp = (float)*(uint32_t *)src;
                } else {
                    temp = (float)*(uint64_t *)src;
                }
            }

            if (has_sat) {
                if (std::isnan(temp)) {
                    *(uint8_t *)dst = 0;
                } else if (temp < 0.0f) {
                    *(uint8_t *)dst = 0;
                } else if (temp > 255.0f) {
                    *(uint8_t *)dst = 255;
                } else {
                    *(uint8_t *)dst = (uint8_t)temp;
                }
            } else {
                *(uint8_t *)dst = (uint8_t)temp;
            }
        } else {
            // 目标是整型
            if (src_is_float) {
                float temp;
                if (src_bytes == 4) {
                    temp = *(float *)src;
                } else {
                    temp = (float)*(double *)src;
                }

                if (has_sat) {
                    if (std::isnan(temp)) {
                        *(uint8_t *)dst = 0;
                    } else if (temp < 0.0f) {
                        *(uint8_t *)dst = 0;
                    } else if (temp > 255.0f) {
                        *(uint8_t *)dst = 255;
                    } else {
                        *(uint8_t *)dst = (uint8_t)temp;
                    }
                } else {
                    *(uint8_t *)dst = (uint8_t)temp;
                }
            } else {
                // 整数到整数转换
                if (src_bytes == 1) {
                    *(uint8_t *)dst = *(uint8_t *)src;
                } else if (src_bytes == 2) {
                    *(uint8_t *)dst = (uint8_t) * (uint16_t *)src;
                } else if (src_bytes == 4) {
                    *(uint8_t *)dst = (uint8_t) * (uint32_t *)src;
                } else {
                    *(uint8_t *)dst = (uint8_t) * (uint64_t *)src;
                }
            }
        }
        break;
    }
    case 2: { // 16-bit
        if (dst_is_float) {
            // 目标是浮点型
            float temp;
            if (src_is_float) {
                if (src_bytes == 4) {
                    temp = *(float *)src;
                } else {
                    temp = (float)*(double *)src;
                }
            } else {
                // 源是整型
                if (src_bytes == 1) {
                    temp = (float)*(uint8_t *)src;
                } else if (src_bytes == 2) {
                    temp = (float)*(uint16_t *)src;
                } else if (src_bytes == 4) {
                    temp = (float)*(uint32_t *)src;
                } else {
                    temp = (float)*(uint64_t *)src;
                }
            }

            if (has_sat) {
                if (std::isnan(temp)) {
                    *(uint16_t *)dst = 0;
                } else if (temp < 0.0f) {
                    *(uint16_t *)dst = 0;
                } else if (temp > 65535.0f) {
                    *(uint16_t *)dst = 65535;
                } else {
                    *(uint16_t *)dst = (uint16_t)temp;
                }
            } else {
                *(uint16_t *)dst = (uint16_t)temp;
            }
        } else {
            // 目标是整型
            if (src_is_float) {
                float temp;
                if (src_bytes == 4) {
                    temp = *(float *)src;
                } else {
                    temp = (float)*(double *)src;
                }

                if (has_sat) {
                    if (std::isnan(temp)) {
                        *(uint16_t *)dst = 0;
                    } else if (temp < 0.0f) {
                        *(uint16_t *)dst = 0;
                    } else if (temp > 65535.0f) {
                        *(uint16_t *)dst = 65535;
                    } else {
                        *(uint16_t *)dst = (uint16_t)temp;
                    }
                } else {
                    *(uint16_t *)dst = (uint16_t)temp;
                }
            } else {
                // 整数到整数转换
                if (src_bytes == 1) {
                    *(uint16_t *)dst = (uint16_t) * (uint8_t *)src;
                } else if (src_bytes == 2) {
                    *(uint16_t *)dst = *(uint16_t *)src;
                } else if (src_bytes == 4) {
                    *(uint16_t *)dst = (uint16_t) * (uint32_t *)src;
                } else {
                    *(uint16_t *)dst = (uint16_t) * (uint64_t *)src;
                }
            }
        }
        break;
    }
    case 4: { // 32-bit
        if (dst_is_float) {
            // 目标是浮点型 (float)
            if (src_is_float) {
                if (src_bytes == 4) {
                    *(float *)dst = *(float *)src;
                } else {
                    *(float *)dst = (float)*(double *)src;
                }
            } else {
                // 源是整型
                if (src_bytes == 1) {
                    *(float *)dst = (float)*(uint8_t *)src;
                } else if (src_bytes == 2) {
                    *(float *)dst = (float)*(uint16_t *)src;
                } else if (src_bytes == 4) {
                    *(float *)dst = (float)*(uint32_t *)src;
                } else {
                    *(float *)dst = (float)*(uint64_t *)src;
                }
            }
        } else {
            // 目标是整型
            if (src_is_float) {
                float temp;
                if (src_bytes == 4) {
                    temp = *(float *)src;
                } else {
                    temp = (float)*(double *)src;
                }

                if (has_sat) {
                    if (std::isnan(temp)) {
                        *(uint32_t *)dst = 0;
                    } else if (temp < 0.0f) {
                        *(uint32_t *)dst = 0;
                    } else if (temp > 4294967295.0f) {
                        *(uint32_t *)dst = 4294967295U;
                    } else {
                        *(uint32_t *)dst = (uint32_t)temp;
                    }
                } else {
                    *(uint32_t *)dst = (uint32_t)temp;
                }
            } else {
                // 整数到整数转换
                if (src_bytes == 1) {
                    *(uint32_t *)dst = (uint32_t) * (uint8_t *)src;
                } else if (src_bytes == 2) {
                    *(uint32_t *)dst = (uint32_t) * (uint16_t *)src;
                } else if (src_bytes == 4) {
                    *(uint32_t *)dst = *(uint32_t *)src;
                } else {
                    *(uint32_t *)dst = (uint32_t) * (uint64_t *)src;
                }
            }
        }
        break;
    }
    case 8: { // 64-bit
        if (dst_is_float) {
            // 目标是双精度浮点型 (double)
            if (src_is_float) {
                if (src_bytes == 4) {
                    *(double *)dst = (double)*(float *)src;
                } else {
                    *(double *)dst = *(double *)src;
                }
            } else {
                // 源是整型
                if (src_bytes == 1) {
                    *(double *)dst = (double)*(uint8_t *)src;
                } else if (src_bytes == 2) {
                    *(double *)dst = (double)*(uint16_t *)src;
                } else if (src_bytes == 4) {
                    *(double *)dst = (double)*(uint32_t *)src;
                } else {
                    *(double *)dst = (double)*(uint64_t *)src;
                }
            }
        } else {
            // 目标是整型
            if (src_is_float) {
                double temp;
                if (src_bytes == 4) {
                    temp = (double)*(float *)src;
                } else {
                    temp = *(double *)src;
                }

                if (has_sat) {
                    if (std::isnan(temp)) {
                        *(uint64_t *)dst = 0;
                    } else if (temp < 0.0) {
                        *(uint64_t *)dst = 0;
                    } else if (temp > 18446744073709551615.0) {
                        *(uint64_t *)dst = 18446744073709551615ULL;
                    } else {
                        *(uint64_t *)dst = (uint64_t)temp;
                    }
                } else {
                    *(uint64_t *)dst = (uint64_t)temp;
                }
            } else {
                // 整数到整数转换
                if (src_bytes == 1) {
                    *(uint64_t *)dst = (uint64_t) * (uint8_t *)src;
                } else if (src_bytes == 2) {
                    *(uint64_t *)dst = (uint64_t) * (uint16_t *)src;
                } else if (src_bytes == 4) {
                    *(uint64_t *)dst = (uint64_t) * (uint32_t *)src;
                } else {
                    *(uint64_t *)dst = *(uint64_t *)src;
                }
            }
        }
        break;
    }
    default:
        assert(0 && "Unsupported destination size for CVT instruction");
    }
}

void CvtaHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::CVTA *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->cvtaOp[0], ss->cvtaQualifier);
    void *from = context->get_operand_addr(ss->cvtaOp[1], ss->cvtaQualifier);

    // 执行CVTA操作，本质上是地址复制
    context->mov(from, to, ss->cvtaQualifier);

    // 如果启用了寄存器跟踪，则更新寄存器信息
    if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,
                                               "reg")) {
        if (ss->cvtaOp[0].operandType == OperandType::O_REG) {
            context->trace_register(
                (OperandContext::REG *)ss->cvtaOp[0].operand, to,
                ss->cvtaQualifier, true);
        }
    }
}

void SelpHandler::execute(ThreadContext *context, StatementContext &stmt) {
    auto ss = (StatementContext::SELP *)stmt.statement;

    // 获取操作数地址
    void *to = context->get_operand_addr(ss->selpOp[0], ss->selpQualifier);
    void *op0 = context->get_operand_addr(ss->selpOp[1], ss->selpQualifier);
    void *op1 = context->get_operand_addr(ss->selpOp[2], ss->selpQualifier);
    void *pred = context->get_operand_addr(ss->selpOp[3], ss->selpQualifier);

    // 执行SELP操作
    process_operation(context, to, op0, op1, pred, ss->selpQualifier);

    // 如果启用了寄存器跟踪，则更新寄存器信息
    if (ptxsim::LoggerConfig::get().is_enabled(ptxsim::log_level::info,
                                               "reg")) {
        if (ss->selpOp[0].operandType == OperandType::O_REG) {
            context->trace_register(
                (OperandContext::REG *)ss->selpOp[0].operand, to,
                ss->selpQualifier, true);
        }
    }
}

void SelpHandler::process_operation(ThreadContext *context, void *dst,
                                    void *src1, void *src2, void *pred,
                                    std::vector<Qualifier> &qualifiers) {
    int len = TypeUtils::get_bytes(qualifiers);
    DTYPE dtype = getDType(qualifiers);

    switch (len) {
    case 1:
        assert(dtype == DINT);
        _selp<uint8_t>(dst, src1, src2, pred);
        return;
    case 2:
        assert(dtype == DINT);
        _selp<uint16_t>(dst, src1, src2, pred);
        return;
    case 4:
        switch (dtype) {
        case DINT:
            _selp<uint32_t>(dst, src1, src2, pred);
            return;
        case DFLOAT:
            _selp<float>(dst, src1, src2, pred);
            return;
        default:
            assert(0);
        }
        return;
    case 8:
        switch (dtype) {
        case DINT:
            _selp<uint64_t>(dst, src1, src2, pred);
            return;
        case DFLOAT:
            _selp<double>(dst, src1, src2, pred);
            return;
        default:
            assert(0);
        }
    default:
        assert(0);
    }
}

template <typename T>
void SelpHandler::_selp(void *to, void *op0, void *op1, void *pred) {
    *(T *)to = *(uint8_t *)pred ? *(T *)op0 : *(T *)op1;
}

void NotHandler::execute(ThreadContext *context, StatementContext &stmt) {
    // TODO: 实现NOT指令
}

void RemHandler::execute(ThreadContext *context, StatementContext &stmt) {
    // TODO: 实现REM指令
}

void RsqrtHandler::execute(ThreadContext *context, StatementContext &stmt) {
    // TODO: 实现RSQRT指令
}

void Lg2Handler::execute(ThreadContext *context, StatementContext &stmt) {
    // TODO: 实现LG2指令
}

void Ex2Handler::execute(ThreadContext *context, StatementContext &stmt) {
    // TODO: 实现EX2指令
}

void WmmaHandler::execute(ThreadContext *context, StatementContext &stmt) {
    // TODO: 实现WMMA指令
}