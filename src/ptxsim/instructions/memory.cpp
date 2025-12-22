#include "memory/memory_manager.h" // 确保包含 MemoryManager
#include "ptxsim/instruction_handlers_decl.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void MOV::process_operation(ThreadContext *context, void *op[2],
                            std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];

    context->mov(src, dst, qualifiers);
}

void CVTA::process_operation(ThreadContext *context, void *op[2],
                             std::vector<Qualifier> &qualifiers) {
    void *to = op[0];
    void *from = op[1];

    // context->mov(from, to, qualifier);
    //  空指针检查
    if (!to || !from) {
        std::cerr << "Error: Null pointer in CVTA instruction" << std::endl;
        return;
    }

    // CVTA 是指针赋值：*to = *(void**)from
    // 即：将 from 指向的指针值，写入 to 指向的位置
    *(void **)to = *(void **)from;
}

void CVT::process_operation(ThreadContext *context, void *op[2],
                            std::vector<Qualifier> &qualifiers) {
    void *dst = op[0];
    void *src = op[1];
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
// // 2. 推断内存空间和大小（从 PTX 修饰符）
// MemorySpace space = get_space_from_qualifiers(quals); // e.g., .global
// size_t size = get_size_from_qualifiers(quals);        // e.g., 4 bytes
// for .b32

// // 3. 调用统一接口（不关心底层实现！）
// MemoryManager->access(space, addr, dst, size, false);

void LD::process_operation(ThreadContext *context, void *op[2],
                           std::vector<Qualifier> &qualifier) {
    void *dst = op[0];
    void *host_ptr = op[1]; // ← 这是 cudaMalloc 返回的主机指针

    // 空指针检查
    if (!dst || !host_ptr) {
        std::cerr << "Error: Null pointer in LD instruction" << std::endl;
        return;
    }

    // 获取单个元素大小
    size_t data_size = TypeUtils::get_bytes(qualifier);

    // ========================
    // 1. 标量 LD（无向量）
    // ========================
    if (!context->QvecHasQ(qualifier, Qualifier::Q_V2) &&
        !context->QvecHasQ(qualifier, Qualifier::Q_V4)) {
        // 单次内存读取
        MemoryManager::instance().access(host_ptr, dst, data_size,
                                         /*is_write=*/false);
        return;
    }

    // ========================
    // 2. 向量 LD（V2/V4）
    // ========================
    size_t step = context->getBytes(qualifier); // 元素步长
    auto vecAddr = context->vec.front()->vec;
    context->vec.pop();

    size_t vec_size = 0;
    if (context->QvecHasQ(qualifier, Qualifier::Q_V2)) {
        vec_size = 2;
        assert(vecAddr.size() == 2);
    } else if (context->QvecHasQ(qualifier, Qualifier::Q_V4)) {
        vec_size = 4;
        assert(vecAddr.size() == 4);
    }

    // 逐元素读取
    for (size_t i = 0; i < vec_size; ++i) {
        void *element_dst = vecAddr[i];
        uint64_t element_host_ptr =
            reinterpret_cast<uint64_t>(host_ptr) + i * step;

        MemoryManager::instance().access(
            reinterpret_cast<void *>(element_host_ptr), element_dst, data_size,
            /*is_write=*/false);
    }
}

void ST::process_operation(ThreadContext *context, void *op[2],
                           std::vector<Qualifier> &qualifiers) {
    void *host_ptr = op[0]; // ← 目标地址：cudaMalloc 返回的主机指针
    void *src = op[1];      // ← 源数据：寄存器或立即数地址

    // 空指针检查
    if (!host_ptr || !src) {
        std::cerr << "Error: Null pointer in ST instruction" << std::endl;
        return;
    }

    // 获取单个元素大小
    size_t data_size = TypeUtils::get_bytes(qualifiers);

    // ========================
    // 1. 标量 ST（无向量）
    // ========================
    if (!context->QvecHasQ(qualifiers, Qualifier::Q_V2) &&
        !context->QvecHasQ(qualifiers, Qualifier::Q_V4)) {
        // 单次内存写入
        MemoryManager::instance().access(host_ptr, src, data_size,
                                         /*is_write=*/true);
        return;
    }

    // ========================
    // 2. 向量 ST（V2/V4）
    // ========================
    size_t step = context->getBytes(qualifiers); // 元素步长
    auto vecAddr = context->vec.front()->vec;
    context->vec.pop();

    size_t vec_size = 0;
    if (context->QvecHasQ(qualifiers, Qualifier::Q_V2)) {
        vec_size = 2;
        assert(vecAddr.size() == 2);
    } else if (context->QvecHasQ(qualifiers, Qualifier::Q_V4)) {
        vec_size = 4;
        assert(vecAddr.size() == 4);
    }

    // 逐元素写入
    for (size_t i = 0; i < vec_size; ++i) {
        void *element_src = vecAddr[i];
        uint64_t element_host_ptr =
            reinterpret_cast<uint64_t>(host_ptr) + i * step;

        MemoryManager::instance().access(
            reinterpret_cast<void *>(element_host_ptr), element_src, data_size,
            /*is_write=*/true);
    }
}