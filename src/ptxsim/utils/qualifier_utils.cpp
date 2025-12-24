#include "ptxsim/utils/qualifier_utils.h"
#include "memory/memory_interface.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/utils/type_utils.h"
#include "utils/logger.h"
#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int Q2bytes(Qualifier q) {
    switch (q) {
    case Qualifier::Q_U64:
    case Qualifier::Q_S64:
    case Qualifier::Q_B64:
    case Qualifier::Q_F64:
        return 8;
    case Qualifier::Q_U32:
    case Qualifier::Q_S32:
    case Qualifier::Q_B32:
    case Qualifier::Q_F32:
        return 4;
    case Qualifier::Q_U16:
    case Qualifier::Q_S16:
    case Qualifier::Q_B16:
    case Qualifier::Q_F16:
        return 2;
    case Qualifier::Q_U8:
    case Qualifier::Q_S8:
    case Qualifier::Q_B8:
    case Qualifier::Q_PRED:
    case Qualifier::Q_F8:
        return 1;
    default:
        return 0;
    }
}

bool Signed(Qualifier q) {
    switch (q) {
    case Qualifier::Q_S64:
    case Qualifier::Q_S32:
    case Qualifier::Q_S16:
    case Qualifier::Q_S8:
        return true;
    default:
        return false;
    }
}

int getBytes(const std::vector<Qualifier> &q) {
    for (auto e : q) {
        int bytes = Q2bytes(e);
        if (bytes)
            return bytes;
    }
    return 0;
}

DTYPE getDType(std::vector<Qualifier> &q) {
    if (q.size() == 0)
        return DNONE;
    Qualifier e = q.back();
    switch (e) {
    case Qualifier::Q_F64:
    case Qualifier::Q_F32:
    case Qualifier::Q_F16:
    case Qualifier::Q_F8:
        return DFLOAT;
    case Qualifier::Q_U64:
    case Qualifier::Q_U32:
    case Qualifier::Q_U16:
    case Qualifier::Q_U8:
    case Qualifier::Q_S64:
    case Qualifier::Q_S32:
    case Qualifier::Q_S16:
    case Qualifier::Q_S8:
    case Qualifier::Q_B64:
    case Qualifier::Q_B32:
    case Qualifier::Q_B16:
    case Qualifier::Q_B8:
        return DINT;
    default:
        return DNONE;
    }
}

DTYPE getDType(Qualifier q) {
    switch (q) {
    case Qualifier::Q_F64:
    case Qualifier::Q_F32:
    case Qualifier::Q_F16:
    case Qualifier::Q_F8:
        return DFLOAT;
    case Qualifier::Q_U64:
    case Qualifier::Q_U32:
    case Qualifier::Q_U16:
    case Qualifier::Q_U8:
    case Qualifier::Q_S64:
    case Qualifier::Q_S32:
    case Qualifier::Q_S16:
    case Qualifier::Q_S8:
    case Qualifier::Q_B64:
    case Qualifier::Q_B32:
    case Qualifier::Q_B16:
    case Qualifier::Q_B8:
        return DINT;
    default:
        return DNONE;
    }
}

Qualifier getDataQualifier(const std::vector<Qualifier> &qualifiers) {
    for (const auto &q : qualifiers) {
        if (Q2bytes(q))
            return q;
    }
    assert(0);
    return Qualifier::S_UNKNOWN; // 添加默认返回值
}

Qualifier getCmpOpQualifier(const std::vector<Qualifier> &qualifiers) {
    for (auto e : qualifiers) {
        switch (e) {
        case Qualifier::Q_EQ:
        case Qualifier::Q_NE:
        case Qualifier::Q_LT:
        case Qualifier::Q_LE:
        case Qualifier::Q_GT:
        case Qualifier::Q_GE:
        case Qualifier::Q_LO:
        case Qualifier::Q_HI:
        case Qualifier::Q_LTU:
        case Qualifier::Q_LEU:
        case Qualifier::Q_GEU:
        case Qualifier::Q_NEU:
        case Qualifier::Q_GTU:
            return e;
        }
    }
    return Qualifier::S_UNKNOWN;
}

void splitDstSrcQualifiers(const std::vector<Qualifier> &qualifiers,
                           std::vector<Qualifier> &dst_qualifiers,
                           std::vector<Qualifier> &src_qualifiers) {
    dst_qualifiers.clear();
    src_qualifiers.clear();

    bool found_dst = false;
    bool found_src = false;

    // 遍历限定符，分离目标和源限定符
    for (const auto &q : qualifiers) {
        int bytes = Q2bytes(q);

        // 如果这个限定符代表一种数据类型
        if (bytes > 0) {
            // 第一个遇到的数据类型通常是目标类型
            if (!found_dst) {
                dst_qualifiers.push_back(q);
                found_dst = true;
            }
            // 第二个遇到的数据类型通常是源类型
            else if (!found_src) {
                src_qualifiers.push_back(q);
                found_src = true;

                // 找到了两个类型，可以退出循环
                if (found_dst && found_src) {
                    break;
                }
            }
        } else {
            // 其他限定符（如舍入模式）添加到两个向量中
            dst_qualifiers.push_back(q);
            src_qualifiers.push_back(q);
        }
    }
}

// 实现获取地址空间的辅助函数
MemorySpace getAddressSpace(const std::vector<Qualifier> &qualifiers) {
    for (const auto &qual : qualifiers) {
        switch (qual) {
        case Qualifier::Q_GLOBAL:
            return MemorySpace::GLOBAL;
        case Qualifier::Q_SHARED:
            return MemorySpace::SHARED;
        case Qualifier::Q_LOCAL:
            return MemorySpace::LOCAL;
        case Qualifier::Q_CONST:
            return MemorySpace::CONST;
        case Qualifier::Q_PARAM:
            return MemorySpace::PARAM;
        default:
            continue;
        }
    }
    // 默认返回GLOBAL空间
    return MemorySpace::GLOBAL;
}

// 解析立即数到缓冲区
void parseImmediate(const std::string &s, Qualifier q, void *out) {
    if (!out)
        return;

    // 清理字符串（移除空格）
    std::string clean = s;
    clean.erase(std::remove_if(clean.begin(), clean.end(), ::isspace),
                clean.end());
    if (clean.empty()) {
        std::vector<Qualifier> q_vec = {q};
        memset(out, 0, getBytes(q_vec));
        return;
    }

    try {
        switch (q) {
        case Qualifier::Q_F32: {
            float *dst = static_cast<float *>(out);
            if (clean.find("0x") == 0 || clean.find("0X") == 0) {
                // 检查是否为标准十六进制浮点数格式（如 0x1.0p0）
                if (clean.find('.') != std::string::npos ||
                    clean.find('p') != std::string::npos) {
                    // 使用标准库函数处理十六进制浮点数
                    *dst =
                        static_cast<float>(std::strtof(clean.c_str(), nullptr));
                } else {
                    // 处理标准十六进制格式，如 0x1234abcd
                    uint32_t bits =
                        static_cast<uint32_t>(std::stoul(clean, nullptr, 16));
                    *dst = *reinterpret_cast<float *>(&bits);
                }
            } else if (clean.size() >= 2 &&
                       (clean[1] == 'f' || clean[1] == 'F')) {
                // 处理PTX特有格式 0f1234，转换为十六进制浮点
                uint32_t bits = static_cast<uint32_t>(
                    std::stoul("0x" + clean.substr(2), nullptr, 16));
                *dst = *reinterpret_cast<float *>(&bits);
            } else {
                *dst = std::stof(clean);
            }
            break;
        }
        case Qualifier::Q_F64: {
            double *dst = static_cast<double *>(out);
            if (clean.find("0x") == 0 || clean.find("0X") == 0) {
                // 检查是否为标准十六进制浮点数格式（如 0x1.0p0）
                if (clean.find('.') != std::string::npos ||
                    clean.find('p') != std::string::npos) {
                    // 使用标准库函数处理十六进制浮点数
                    *dst = std::strtod(clean.c_str(), nullptr);
                } else {
                    // 处理标准十六进制格式
                    uint64_t bits = std::stoull(clean, nullptr, 16);
                    *dst = *reinterpret_cast<double *>(&bits);
                }
            } else if (clean.size() >= 2 &&
                       (clean[1] == 'd' || clean[1] == 'D')) {
                // 处理PTX特有格式 0d1234
                uint64_t bits =
                    std::stoull("0x" + clean.substr(2), nullptr, 16);
                *dst = *reinterpret_cast<double *>(&bits);
            } else {
                *dst = std::stod(clean);
            }
            break;
        }
        // 整型统一用 stoll + 截断
        case Qualifier::Q_S64:
        case Qualifier::Q_U64:
        case Qualifier::Q_B64: {
            int64_t val = std::stoll(clean, nullptr, 0);
            *static_cast<uint64_t *>(out) = static_cast<uint64_t>(val);
            break;
        }
        case Qualifier::Q_S32:
        case Qualifier::Q_U32:
        case Qualifier::Q_B32: {
            int64_t val = std::stoll(clean, nullptr, 0); // 使用stoll防止溢出
            *static_cast<uint32_t *>(out) = static_cast<uint32_t>(val);
            break;
        }
        case Qualifier::Q_S16:
        case Qualifier::Q_U16:
        case Qualifier::Q_B16: {
            int64_t val = std::stoll(clean, nullptr, 0);
            *static_cast<uint16_t *>(out) = static_cast<uint16_t>(val);
            break;
        }
        case Qualifier::Q_S8:
        case Qualifier::Q_U8:
        case Qualifier::Q_B8: {
            int64_t val = std::stoll(clean, nullptr, 0);
            *static_cast<uint8_t *>(out) = static_cast<uint8_t>(val);
            break;
        }
        case Qualifier::Q_PRED: {
            // 谓词类型应为1-bit，但存储为uint8_t
            bool val = (std::stoll(clean, nullptr, 0) != 0);
            *static_cast<uint8_t *>(out) = static_cast<uint8_t>(val);
            break;
        }
        default:
            // 未知类型，清零
            PTX_WARN_EMU("Unsupported immediate qualifier: %s, zeroing value",
                         Q2s(q).c_str());
            {
                std::vector<Qualifier> q_vec = {q};
                memset(out, 0, getBytes(q_vec));
            }
            return;
        }
    } catch (const std::exception &e) {
        PTX_ERROR_EMU("Failed to parse immediate value '%s' as %s: %s",
                      clean.c_str(), Q2s(q).c_str(), e.what());
        {
            std::vector<Qualifier> q_vec = {q};
            memset(out, 0, getBytes(q_vec));
        }
    }
}

bool QvecHasQ(const std::vector<Qualifier> &qvec, Qualifier q) {
    return std::find(qvec.begin(), qvec.end(), q) != qvec.end();
}
