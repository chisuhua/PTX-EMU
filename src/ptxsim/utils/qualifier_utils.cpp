#include "ptxsim/utils/qualifier_utils.h"
#include "ptx_ir/ptx_types.h"

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

int getBytes(std::vector<Qualifier> &q) {
    for (auto e : q) {
        int bytes = Q2bytes(e);
        if (bytes)
            return bytes;
    }
    return 0;
}

DTYPE getDType(std::vector<Qualifier> &q) {
    for (auto e : q) {
        switch (e) {
        case Qualifier::Q_F64:
        case Qualifier::Q_F32:
        case Qualifier::Q_F16:
        case Qualifier::Q_F8:
            return DFLOAT;
        case Qualifier::Q_S64:
        case Qualifier::Q_B64:
        case Qualifier::Q_U64:
        case Qualifier::Q_S32:
        case Qualifier::Q_B32:
        case Qualifier::Q_U32:
        case Qualifier::Q_S16:
        case Qualifier::Q_B16:
        case Qualifier::Q_U16:
        case Qualifier::Q_S8:
        case Qualifier::Q_B8:
        case Qualifier::Q_U8:
        case Qualifier::Q_PRED:
            return DINT;
        case Qualifier::Q_V2:
        case Qualifier::Q_V4:
        case Qualifier::Q_CONST:
        case Qualifier::Q_PARAM:
        case Qualifier::Q_GLOBAL:
        case Qualifier::Q_LOCAL:
        case Qualifier::Q_SHARED:
        case Qualifier::Q_GT:
        case Qualifier::Q_GE:
        case Qualifier::Q_EQ:
        case Qualifier::Q_NE:
        case Qualifier::Q_LT:
        case Qualifier::Q_TO:
        case Qualifier::Q_WIDE:
        case Qualifier::Q_SYNC:
        case Qualifier::Q_LO:
        case Qualifier::Q_HI:
        case Qualifier::Q_UNI:
        case Qualifier::Q_RN:
        case Qualifier::Q_A:
        case Qualifier::Q_B:
        case Qualifier::Q_D:
        case Qualifier::Q_ROW:
        case Qualifier::Q_ALIGNED:
        case Qualifier::Q_M8N8K4:
        case Qualifier::Q_M16N16K16:
        case Qualifier::Q_NEU:
        case Qualifier::Q_NC:
        case Qualifier::Q_FTZ:
        case Qualifier::Q_APPROX:
        case Qualifier::Q_LTU:
        case Qualifier::Q_LE:
        case Qualifier::Q_GTU:
        case Qualifier::Q_LEU:
        case Qualifier::Q_DOTADD:
        case Qualifier::Q_GEU:
        case Qualifier::Q_RZI:
        case Qualifier::Q_DOTOR:
        case Qualifier::Q_SAT:
        case Qualifier::S_UNKNOWN:
            break;
        }
    }
    return DNONE;
}

DTYPE getDType(Qualifier q) {
    switch (q) {
    case Qualifier::Q_F64:
    case Qualifier::Q_F32:
    case Qualifier::Q_F16:
    case Qualifier::Q_F8:
        return DFLOAT;
    case Qualifier::Q_S64:
    case Qualifier::Q_B64:
    case Qualifier::Q_U64:
    case Qualifier::Q_S32:
    case Qualifier::Q_B32:
    case Qualifier::Q_U32:
    case Qualifier::Q_S16:
    case Qualifier::Q_B16:
    case Qualifier::Q_U16:
    case Qualifier::Q_S8:
    case Qualifier::Q_B8:
    case Qualifier::Q_U8:
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
}

void splitCvtQualifiers(const std::vector<Qualifier> &qualifiers,
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

void splitDstSrcQualifiers(const std::vector<Qualifier> &qualifiers,
                           std::vector<Qualifier> &dst_qualifiers,
                           std::vector<Qualifier> &src1_qualifiers,
                           std::vector<Qualifier> &src2_qualifiers) {
    dst_qualifiers.clear();
    src1_qualifiers.clear();
    src2_qualifiers.clear();

    int data_types_found = 0;

    // 遍历限定符，分离目标和源限定符
    for (const auto &q : qualifiers) {
        int bytes = Q2bytes(q);

        // 如果这个限定符代表一种数据类型
        if (bytes > 0) {
            data_types_found++;
            switch (data_types_found) {
            case 1: // 目标操作数类型
                dst_qualifiers.push_back(q);
                break;
            case 2: // 第一个源操作数类型
                src1_qualifiers.push_back(q);
                break;
            case 3: // 第二个源操作数类型
                src2_qualifiers.push_back(q);
                break;
            }
        } else {
            // 其他限定符添加到所有向量中
            dst_qualifiers.push_back(q);
            src1_qualifiers.push_back(q);
            src2_qualifiers.push_back(q);
        }
    }
}

void splitDstSrcQualifiers(const std::vector<Qualifier> &qualifiers,
                           std::vector<Qualifier> &dst_qualifiers,
                           std::vector<Qualifier> &src2_qualifiers) {
    dst_qualifiers.clear();
    src2_qualifiers.clear();

    int data_types_found = 0;

    // 遍历限定符，分离目标和源限定符
    for (const auto &q : qualifiers) {
        int bytes = Q2bytes(q);

        // 如果这个限定符代表一种数据类型
        if (bytes > 0) {
            data_types_found++;
            switch (data_types_found) {
            case 1: // 目标操作数类型
                dst_qualifiers.push_back(q);
                break;
            case 2: // 第二个源操作数类型
                src2_qualifiers.push_back(q);
                break;
            }
        } else {
            // 其他限定符添加到所有向量中
            dst_qualifiers.push_back(q);
            src2_qualifiers.push_back(q);
        }
    }
}
