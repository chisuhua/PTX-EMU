#include "ptx_ir/ptx_types.h"
#include "ptx_ir/operand_context.h"
#include <cmath>
#include <cstdlib>
#include <string>

void extractREG(std::string s, int &idx, std::string &name) {
    // Handle special registers like %tid.x, %ctaid.x, etc.
    size_t dotPos = s.find('.');
    if (dotPos != std::string::npos) {
        name = s; // Keep the full name including the dot part
        idx = 0;
        return;
    }

    // Handle regular registers like %r1, %rd2, etc.
    int ret = 0;
    for (char c : s) {
        if (c >= '0' && c <= '9') {
            ret = ret * 10 + c - '0';
        }
    }
    idx = ret;
    for (int i = 0; i < s.size(); i++) {
        if ((s[i] >= '0' && s[i] <= '9')) {
            name = s.substr(0, i);
            return;
        }
    }
    idx = -1;
    name = s;
}

std::string Q2s(Qualifier q) {
    switch (q) {
#define X(enum_val, enum_name, str_val)                                        \
    case Qualifier::enum_val:                                                  \
        return str_val;
#include "ptx_ir/ptx_qualifier.def"
#undef X
    case Qualifier::Q_UNKNOWN:
        return "";
    default:
        assert(0 && "Unsupported qualifier");
        return "";
    }
}

int Q2bytes(Qualifier q) {
    switch (q) {
    case Qualifier::Q_U8:
    case Qualifier::Q_B8:
    case Qualifier::Q_S8:
    case Qualifier::Q_F8:
    case Qualifier::Q_E4M3:
    case Qualifier::Q_E5M2:
        return 1;
    case Qualifier::Q_U16:
    case Qualifier::Q_B16:
    case Qualifier::Q_S16:
    case Qualifier::Q_F16:
    case Qualifier::Q_E4M3X4:
        return 2;
    case Qualifier::Q_U32:
    case Qualifier::Q_B32:
    case Qualifier::Q_S32:
    case Qualifier::Q_F32:
        return 4;
    case Qualifier::Q_U64:
    case Qualifier::Q_B64:
    case Qualifier::Q_S64:
    case Qualifier::Q_F64:
        return 8;
    case Qualifier::Q_PRED:
        return 1; // Assume predicate is 1 byte
    default:
        assert(0 && "Unsupported qualifier for byte calculation");
        return 4; // Default return 4 bytes
    }
}
