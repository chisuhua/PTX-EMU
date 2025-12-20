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
    name = s;
}

std::string Q2s(Qualifier q) {
    switch (q) {
    case Qualifier::Q_U64:
        return ".u64";
    case Qualifier::Q_U32:
        return ".u32";
    case Qualifier::Q_U16:
        return ".u16";
    case Qualifier::Q_U8:
        return ".u8";
    case Qualifier::Q_PRED:
        return ".pred";
    case Qualifier::Q_B8:
        return ".b8";
    case Qualifier::Q_B16:
        return ".b16";
    case Qualifier::Q_B32:
        return ".b32";
    case Qualifier::Q_B64:
        return ".b64";
    case Qualifier::Q_F8:
        return ".f8";
    case Qualifier::Q_F16:
        return ".f16";
    case Qualifier::Q_F32:
        return ".f32";
    case Qualifier::Q_F64:
        return ".f64";
    case Qualifier::Q_S8:
        return ".s8";
    case Qualifier::Q_S16:
        return ".s16";
    case Qualifier::Q_S32:
        return ".s32";
    case Qualifier::Q_S64:
        return ".s64";
    case Qualifier::Q_V2:
        return ".v2";
    case Qualifier::Q_V4:
        return ".v4";
    case Qualifier::Q_CONST:
        return ".const";
    case Qualifier::Q_PARAM:
        return ".param";
    case Qualifier::Q_GLOBAL:
        return ".global";
    case Qualifier::Q_LOCAL:
        return ".local";
    case Qualifier::Q_SHARED:
        return ".shared";
    case Qualifier::Q_GT:
        return ".gt";
    case Qualifier::Q_GE:
        return ".ge";
    case Qualifier::Q_EQ:
        return ".eq";
    case Qualifier::Q_NE:
        return ".ne";
    case Qualifier::Q_LT:
        return ".lt";
    case Qualifier::Q_TO:
        return ".to";
    case Qualifier::Q_WIDE:
        return ".wide";
    case Qualifier::Q_SYNC:
        return ".sync";
    case Qualifier::Q_LO:
        return ".lo";
    case Qualifier::Q_HI:
        return ".hi";
    case Qualifier::Q_UNI:
        return ".uni";
    case Qualifier::Q_RN:
        return ".rn";
    case Qualifier::Q_A:
        return ".a";
    case Qualifier::Q_B:
        return ".b";
    case Qualifier::Q_D:
        return ".d";
    case Qualifier::Q_ROW:
        return ".row";
    case Qualifier::Q_ALIGNED:
        return ".aligned";
    case Qualifier::Q_M8N8K4:
        return ".m8n8k4";
    case Qualifier::Q_M16N16K16:
        return ".m16n16k16";
    case Qualifier::Q_NEU:
        return ".neu";
    case Qualifier::Q_NC:
        return ".nc";
    case Qualifier::Q_FTZ:
        return ".ftz";
    case Qualifier::Q_APPROX:
        return ".approx";
    case Qualifier::Q_LTU:
        return ".ltu";
    case Qualifier::Q_LE:
        return ".le";
    case Qualifier::Q_GTU:
        return ".gtu";
    case Qualifier::Q_LEU:
        return ".leu";
    case Qualifier::Q_DOTADD:
        return ".add";
    case Qualifier::Q_GEU:
        return ".geu";
    case Qualifier::Q_RZI:
        return ".rzi";
    case Qualifier::Q_DOTOR:
        return ".or";
    case Qualifier::Q_SAT:
        return ".sat";
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
        return 1;
    case Qualifier::Q_U16:
    case Qualifier::Q_B16:
    case Qualifier::Q_S16:
    case Qualifier::Q_F16:
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
