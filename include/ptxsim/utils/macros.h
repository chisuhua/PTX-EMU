#ifndef PTXSIM_UTILS_MACROS_H
#define PTXSIM_UTILS_MACROS_H

#include <cassert>

// UNSUPPORTED_TYPESIZE(context) — 标准化的 "data size not supported" 断言。
// 用于所有 PTX handler 中的 default: switch case，统一 error message 格式。
//
// 用法:
//   default:
//       UNSUPPORTED_TYPESIZE("floating point comparison");
//   }
#define UNSUPPORTED_TYPESIZE(context) \
    assert(0 && "Unsupported data size for " context)

#endif // PTXSIM_UTILS_MACROS_H