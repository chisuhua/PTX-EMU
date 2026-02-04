// param_context.h
#ifndef PARAM_CONTEXT_H
#define PARAM_CONTEXT_H

#include "ptx_types.h"
#include <optional>
#include <string>
#include <vector>

struct ParamContext {
    std::string paramName;
    std::vector<Qualifier> paramTypes; // e.g., Q_U32, Q_PTR
    size_t byteSize = 0;               // total size in bytes
    std::optional<size_t> align;       // alignment in bytes (e.g., 4, 8, 16)

    // 兼容旧解析器与运行时逻辑
    int paramAlign = 0;
    int paramNum = 1;

    bool isPtr = false;

    // Helper: get effective alignment (default to natural alignment if not
    // specified)
    size_t effectiveAlignment() const {
        if (align.has_value()) {
            return *align;
        }
        // Natural alignment: min(8, power-of-two >= byteSize for scalars)
        // For pointers, typically 8 on 64-bit
        if (isPtr)
            return 8;
        if (byteSize == 0)
            return 1;
        // Round up to next power of two, capped at 8 (PTX typical max)
        size_t a = 1;
        while (a < byteSize && a < 8)
            a *= 2;
        return a;
    }
};
#endif