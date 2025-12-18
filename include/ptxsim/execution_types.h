#ifndef EXECUTION_TYPES_H
#define EXECUTION_TYPES_H

#include <cstdint>
#include <sstream>
#include <string>

enum EXE_STATE { RUN, EXIT, BAR_SYNC };

struct Dim3 {
    uint32_t x, y, z;
    Dim3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) : x(x), y(y), z(z) {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "[" << x << "," << y << "," << z << "]";
        return oss.str();
    }
};

#endif // EXECUTION_TYPES_H