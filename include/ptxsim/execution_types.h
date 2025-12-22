#ifndef EXECUTION_TYPES_H
#define EXECUTION_TYPES_H

#include <cstdint>
#include <sstream>
#include <string>

enum EXE_STATE { RUN, EXIT, BAR_SYNC };
enum BAR_TYPE { SYNC };

struct Dim3 {
    uint32_t x, y, z;
    Dim3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) : x(x), y(y), z(z) {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << "[" << x << "," << y << "," << z << "]";
        return oss.str();
    }
};

// 定义 CTA 唯一标识符（可扩展为多 GPU）
struct CTAId {
    uint32_t grid_id;
    uint32_t cta_x;
    uint32_t cta_y;
    uint32_t cta_z;

    bool operator==(const CTAId &other) const {
        return grid_id == other.grid_id && cta_x == other.cta_x &&
               cta_y == other.cta_y && cta_z == other.cta_z;
    }
};

// 为 CTAId 提供 hash（用于 unordered_map）
namespace std {
template <> struct hash<CTAId> {
    size_t operator()(const CTAId &id) const {
        return ((size_t)id.grid_id << 32) ^
               (id.cta_x | (id.cta_y << 10) | (id.cta_z << 20));
    }
};

} // namespace std

#endif // EXECUTION_TYPES_H