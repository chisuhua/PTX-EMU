#pragma once
#include <fstream>
#include <vector>
#include <cstdint>

inline std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

inline void write_file(const char* path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)data.data(), data.size());
}
