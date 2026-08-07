#include "cudart/ptxir_loader.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
}

bool write_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
    return f.good();
}

void print_usage() {
    std::cout << "Usage: ptxir_extract --in <path> [--out-cubin <X>] [--out-ptxir <Y>]\n";
}

void print_version() { std::cout << "ptxir_extract v1.0\n"; }

}  // namespace

int main(int argc, char** argv) {
    std::string in_path, out_cubin, out_ptxir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--in" && i + 1 < argc) {
            in_path = argv[++i];
        } else if (a == "--out-cubin" && i + 1 < argc) {
            out_cubin = argv[++i];
        } else if (a == "--out-ptxir" && i + 1 < argc) {
            out_ptxir = argv[++i];
        } else if (a == "--help") {
            print_usage();
            return 0;
        } else if (a == "--version") {
            print_version();
            return 0;
        }
    }

    if (in_path.empty()) {
        std::cerr << "Error: --in is required\n";
        return 4;
    }

    auto data = read_file(in_path);
    if (data.empty()) {
        std::cerr << "Error: cannot open " << in_path << "\n";
        return 2;
    }

    if (!cudart::PTXIRLoader::hasEmbeddedPTXIR(data.data(), data.size())) {
        if (!out_cubin.empty()) {
            write_file(out_cubin, data);
        }
        return 0;
    }

    auto pure = cudart::PTXIRLoader::extractPureCubin(data.data(), data.size());
    if (!pure) {
        std::cerr << "Error: cubin hash mismatch\n";
        return 3;
    }
    if (!out_cubin.empty()) {
        if (!write_file(out_cubin, *pure)) {
            std::cerr << "Error: cannot write " << out_cubin << "\n";
            return 2;
        }
    }
    if (!out_ptxir.empty()) {
        size_t section_size = 0;
        auto section = cudart::PTXIRLoader::extractPTXIR(data.data(), data.size(), &section_size);
        if (section) {
            std::vector<uint8_t> v(section.get(), section.get() + section_size);
            if (!write_file(out_ptxir, v)) {
                std::cerr << "Error: cannot write " << out_ptxir << "\n";
                return 2;
            }
        }
    }
    return 0;
}
