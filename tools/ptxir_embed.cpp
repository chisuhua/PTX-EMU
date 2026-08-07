#include "cudart/ptxir_loader.h"
#include "ptx_ir/ptxir_format.h"
#include "ptxir/ptxir_serialization.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openssl/sha.h>
#include <sstream>
#include <unistd.h>
#include <vector>

namespace {

uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

std::vector<uint8_t> sha256(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> hash(32);
    SHA256(data.data(), data.size(), hash.data());
    return hash;
}

bool find_manifest_offset(const std::vector<uint8_t>& ptxir, size_t* offset) {
    if (ptxir.size() < sizeof(PtxirHeader)) return false;
    PtxirHeader hdr;
    std::memcpy(&hdr, ptxir.data(), sizeof(hdr));
    if (std::memcmp(hdr.magic, PTXIR_MAGIC, 4) != 0) return false;
    size_t toc_pos = sizeof(PtxirHeader);
    for (uint16_t i = 0; i < hdr.section_count; ++i) {
        if (toc_pos + 6 > ptxir.size()) return false;
        uint8_t type = ptxir[toc_pos];
        uint32_t section_offset = read_u32(ptxir.data() + toc_pos + 2);
        if (type == static_cast<uint8_t>(PtxirSectionType::MANIFEST)) {
            *offset = section_offset;
            return true;
        }
        toc_pos += 6;
    }
    return false;
}

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
    std::cout << "Usage: ptxir_embed [--in-exe <path> | --in-cubin <path>] "
                 "(--in-ptxir <path> | --in-ptx <path>) "
                 "--kernel-name <name> --out <path>\n";
}

void print_version() { std::cout << "ptxir_embed v1.0\n"; }

}  // namespace

int main(int argc, char** argv) {
    std::string in_exe, in_cubin, in_ptxir, in_ptx, out_path, kernel_name;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--in-exe" && i + 1 < argc) {
            in_exe = argv[++i];
        } else if (a == "--in-cubin" && i + 1 < argc) {
            in_cubin = argv[++i];
        } else if (a == "--in-ptxir" && i + 1 < argc) {
            in_ptxir = argv[++i];
        } else if (a == "--in-ptx" && i + 1 < argc) {
            in_ptx = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--kernel-name" && i + 1 < argc) {
            kernel_name = argv[++i];
        } else if (a == "--help") {
            print_usage();
            return 0;
        } else if (a == "--version") {
            print_version();
            return 0;
        }
    }

    if (kernel_name.empty()) {
        std::cerr << "Error: --kernel-name is required\n";
        return 4;
    }
    if ((in_exe.empty() && in_cubin.empty()) || (!in_exe.empty() && !in_cubin.empty())) {
        std::cerr << "Error: exactly one of --in-exe or --in-cubin is required\n";
        return 4;
    }
    if ((in_ptxir.empty() && in_ptx.empty()) || (!in_ptxir.empty() && !in_ptx.empty())) {
        std::cerr << "Error: exactly one of --in-ptxir or --in-ptx is required\n";
        return 4;
    }
    if (out_path.empty()) {
        std::cerr << "Error: --out is required\n";
        return 4;
    }

    std::string prefix_path = in_exe.empty() ? in_cubin : in_exe;
    auto prefix = read_file(prefix_path);
    if (prefix.empty()) {
        std::cerr << "Error: cannot open " << prefix_path << "\n";
        return 2;
    }

    std::vector<uint8_t> section;
    std::string generated_ptxir;
    if (!in_ptx.empty()) {
        char tmpl[] = "/tmp/ptxir_embed_generated_XXXXXX.ptxir";
        int fd = mkstemps(tmpl, 6);
        if (fd == -1) {
            std::cerr << "Error: cannot create temp ptxir file\n";
            return 2;
        }
        close(fd);
        generated_ptxir = tmpl;
        if (!generate_ptxir(in_ptx, generated_ptxir, kernel_name)) {
            std::cerr << "Error: failed to generate PTXIR from " << in_ptx << "\n";
            return 3;
        }
        section = read_file(generated_ptxir);
    } else {
        section = read_file(in_ptxir);
    }
    if (section.empty()) {
        std::cerr << "Error: cannot open " << in_ptxir << "\n";
        return 2;
    }

    size_t manifest_offset = 0;
    if (!find_manifest_offset(section, &manifest_offset)) {
        std::cerr << "Error: PTXIR section has no MANIFEST; generate with --kernel-name\n";
        return 3;
    }
    if (manifest_offset + 32 > section.size()) {
        std::cerr << "Error: MANIFEST section too small\n";
        return 3;
    }

    auto hash = sha256(prefix);
    std::copy(hash.begin(), hash.end(), section.begin() + manifest_offset);

    std::vector<uint8_t> out = prefix;
    out.insert(out.end(), section.begin(), section.end());
    uint32_t size_le = static_cast<uint32_t>(section.size());
    out.push_back(static_cast<uint8_t>(size_le & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((size_le >> 24) & 0xFF));
    out.insert(out.end(), cudart::PTXIR_EMBED_MAGIC, cudart::PTXIR_EMBED_MAGIC + 8);

    if (!write_file(out_path, out)) {
        std::cerr << "Error: cannot write " << out_path << "\n";
        return 2;
    }
    return 0;
}
