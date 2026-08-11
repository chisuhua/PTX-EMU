// image_classifier.cpp
// =============================================================================
// 6-class image classifier implementation (Task 2.2, Phase 12.3.A)
//
// Pure function — no /proc/self/exe, no cuobjdump, no PTXIR_MODE reads.
// =============================================================================

#include "cudart/image_classifier.h"
#include <algorithm>
#include <cstring>

namespace ptxemu::cudart {

namespace {

bool ends_with_sig(const uint8_t* p, size_t n, const char* sig, size_t sig_n) {
    return n >= sig_n && std::memcmp(p + n - sig_n, sig, sig_n) == 0;
}

bool looks_like_ptx_text(const uint8_t* p, size_t n) {
    if (n < 2) return false;
    size_t scan = std::min<size_t>(n, 256);
    for (size_t i = 0; i < scan; ++i) {
        if (p[i] == '.') return true;
        if (p[i] == '\n' || p[i] == '\r') break;
    }
    return false;
}

}  // anonymous namespace

ImageKind classifyImage(const uint8_t* bytes, size_t size) {
    if (!bytes || size == 0) return ImageKind::kUnknown;

    // Executable-tail PTXIR must be checked BEFORE prefix-based checks
    if (ends_with_sig(bytes, size, "PTXIR", 5)) return ImageKind::kExecutableTailPtxir;

    // NVIDIA cubin = ELF magic (0x7f + "ELF")
    if (size >= 4 && bytes[0] == 0x7f &&
        bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F') return ImageKind::kCubin;

    // NVIDIA fatbin magic (0xBA + 0x55 + 0xED + 0x10)
    if (size >= 4 && bytes[0] == 0xBA && bytes[1] == 0x55 &&
        bytes[2] == 0xED && bytes[3] == 0x10) return ImageKind::kFatbin;

    // Tile IR prefix "TILEIR"
    if (size >= 6 && bytes[0] == 'T' && bytes[1] == 'I' &&
        bytes[2] == 'L' && bytes[3] == 'E' && bytes[4] == 'I' && bytes[5] == 'R')
        return ImageKind::kTileIR;

    // Standalone PTXIR — "PTXIR" at start
    if (size >= 5 && bytes[0] == 'P' && bytes[1] == 'T' &&
        bytes[2] == 'X' && bytes[3] == 'I' && bytes[4] == 'R')
        return ImageKind::kPtxirStandalone;

    // PTX text heuristic
    if (looks_like_ptx_text(bytes, size)) return ImageKind::kPtxText;

    return ImageKind::kUnknown;
}

}  // namespace ptxemu::cudart
