// image_classifier.h
// =============================================================================
// 6-class image classifier for CUDA module loading (Task 2.2, Phase 12.3.A)
//
// Pure function — no /proc/self/exe, no cuobjdump, no PTXIR_MODE reads.
// Used as a gate inside cuModuleLoadData to accept/reject image types.
// =============================================================================
#pragma once
#include <cstdint>
#include <cstddef>

namespace ptxemu::cudart {

enum class ImageKind {
    kPtxText,             // SUPPORTED — PTX assembly text
    kPtxirStandalone,    // SUPPORTED — standalone PTXIR binary
    kExecutableTailPtxir, // REJECTED — PTXIR embedded in executable tail
    kCubin,               // REJECTED → CUDA_ERROR_INVALID_IMAGE
    kFatbin,              // REJECTED → CUDA_ERROR_INVALID_IMAGE
    kTileIR,              // REJECTED → CUDA_ERROR_INVALID_IMAGE
    kUnknown              // REJECTED
};

// Classify image bytes into one of 6 kinds (pure function, no side effects).
ImageKind classifyImage(const uint8_t* bytes, size_t size);

}  // namespace ptxemu::cudart
