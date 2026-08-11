// test_image_classifier.cpp
// =============================================================================
// Unit tests: 6-class image classifier (Task 2.1, Phase 12.3.A)
//
// Verifies classifyImage() categorizes image bytes into:
//   kPtxText, kPtxirStandalone, kExecutableTailPtxir,
//   kCubin, kFatbin, kTileIR
//
// RED PHASE: image_classifier.h does not exist yet.
// =============================================================================

#include "catch_amalgamated.hpp"
#include "cudart/image_classifier.h"
#include <vector>

using ptxemu::cudart::classifyImage;
using ptxemu::cudart::ImageKind;

TEST_CASE("ImageClassifier: PTX text → SUPPORTED", "[unit][cudart][classifier]") {
    // PTX text starts with '.' and contains ".version"
    std::string ptx = ".version 7.0\n.target sm_100\n.address_size 64\n";
    std::vector<uint8_t> bytes(ptx.begin(), ptx.end());
    REQUIRE(classifyImage(bytes.data(), bytes.size()) == ImageKind::kPtxText);
}

TEST_CASE("ImageClassifier: standalone PTXIR → SUPPORTED", "[unit][cudart][classifier]") {
    // PTXIR standalone: magic "PTXIR" + size + body
    std::vector<uint8_t> bytes = {'P','T','X','I','R', 0,0,0,0};
    REQUIRE(classifyImage(bytes.data(), bytes.size()) == ImageKind::kPtxirStandalone);
}

TEST_CASE("ImageClassifier: executable-tail PTXIR → REJECTED", "[unit][cudart][classifier]") {
    // Mimics __cudaRegisterFatBinary trailing section shape: ELF prefix
    // followed by PTXIR at the end (executable tail).
    std::vector<uint8_t> bytes(1024, 0x7f);
    bytes.insert(bytes.end(), {'P','T','X','I','R'});
    REQUIRE(classifyImage(bytes.data(), bytes.size()) == ImageKind::kExecutableTailPtxir);
}

TEST_CASE("ImageClassifier: NVIDIA cubin (ELF magic) → INVALID", "[unit][cudart][classifier]") {
    // ELF magic 0x7F + "ELF"
    std::vector<uint8_t> bytes = {0x7f, 'E','L','F'};
    REQUIRE(classifyImage(bytes.data(), bytes.size()) == ImageKind::kCubin);
}

TEST_CASE("ImageClassifier: NVIDIA fatbin → INVALID", "[unit][cudart][classifier]") {
    // Fatbin magic 0xBA55ED10
    std::vector<uint8_t> bytes = {0xBA, 0x55, 0xED, 0x10};
    REQUIRE(classifyImage(bytes.data(), bytes.size()) == ImageKind::kFatbin);
}

TEST_CASE("ImageClassifier: Tile IR → INVALID", "[unit][cudart][classifier]") {
    // Tile IR prefix "TILEIR"
    std::vector<uint8_t> bytes = {'T','I','L','E','I','R'};
    REQUIRE(classifyImage(bytes.data(), bytes.size()) == ImageKind::kTileIR);
}
