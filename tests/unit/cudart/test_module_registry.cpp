#include "catch_amalgamated.hpp"
#include "cudart/module_registry.h"
#include <cstring>

namespace {

constexpr size_t TEST_IMAGE_SIZE = 64;

}  // namespace

TEST_CASE("ModuleRegistry: insert/lookup round-trip", "[module_registry][cudart]") {
    auto& registry = ptxemu::cudart::global_registry();

    uint8_t image_bytes[TEST_IMAGE_SIZE] = {0x01, 0x02, 0x03, 0x04};
    CUmodule mod = nullptr;
    CUresult res = registry.insert(image_bytes, TEST_IMAGE_SIZE, &mod);

    REQUIRE(res == CUDA_SUCCESS);
    REQUIRE(mod != nullptr);

    // Lookup the inserted module
    auto* rec = registry.lookup(mod);
    REQUIRE(rec != nullptr);
    REQUIRE(rec->image_size == TEST_IMAGE_SIZE);
    REQUIRE(std::memcmp(rec->image_bytes.get(), image_bytes, TEST_IMAGE_SIZE) == 0);

    // Cleanup
    registry.remove(mod);
}

TEST_CASE("ModuleRegistry: deep copy isolates caller pointer", "[module_registry][cudart]") {
    auto& registry = ptxemu::cudart::global_registry();

    uint8_t original_bytes[TEST_IMAGE_SIZE] = {0xDE, 0xAD, 0xBE, 0xEF};
    CUmodule mod = nullptr;
    CUresult res = registry.insert(original_bytes, TEST_IMAGE_SIZE, &mod);
    REQUIRE(res == CUDA_SUCCESS);

    // Mutate the caller's buffer after insert
    original_bytes[0] = 0xFF;
    original_bytes[1] = 0xFF;

    // Registry copy must be unchanged
    auto* rec = registry.lookup(mod);
    REQUIRE(rec != nullptr);
    REQUIRE(rec->image_bytes[0] == 0xDE);
    REQUIRE(rec->image_bytes[1] == 0xAD);
    REQUIRE(rec->image_bytes[2] == 0xBE);
    REQUIRE(rec->image_bytes[3] == 0xEF);

    registry.remove(mod);
}

TEST_CASE("ModuleRegistry: insert returns handle not pointer", "[module_registry][cudart]") {
    auto& registry = ptxemu::cudart::global_registry();

    uint8_t image[8] = {0x11, 0x22};
    CUmodule mod1 = nullptr;
    CUmodule mod2 = nullptr;

    REQUIRE(registry.insert(image, sizeof(image), &mod1) == CUDA_SUCCESS);
    REQUIRE(registry.insert(image, sizeof(image), &mod2) == CUDA_SUCCESS);

    // Each insert returns a unique handle
    REQUIRE(mod1 != nullptr);
    REQUIRE(mod2 != nullptr);
    REQUIRE(mod1 != mod2);

    registry.remove(mod1);
    registry.remove(mod2);
}

TEST_CASE("ModuleRegistry: lookup invalid handle returns nullptr", "[module_registry][cudart]") {
    auto& registry = ptxemu::cudart::global_registry();

    // A module handle that was never inserted
    CUmodule invalid_mod = reinterpret_cast<CUmodule>(0x12345678);
    REQUIRE(registry.lookup(invalid_mod) == nullptr);
}

TEST_CASE("ModuleRegistry: insert function and lookup round-trip", "[module_registry][cudart]") {
    auto& registry = ptxemu::cudart::global_registry();

    uint8_t image[8] = {0xAA};
    CUmodule mod = nullptr;
    REQUIRE(registry.insert(image, sizeof(image), &mod) == CUDA_SUCCESS);

    CUfunction func = nullptr;
    CUresult res = registry.insert_function(mod, "my_kernel", &func);
    REQUIRE(res == CUDA_SUCCESS);
    REQUIRE(func != nullptr);

    auto* func_rec = registry.lookup_function(func);
    REQUIRE(func_rec != nullptr);
    REQUIRE(func_rec->name == "my_kernel");
    REQUIRE(func_rec->parent == mod);

    registry.remove(mod);
}

TEST_CASE("ModuleRegistry: lookup_function invalid handle returns nullptr", "[module_registry][cudart]") {
    auto& registry = ptxemu::cudart::global_registry();

    CUfunction invalid_func = reinterpret_cast<CUfunction>(0x87654321);
    REQUIRE(registry.lookup_function(invalid_func) == nullptr);
}
