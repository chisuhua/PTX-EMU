// tests/integration/tmem/test_tmem_with_cta_context.cpp
// Phase 0.5.2 (Fix #9b): CTAContext TMEM integration test.
//
// Verifies that CTAContext exposes a per-CTA Tmem instance via the
// tmem() accessor, that the instance persists across the CTA lifetime,
// and that independent CTAContext instances have isolated TMEM stores.

#include "catch_amalgamated.hpp"
#include "ptxsim/cta_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"

namespace {

static void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

} // anonymous namespace

TEST_CASE("Fix #9b: Tmem default-init on CTAContext construction",
          "[integration][tmem][cta]") {
    init_factory_once();

    CTAContext cta;
    Tmem& tmem = cta.tmem();

    SECTION("All slots are initially zero") {
        uint8_t buf[Tmem::kSlotSize];
        for (size_t s = 0; s < 4; ++s) {
            std::memset(buf, 0xFF, sizeof(buf));
            tmem.read(s, buf, sizeof(buf));
            for (size_t b = 0; b < sizeof(buf); ++b) {
                REQUIRE(buf[b] == 0);
            }
        }
    }

    SECTION("validate_slot_id accepts valid indices") {
        REQUIRE(tmem.validate_slot_id(0));
        REQUIRE(tmem.validate_slot_id(127));
        REQUIRE(tmem.validate_slot_id(255));
        REQUIRE_FALSE(tmem.validate_slot_id(256));
        REQUIRE_FALSE(tmem.validate_slot_id(999));
    }
}

TEST_CASE("Fix #9b: Tmem write-read roundtrip through CTAContext",
          "[integration][tmem][cta][roundtrip]") {
    init_factory_once();

    CTAContext cta;
    Tmem& tmem = cta.tmem();

    uint8_t data[Tmem::kSlotSize];
    std::memset(data, 0xAB, sizeof(data));
    tmem.write(5, data, sizeof(data));

    uint8_t buf[Tmem::kSlotSize];
    std::memset(buf, 0, sizeof(buf));
    tmem.read(5, buf, sizeof(buf));
    for (size_t b = 0; b < sizeof(buf); ++b) {
        REQUIRE(buf[b] == 0xAB);
    }

    SECTION("Adjacent slots are not affected by writes") {
        uint8_t zero_buf[Tmem::kSlotSize];
        tmem.read(4, zero_buf, sizeof(zero_buf));
        for (size_t b = 0; b < sizeof(zero_buf); ++b) {
            REQUIRE(zero_buf[b] == 0);
        }
        tmem.read(6, zero_buf, sizeof(zero_buf));
        for (size_t b = 0; b < sizeof(zero_buf); ++b) {
            REQUIRE(zero_buf[b] == 0);
        }
    }
}

TEST_CASE("Fix #9b: Tmem clear zeros all slots through CTAContext",
          "[integration][tmem][cta][clear]") {
    init_factory_once();

    CTAContext cta;
    Tmem& tmem = cta.tmem();

    uint8_t data[Tmem::kSlotSize];
    std::memset(data, 0x55, sizeof(data));
    tmem.write(0, data, sizeof(data));
    tmem.write(128, data, sizeof(data));
    tmem.write(255, data, sizeof(data));

    tmem.clear();

    uint8_t buf[Tmem::kSlotSize];
    for (size_t s : {0, 128, 255}) {
        std::memset(buf, 0xFF, sizeof(buf));
        tmem.read(s, buf, sizeof(buf));
        for (size_t b = 0; b < sizeof(buf); ++b) {
            REQUIRE(buf[b] == 0);
        }
    }
}

TEST_CASE("Fix #9b: Two CTAContext instances have isolated TMEM",
          "[integration][tmem][cta][isolation]") {
    init_factory_once();

    CTAContext cta1;
    CTAContext cta2;

    uint8_t data1[Tmem::kSlotSize];
    std::memset(data1, 0x11, sizeof(data1));
    cta1.tmem().write(10, data1, sizeof(data1));

    uint8_t data2[Tmem::kSlotSize];
    std::memset(data2, 0x22, sizeof(data2));
    cta2.tmem().write(10, data2, sizeof(data2));

    uint8_t buf[Tmem::kSlotSize];
    cta1.tmem().read(10, buf, sizeof(buf));
    for (size_t b = 0; b < sizeof(buf); ++b) {
        REQUIRE(buf[b] == 0x11);
    }

    cta2.tmem().read(10, buf, sizeof(buf));
    for (size_t b = 0; b < sizeof(buf); ++b) {
        REQUIRE(buf[b] == 0x22);
    }

    SECTION("Updating CTA1 does not affect CTA2") {
        std::memset(data1, 0x33, sizeof(data1));
        cta1.tmem().write(10, data1, sizeof(data1));

        cta1.tmem().read(10, buf, sizeof(buf));
        for (size_t b = 0; b < sizeof(buf); ++b) {
            REQUIRE(buf[b] == 0x33);
        }

        cta2.tmem().read(10, buf, sizeof(buf));
        for (size_t b = 0; b < sizeof(buf); ++b) {
            REQUIRE(buf[b] == 0x22);
        }
    }
}

TEST_CASE("Fix #9b: Tmem const accessor through CTAContext",
          "[integration][tmem][cta][const]") {
    init_factory_once();

    CTAContext cta;

    uint8_t data[Tmem::kSlotSize];
    std::memset(data, 0x7E, sizeof(data));
    cta.tmem().write(99, data, sizeof(data));

    const CTAContext& const_cta = cta;
    const Tmem& const_tmem = const_cta.tmem();

    REQUIRE(const_tmem.validate_slot_id(99));

    uint8_t buf[Tmem::kSlotSize];
    std::memset(buf, 0, sizeof(buf));
    const_tmem.read(99, buf, sizeof(buf));
    for (size_t b = 0; b < sizeof(buf); ++b) {
        REQUIRE(buf[b] == 0x7E);
    }
}