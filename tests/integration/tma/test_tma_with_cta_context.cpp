// tests/integration/tma/test_tma_with_cta_context.cpp
// Phase 0.5.1 (Fix #9a): CTAContext TMA descriptor store integration test.
//
// Verifies that CTAContext exposes a per-CTA TmaDescriptorStore via the
// tma_descriptor_store() accessor, that the store persists across the
// CTA lifetime, and that independent CTAContext instances have isolated
// descriptor stores.

#include "catch_amalgamated.hpp"
#include "ptxsim/cta_context.h"
#include "ptxsim/memory/tma_descriptor.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/barrier/barrier_module.h"
#include "ptxsim/instruction_factory.h"

namespace {

static void init_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

static TmaDescriptor make_test_descriptor(uint64_t addr, uint32_t dim0) {
    TmaDescriptor desc;
    desc.global_address = addr;
    desc.global_dim[0] = dim0;
    desc.global_dim[1] = 0;
    desc.global_dim[2] = 0;
    desc.global_dim[3] = 0;
    desc.global_dim[4] = 0;
    desc.rank = 1;
    desc.elemtype = 1;
    return desc;
}

} // anonymous namespace

TEST_CASE("Fix #9a: TmaDescriptorStore default-init on CTAContext construction",
          "[integration][tma][cta]") {
    init_factory_once();

    CTAContext cta;
    TmaDescriptorStore& store = cta.tma_descriptor_store();

    SECTION("Store is initially empty") {
        REQUIRE_FALSE(store.has(0));
        REQUIRE_FALSE(store.has(1));
        REQUIRE_FALSE(store.has(99));
    }

    SECTION("load returns nullptr for unknown key") {
        REQUIRE(store.load(0) == nullptr);
        REQUIRE(store.load(1) == nullptr);
    }
}

TEST_CASE("Fix #9a: TmaDescriptorStore accessor through SMContext+CTAContext",
          "[integration][tma][cta][sm]") {
    init_factory_once();

    SMContext sm(4, 128, 4096, 0);

    auto block = std::make_unique<CTAContext>();
    Dim3 gridDim = {1, 1, 1};
    Dim3 blockDim = {32, 1, 1};
    Dim3 blockIdx = {0, 0, 0};
    std::vector<ptxemu::ir::StatementContext> stmts;
    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;
    block->init(gridDim, blockDim, blockIdx, stmts, &name2Sym, label2pc);

    CTAContext* cta = block.get();
    REQUIRE(cta != nullptr);

    TmaDescriptorStore& store = cta->tma_descriptor_store();
    REQUIRE_FALSE(store.has(0));

    sm.add_block(std::move(block));

    WarpContext* warp = sm.get_warp(0);
    REQUIRE(warp != nullptr);
    CTAContext* cta_from_sm = warp->get_cta_context();
    REQUIRE(cta_from_sm == cta);

    TmaDescriptorStore& store2 = cta_from_sm->tma_descriptor_store();
    // Same store reference after transfer to SMContext
    REQUIRE_FALSE(store2.has(0));

    TmaDescriptor desc = make_test_descriptor(0x1000, 16);
    cta_from_sm->tma_descriptor_store().store(0, desc);

    REQUIRE(store2.has(0));
    const TmaDescriptor* loaded = store2.load(0);
    REQUIRE(loaded != nullptr);
    REQUIRE(loaded->global_address == 0x1000);
    REQUIRE(loaded->global_dim[0] == 16);
    REQUIRE(loaded->rank == 1);
    REQUIRE(loaded->elemtype == 1);
}

TEST_CASE("Fix #9a: Two CTAContext instances have isolated TmaDescriptorStores",
          "[integration][tma][cta][isolation]") {
    init_factory_once();

    CTAContext cta1;
    CTAContext cta2;

    TmaDescriptor desc1 = make_test_descriptor(0x1000, 16);
    TmaDescriptor desc2 = make_test_descriptor(0x2000, 32);

    cta1.tma_descriptor_store().store(0, desc1);
    cta2.tma_descriptor_store().store(0, desc2);

    const TmaDescriptor* loaded1 = cta1.tma_descriptor_store().load(0);
    const TmaDescriptor* loaded2 = cta2.tma_descriptor_store().load(0);

    REQUIRE(loaded1 != nullptr);
    REQUIRE(loaded2 != nullptr);
    REQUIRE(loaded1->global_address == 0x1000);
    REQUIRE(loaded2->global_address == 0x2000);
    REQUIRE(loaded1->global_dim[0] == 16);
    REQUIRE(loaded2->global_dim[0] == 32);

    REQUIRE(cta2.tma_descriptor_store().has(0));
    // Update cta1 only — verify cta2 unchanged
    TmaDescriptor desc1b = make_test_descriptor(0x3000, 48);
    cta1.tma_descriptor_store().store(0, desc1b);

    const TmaDescriptor* loaded1b = cta1.tma_descriptor_store().load(0);
    REQUIRE(loaded1b->global_address == 0x3000);
    REQUIRE(loaded1b->global_dim[0] == 48);

    const TmaDescriptor* loaded2_unchanged = cta2.tma_descriptor_store().load(0);
    REQUIRE(loaded2_unchanged->global_address == 0x2000);
    REQUIRE(loaded2_unchanged->global_dim[0] == 32);
}

TEST_CASE("Fix #9a: TmaDescriptorStore clear semantics on CTAContext",
          "[integration][tma][cta]") {
    init_factory_once();

    CTAContext cta;

    cta.tma_descriptor_store().store(0, make_test_descriptor(0x1000, 16));
    cta.tma_descriptor_store().store(1, make_test_descriptor(0x2000, 32));
    REQUIRE(cta.tma_descriptor_store().has(0));
    REQUIRE(cta.tma_descriptor_store().has(1));

    cta.tma_descriptor_store().clear();
    REQUIRE_FALSE(cta.tma_descriptor_store().has(0));
    REQUIRE_FALSE(cta.tma_descriptor_store().has(1));
}

TEST_CASE("Fix #9a: TmaDescriptorStore const accessor",
          "[integration][tma][cta][const]") {
    init_factory_once();

    CTAContext cta;
    cta.tma_descriptor_store().store(0, make_test_descriptor(0xDEAD, 42));

    const CTAContext& const_cta = cta;
    const TmaDescriptorStore& const_store = const_cta.tma_descriptor_store();

    REQUIRE(const_store.has(0));
    const TmaDescriptor* loaded = const_store.load(0);
    REQUIRE(loaded != nullptr);
    REQUIRE(loaded->global_address == 0xDEAD);
    REQUIRE(loaded->global_dim[0] == 42);
}