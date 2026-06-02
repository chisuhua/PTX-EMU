#ifndef PTXSIM_TESTING_PREDICATES_H
#define PTXSIM_TESTING_PREDICATES_H

#include <cstdint>
#include <string>

#include "ptxsim/warp_context.h"
#include "register/register_bank_manager.h"

namespace ptxsim::testing {

/**
 * Setup predicate register p1 with per-lane values.
 * @param w Warp context
 * @param taken 32-bit mask, bit i=1 means lane i takes branch
 */
static inline void setup_pred(WarpContext *w, uint32_t taken) {
    auto rbm = w->get_register_bank_manager();
    rbm->create_register("p1", 1);
    for (int i = 0; i < 32; i++) {
        auto *p = static_cast<uint8_t*>(rbm->get_register("p1", 0, i));
        *p = (taken & (1u << i)) ? 1 : 0;
    }
}

/**
 * Set predicate value for a single lane.
 * @param w Warp context
 * @param lane Lane index [0, 31]
 * @param value Predicate value (0 or 1)
 */
static inline void set_predicate_per_lane(WarpContext *w, int lane, uint8_t value) {
    auto rbm = w->get_register_bank_manager();
    if (!rbm->get_register("p1", 0, lane)) {
        rbm->create_register("p1", 1);
    }
    auto *p = static_cast<uint8_t*>(rbm->get_register("p1", 0, lane));
    if (p) {
        *p = value;
    }
}

/**
 * Get predicate mask from register p1.
 * @param w Warp context
 * @return 32-bit mask, bit i represents lane i's predicate value
 */
static inline uint32_t get_predicate_mask(WarpContext *w) {
    auto rbm = w->get_register_bank_manager();
    uint32_t mask = 0;
    for (int i = 0; i < 32; i++) {
        auto *p = static_cast<uint8_t*>(rbm->get_register("p1", 0, i));
        if (p && *p) {
            mask |= (1u << i);
        }
    }
    return mask;
}

} // namespace ptxsim::testing

#endif // PTXSIM_TESTING_PREDICATES_H