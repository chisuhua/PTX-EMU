#include "memory/hardware_memory_manager.h"
#include "memory/hardware_memory_manager.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/memory/tmem.h"
#include "ptxsim/async/tc_queue.h"
#include "ptxsim/utils/half_utils.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "utils/logger.h"
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

bool has_qualifier(const std::vector<Qualifier>& qualifiers, Qualifier q) {
    for (const auto& x : qualifiers) {
        if (x == q)
            return true;
    }
    return false;
}

bool is_tcgen05_mma_f16(const std::vector<Qualifier>& qualifiers) {
    return has_qualifier(qualifiers, Qualifier::Q_CLUSTER) &&
           has_qualifier(qualifiers, Qualifier::Q_F16) &&
           !has_qualifier(qualifiers, Qualifier::Q_TCGEN05_LD) &&
           !has_qualifier(qualifiers, Qualifier::Q_TCGEN05_ST) &&
           !has_qualifier(qualifiers, Qualifier::Q_TCGEN05_COMMIT) &&
           !has_qualifier(qualifiers, Qualifier::Q_TCGEN05_WAIT);
}

bool is_tcgen05_ld(const std::vector<Qualifier>& qualifiers) {
    return has_qualifier(qualifiers, Qualifier::Q_CLUSTER) &&
           has_qualifier(qualifiers, Qualifier::Q_F16) &&
           has_qualifier(qualifiers, Qualifier::Q_TCGEN05_LD);
}

bool is_tcgen05_st(const std::vector<Qualifier>& qualifiers) {
    return has_qualifier(qualifiers, Qualifier::Q_CLUSTER) &&
           has_qualifier(qualifiers, Qualifier::Q_F16) &&
           has_qualifier(qualifiers, Qualifier::Q_TCGEN05_ST);
}

bool is_tcgen05_commit(const std::vector<Qualifier>& qualifiers) {
    return has_qualifier(qualifiers, Qualifier::Q_CLUSTER) &&
           has_qualifier(qualifiers, Qualifier::Q_F16) &&
           has_qualifier(qualifiers, Qualifier::Q_TCGEN05_COMMIT);
}

bool is_tcgen05_wait(const std::vector<Qualifier>& qualifiers) {
    return has_qualifier(qualifiers, Qualifier::Q_CLUSTER) &&
           has_qualifier(qualifiers, Qualifier::Q_F16) &&
           has_qualifier(qualifiers, Qualifier::Q_TCGEN05_WAIT);
}

// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[0][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[0][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[0][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[0][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[1][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[1][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[1][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[1][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[2][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[2][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[2][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[2][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[3][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[3][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[3][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[3][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[4][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[4][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[4][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[4][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[5][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[5][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[5][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[5][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[6][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[6][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[6][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[6][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[7][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[7][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[7][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 0 C[7][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[0][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[0][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[0][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[0][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[1][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[1][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[1][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[1][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[2][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[2][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[2][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[2][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[3][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[3][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[3][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[3][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[4][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[4][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[4][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[4][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[5][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[5][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[5][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[5][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[6][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[6][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[6][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[6][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[7][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[7][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[7][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 1 C[7][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[0][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[0][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[0][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[0][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[1][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[1][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[1][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[1][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[2][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[2][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[2][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[2][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[3][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[3][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[3][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[3][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[4][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[4][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[4][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[4][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[5][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[5][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[5][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[5][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[6][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[6][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[6][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[6][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[7][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[7][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[7][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 2 C[7][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[0][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[0][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[0][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[0][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[1][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[1][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[1][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[1][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[2][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[2][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[2][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[2][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[3][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[3][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[3][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[3][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[4][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[4][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[4][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[4][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[5][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[5][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[5][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[5][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[6][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[6][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[6][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[6][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[7][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[7][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[7][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 3 C[7][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[0][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[0][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[0][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[0][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[1][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[1][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[1][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[1][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[2][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[2][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[2][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[2][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[3][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[3][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[3][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[3][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[4][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[4][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[4][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[4][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[5][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[5][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[5][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[5][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[6][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[6][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[6][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[6][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[7][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[7][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[7][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 4 C[7][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[0][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[0][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[0][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[0][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[1][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[1][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[1][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[1][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[2][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[2][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[2][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[2][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[3][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[3][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[3][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[3][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[4][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[4][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[4][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[4][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[5][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[5][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[5][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[5][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[6][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[6][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[6][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[6][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[7][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[7][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[7][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 5 C[7][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[0][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[0][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[0][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[0][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[1][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[1][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[1][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[1][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[2][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[2][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[2][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[2][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[3][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[3][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[3][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[3][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[4][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[4][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[4][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[4][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[5][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[5][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[5][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[5][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[6][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[6][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[6][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[6][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[7][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[7][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[7][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 6 C[7][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[0][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[0][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[0][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[0][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[1][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[1][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[1][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[1][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[2][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[2][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[2][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[2][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[3][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[3][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[3][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[3][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[4][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[4][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[4][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[4][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[5][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[5][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[5][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[5][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[6][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[6][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[6][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[6][3] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[7][0] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[7][1] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[7][2] PTX ISA §9.7.13
// UNVERIFIED-AGAINST-HARDWARE — fragment element lane 7 C[7][3] PTX ISA §9.7.13

}

void execute_tcgen05_ld(ThreadContext* context,
                         const std::vector<Qualifier>& qualifiers);
void execute_tcgen05_st(ThreadContext* context,
                         const std::vector<Qualifier>& qualifiers);
void execute_tcgen05_commit(ThreadContext* context,
                             const std::vector<Qualifier>& qualifiers);
void execute_tcgen05_wait(ThreadContext* context,
                           const std::vector<Qualifier>& qualifiers);

void WmmaHandler::processWmmaOperation(ThreadContext *context, void **operands,
                                        const std::vector<Qualifier> &qualifiers) {
    (void)operands;

    // Dispatch based on qualifier sub-operation markers
    if (is_tcgen05_ld(qualifiers)) {
        execute_tcgen05_ld(context, qualifiers);
        return;
    }
    if (is_tcgen05_st(qualifiers)) {
        execute_tcgen05_st(context, qualifiers);
        return;
    }
    if (is_tcgen05_commit(qualifiers)) {
        execute_tcgen05_commit(context, qualifiers);
        return;
    }
    if (is_tcgen05_wait(qualifiers)) {
        execute_tcgen05_wait(context, qualifiers);
        return;
    }

    if (!is_tcgen05_mma_f16(qualifiers)) {
        PTX_ERROR_EMU("WMMA / Tensor Core instruction not implemented "
                      "(qualifiers=%zu) - see implement-wmma-tensor-core",
                      qualifiers.size());
        throw UnsupportedInstructionException(
            "wmma.*",
            "Tensor Core not yet implemented in ptx-emu (ref: c5 Fix #1)");
    }

    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.mma: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "wmma.*", "tcgen05.mma requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.mma: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "wmma.*", "tcgen05.mma requires an active CTAContext");
    }

    Tmem& tmem = cta->tmem();
    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        size_t a_slot = static_cast<size_t>(lane_id) * 2;
        size_t b_slot = static_cast<size_t>(lane_id) * 2 + 1;
        size_t c_slot = static_cast<size_t>(64) + static_cast<size_t>(lane_id);

        std::array<uint8_t, Tmem::kSlotSize> a_buf{};
        tmem.read(a_slot, a_buf.data(), Tmem::kSlotSize);
        const uint16_t* a_raw =
            reinterpret_cast<const uint16_t*>(a_buf.data());

        std::array<uint8_t, Tmem::kSlotSize> b_buf{};
        tmem.read(b_slot, b_buf.data(), Tmem::kSlotSize);
        const uint16_t* b_raw =
            reinterpret_cast<const uint16_t*>(b_buf.data());

        float a_flat[ROWS * COLS_A];
        float b_flat[ROWS * COLS_B];
        for (int k = 0; k < ROWS * COLS_A; ++k)
            a_flat[k] = f16_to_f32(a_raw[k]);
        for (int k = 0; k < ROWS * COLS_B; ++k)
            b_flat[k] = f16_to_f32(b_raw[k]);

        std::array<uint16_t, ROWS * COLS_B> c_frag{};
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS_B; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < COLS_A; ++k) {
                    sum += a_flat[i * COLS_A + k] *
                           b_flat[k * COLS_B + j];
                }
                c_frag[i * COLS_B + j] = f32_to_f16(sum);
            }
        }

        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        std::memcpy(c_buf.data(), c_frag.data(),
                    c_frag.size() * sizeof(uint16_t));
        tmem.write(c_slot, c_buf.data(), Tmem::kSlotSize);
    }

    PTX_DEBUG_EMU("tcgen05.mma.cta_group::1.kind::f16 executed "
                  "(32 lanes x 8x4 fragments)");
}

void execute_tcgen05_ld(ThreadContext* context,
                         const std::vector<Qualifier>& qualifiers) {
    (void)qualifiers;
    // PTX ISA §9.7.13: tcgen05.ld — load from TMA descriptor to TMEM
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.ld: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "wmma.ld", "tcgen05.ld requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.ld: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "wmma.ld", "tcgen05.ld requires an active CTAContext");
    }

    TmaDescriptorStore& desc_store = cta->tma_descriptor_store();
    const TmaDescriptor* desc = desc_store.load(0);
    if (!desc) {
        PTX_ERROR_EMU("tcgen05.ld: no TMA descriptor found for cta_id=0");
        throw UnsupportedInstructionException(
            "wmma.ld", "tcgen05.ld requires a TMA descriptor");
    }

    // UNVERIFIED-AGAINST-HARDWARE — 128-byte transfer per PTX ISA §9.7.13
    uint8_t tmp[Tmem::kSlotSize];
    std::memcpy(tmp, reinterpret_cast<const void*>(desc->global_address),
                Tmem::kSlotSize);

    Tmem& tmem = cta->tmem();
    // UNVERIFIED-AGAINST-HARDWARE — target slot 0 per PTX ISA §9.7.13
    tmem.write(0, tmp, Tmem::kSlotSize);

    PTX_DEBUG_EMU("tcgen05.ld: TMA desc global=0x%016lx → TMEM slot 0 "
                  "(%zu bytes)",
                  desc->global_address, Tmem::kSlotSize);
}

void execute_tcgen05_st(ThreadContext* context,
                         const std::vector<Qualifier>& qualifiers) {
    (void)qualifiers;
    // PTX ISA §9.7.13: tcgen05.st — store from TMEM to TMA descriptor
    // UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13
    WarpContext* warp = context->get_warp_context();
    if (!warp) {
        PTX_ERROR_EMU("tcgen05.st: no WarpContext attached to thread");
        throw UnsupportedInstructionException(
            "wmma.st", "tcgen05.st requires an active WarpContext");
    }
    CTAContext* cta = warp->get_cta_context();
    if (!cta) {
        PTX_ERROR_EMU("tcgen05.st: no CTAContext attached to warp");
        throw UnsupportedInstructionException(
            "wmma.st", "tcgen05.st requires an active CTAContext");
    }

    TmaDescriptorStore& desc_store = cta->tma_descriptor_store();
    const TmaDescriptor* desc = desc_store.load(0);
    if (!desc) {
        PTX_ERROR_EMU("tcgen05.st: no TMA descriptor found for cta_id=0");
        throw UnsupportedInstructionException(
            "wmma.st", "tcgen05.st requires a TMA descriptor");
    }

    // UNVERIFIED-AGAINST-HARDWARE — 128-byte transfer per PTX ISA §9.7.13
    uint8_t tmp[Tmem::kSlotSize];
    Tmem& tmem = cta->tmem();
    tmem.read(0, tmp, Tmem::kSlotSize);

    std::memcpy(reinterpret_cast<void*>(desc->global_address),
                tmp, Tmem::kSlotSize);

    PTX_DEBUG_EMU("tcgen05.st: TMEM slot 0 → TMA desc global=0x%016lx "
                  "(%zu bytes)",
                  desc->global_address, Tmem::kSlotSize);
}

void execute_tcgen05_commit(ThreadContext* context,
                             const std::vector<Qualifier>& qualifiers) {
    (void)context;
    (void)qualifiers;
    PTX_ERROR_EMU("tcgen05.commit: not yet implemented (Phase 2.2)");
    throw UnsupportedInstructionException(
        "wmma.commit",
        "tcgen05.commit not yet implemented in ptx-emu");
}

void execute_tcgen05_wait(ThreadContext* context,
                           const std::vector<Qualifier>& qualifiers) {
    (void)context;
    (void)qualifiers;
    PTX_ERROR_EMU("tcgen05.wait: not yet implemented (Phase 2.2)");
    throw UnsupportedInstructionException(
        "wmma.wait",
        "tcgen05.wait not yet implemented in ptx-emu");
}