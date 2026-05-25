// barrier_types.h - Common barrier types and constants
#ifndef BARRIER_TYPES_H
#define BARRIER_TYPES_H

#include <cstdint>

namespace ptxsim {

// Hardware maximum barriers per CTA (NVIDIA standard)
static constexpr int MAX_BARRIERS_PER_CTA = 16;

// Maximum warp-level barriers (per warp)
static constexpr int MAX_WARP_BARRIERS = 4;

// Warp size (NVIDIA standard)
static constexpr int WARP_SIZE = 32;

// Default barrier for __syncthreads()
static constexpr int DEFAULT_CTA_BARRIER_ID = 0;

// Default warp barrier
static constexpr int DEFAULT_WARP_BARRIER_ID = 0;

} // namespace ptxsim

#endif // BARRIER_TYPES_H