#ifndef PTXSIM_SCOREBOARD_INTERFACE_H
#define PTXSIM_SCOREBOARD_INTERFACE_H
#include <cstdint>

/// Pure virtual interface for CppTLM Scoreboard injection.
/// Zero external dependencies (only <cstdint>).
/// Ref: ADR-0020, CppTLM RFC-P1-001 §3.1
class IScoreboard {
public:
    virtual ~IScoreboard() = default;
    virtual bool has_free_entry() const = 0;
    virtual bool allocate(uint32_t reg_id, uint32_t warp_id) = 0;
    virtual bool release(uint32_t reg_id, uint32_t warp_id) = 0;
    virtual void tick() = 0;
};
#endif
