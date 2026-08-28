#include "ptx_ir/instruction_latency_table.h"

namespace ptxsim {

InstructionLatencyTable& InstructionLatencyTable::instance() {
    static InstructionLatencyTable t;
    return t;
}

InstructionLatencyTable::InstructionLatencyTable() {
    reset_to_defaults();
}

void InstructionLatencyTable::reset_to_defaults() {
    table_.clear();
    for (int i = 0; i <= static_cast<int>(S_UNKNOWN); ++i) {
        auto type = static_cast<ptxemu::ir::StatementType>(i);
        table_[type] = resolve_default(type);
    }
}

void InstructionLatencyTable::load(const InstructionLatencyConfig& cfg) {
    if (cfg.ld_global_cycles > 0) {
        table_[S_LD] = InstructionLatency(static_cast<uint32_t>(cfg.ld_global_cycles),
                                          cfg.ld_global_long_delay);
    }
    if (cfg.st_global_cycles > 0) {
        table_[S_ST] = InstructionLatency(static_cast<uint32_t>(cfg.st_global_cycles),
                                          cfg.st_global_long_delay);
    }
    if (cfg.mul_cycles > 0) {
        InstructionLatency v(static_cast<uint32_t>(cfg.mul_cycles),
                             cfg.mul_long_delay);
        table_[S_MUL]   = v;
        table_[S_MUL24] = v;
        table_[S_MAD]   = v;
        table_[S_MAD24] = v;
        table_[S_FMA]   = v;
    }
    if (cfg.div_cycles > 0) {
        InstructionLatency v(static_cast<uint32_t>(cfg.div_cycles),
                             cfg.div_long_delay);
        table_[S_DIV] = v;
        table_[S_REM] = v;
    }
    if (cfg.bar_sync_cycles > 0) {
        InstructionLatency v(static_cast<uint32_t>(cfg.bar_sync_cycles),
                             cfg.bar_sync_long_delay);
        table_[S_BAR]           = v;
        table_[S_BAR_WARP_SYNC] = v;
        table_[S_MEMBAR]        = v;
        table_[S_FENCE]         = v;
    }
    if (cfg.default_cycles > 0) {
        InstructionLatency v(static_cast<uint32_t>(cfg.default_cycles),
                             cfg.default_long_delay);
        for (auto& kv : table_) {
            if (kv.second.cycles == DEFAULT_LATENCY.cycles &&
                kv.second.is_long_delay == DEFAULT_LATENCY.is_long_delay) {
                kv.second = v;
            }
        }
    }
}

InstructionLatency InstructionLatencyTable::get(ptxemu::ir::StatementType type) const {
    auto it = table_.find(type);
    if (it != table_.end()) return it->second;
    return DEFAULT_LATENCY;
}

InstructionLatency InstructionLatencyTable::resolve_default(ptxemu::ir::StatementType type) const {
    switch (type) {
    case S_LD:                return LD_GLOBAL_LATENCY;
    case S_ST:                return ST_GLOBAL_LATENCY;
    case S_MUL:
    case S_MUL24:
    case S_MAD:
    case S_MAD24:
    case S_FMA:               return MUL_LATENCY;
    case S_DIV:
    case S_REM:               return DIV_LATENCY;
    case S_BAR:
    case S_BAR_WARP_SYNC:
    case S_MEMBAR:
    case S_FENCE:             return BAR_SYNC_LATENCY;
    default:                  return DEFAULT_LATENCY;
    }
}

InstructionLatency getLatency(ptxemu::ir::StatementType type) {
    return InstructionLatencyTable::instance().get(type);
}

} // namespace ptxsim
