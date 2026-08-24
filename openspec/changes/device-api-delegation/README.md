# device-api-delegation

PTX-EMU HSK-8 Phase 2.2/2.3 delegation — implement 4 stubbed IPtxEmuDevice methods (set_scoreboard, set_active_mask, set_next_pc, attach_timing) by delegating to existing SMContext/WarpContext/ThreadContext APIs. No public API signature changes (PTXEMU_API_VERSION=1 frozen). See parent plan 2026-08-24-hsk8-followup-task-path.md §Phase 2.
