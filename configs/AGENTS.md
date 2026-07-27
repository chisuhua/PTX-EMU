# configs/ — GPU Architecture JSON + Debug/Logging INI

**16 files.** GPU hardware configs drive SM count, memory hierarchy, interconnect, and instruction latencies. INI configs control logging levels, trace filters, warp scheduler parameters, and GPU selection at startup.

## STRUCTURE

```
configs/
├── ampere_a100.json       # Ampere A100: 108 SM, HBM2e, NVLink 3.0, crossbar
├── hopper_h100.json        # Hopper H100: 132 SM, HBM3, NVLink 4.0, ring, Transformer Engine
├── blackwell_b200.json     # Blackwell B200: 144 SM, HBM3E, NVLink 5.0, mesh, FP4/FP6
├── mini.json               # Reduced A100 (2 SM, fast latencies) — test/debug default
├── gpu_config.json         # Minimal flat config (80 SM, no tensor/memory/interconnect)
├── config.ini              # Default dev config: debug global, emu=debug, thread=warning
├── debug_config.ini        # Balanced debug: all key components at debug, mini.json GPU
├── dev_debug_config.ini    # Dev debug: similar to debug_config + convergence trace
├── verbose_trace_config.ini# Max verbosity: global=trace, all components=trace, file-only
├── trace_config.ini        # Instruction trace: exec=trace, minimal other noise
├── instruction_debug_config.ini # Instruction focus: exec/instr/reg=trace, others=warning
├── memory_debug_config.ini # Memory focus: mem=trace, reg=debug, only memory instruction trace
├── perf_config.ini         # Perf analysis: global=warning, all tracing off
├── release_config.ini      # Production: global=info, console-only, no tracing
├── scheduler_config.ini    # Warp scheduler: priority algorithm, anti-starvation, barrier-aware
├── config_backup.ini       # Reference backup of all format options
```

## WHERE TO LOOK

| Need | File |
|------|------|
| Default GPU params | `gpu_config.json` (flat, minimal) |
| Full arch spec (Ampere) | `ampere_a100.json` |
| Full arch spec (Hopper) | `hopper_h100.json` |
| Full arch spec (Blackwell) | `blackwell_b200.json` |
| Quick test GPU | `mini.json` (2 SM, low latencies) |
| Day-to-day debugging | `config.ini` or `debug_config.ini` |
| Deep trace (all components) | `verbose_trace_config.ini` |
| Memory-only debugging | `memory_debug_config.ini` |
| Instruction-only debugging | `instruction_debug_config.ini` |
| Scheduler tuning | `scheduler_config.ini` |
| Performance measurement | `perf_config.ini` |
| Production run | `release_config.ini` |

## LOGGING COMPONENTS AND LEVELS

**Components** (set via `component.<name>=<level>` in INI `[logger]`):

| Component | Scope |
|-----------|-------|
| `emu` | Warp PC, active mask, warp-level state |
| `exec` | Instruction execution flow |
| `mem` | Memory load/store operations |
| `reg` | Register reads/writes |
| `thread` | Per-lane instruction trace (lane=X pc=Y text=Z) |
| `func` | Function call/return tracking |
| `instr` | Instruction dispatch |
| `cudart` | CUDA runtime API interception |

**Levels** (ascending severity): `trace` < `debug` < `info` < `warning` < `error` < `fatal`

## ANTI-PATTERNS

- ❌ Editing `gpu_config.json` thinking it's the active config — the INI's `[gpu]` section's `gpu_config_file` selects the actual GPU JSON
- ❌ Setting `global_level=info` while debugging memory — `mem` component needs `trace` or `debug`
- ❌ Using `verbose_trace_config.ini` in console mode (`target=console`) — output is massive; use `target=file`
- ❌ Modifying `config_backup.ini` directly — it's a reference template, not loaded by the emulator
- ❌ Confusing `trace_lanes=0x1` (bitmask, lane 0 only) with lane index — it's a bitmask, not a decimal index
- ❌ Expecting `scheduler_config.ini` parameters to take effect without restart — all configs are read at startup