# PTX-EMU Configuration

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
GPU architecture configs, debug/logging settings.

## STRUCTURE
```
configs/
├── *.json           # GPU architecture (ampere_a100.json, hopper_h100.json, etc.)
├── config.ini       # Default logging
├── debug_config.ini # Debug logging
└── *-debug_config.ini  # Component-specific debug
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| GPU arch | `ampere_a100.json`, `hopper_h100.json` | SM counts, warp size, memory |
| Debug logging | `debug_config.ini` | Components: emu, exec, mem, reg, thread, func |
| Memory debug | `memory_debug_config.ini` | Memory subsystem logging |
| Instruction debug | `instruction_debug_config.ini` | Instruction trace |

## LOGGING COMPONENTS
- `emu` - Emulator state
- `exec` - Execution flow
- `mem` - Memory operations
- `reg` - Register operations
- `thread` - Thread state
- `func` - Function calls

## LOG LEVELS
`trace`, `debug`, `info`, `warning`, `error`, `fatal`

## ANTI-PATTERNS
- DO NOT set all components to `trace` in production - performance impact
- DO NOT modify configs/ without understanding GPU architecture JSON schema
