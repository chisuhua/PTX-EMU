# PTX Instruction Documentation

> **Purpose**: Document PTX instruction support status in PTX-EMU emulator.
> **Status**: Initial scaffold (2026-06-24). Full content extraction deferred to Phase 4.

## Why This Directory Exists

The `ptx-grammar-modification` skill (`.opencode/skills/ptx-grammar-modification/SKILL.md`) mandates:

> "□ 2. 阅读 docs/ptx/ 对应章节"

Before any ANTLR grammar modification, the corresponding PTX documentation section MUST be read. This directory is the canonical location for that documentation.

## Structure

Each PTX instruction or instruction family should have a `.md` file:

```
docs/ptx/
├── README.md             # This file
├── arithmetic.md         # add, sub, mul, mad, etc.
├── memory.md             # ld, st, atom, etc.
├── control.md            # bra, ret, call, exit
├── barrier.md            # bar.sync, bar.warp.sync
├── conversion.md         # cvt, cvta
├── st-async.md           # st.async (PTX 8.7+ placeholder — to be removed)
├── red-async.md          # red.async (PTX 8.7+ placeholder — to be removed)
├── tcgen05.md            # tcgen05.* family (PTX 9.0+ placeholder — to be removed)
└── tensormap.md          # tensormap.replace (PTX 8.7+ placeholder — to be removed)
```

## Support Status Table

| Instruction | Grammar | Handler | Tests | Status |
|------------|---------|---------|-------|--------|
| `add` | ✅ | ✅ | ✅ | Full |
| `sub` | ✅ | ✅ | ✅ | Full |
| `mul` | ✅ | ✅ | ✅ | Full |
| `mad` | ✅ | ✅ | ✅ | Full |
| `bra` | ✅ | ✅ | ✅ | Full |
| `ret` | ✅ | ✅ | ✅ | Full |
| `ld.global` | ✅ | ✅ | ✅ | Full |
| `st.global` | ✅ | ✅ | ✅ | Full |
| `atom.global.add` | ✅ | ✅ | ✅ | Full |
| `atom.global.exch` | ✅ | ✅ | ✅ | Full (T2-5) |
| `bar.sync` | ✅ | ✅ | ✅ | Full |
| `bar.warp.sync` | ✅ | ✅ | ✅ | Full (legacy wbar path, Phase 5) |
| `cvt.f32.f16` | ✅ | ✅ | ✅ | Full (T2-6 strategy pattern) |
| **`st.async`** | ✅ | ❌ stub | ❌ | **Placeholder (T2-4 to remove)** |
| **`red.async`** | ✅ | ❌ stub | ❌ | **Placeholder (T2-4 to remove)** |
| **`tcgen05.*`** | ✅ | ✅ 11/11 | ✅ | **Full (ADR-0016, implement-tcgen05-handlers-extended Phase 4 `718095a`)** |
| **`tensormap.replace`** | ✅ | ❌ stub | ❌ | **Placeholder (T2-4 to remove)** |
| `st.bulk` | ✅ | ❌ stub | ❌ | Out of T2-4 scope (separate cleanup) |
| `wmma.*` | ✅ | ❌ stub | ❌ | Stub (Tensor Core) |
| `mma.*` | ✅ | ❌ stub | ❌ | Stub (Tensor Core) |

## Extraction Methodology

For Phase 4 expansion of this directory, extract PTX instruction specifications from:

1. **NVIDIA PTX ISA Reference** (primary source): https://docs.nvidia.com/cuda/parallel-thread-execution/
2. **CUDA Toolkit Documentation** (supplementary): https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

### Extraction Template

Each `.md` file should contain:

```markdown
# {instruction_name} - PTX {version}

## Syntax
{grammar rule from PTX spec}

## Semantics
{behavioral description}

## Examples
{real PTX code samples (from cuobjdump -xptx if available)}

## PTX-EMU Implementation Status
- Grammar: ✅ / ❌
- Handler: ✅ / ❌ / ⚠️ partial
- Tests: ✅ / ❌ / ⚠️ partial

## Notes
{any deviations from spec, limitations, etc.}
```

## Phase 4 Tasks

1. **Extract `st.async`** spec from NVIDIA PTX 8.7+ docs (2-3 hours)
2. **Extract `red.async`** spec from NVIDIA PTX 8.7+ docs (1-2 hours)
3. **Extract `tcgen05.*` family** spec from NVIDIA PTX 9.0+ docs (4-6 hours, most complex)
4. **Extract `tensormap.replace`** spec from NVIDIA PTX 8.7+ docs (1-2 hours)
5. **Fill in placeholder docs for `st.bulk`, `wmma.*`, `mma.*`** (parallel work)

**Estimated effort**: 1 working day for foundational docs/ptx/ infrastructure

## Current Status (2026-06-24)

- ✅ Directory created (this file)
- ❌ Instruction-specific `.md` files not yet created
- ⏸ Phase 4 work pending