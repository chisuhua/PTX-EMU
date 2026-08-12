# libptxemu_abi_baseline.txt — Regeneration Guide

This file captures the externally-visible ABI of `libptxemu_device.so` (per ADR-0029).
Format: `<type-letter> <symbol-name>` — load addresses are stripped.

## Regeneration

When the ABI legitimately changes (a `ptxemu_*` symbol is added/removed/renamed):

```bash
nm -D build/lib/libptxemu_device.so \
  | awk '{print $2, $3}' \
  | grep ptxemu_ \
  | sort -u \
  > tests/integration/cpptlm/baselines/libptxemu_abi_baseline.txt
git add tests/integration/cpptlm/baselines/libptxemu_abi_baseline.txt
git commit -m "test(cpptlm): regenerate ABI baseline after <reason>"
```

## Current ABI (5 originally defined in ADR-0029, expanded as needed)

The baseline contains all `ptxemu_*` symbols defined in `libptxemu_device.so`. Original 5 from ADR-0029:

- `ptxemu_image_load` — load PTXIR image from blob
- `ptxemu_image_execute` — execute a kernel
- `ptxemu_image_unload` — unload image
- `ptxemu_image_kernel_name` — query kernel name
- `ptxemu_module_version` — module version

Additional symbols may appear in the baseline after later changes (e.g. `ptxemu_image_execute_named`).

## Test Reference

`tests/integration/cpptlm/test_libptxemu_abi_baseline.cpp` runs `nm -D` with the same `awk` filter and compares byte-equal to this file. Adding/removing any `ptxemu_*` symbol will fail the test until the baseline is regenerated.