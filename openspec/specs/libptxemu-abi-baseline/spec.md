# libptxemu-abi-baseline Specification

## Purpose
TBD - created by archiving change fix-path-coverage-gaps-followup. Update Purpose after archive.
## Requirements
### Requirement: Address-stripping for ABI baseline comparison

The `integration_libptxemu_abi_baseline` test MUST strip load addresses from comparison and compare only symbol type + name.

#### Scenario: Strip load addresses from nm output

- **WHEN** the test runs `nm -D libptxemu_device.so`
- **THEN** output MUST contain only `<type-letter> <symbol-name>` format
- **AND** load addresses (hex `00000000` prefix) MUST be absent

#### Scenario: Sort and deduplicate

- **WHEN** multiple symbols have the same name and type
- **THEN** the comparison output MUST be sorted alphabetically and deduplicated

#### Scenario: Regeneration command documented

- **WHEN** ABI legitimately changes (symbol added/removed/renamed)
- **THEN** `tests/integration/cpptlm/baselines/README.md` MUST provide the regeneration command

### Requirement: ABI baseline contains exactly the ptxemu_ symbols

The baseline file `libptxemu_abi_baseline.txt` MUST contain exactly one line per `ptxemu_*` symbol defined in `libptxemu_device.so`.

#### Scenario: ADR-0029 5-symbol baseline

- **WHEN** the baseline is regenerated
- **THEN** it MUST include at minimum the 5 ABI symbols defined in ADR-0029:
  - `T ptxemu_image_load`
  - `T ptxemu_image_execute`
  - `T ptxemu_image_unload`
  - `T ptxemu_image_kernel_name`
  - `T ptxemu_module_version`

#### Scenario: Additional symbols allowed

- **WHEN** new `ptxemu_*` symbols are added to the library after ADR-0029
- **THEN** they MUST also appear in the baseline after regeneration

### Requirement: Baseline regeneration via documented command

A `baselines/README.md` file MUST document how to regenerate the baseline.

#### Scenario: nm + awk + grep + sort -u pipeline

- **WHEN** the documented regeneration command is executed
- **THEN** the new baseline MUST match the format `<type-letter> <symbol-name>` per line

#### Scenario: No manual editing

- The baseline file MUST be regenerated via the documented command, not manually edited

