## ADDED Requirements

### Requirement: PTX regression tests restored
The system SHALL provide regression test coverage for 7 categories of PTX instructions via restored unit tests.

#### Scenario: All 7 tests are registered in ctest
- **WHEN** running `ctest -L "unit;ptx"`
- **THEN** the output SHALL include `unit_ptx_integer`, `unit_ptx_float`, `unit_ptx_extended`, `unit_ptx_bitwise`, `unit_ptx_cvt`, `unit_ptx_ld_st`, and `unit_ptx_cvta`

#### Scenario: Each test passes when run individually
- **WHEN** running each restored test with `ctest -R <test_name> -V`
- **THEN** each test SHALL exit with code 0

#### Scenario: Full sanity check passes
- **WHEN** running `./scripts/sanity.sh`
- **THEN** all tests SHALL pass with no new failures beyond pre-existing known failures