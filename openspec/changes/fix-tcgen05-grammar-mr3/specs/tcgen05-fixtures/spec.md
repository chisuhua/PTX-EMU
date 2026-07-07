## ADDED Requirements

### Requirement: All 12 tcgen05 PTX fixtures SHALL parse successfully
The system SHALL provide 12 `tests/ptx/tcgen05_*.ptx` files covering all
12 Blackwell tcgen05 instruction families (per PTX ISA §9.7.16), and
all of them SHALL parse successfully via `./tests/ptx/test_all_ptx.sh`.

The 12 fixtures are: `tcgen05_alloc.ptx`, `tcgen05_dealloc.ptx`,
`tcgen05_relinquish.ptx`, `tcgen05_ld.ptx`, `tcgen05_st.ptx`,
`tcgen05_cp.ptx`, `tcgen05_cp_multicast.ptx`, `tcgen05_mma.ptx`,
`tcgen05_mma_block_scale.ptx`, `tcgen05_commit.ptx`,
`tcgen05_wait.ptx`, `tcgen05_fence.ptx`.

#### Scenario: test_all_ptx.sh passes 12/12
- **WHEN** `./tests/ptx/test_all_ptx.sh` is run after all 12 fixtures are added
- **THEN** all 12 tcgen05 fixtures parse successfully
- **AND** the test suite output shows PASS for each fixture

#### Scenario: fixtures use real PTX syntax
- **WHEN** the fixture files are inspected
- **THEN** each fixture contains syntactically correct PTX per PTX ISA §9.7.16
- **AND** the syntax matches examples from the NVIDIA official spec
