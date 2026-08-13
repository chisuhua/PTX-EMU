# global-memory-address-normalization Specification

## ADDED Requirements

### Requirement: Normalize fake-runtime global pool addresses

The PTX-EMU global-memory access path SHALL accept absolute addresses returned from the fake runtime's `CudaDriver` global-memory pool. When an address lies within `[pool_base, pool_base + pool_size)`, the access path SHALL convert it to the pool-relative representation expected by `SimpleMemory` before validation and access.

#### Scenario: Absolute pool address

- **WHEN** an `ld.global` or `st.global` operation uses an address inside the configured global pool absolute range
- **THEN** the access is translated to the corresponding pool-relative address and completes without an out-of-bounds error

#### Scenario: Existing relative address

- **WHEN** a global-memory operation already uses a valid pool-relative address
- **THEN** the address remains valid and is not translated a second time

#### Scenario: Address outside the pool

- **WHEN** a global-memory operation uses an address outside both the valid absolute pool range and the valid relative pool range
- **THEN** the existing invalid-memory error behavior is preserved

### Requirement: Preserve non-global address spaces

The address normalization SHALL apply only at the verified common global-memory access boundary and SHALL NOT alter shared, local, parameter, constant, or texture address-space semantics.

#### Scenario: Shared-memory access

- **WHEN** a shared-memory instruction executes after the global-address fix
- **THEN** it uses the existing shared-memory addressing behavior without global-pool translation

#### Scenario: Parameter-space access

- **WHEN** a parameter-space load executes after the global-address fix
- **THEN** it uses the existing parameter-space addressing behavior without global-pool translation
