## ADDED Requirements

### Requirement: cpptlm_set_driver ABI entry point

The system SHALL provide an `extern "C"` ABI entry point `cpptlm_set_driver` for registering the `IPtxEmuDriver*` across `.so` boundary.

- SHALL be declared in `include/cudart/cpptlm_bridge.h`
- SHALL be implemented in `src/cudart/cudart_sim.cpp`
- SHALL use `PTXEMU_BRIDGE_API` visibility attribute
- SHALL accept `tlm::IPtxEmuDriver*` as parameter (nullable)
- SHALL store in global pointer `g_ptx_emu_driver`
- SHALL be idempotent (nullptr resets to no-driver state)
- SHALL bump `CPPTLMBRIDGE_VERSION` to 2

#### Scenario: cpptlm_set_driver stores valid driver pointer

- **WHEN** `cpptlm_set_driver(driver)` is called with a non-null `IPtxEmuDriver*`
- **THEN** `g_ptx_emu_driver` SHALL point to the provided driver
- **THEN** driver SHALL be usable by bridge path `on_complete` callbacks

#### Scenario: cpptlm_set_driver resets on nullptr

- **WHEN** `cpptlm_set_driver(nullptr)` is called
- **THEN** `g_ptx_emu_driver` SHALL be set to `nullptr`
- **THEN** subsequent bridge path callbacks SHALL skip driver operations (no crash)

#### Scenario: initialize_environment creates and sets driver

- **WHEN** `initialize_environment()` is called
- **THEN** SHALL create `PtxEmuDriverShim` with the initialized `g_gpu_context`
- **THEN** SHALL call `cpptlm_set_driver(shim)` to register it
- **THEN** `g_ptx_emu_driver` SHALL be non-null after initialization

#### Scenario: CPPTLMBRIDGE_VERSION is bumped to 2

- **WHEN** the header `include/cudart/cpptlm_bridge.h` is compiled
- **THEN** `CPPTLMBRIDGE_VERSION` SHALL equal 2