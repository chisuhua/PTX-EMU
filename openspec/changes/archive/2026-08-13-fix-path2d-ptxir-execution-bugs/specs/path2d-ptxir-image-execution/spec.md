# path2d-ptxir-image-execution Specification

## ADDED Requirements

### Requirement: Normalize PTXIR image formats before inspection

The path_2D image executor SHALL accept standalone PTXIR and PTXIR-Embedded images. Before reading a manifest, enumerating kernels, selecting a named kernel, or deserializing execution statements, it SHALL normalize an embedded image to its PTXIR section using the existing PTXIR loader. Malformed or unsupported images SHALL return a nonzero error without reporting a valid kernel.

#### Scenario: Standalone PTXIR image

- **WHEN** `ptxemu_image_load` receives a valid standalone PTXIR image
- **THEN** kernel-name lookup and named execution use the standalone PTXIR bytes and succeed for a valid kernel

#### Scenario: PTXIR-Embedded image

- **WHEN** `ptxemu_image_load` receives a valid PTXIR-Embedded executable or object image
- **THEN** manifest lookup extracts the embedded PTXIR section before parsing and returns the correct kernel metadata

#### Scenario: Malformed embedded image

- **WHEN** an image has an embedded-image marker but an invalid section size or malformed PTXIR body
- **THEN** load, manifest lookup, or execution returns a nonzero error and does not use cubin bytes as PTXIR input

### Requirement: Synchronous image execution

`ptxemu_image_execute` and `ptxemu_image_execute_named` SHALL return only after the submitted kernel has completed and completion callbacks have run, or after returning an execution error. The executor SHALL preserve its existing single-GPU-instance and serialized-launch invariants while driving completion.

#### Scenario: Store kernel completion

- **WHEN** a valid image-executor kernel writes a device buffer
- **THEN** the execute API returns only after the write is observable through the simulator's device-to-host copy path

#### Scenario: Missing GPU context

- **WHEN** image execution is requested while the shared GPU context is unavailable
- **THEN** the API returns a nonzero error and does not report successful execution

#### Scenario: Repeated launches

- **WHEN** the same valid image handle is executed multiple times
- **THEN** each call completes before returning and the stored image bytes remain unchanged

### Requirement: Image executor ABI compatibility

The implementation SHALL preserve the public declarations and version values in `cpptlm_module.h` and SHALL not modify `cpptlm_bridge.h` or `CPPTLMBRIDGE_VERSION`.

#### Scenario: ABI symbols remain stable

- **WHEN** `libptxemu_device.so` is rebuilt after the fix
- **THEN** the existing `ptxemu_image_*` symbols and module-version contract remain available without a public signature change
