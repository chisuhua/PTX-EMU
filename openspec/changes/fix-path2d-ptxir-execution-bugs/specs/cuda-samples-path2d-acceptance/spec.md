# cuda-samples-path2d-acceptance Specification

## ADDED Requirements

### Requirement: Compile an unmodified CUDA Sample into a PTXIR image

The acceptance harness SHALL compile an unmodified NVIDIA CUDA Samples kernel with the available CUDA toolkit using `nvcc -arch=sm_100`, extract its PTX, and create a standalone or embedded PTXIR image using the repository's existing PTXIR tools.

#### Scenario: NVIDIA vectorAdd compilation

- **WHEN** the harness processes `cpp/0_Introduction/vectorAdd/vectorAdd.cu` from the NVIDIA CUDA Samples checkout
- **THEN** compilation and PTXIR image generation succeed without modifying the sample source

### Requirement: Execute the Sample through path_2D

The acceptance harness SHALL load the generated PTXIR image through `libptxemu_device.so`, select the compiled kernel symbol, execute it through `ptxemu_image_execute_named`, and copy the output buffer back through the simulator memory path.

#### Scenario: vectorAdd execution

- **WHEN** the generated vectorAdd image is loaded and executed with valid input buffers and launch dimensions
- **THEN** execution completes without an image, completion, or global-memory address error

### Requirement: Verify numerical output

The acceptance harness SHALL verify every vectorAdd output element against `A[i] + B[i]` with an absolute error below `1e-5`.

#### Scenario: Correct vectorAdd result

- **WHEN** the path_2D execution completes
- **THEN** all output elements satisfy `abs(A[i] + B[i] - C[i]) < 1e-5`

### Requirement: Third-party source transparency

The acceptance workflow SHALL keep the NVIDIA CUDA Sample source unchanged. Any include paths, compiler flags, temporary files, launcher metadata, or PTXIR extraction steps SHALL be supplied by the harness or PTX-EMU tooling.

#### Scenario: Source checksum unchanged

- **WHEN** the harness compiles and executes the sample
- **THEN** the sample source file has no source-tree modifications
