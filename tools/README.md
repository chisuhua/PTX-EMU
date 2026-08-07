# PTXIR Embed/Extract Tools

Tools for creating and manipulating PTXIR-Embedded binaries.

## ptxir_embed

Embed a PTXIR section (and optional manifest) into a host binary (cubin/object/executable).

```bash
# Embed PTXIR generated from a PTX file on the fly
build/bin/ptxir_embed \
  --in-cubin kernel.o \
  --in-ptx kernel.ptx \
  --kernel-name kernel_add \
  --out kernel.embedded.o

# Embed a pre-generated PTXIR file
build/bin/ptxir_embed \
  --in-exe program \
  --in-ptxir kernel.ptxir \
  --kernel-name kernel_add \
  --out program.embedded
```

Options:
- `--in-exe <path>` or `--in-cubin <path>`: the host binary to append the section to.
- `--in-ptxir <path>` or `--in-ptx <path>`: the PTXIR payload, or a PTX file to convert.
- `--kernel-name <name>`: required; name of the `.entry` kernel in the PTXIR manifest.
- `--out <path>`: output file.

The tool computes the SHA-256 hash of the host prefix and writes it into the MANIFEST section of the PTXIR, so that `ptxir_extract` can detect tampering.

## ptxir_extract

Extract the pure host binary and/or PTXIR section from a PTXIR-Embedded binary.

```bash
# Extract both the original binary and the PTXIR payload
build/bin/ptxir_extract \
  --in kernel.embedded.o \
  --out-cubin kernel.pure.o \
  --out-ptxir kernel.ptxir

# Pass-through for non-embedded input
build/bin/ptxir_extract --in kernel.o --out-cubin kernel.out.o
```

If the input is not a PTXIR-Embedded binary and `--out-cubin` is provided, the input is copied unchanged (pass-through).

## PTXIR_MODE

`PTXIR_MODE=auto` in the environment enables the PTXIR-Embedded dispatch path inside `__cudaRegisterFatBinary`. When enabled, PTX-EMU reads `/proc/self/exe`, detects the magic footer, and loads the embedded PTXIR instead of extracting PTX with `cuobjdump`. The default in `configs/*.ini` is `mode = off`, which keeps the original behavior. The environment variable overrides the INI setting.
