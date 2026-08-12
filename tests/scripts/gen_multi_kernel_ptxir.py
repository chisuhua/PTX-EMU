#!/usr/bin/env python3
"""Generate multi-kernel PTXIR binary from multi_kernel_basic.ptx.

Emits a valid PTXIR v4 binary with 3 KernelEntry records in the manifest.
This is a static generator — it does not invoke the full PTX parser.
Used to bootstrap the multi-entry fixture for Phase C2 tests.
"""
import argparse
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PTXIR_MAGIC = b"PTXI"
PTXIR_VERSION = 4
HEADER_SIZE = 24
TOC_ENTRY_SIZE = 6

# Section types
REGDECL = 1
KERNEL = 3
MANIFEST = 6
STRING_TABLE = 5


def emit_manifest_section(kernel_names):
    """Build MANIFEST section bytes for the given kernel names."""
    buf = bytearray()

    # cubin_hash: 32 bytes (zero-padded)
    buf.extend(b"\x00" * 32)

    # kernel_name (backward-compat): first kernel name, NUL-terminated
    first_name = kernel_names[0].encode("ascii")
    buf.extend(first_name)
    buf.append(0)  # NUL terminator

    # ptx_address_size
    buf.append(64)

    # params: uint16 count (0) + no params
    buf.extend(struct.pack("<H", 0))

    # v2 kernels vector
    kernel_count = len(kernel_names)
    buf.extend(struct.pack("<H", kernel_count))

    for name in kernel_names:
        name_bytes = name.encode("ascii")
        # kernel name: NUL-terminated
        buf.extend(name_bytes)
        buf.append(0)
        # arg_count: 4 bytes little-endian = 0
        buf.extend(struct.pack("<I", 0))
        # arg_byte_size: 4 bytes little-endian = 0
        buf.extend(struct.pack("<I", 0))

    return bytes(buf)


def build_ptxir(manifest_bytes):
    """Build complete PTXIR v4 binary with given manifest bytes."""
    # Layout: header(24) + TOC(4×6=24) + REGDECL + KERNEL + MANIFEST + STRING_TABLE
    # Offsets computed sequentially
    header_end = HEADER_SIZE
    toc_offset = header_end
    regdecl_offset = toc_offset + 4 * TOC_ENTRY_SIZE
    kernel_offset = regdecl_offset + 4  # uint32 count = 0
    manifest_offset = kernel_offset + 4  # uint32 count = 0
    string_table_offset = manifest_offset + len(manifest_bytes)

    # String table: count=0
    string_table = struct.pack("<I", 0)

    # Build TOC entries [REGDECL, KERNEL, MANIFEST, STRING_TABLE]
    toc = bytearray()
    toc.extend(struct.pack("BB", REGDECL, 0))  # type, reserved
    toc.extend(struct.pack("<I", regdecl_offset))
    toc.extend(struct.pack("BB", KERNEL, 0))
    toc.extend(struct.pack("<I", kernel_offset))
    toc.extend(struct.pack("BB", MANIFEST, 0))
    toc.extend(struct.pack("<I", manifest_offset))
    toc.extend(struct.pack("BB", STRING_TABLE, 0))
    toc.extend(struct.pack("<I", string_table_offset))

    # REGDECL section: count = 0
    regdecl_section = struct.pack("<I", 0)

    # KERNEL section: count = 0 (no statements, just manifest)
    kernel_section = struct.pack("<I", 0)

    # Build header
    header = bytearray(HEADER_SIZE)
    header[0:4] = PTXIR_MAGIC
    header[4:6] = struct.pack("<H", PTXIR_VERSION)
    header[6:8] = struct.pack("<H", 0)  # flags
    header[8:10] = struct.pack("<H", 4)  # section_count
    header[10:12] = struct.pack("<H", 0)  # reserved
    header[12:16] = struct.pack("<I", string_table_offset)
    header[16:20] = struct.pack("<I", len(string_table))
    header[20:24] = struct.pack("<I", HEADER_SIZE)

    # Assemble
    out = bytearray()
    out.extend(header)
    out.extend(toc)
    out.extend(regdecl_section)
    out.extend(kernel_section)
    out.extend(manifest_bytes)
    out.extend(string_table)

    return bytes(out)


def main():
    parser = argparse.ArgumentParser(description="Generate multi-kernel PTXIR binary")
    parser.add_argument("--ptx", required=True, type=Path,
                        help="Input PTX file (for reference, not parsed)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output PTXIR binary path")
    args = parser.parse_args()

    if not args.ptx.exists():
        print(f"ERROR: PTX file not found: {args.ptx}", file=sys.stderr)
        sys.exit(1)

    # The 3 kernel names must match those declared in the PTX source
    kernel_names = ["vec_add", "mat_mul", "reduce_sum"]

    manifest_bytes = emit_manifest_section(kernel_names)
    ptxir_bytes = build_ptxir(manifest_bytes)

    args.output.write_bytes(ptxir_bytes)
    print(f"Wrote {len(ptxir_bytes)}-byte PTXIR to {args.output}")
    print(f"  Kernels: {', '.join(kernel_names)}")


if __name__ == "__main__":
    main()
