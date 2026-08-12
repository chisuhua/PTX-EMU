#!/usr/bin/env python3
"""
Build PTXIR-prefixed blob for cuModuleLoadData.
cuModuleLoadData expects: "PTXIR" (5 bytes) + LE u32 size + body.
ptxir_embed appends a different format (prefix + section + LE u32 size + 8-byte magic at END).
So we extract the body from ptxir_embed output and re-prefix it.
"""
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PTXEMU_ROOT = HERE.parent.parent.parent
PTXIR_EMBED = PTXEMU_ROOT / "build" / "bin" / "ptxir_embed"

KERNELS_CU = HERE / "vec_add.cu"
CUBIN_OUT = HERE / "vec_add.cubin"
PTX_OUT = HERE / "vec_add.ptx"
EMBED_OUT = HERE / "vec_add.embedded.cubin"
BLOB_OUT = HERE / "vec_add.ptxir_blob"

def main():
    # 1. nvcc -> .cubin
    subprocess.check_call([
        "nvcc", "-arch=sm_100", "-O2", "-c", str(KERNELS_CU),
        "-o", str(CUBIN_OUT), "--cubin"
    ])

    # 2. ptxir_embed --in-cubin X.cubin --in-ptx X.ptx --kernel-name K --out X.embedded
    subprocess.check_call([
        str(PTXIR_EMBED),
        "--in-cubin", str(CUBIN_OUT),
        "--in-ptx", str(PTX_OUT),
        "--kernel-name", "vec_add",
        "--out", str(EMBED_OUT),
    ])

    # 3. Extract PTXIR section body from embedded cubin
    embedded = EMBED_OUT.read_bytes()
    trailer = embedded[-16:]
    size1, size2 = struct.unpack("<II", trailer[:8])
    magic = trailer[8:16]
    print(f"size1={size1} size2={size2} magic={magic!r}")

    body = embedded[:-(16)]
    blob = b"PTXIR" + struct.pack("<I", len(body)) + body
    BLOB_OUT.write_bytes(blob)
    print(f"wrote {len(blob)} bytes to {BLOB_OUT}")

if __name__ == "__main__":
    main()
