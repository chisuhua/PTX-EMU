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
    OBJ_OUT = HERE / "vec_add.o"
    # 1. nvcc -c -> .o (compute_100 arch for ptxir_embed compatibility)
    subprocess.check_call([
        "nvcc", "-c", "-arch=compute_100", "-O2", str(KERNELS_CU),
        "-o", str(OBJ_OUT)
    ])

    # 2. cuobjdump -ptx -> .ptx (then strip to kernel entry)
    raw_ptx = HERE / "vec_add.raw.ptx"
    subprocess.check_call([
        "cuobjdump", "-ptx", str(OBJ_OUT)
    ], stdout=raw_ptx.open("w"))

    ptx_content = raw_ptx.read_text()
    entry_idx = ptx_content.find(".visible .entry")
    if entry_idx != -1:
        PTX_OUT.write_text(ptx_content[entry_idx:])
    else:
        PTX_OUT.write_text(ptx_content)

    # 3. ptxir_embed --in-cubin X.o --in-ptx X.ptx --kernel-name K --out X.embedded
    subprocess.check_call([
        str(PTXIR_EMBED),
        "--in-cubin", str(OBJ_OUT),
        "--in-ptx", str(PTX_OUT),
        "--kernel-name", "vec_add",
        "--out", str(EMBED_OUT),
    ])

    # 4. Extract PTXIR section body from embedded cubin
    # Format from ptxir_embed: [prefix][ptxir_body][u32_size_LE][8_byte_magic "PTXEMB\x01\x00"]
    embedded = EMBED_OUT.read_bytes()
    magic = embedded[-8:]
    if magic != b"PTXEMB\x01\x00":
        raise RuntimeError(f"expected PTXEMB trailer magic, got {magic!r}")
    size = struct.unpack("<I", embedded[-12:-8])[0]
    body = embedded[-(12 + size):-12]
    print(f"body size={size} magic={magic!r}")

    blob = b"PTXIR" + struct.pack("<I", len(body)) + body
    BLOB_OUT.write_bytes(blob)
    print(f"wrote {len(blob)} bytes to {BLOB_OUT}")

if __name__ == "__main__":
    main()
