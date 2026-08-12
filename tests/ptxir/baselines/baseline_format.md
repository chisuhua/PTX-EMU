# Cute RMSNorm Output Baseline Format

Each `.bin` file in this directory is a magic-prefixed binary:

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0      | 10   | magic | `PTXR_OUT\0\0` (10 bytes literal: P T X R _ O U T NUL NUL) |
| 10     | 4    | size  | LE u32 — byte count of payload that follows |
| 14     | size | bytes | Raw output buffer from simulator |

**Total minimum size: 14 bytes** (empty payload = 10 + 4 + 0).

**Stability contract**: baselines are immutable once committed. Any change requires:
1. New file with versioned name (e.g. `cute_rmsnorm_output_v2.bin`)
2. Update `current_baseline` symlink
3. Document D3 mutation in design doc

**Regeneration**:
```bash
cd build && ctest -R "cute_rmsnorm output dump" -V --environment DUMP_OUTPUT=1
python3 -c "import struct; data=open('/tmp/cute_rmsnorm_out.bin','rb').read(); \
    open('tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin','wb') \
    .write(b'PTXR_OUT\0\0' + struct.pack('<I',len(data)) + data)"
```
