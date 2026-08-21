# fix-phase0-gate1-dgpu-bar-leak

Phase 0 byte-identical gate 1 failure: 10 tlm::gpu::DGpuBar symbols leaked into libcudart.so via --whole-archive cpptlm_core. Minimal fix: remove the --whole-archive flag (preserving ABI consumers), regenerate baseline with audit log.
