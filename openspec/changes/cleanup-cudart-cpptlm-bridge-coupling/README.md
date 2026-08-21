# cleanup-cudart-cpptlm-bridge-coupling

Complete removal of PTX-EMU-side cpptlm bridge coupling: delete PtxEmuDriverShim.{h,cpp}, stub_bridge.h, all g_cpptlm_bridge consumers in cudart_sim.cpp + memory.cpp, bridge-specific tests, BUILD_LIB_CPPTLM_CUDART macro, EMU_COSIM/PTX_EMU_MAX_ADVANCE_CYCLES env vars. KEEP generate_kernel_id() (used by cudaStreamCreate non-bridge path). NOT in scope: reversal direction (no PtxEmuSubmodule, no dlopen to libptxemu_device.so).
