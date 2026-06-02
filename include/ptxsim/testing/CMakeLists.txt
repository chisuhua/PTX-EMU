# PTXSIM Testing Helpers - Header-Only Infrastructure
#
# This directory contains header-only testing utilities for PTX simulation tests.
# These helpers provide instruction construction, verification, and shared memory
# support for unit/integration testing without depending on ANTLR or Catch2.
#
# Headers:
#   - instruction_helpers.h  : make_*() functions for StatementContext construction
#   - assertion_utils.h      : count_active_lanes(), count_at_pc(), check_mask()
#   - shared_memory.h        : allocate_shared(), write_shared(), read_shared()
#   - predicates.h           : Predicate manipulation helpers
#   - scheduler_utils.h      : Scheduler-related test utilities
#   - debug_utils.h          : Debug formatting helpers
#   - warp_test_utils.h      : Warp-level test utilities
#
# Usage in tests:
#   target_include_directories(my_test PRIVATE
#       ${PROJECT_SOURCE_DIR}/include
#   )
#
# Note: No library target is created - these are pure header utilities.
# Link against ptxsim_core or ptxir as needed for your test binary.

# Header-only directory - no targets needed
set(PTXSIM_TESTING_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/instruction_helpers.h
    ${CMAKE_CURRENT_SOURCE_DIR}/assertion_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_memory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/predicates.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/debug_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/warp_test_utils.h
)

# Expose headers for external tests via generated header collection
set(PTXSIM_TESTING_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Path to ptxsim testing headers")