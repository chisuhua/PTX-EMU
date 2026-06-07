# tests/unit/testing/CMakeLists.txt
# Unit tests for ptxsim::testing helpers in include/ptxsim/testing/

add_catch_test(unit_memory_test_utils
    test_memory_test_utils.cpp
)
set_tests_properties(unit_memory_test_utils PROPERTIES LABELS "unit;testing")
