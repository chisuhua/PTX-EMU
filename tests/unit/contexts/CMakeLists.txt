# T2-3: Unit tests for god-class POD split contexts/

add_catch_test(unit_exec_state_pod
    contexts/test_exec_state_pod.cpp
)
set_tests_properties(unit_exec_state_pod PROPERTIES LABELS "unit;contexts")

add_catch_test(unit_register_predicate_pod
    contexts/test_register_predicate_pod.cpp
)
set_tests_properties(unit_register_predicate_pod PROPERTIES LABELS "unit;contexts")

add_catch_test(unit_memory_ref_pod
    contexts/test_memory_ref_pod.cpp
)
set_tests_properties(unit_memory_ref_pod PROPERTIES LABELS "unit;contexts")

add_catch_test(unit_program_ref_pod
    contexts/test_program_ref_pod.cpp
)
set_tests_properties(unit_program_ref_pod PROPERTIES LABELS "unit;contexts")

add_catch_test(unit_lane_mask_pod
    contexts/test_lane_mask_pod.cpp
)
set_tests_properties(unit_lane_mask_pod PROPERTIES LABELS "unit;contexts")

add_catch_test(unit_warp_identity_pod
    contexts/test_warp_identity_pod.cpp
)
set_tests_properties(unit_warp_identity_pod PROPERTIES LABELS "unit;contexts")

add_catch_test(unit_backend_links_pod
    contexts/test_backend_links_pod.cpp
)
set_tests_properties(unit_backend_links_pod PROPERTIES LABELS "unit;contexts")

# T2-3 A3a: ThreadContext 4-POD facade members (additive, defaults only)
add_catch_test(unit_thread_context_pod_facade
    contexts/test_thread_context_pod_facade.cpp
)
set_tests_properties(unit_thread_context_pod_facade PROPERTIES LABELS "unit;contexts;pod;thread_context")