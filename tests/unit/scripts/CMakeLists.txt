# CMakeLists.txt for tests/unit/scripts/
#
# Python unit tests for scripts/check-docs-index.py.
# Different from Catch2 pattern (those use add_catch_test).

set(SCRIPTS_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/scripts)

# Aggregate runner
add_test(NAME unit_doc_validator_all
    COMMAND bash ${SCRIPTS_TESTS_DIR}/run_all_tests.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
set_tests_properties(unit_doc_validator_all PROPERTIES LABELS "unit;docs_validator")

# Individual test targets (for granular ctest filtering)
foreach(TEST_SCRIPT
    test_check_1_subdirs.py
    test_check_1_mismatch.py
    test_check_2_links_pass.py
    test_check_2_links_fail.py
    test_check_3_stats.py
    test_check_4_orphan.py
    test_check_5_banner.py
    test_check_6_skills.py
)
    get_filename_component(TEST_NAME ${TEST_SCRIPT} NAME_WLE)
    add_test(NAME unit_doc_${TEST_NAME}
        COMMAND python3 ${SCRIPTS_TESTS_DIR}/${TEST_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(unit_doc_${TEST_NAME} PROPERTIES LABELS "unit;docs_validator")
endforeach()
