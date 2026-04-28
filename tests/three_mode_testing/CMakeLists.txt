# tests/three_mode_testing/CMakeLists.txt
#
# Three-Mode Testing Framework
# Auto-detects all *_mode*.cpp files in this directory
#

set(THREE_MODE_BASE
    ${CMAKE_SOURCE_DIR}/tests/catch_amalgamated.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
)

set(THREE_MODE_INCLUDES
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Auto-detect all *_mode*.cpp files
file(GLOB THREE_MODE_SOURCES CONFIGURE_DEPENDS "*.cpp")

# Filter to only mode test files (exclude test_helpers.cpp and any non-test files)
set(THREE_MODE_TESTS "")
foreach(source IN LISTS THREE_MODE_SOURCES)
    get_filename_component(basename ${source} NAME)
    # Match *_modeN.cpp pattern (N is a digit)
    if(basename MATCHES "_mode[0-9]+\\.cpp$")
        list(APPEND THREE_MODE_TESTS ${source})
    endif()
endforeach()

message(STATUS "Three-mode tests detected: ${THREE_MODE_TESTS}")

# Create executable and test for each detected file
foreach(source IN LISTS THREE_MODE_TESTS)
    get_filename_component(test_name ${source} NAME_WE)  # e.g., test_divergence_sync_standalone_mode1
    message(STATUS "Registering three-mode test: ${test_name}")

    add_executable(${test_name}
        ${source}
        ${THREE_MODE_BASE}
    )

    target_include_directories(${test_name} PRIVATE ${THREE_MODE_INCLUDES})
    target_link_directories(${test_name} PRIVATE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${CMAKE_SOURCE_DIR}/lib)
    target_link_libraries(${test_name} PRIVATE ptxsim cudart antlr4_shared -Wl,--as-needed -ldl -lpthread)
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests)
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

# Group all detected tests under "three_mode" label
if(THREE_MODE_TESTS)
    list(TRANSFORM THREE_MODE_TESTS GET NAME_WE THREE_MODE_TEST_NAMES)
    set_tests_properties(${THREE_MODE_TEST_NAMES} PROPERTIES LABELS "three_mode")
endif()
