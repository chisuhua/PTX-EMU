# tests/three_mode_testing/CMakeLists.txt
#
# Four-Mode Testing Framework
# Auto-detects all *_mode*.cpp files (mode1, mode2, mode3a, mode3b)
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

file(GLOB THREE_MODE_SOURCES CONFIGURE_DEPENDS "*.cpp")

set(THREE_MODE_TESTS "")
foreach(source IN LISTS THREE_MODE_SOURCES)
    get_filename_component(basename ${source} NAME)
    if(basename MATCHES "_mode[0-9]+[abc]?\\.cpp$")
        list(APPEND THREE_MODE_TESTS ${source})
    endif()
    if(basename MATCHES "_ptxir_serialization\\.cpp$")
        list(APPEND THREE_MODE_TESTS ${source})
    endif()
    if(basename MATCHES "_four_mode_flow\\.cpp$")
        list(APPEND THREE_MODE_TESTS ${source})
    endif()
endforeach()

message(STATUS "Four-mode tests detected: ${THREE_MODE_TESTS}")

foreach(source IN LISTS THREE_MODE_TESTS)
    get_filename_component(test_name ${source} NAME_WE)

    add_executable(${test_name}
        ${source}
        ${THREE_MODE_BASE}
    )

    target_include_directories(${test_name} PRIVATE ${THREE_MODE_INCLUDES})
    target_link_directories(${test_name} PRIVATE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${CMAKE_SOURCE_DIR}/lib)
    target_link_libraries(${test_name} PRIVATE ptxsim cudart antlr4_shared ptxir_writer ptxir_reader -Wl,--as-needed -ldl -lpthread)
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests)
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

if(THREE_MODE_TESTS)
    set(THREE_MODE_TEST_NAMES "")
    foreach(src IN LISTS THREE_MODE_TESTS)
        get_filename_component(name ${src} NAME_WE)
        list(APPEND THREE_MODE_TEST_NAMES ${name})
    endforeach()
    set_tests_properties(${THREE_MODE_TEST_NAMES} PROPERTIES LABELS "three_mode")
endif()