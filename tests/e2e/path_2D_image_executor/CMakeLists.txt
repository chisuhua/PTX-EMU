add_catch_test(e2e_image_executor_output test_image_executor_output.cpp)
set_tests_properties(e2e_image_executor_output PROPERTIES
    LABELS "e2e;path_2D" TIMEOUT 60
)
