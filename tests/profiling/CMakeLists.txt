add_executable(test_gpu_profile
    test_gpu_profile.cpp
)

target_link_libraries(test_gpu_profile PUBLIC
    drishti_profiling
    drishti_backends
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_gpu_profile COMMAND test_gpu_profile)
set_tests_properties(test_gpu_profile PROPERTIES RUN_SERIAL TRUE)

