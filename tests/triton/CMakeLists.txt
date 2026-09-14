add_executable(test_triton
    test_triton_integration.cpp
)

target_link_libraries(test_triton PRIVATE
    drishti_common
    drishti_core
    drishti_backends
    drishti_provenance
    drishti_profiling
    drishti_diagnosis
    drishti_optimizer
    drishti_correlation
    drishti::triton
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(test_triton PROPERTIES RUN_SERIAL TRUE)
