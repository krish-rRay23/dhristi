add_executable(test_rocm_backend
    test_rocm_backend.cpp
)

target_link_libraries(test_rocm_backend PUBLIC
    drishti_backends
    drishti_profiling
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_rocm_backend COMMAND test_rocm_backend)
