add_executable(test_diagnosis
    test_diagnosis.cpp
    test_cross_vendor.cpp
)

target_link_libraries(test_diagnosis PUBLIC
    drishti_diagnosis
    drishti_backends
    drishti_optimizer
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_diagnosis COMMAND test_diagnosis)
set_tests_properties(test_diagnosis PROPERTIES RUN_SERIAL TRUE)
