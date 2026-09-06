add_executable(test_correlation
    test_correlation.cpp
)

target_link_libraries(test_correlation PUBLIC
    drishti_correlation
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_correlation COMMAND test_correlation)
set_tests_properties(test_correlation PROPERTIES RUN_SERIAL TRUE)
