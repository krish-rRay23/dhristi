add_executable(drishti_core_tests
    test_version.cpp
)

target_link_libraries(drishti_core_tests PRIVATE
    drishti::core
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(drishti_core_tests)
