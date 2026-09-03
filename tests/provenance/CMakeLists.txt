add_executable(test_provenance
    test_provenance.cpp
)

target_link_libraries(test_provenance PUBLIC
    drishti_provenance
    drishti_analysis
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME test_provenance COMMAND test_provenance)