add_executable(drishti_analysis_tests
    test_mlir_parse.cpp
    test_llvm_integration.cpp
)

target_link_libraries(drishti_analysis_tests PRIVATE
    drishti::analysis
    drishti::provenance
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(drishti_analysis_tests)
