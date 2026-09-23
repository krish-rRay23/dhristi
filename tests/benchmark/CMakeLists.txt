add_executable(test_benchmark
    test_benchmark.cpp
    test_journal_experiments.cpp
)

target_link_libraries(test_benchmark
    PRIVATE
        drishti_benchmark
        gtest
        gtest_main
)

gtest_discover_tests(test_benchmark)
