#include <gtest/gtest.h>

#include "drishti/benchmark/benchmark.h"

namespace drishti::benchmark {

TEST(BenchmarkSuiteTest, ParseWorkloadTypes) {
    EXPECT_EQ(parse_workload_type("gemm"), WorkloadType::GEMM);
    EXPECT_EQ(parse_workload_type("reduction"), WorkloadType::Reduction);
    EXPECT_EQ(parse_workload_type("fusion"), WorkloadType::Fusion);
    EXPECT_EQ(parse_workload_type("transpose"), WorkloadType::Transpose);
    EXPECT_EQ(parse_workload_type("attention"), WorkloadType::Attention);
    EXPECT_EQ(parse_workload_type("all"), WorkloadType::All);
    EXPECT_EQ(parse_workload_type("unknown"), WorkloadType::All);
}

TEST(BenchmarkSuiteTest, RunFullBenchmarkSuite) {
    BenchmarkConfig cfg;
    cfg.filter = WorkloadType::All;
    cfg.num_elements = 262144;
    cfg.repeats = 3;

    auto rep = run_benchmark_suite(cfg);

    EXPECT_TRUE(rep.ok);
    EXPECT_EQ(rep.total_workloads, 5u);
    EXPECT_EQ(rep.passed_workloads, 5u);
    EXPECT_GE(rep.rank_matched_workloads, 4u); // High prediction ranking accuracy
    EXPECT_LT(rep.avg_prediction_error_pct, 25.0); // Within tolerance

    for (const auto& w : rep.workloads) {
        EXPECT_TRUE(w.ok);
        EXPECT_TRUE(w.correct);
        EXPECT_FALSE(w.name.empty());
        EXPECT_FALSE(w.regime.empty());
        EXPECT_GT(w.candidates.size(), 1u);
        EXPECT_GT(w.speedup_factor, 1.0);
    }

    const std::string json = benchmark_to_json(rep);
    EXPECT_NE(json.find("\"schema\": \"drishti.benchmark/v1\""), std::string::npos);

    const std::string text = format_benchmark_report(rep);
    EXPECT_NE(text.find("PHASE 16: GENERALIZATION BENCHMARK SUITE REPORT"), std::string::npos);
}

TEST(BenchmarkSuiteTest, RunSingleWorkload) {
    BenchmarkConfig cfg;
    cfg.filter = WorkloadType::GEMM;
    cfg.num_elements = 65536;
    cfg.repeats = 3;

    auto rep = run_benchmark_suite(cfg);

    EXPECT_TRUE(rep.ok);
    EXPECT_EQ(rep.total_workloads, 1u);
    EXPECT_EQ(rep.workloads[0].name, "GEMM");
    EXPECT_TRUE(rep.workloads[0].correct);
}

}  // namespace drishti::benchmark
