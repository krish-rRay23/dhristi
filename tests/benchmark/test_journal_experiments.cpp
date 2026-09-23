#include <gtest/gtest.h>
#include "drishti/benchmark/journal_experiments.h"

namespace drishti::benchmark {

TEST(JournalExperimentsTest, RunSuiteReturnsValidResults) {
    JournalBenchmarkReport rep = run_journal_experiments();
    EXPECT_TRUE(rep.ok);
    EXPECT_GT(rep.total_experiments, 0u);
    EXPECT_FALSE(rep.hardware_device.empty());
    EXPECT_FALSE(rep.timestamp.empty());
}

TEST(JournalExperimentsTest, JsonSerializationContainsRequiredFields) {
    JournalBenchmarkReport rep = run_journal_experiments();
    std::string json_str = journal_report_to_json(rep);
    EXPECT_NE(json_str.find("\"timestamp\""), std::string::npos);
    EXPECT_NE(json_str.find("\"hardware_device\""), std::string::npos);
    EXPECT_NE(json_str.find("\"total_experiments\""), std::string::npos);
    EXPECT_NE(json_str.find("\"results\""), std::string::npos);
    EXPECT_NE(json_str.find("\"latency_ms\""), std::string::npos);
    EXPECT_NE(json_str.find("\"speedup_vs_baseline\""), std::string::npos);
    EXPECT_NE(json_str.find("\"vram_bandwidth_gbps\""), std::string::npos);
    EXPECT_NE(json_str.find("\"correctness\""), std::string::npos);
}

TEST(JournalExperimentsTest, FormatJournalReportIsNonEmpty) {
    JournalBenchmarkReport rep = run_journal_experiments();
    std::string formatted = format_journal_report(rep);
    EXPECT_FALSE(formatted.empty());
    EXPECT_NE(formatted.find("DRISHTI REPRODUCIBLE JOURNAL RESEARCH BENCHMARK"), std::string::npos);
}

TEST(JournalExperimentsTest, AblationsContainsFullDrishtiAndMinusVectorization) {
    JournalBenchmarkReport rep = run_journal_experiments();
    bool found_full = false;
    bool found_vec = false;
    bool found_fusion = false;
    for (const auto& r : rep.results) {
        if (r.ablation_id == "Full Drishti") found_full = true;
        if (r.ablation_id == "Drishti - Vectorization") found_vec = true;
        if (r.ablation_id == "Drishti - Kernel Fusion") found_fusion = true;
    }
    EXPECT_TRUE(found_full);
    EXPECT_TRUE(found_vec);
    EXPECT_TRUE(found_fusion);
}

}  // namespace drishti::benchmark
