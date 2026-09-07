// Phase 14 tests: Cost-Model-Guided Optimization Search.

#include "drishti/optimizer/search.h"
#include "drishti/profiling/gpu_metrics.h"

#include <gtest/gtest.h>
#include <string>

namespace {

using drishti::optimizer::SearchConfig;
using drishti::optimizer::SearchReport;
using drishti::optimizer::run_optimization_search;
using drishti::optimizer::search_to_json;
using drishti::optimizer::format_search_report;

TEST(OptimizationSearch, PreExecutionRankingRanksFusedBeforeBaseline) {
    SearchConfig cfg;
    cfg.num_elements = 262144;
    cfg.candidate_budget = 4;

    SearchReport rep = run_optimization_search(cfg);
    EXPECT_TRUE(rep.ok);
    EXPECT_EQ(rep.candidates.size(), 4u);

    // Rank 1 candidate must be predicted to be faster than baseline
    EXPECT_NE(rep.predicted_winner_id, "baseline_unfused");
    EXPECT_FALSE(rep.predicted_winner_id.empty());

    // Verify candidate ranks are 1..4
    for (std::size_t i = 0; i < rep.candidates.size(); ++i) {
        EXPECT_EQ(rep.candidates[i].predicted_rank, i + 1);
        EXPECT_TRUE(rep.candidates[i].executed);
        EXPECT_TRUE(rep.candidates[i].measured_metrics.correct);
        EXPECT_GT(rep.candidates[i].measured_metrics.kernel_ms_min, 0.0);
    }

    // Verify actual winner match
    EXPECT_FALSE(rep.actual_winner_id.empty());
    EXPECT_NE(rep.actual_winner_id, "baseline_unfused");

    // JSON and Text formatting
    std::string json = search_to_json(rep);
    EXPECT_NE(json.find("drishti.search/v2"), std::string::npos);

    std::string text = format_search_report(rep);
    EXPECT_NE(text.find("Dṛṣṭi Hardware-Calibrated Optimization Search"), std::string::npos);
}

}  // namespace
