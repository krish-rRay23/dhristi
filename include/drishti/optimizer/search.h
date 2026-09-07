#ifndef DRISHTI_OPTIMIZER_SEARCH_H
#define DRISHTI_OPTIMIZER_SEARCH_H

// Phase 14: Cost-Model-Guided Optimization Search for Drishti.
//
// Pre-execution analytical ranking of bounded optimization candidates followed by
// empirical GPU experiment execution, accuracy validation, and winner reporting.

#include <cstddef>
#include <string>
#include <vector>

#include "drishti/optimizer/cost_model.h"
#include "drishti/profiling/gpu_metrics.h"

namespace drishti::optimizer {

struct SearchCandidate {
    std::string candidate_id;    // e.g. "fused_block256"
    std::string title;           // human-readable description
    std::string transformation;  // e.g. "affine-loop-fusion"

    // Cost Model Features & Analytical Prediction (Phase 1)
    CostModelFeatures features;
    CostModelEstimate predicted_cost;
    std::size_t predicted_rank = 0;  // 1-indexed

    // Empirical GPU Execution Results (Phase 2)
    bool executed = false;
    profiling::GpuProfileMetrics measured_metrics;
    double measured_speedup_percent = 0.0;  // relative to baseline candidate
    std::size_t measured_rank = 0;   // 1-indexed

    // Validation & Prediction Accuracy
    PredictionValidation validation;
};

struct SearchReport {
    bool ok = false;
    std::string error;

    std::size_t num_elements = 65536;
    profiling::GpuDeviceModel device;

    std::vector<SearchCandidate> candidates;  // sorted by predicted_rank

    std::string predicted_winner_id;
    std::string actual_winner_id;
    bool prediction_matches_actual = false;

    std::string summary;
};

struct SearchConfig {
    std::size_t num_elements = 262144;
    std::size_t candidate_budget = 4;
    bool execute_all = true;
};

// Run the full cost-model-guided optimization search flow. Never throws.
[[nodiscard]] SearchReport run_optimization_search(const SearchConfig& cfg = {});

// Machine-readable rendering for search reports.
[[nodiscard]] std::string search_to_json(const SearchReport& r);

// Human-readable report for `drishti optimize --search`.
[[nodiscard]] std::string format_search_report(const SearchReport& r);

}  // namespace drishti::optimizer

#endif  // DRISHTI_OPTIMIZER_SEARCH_H
