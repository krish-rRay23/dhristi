// Phase 13 tests: Hardware-Aware Analytical Cost Model.

#include "drishti/optimizer/cost_model.h"
#include "drishti/profiling/gpu_metrics.h"

#include <gtest/gtest.h>
#include <string>

namespace {

using drishti::optimizer::CostModelFeatures;
using drishti::optimizer::CostModelEstimate;
using drishti::optimizer::estimate_kernel_cost;
using drishti::optimizer::validate_prediction;
using drishti::optimizer::cost_estimate_to_json;
using drishti::optimizer::format_cost_estimate_report;

TEST(CostModel, RooflineIdentifiesMemoryBoundRegime) {
    CostModelFeatures f;
    f.kernel_label = "vecadd";
    f.num_elements = 1048576;
    f.block_size = 256;
    f.bytes_per_element = 12.0;  // 2 ld + 1 st f32
    f.flops_per_element = 1.0;   // 1 addf
    f.launch_count = 1;

    CostModelEstimate est = estimate_kernel_cost(f);
    EXPECT_TRUE(est.ok);
    EXPECT_EQ(est.bottleneck_regime, "memory-bound");
    EXPECT_GT(est.memory_ms, est.compute_ms);
    EXPECT_GT(est.predicted_kernel_ms, 0.0);
    EXPECT_GT(est.predicted_total_ms, est.predicted_kernel_ms);
    EXPECT_FALSE(est.reasoning.empty());
}

TEST(CostModel, LaunchOverheadScalesWithKernelLaunches) {
    CostModelFeatures f_single;
    f_single.num_elements = 65536;
    f_single.launch_count = 1;

    CostModelFeatures f_double = f_single;
    f_double.launch_count = 2;

    CostModelEstimate est_single = estimate_kernel_cost(f_single);
    CostModelEstimate est_double = estimate_kernel_cost(f_double);

    EXPECT_GT(est_double.launch_overhead_ms, est_single.launch_overhead_ms);
    EXPECT_DOUBLE_EQ(est_double.launch_overhead_ms, est_single.launch_overhead_ms * 2.0);
}

TEST(CostModel, PredictsFusionSpeedupAgainstRTX3050Measurements) {
    // 1. Baseline unfused pipeline: 2 launches, 24 B/element
    CostModelFeatures f_base;
    f_base.kernel_label = "fusedemo_unfused";
    f_base.num_elements = 262144;  // 256x1024
    f_base.block_size = 256;
    f_base.bytes_per_element = 24.0;
    f_base.flops_per_element = 2.0;
    f_base.launch_count = 2;

    // 2. Candidate fused pipeline: 1 launch, 16 B/element
    CostModelFeatures f_fused = f_base;
    f_fused.kernel_label = "fusedemo_fused";
    f_fused.bytes_per_element = 16.0;
    f_fused.launch_count = 1;

    CostModelEstimate est_base = estimate_kernel_cost(f_base);
    CostModelEstimate est_fused = estimate_kernel_cost(f_fused);

    EXPECT_GT(est_base.predicted_total_ms, est_fused.predicted_total_ms);

    // Speedup prediction
    const double predicted_speedup = (est_base.predicted_total_ms - est_fused.predicted_total_ms) /
                                     est_base.predicted_total_ms * 100.0;
    EXPECT_GT(predicted_speedup, 15.0);  // Predicts >= 15% speedup

    // Validate against physical RTX 3050 measurements (0.0440 ms vs 0.0338 ms)
    const double measured_base_ms = 0.0440;
    const double measured_fused_ms = 0.0338;

    auto val_base = validate_prediction(est_base.predicted_kernel_ms, measured_base_ms);
    auto val_fused = validate_prediction(est_fused.predicted_kernel_ms, measured_fused_ms);

    EXPECT_TRUE(val_base.is_accurate) << val_base.assessment;
    EXPECT_TRUE(val_fused.is_accurate) << val_fused.assessment;
}

TEST(CostModel, FormatsJsonAndTextReport) {
    CostModelFeatures f;
    f.kernel_label = "test_kernel";
    CostModelEstimate est = estimate_kernel_cost(f);

    std::string json = cost_estimate_to_json(est);
    EXPECT_NE(json.find("drishti.cost_estimate/v1"), std::string::npos);

    std::string text = format_cost_estimate_report(est);
    EXPECT_NE(text.find("ANALYTICAL & CALIBRATED LATENCY ESTIMATES"), std::string::npos);
}

}  // namespace
