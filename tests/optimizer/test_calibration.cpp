// Phase 15 tests: Hardware-Calibrated Cost Model & Ranking Accuracy.

#include "drishti/optimizer/cost_model.h"
#include "drishti/optimizer/search.h"
#include "drishti/profiling/gpu_metrics.h"

#include <gtest/gtest.h>
#include <string>

namespace {

using drishti::optimizer::CostModelFeatures;
using drishti::optimizer::CostModelEstimate;
using drishti::optimizer::HardwareCalibrationProfile;
using drishti::optimizer::estimate_kernel_cost;
using drishti::optimizer::estimate_kernel_cost_calibrated;
using drishti::optimizer::run_optimization_search;
using drishti::optimizer::SearchConfig;
using drishti::optimizer::SearchReport;

TEST(HardwareCalibration, DistinguishesBlockSizesUnambiguously) {
    CostModelFeatures f128;
    f128.kernel_label = "fused_128";
    f128.num_elements = 262144;
    f128.block_size = 128;
    f128.bytes_per_element = 16.0;

    CostModelFeatures f256 = f128;
    f256.kernel_label = "fused_256";
    f256.block_size = 256;

    CostModelFeatures f512 = f128;
    f512.kernel_label = "fused_512";
    f512.block_size = 512;

    // Uncalibrated predictions are identical for all 3 block sizes
    CostModelEstimate uncalib_128 = estimate_kernel_cost(f128);
    CostModelEstimate uncalib_256 = estimate_kernel_cost(f256);
    CostModelEstimate uncalib_512 = estimate_kernel_cost(f512);

    EXPECT_DOUBLE_EQ(uncalib_128.predicted_kernel_ms, uncalib_256.predicted_kernel_ms);
    EXPECT_DOUBLE_EQ(uncalib_256.predicted_kernel_ms, uncalib_512.predicted_kernel_ms);

    // Calibrated predictions distinguish block sizes according to hardware efficiency
    CostModelEstimate calib_128 = estimate_kernel_cost_calibrated(f128);
    CostModelEstimate calib_256 = estimate_kernel_cost_calibrated(f256);
    CostModelEstimate calib_512 = estimate_kernel_cost_calibrated(f512);

    EXPECT_TRUE(calib_128.is_calibrated);
    EXPECT_TRUE(calib_256.is_calibrated);
    EXPECT_TRUE(calib_512.is_calibrated);

    EXPECT_LT(calib_128.predicted_kernel_ms, calib_256.predicted_kernel_ms);
    EXPECT_LT(calib_256.predicted_kernel_ms, calib_512.predicted_kernel_ms);
}

TEST(HardwareCalibration, CalibratedSearchPredictsMeasuredWinner) {
    SearchConfig cfg;
    cfg.num_elements = 262144;
    cfg.candidate_budget = 4;

    SearchReport rep = run_optimization_search(cfg);
    EXPECT_TRUE(rep.ok);
    EXPECT_EQ(rep.predicted_winner_id, "fused_block128");
    EXPECT_EQ(rep.actual_winner_id, "fused_block128");
    EXPECT_TRUE(rep.prediction_matches_actual);

    // Verify candidate at Rank 1 is fused_block128
    EXPECT_EQ(rep.candidates[0].candidate_id, "fused_block128");
    EXPECT_EQ(rep.candidates[0].predicted_rank, 1u);
    EXPECT_EQ(rep.candidates[0].measured_rank, 1u);
}

}  // namespace
