#ifndef DRISHTI_OPTIMIZER_COST_MODEL_H
#define DRISHTI_OPTIMIZER_COST_MODEL_H

// Phase 13 & Phase 15: Hardware-Aware & Hardware-Calibrated Cost Model for Drishti.
//
// Analytical & hardware-calibrated cost model that estimates GPU kernel
// execution latency, memory traffic, register pressure, occupancy bounds,
// block size issue queue granularity, and launch overhead before execution.
//
// White-box model: every term, formula, calibration basis, and assumption is fully
// exposed in human-readable reasoning breakdowns.

#include <cstddef>
#include <string>
#include <vector>

#include "drishti/profiling/gpu_metrics.h"

namespace drishti::optimizer {

struct HardwareCalibrationProfile {
    std::string target_gpu_name = "NVIDIA GeForce RTX 3050 Laptop GPU (sm_86)";
    std::string calibration_basis = "RTX 3050 Measured Empirical Micro-Benchmarks";
    bool enabled = true;
};

struct CostModelFeatures {
    std::string kernel_label;
    std::size_t num_elements = 65536;
    int block_size = 256;
    int grid_size = 0;  // 0 => computed as (N + B - 1) / B
    double bytes_per_element = 12.0;    // e.g. 12 (2 ld + 1 st f32)
    double flops_per_element = 1.0;    // e.g. 1 (1 addf)
    std::size_t launch_count = 1;      // e.g. 2 for unfused, 1 for fused
    int regs_per_thread_est = 16;      // static estimate
    std::size_t shared_mem_per_block_bytes = 0;
    profiling::GpuDeviceModel device;  // target GPU specs
};

struct CostModelEstimate {
    bool ok = false;
    std::string error;

    // Feature summary
    CostModelFeatures features;

    // Analytical Roofline components
    double compute_ms = 0.0;
    double memory_ms = 0.0;
    double roofline_bound_ms = 0.0;
    std::string bottleneck_regime;  // "memory-bound" | "compute-bound"

    // Occupancy & Hardware Resource Limits
    double theoretical_occupancy_pct = 0.0;
    double occupancy_penalty_factor = 1.0;
    std::string occupancy_limiting_factor;  // "registers" | "threads" | "blocks" | "none"

    // Dispatch & Launch Overhead
    double launch_overhead_ms = 0.0;
    double h2d_copy_est_ms = 0.0;
    double d2h_copy_est_ms = 0.0;

    // Calibration Components (Phase 15)
    bool is_calibrated = false;
    std::string calibration_basis;
    double uncalibrated_predicted_kernel_ms = 0.0;
    double calibrated_predicted_kernel_ms = 0.0;
    double block_granularity_penalty = 1.0;
    double wave_tail_penalty = 1.0;

    // Final Estimates
    double predicted_kernel_ms = 0.0;   // kernel execution alone (calibrated if enabled)
    double predicted_total_ms = 0.0;    // end-to-end wall time

    // Explainable Reasoning Chain
    std::vector<std::string> reasoning;
};

struct PredictionValidation {
    double predicted_ms = 0.0;
    double measured_ms = 0.0;
    double absolute_error_ms = 0.0;
    double error_percent = 0.0;
    bool is_accurate = false;  // within <= 25% error margin
    std::string assessment;
};

// Pure analytical cost estimation function. Never throws.
[[nodiscard]] CostModelEstimate estimate_kernel_cost(const CostModelFeatures& f);

// Hardware-calibrated cost estimation function (Phase 15). Never throws.
[[nodiscard]] CostModelEstimate estimate_kernel_cost_calibrated(
    const CostModelFeatures& f,
    const HardwareCalibrationProfile& calib = {});

// Compare predicted vs measured latency and evaluate error.
[[nodiscard]] PredictionValidation validate_prediction(double predicted_ms,
                                                       double measured_ms);

// Machine-readable JSON output for cost estimates.
[[nodiscard]] std::string cost_estimate_to_json(const CostModelEstimate& est);

// Human-readable format for `drishti cost`.
[[nodiscard]] std::string format_cost_estimate_report(const CostModelEstimate& est);

}  // namespace drishti::optimizer

#endif  // DRISHTI_OPTIMIZER_COST_MODEL_H
