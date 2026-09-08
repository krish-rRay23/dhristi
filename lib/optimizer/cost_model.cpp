#include "drishti/optimizer/cost_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace drishti::optimizer {
namespace {

std::string fmt(double v, int prec = 4) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(prec);
    oss << v;
    return oss.str();
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

}  // namespace

CostModelEstimate estimate_kernel_cost(const CostModelFeatures& orig_features) {
    CostModelEstimate est;  
    est.ok = true;
    est.features = orig_features;

    if (est.features.block_size <= 0) est.features.block_size = 256;
    if (est.features.grid_size <= 0 && est.features.num_elements > 0) {
        const std::size_t bs = static_cast<std::size_t>(est.features.block_size);
        est.features.grid_size = static_cast<int>((est.features.num_elements + bs - 1) / bs);
    }
    if (est.features.launch_count == 0) est.features.launch_count = 1;

    auto& dev = est.features.device;
    if (!dev.present || dev.sm_count <= 0) {
        dev.present = true;
        dev.backend = "cuda";
        dev.name = "NVIDIA GeForce RTX 3050 Laptop GPU (sm_86)";
        dev.compute_major = 8;
        dev.compute_minor = 6;
        dev.sm_count = 16;
        dev.max_threads_per_block = 1024;
        dev.clock_mhz = 1500;
        dev.mem_theoretical_gbps = 192.0;
        dev.total_mem_bytes = 4294443008ULL;
    }

    const double n = static_cast<double>(est.features.num_elements);

    // 1. Compute Pressure Calculation
    const double alus_per_sm = 128.0;
    const double clock_hz = (dev.clock_mhz > 0 ? static_cast<double>(dev.clock_mhz) : 1500.0) * 1e6;
    const double peak_flops_per_sec = static_cast<double>(dev.sm_count) * alus_per_sm * 2.0 * clock_hz;
    const double total_flops = n * est.features.flops_per_element;

    if (peak_flops_per_sec > 0.0) {
        est.compute_ms = (total_flops / peak_flops_per_sec) * 1000.0;
    }

    // 2. Memory Pressure & Traffic Calculation
    const double effective_gbps = (dev.mem_theoretical_gbps > 0.0)
                                      ? dev.mem_theoretical_gbps * 0.70
                                      : 140.0;
    const double bytes_moved = n * est.features.bytes_per_element;
    const double bw_bytes_per_sec = effective_gbps * 1e9;

    if (bw_bytes_per_sec > 0.0) {
        est.memory_ms = (bytes_moved / bw_bytes_per_sec) * 1000.0;
    }

    est.roofline_bound_ms = std::max(est.compute_ms, est.memory_ms);
    if (est.memory_ms >= est.compute_ms) {
        est.bottleneck_regime = "memory-bound";
    } else {
        est.bottleneck_regime = "compute-bound";
    }

    // 3. Register Pressure & Occupancy Calculation
    const int regs_per_sm = 65536;
    const int max_threads_per_sm = 1536;
    const int max_blocks_per_sm = 32;

    const int reg_est = est.features.regs_per_thread_est > 0 ? est.features.regs_per_thread_est : 16;
    const int regs_per_block = reg_est * est.features.block_size;

    const int by_regs = (regs_per_block > 0) ? std::min(regs_per_sm / regs_per_block, max_blocks_per_sm) : max_blocks_per_sm;
    const int by_threads = std::min(max_threads_per_sm / est.features.block_size, max_blocks_per_sm);
    const int active_blocks = std::max(0, std::min(by_regs, by_threads));

    est.theoretical_occupancy_pct = (static_cast<double>(active_blocks * est.features.block_size) / static_cast<double>(max_threads_per_sm)) * 100.0;

    if (by_regs < by_threads) {
        est.occupancy_limiting_factor = "registers";
    } else if (by_threads < max_blocks_per_sm) {
        est.occupancy_limiting_factor = "threads";
    } else {
        est.occupancy_limiting_factor = "none";
    }

    if (est.theoretical_occupancy_pct < 50.0) {
        est.occupancy_penalty_factor = 1.0 + 0.5 * ((50.0 - est.theoretical_occupancy_pct) / 50.0);
    } else {
        est.occupancy_penalty_factor = 1.0;
    }

    // 4. Launch & Dispatch Overhead Calculation
    constexpr double kLaunchDispatchOverheadMs = 0.015;
    est.launch_overhead_ms = static_cast<double>(est.features.launch_count) * kLaunchDispatchOverheadMs;

    // 5. Uncalibrated Latency
    est.uncalibrated_predicted_kernel_ms = est.roofline_bound_ms * est.occupancy_penalty_factor;
    est.calibrated_predicted_kernel_ms = est.uncalibrated_predicted_kernel_ms;
    est.predicted_kernel_ms = est.uncalibrated_predicted_kernel_ms;
    est.predicted_total_ms = est.predicted_kernel_ms + est.launch_overhead_ms;

    // 6. Reasoning Chain
    est.reasoning.push_back("Analytical Roofline: Workload is " + est.bottleneck_regime + " (AI = " +
                            fmt(est.features.flops_per_element / est.features.bytes_per_element, 4) + " FLOP/B).");
    est.reasoning.push_back("Memory Traffic: " + fmt(bytes_moved, 0) + " bytes over " + fmt(effective_gbps, 1) +
                            " GB/s effective bandwidth yields T_mem = " + fmt(est.memory_ms, 4) + " ms.");
    est.reasoning.push_back("Compute Latency: " + fmt(total_flops, 0) + " FLOPs over " + fmt(peak_flops_per_sec / 1e12, 2) +
                            " TFLOPS peak capacity yields T_compute = " + fmt(est.compute_ms, 4) + " ms.");
    est.reasoning.push_back("Occupancy Bounds: Theoretical occupancy is " + fmt(est.theoretical_occupancy_pct, 1) + "% (" +
                            std::to_string(active_blocks) + " active blocks/SM, limited by " + est.occupancy_limiting_factor +
                            ", penalty factor = " + fmt(est.occupancy_penalty_factor, 2) + ").");
    est.reasoning.push_back("Kernel Dispatch: " + std::to_string(est.features.launch_count) + " kernel launch(es) add " +
                            fmt(est.launch_overhead_ms, 4) + " ms of stream launch overhead.");
    est.reasoning.push_back("Uncalibrated Prediction: T_kernel_uncalib = " + fmt(est.uncalibrated_predicted_kernel_ms, 4) + " ms (" +
                            fmt(est.uncalibrated_predicted_kernel_ms * 1000.0, 1) + " μs).");

    return est;
}

CostModelEstimate estimate_kernel_cost_calibrated(
    const CostModelFeatures& features,
    const HardwareCalibrationProfile& calib) {
    CostModelEstimate est = estimate_kernel_cost(features);
    if (!calib.enabled) return est;

    est.is_calibrated = true;
    est.calibration_basis = calib.calibration_basis;

    // 1. Block Size Granularity & Memory Issue Queue Contention Penalty
    // B=128 => 1.000, B=256 => 1.040, B=512 => 1.0833
    const double b_ratio = static_cast<double>(est.features.block_size) / 128.0;
    if (b_ratio >= 1.0) {
        est.block_granularity_penalty = 1.0 + 0.04 * (b_ratio - 1.0);
    } else {
        est.block_granularity_penalty = 1.0 + 0.05 * (1.0 - b_ratio);
    }

    // 2. SM Wave Tail Scheduling Penalty
    const int sm_count = est.features.device.sm_count > 0 ? est.features.device.sm_count : 16;
    const int active_blocks_per_sm = std::max(1, 1536 / est.features.block_size);
    const int concurrent_capacity = sm_count * active_blocks_per_sm;

    if (concurrent_capacity > 0 && est.features.grid_size > 0) {
        const int tail_blocks = est.features.grid_size % concurrent_capacity;
        if (tail_blocks > 0) {
            const double frac = static_cast<double>(tail_blocks) / static_cast<double>(concurrent_capacity);
            if (frac < 0.25) {
                est.wave_tail_penalty = 1.0 + 0.03 * (1.0 - frac);
            }
        }
    }

    // Apply Calibration Factors
    est.calibrated_predicted_kernel_ms = est.uncalibrated_predicted_kernel_ms *
                                         est.block_granularity_penalty *
                                         est.wave_tail_penalty;

    est.predicted_kernel_ms = est.calibrated_predicted_kernel_ms;
    est.predicted_total_ms = est.predicted_kernel_ms + est.launch_overhead_ms;

    est.reasoning.push_back("Hardware Calibration [" + calib.calibration_basis + "]:");
    est.reasoning.push_back("  - Block Granularity Penalty (B=" + std::to_string(est.features.block_size) +
                            "): multiplier = " + fmt(est.block_granularity_penalty, 4) +
                            " (models L2/issue queue memory contention).");
    if (est.wave_tail_penalty > 1.0) {
        est.reasoning.push_back("  - SM Wave Tail Scheduling Penalty: multiplier = " +
                                fmt(est.wave_tail_penalty, 4) + ".");
    }
    est.reasoning.push_back("Calibrated Prediction: T_kernel_calib = " + fmt(est.calibrated_predicted_kernel_ms, 4) + " ms (" +
                            fmt(est.calibrated_predicted_kernel_ms * 1000.0, 1) + " μs), T_total_calib = " +
                            fmt(est.predicted_total_ms, 4) + " ms.");

    return est;
}

PredictionValidation validate_prediction(double predicted_ms, double measured_ms) {
    PredictionValidation v;
    v.predicted_ms = predicted_ms;
    v.measured_ms = measured_ms;

    if (measured_ms <= 0.0) {
        v.absolute_error_ms = 0.0;
        v.error_percent = 0.0;
        v.is_accurate = false;
        v.assessment = "Invalid measured time (<= 0.0 ms)";
        return v;
    }

    v.absolute_error_ms = std::abs(predicted_ms - measured_ms);
    v.error_percent = (v.absolute_error_ms / measured_ms) * 100.0;
    v.is_accurate = (v.error_percent <= 25.0);

    if (v.is_accurate) {
        v.assessment = "HIGH ACCURACY (Error " + fmt(v.error_percent, 1) + "% <= 25% margin)";
    } else {
        v.assessment = "MODERATE MARGIN (Error " + fmt(v.error_percent, 1) + "% > 25% margin)";
    }

    return v;
}

std::string cost_estimate_to_json(const CostModelEstimate& est) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"schema\": \"drishti.cost_estimate/v1\",\n"
        << "  \"ok\": " << (est.ok ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(est.error) << "\",\n"
        << "  \"kernel_label\": \"" << json_escape(est.features.kernel_label) << "\",\n"
        << "  \"num_elements\": " << est.features.num_elements << ",\n"
        << "  \"block_size\": " << est.features.block_size << ",\n"
        << "  \"launch_count\": " << est.features.launch_count << ",\n"
        << "  \"bottleneck_regime\": \"" << json_escape(est.bottleneck_regime) << "\",\n"
        << "  \"is_calibrated\": " << (est.is_calibrated ? "true" : "false") << ",\n"
        << "  \"calibration_basis\": \"" << json_escape(est.calibration_basis) << "\",\n"
        << "  \"uncalibrated_predicted_kernel_ms\": " << fmt(est.uncalibrated_predicted_kernel_ms, 4) << ",\n"
        << "  \"calibrated_predicted_kernel_ms\": " << fmt(est.calibrated_predicted_kernel_ms, 4) << ",\n"
        << "  \"block_granularity_penalty\": " << fmt(est.block_granularity_penalty, 4) << ",\n"
        << "  \"compute_ms\": " << fmt(est.compute_ms, 4) << ",\n"
        << "  \"memory_ms\": " << fmt(est.memory_ms, 4) << ",\n"
        << "  \"roofline_bound_ms\": " << fmt(est.roofline_bound_ms, 4) << ",\n"
        << "  \"theoretical_occupancy_pct\": " << fmt(est.theoretical_occupancy_pct, 1) << ",\n"
        << "  \"launch_overhead_ms\": " << fmt(est.launch_overhead_ms, 4) << ",\n"
        << "  \"predicted_kernel_ms\": " << fmt(est.predicted_kernel_ms, 4) << ",\n"
        << "  \"predicted_total_ms\": " << fmt(est.predicted_total_ms, 4) << ",\n"
        << "  \"reasoning\": [";
    for (std::size_t i = 0; i < est.reasoning.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "\n    \"" << json_escape(est.reasoning[i]) << "\"";
    }
    oss << (est.reasoning.empty() ? "]" : "\n  ]") << "\n}\n";
    return oss.str();
}

std::string format_cost_estimate_report(const CostModelEstimate& est) {
    std::ostringstream oss;
    oss << "================================================================================\n"
        << "  Dṛṣṭi Hardware-Aware & Calibrated Cost Model (Phase 13 / Phase 15)\n"
        << "================================================================================\n"
        << "  Kernel Workload : " << (est.features.kernel_label.empty() ? "vecadd" : est.features.kernel_label) << "\n"
        << "  Target Device   : " << est.features.device.name << " (" << est.features.device.sm_count << " SMs)\n"
        << "  Workload Size   : N = " << est.features.num_elements << " elements\n"
        << "  Launch Grid     : " << est.features.launch_count << " launch(es) x ("
        << est.features.grid_size << " blocks x " << est.features.block_size << " threads)\n"
        << "  Memory Traffic  : " << fmt(est.features.bytes_per_element, 1) << " B/elem ("
        << fmt(static_cast<double>(est.features.num_elements) * est.features.bytes_per_element, 0) << " bytes total)\n"
        << "  Calibration     : " << (est.is_calibrated ? "ENABLED (" + est.calibration_basis + ")" : "DISABLED") << "\n"
        << "--------------------------------------------------------------------------------\n"
        << "  ANALYTICAL & CALIBRATED LATENCY ESTIMATES:\n"
        << "    Bottleneck Regime : " << est.bottleneck_regime << "\n"
        << "    Compute Latency   : " << fmt(est.compute_ms, 4) << " ms\n"
        << "    Memory Latency    : " << fmt(est.memory_ms, 4) << " ms\n"
        << "    Roofline Bound    : " << fmt(est.roofline_bound_ms, 4) << " ms\n"
        << "    Occupancy Bounds  : " << fmt(est.theoretical_occupancy_pct, 1) << "% theoretical (limited by "
        << est.occupancy_limiting_factor << ")\n"
        << "    Launch Overhead   : " << fmt(est.launch_overhead_ms, 4) << " ms ("
        << est.features.launch_count << " launch(es))\n";

    if (est.is_calibrated) {
        oss << "    ----------------------------------------------------------------------------\n"
            << "    Uncalibrated Kernel Latency : " << fmt(est.uncalibrated_predicted_kernel_ms, 4) << " ms ("
            << fmt(est.uncalibrated_predicted_kernel_ms * 1000.0, 1) << " μs)\n"
            << "    Block Granularity Penalty   : x" << fmt(est.block_granularity_penalty, 4) << " (B="
            << est.features.block_size << ")\n"
            << "    Calibrated Kernel Latency   : " << fmt(est.calibrated_predicted_kernel_ms, 4) << " ms ("
            << fmt(est.calibrated_predicted_kernel_ms * 1000.0, 1) << " μs)\n";
    }

    oss << "    ----------------------------------------------------------------------------\n"
        << "    Predicted Kernel Latency    : " << fmt(est.predicted_kernel_ms, 4) << " ms ("
        << fmt(est.predicted_kernel_ms * 1000.0, 1) << " μs)\n"
        << "    Predicted Total Latency     : " << fmt(est.predicted_total_ms, 4) << " ms ("
        << fmt(est.predicted_total_ms * 1000.0, 1) << " μs)\n"
        << "================================================================================\n"
        << "  ANALYTICAL & CALIBRATION REASONING BREAKDOWN:\n";

    for (const auto& r : est.reasoning) {
        oss << "    - " << r << "\n";
    }
    oss << "================================================================================\n";
    return oss.str();
}

}  // namespace drishti::optimizer
