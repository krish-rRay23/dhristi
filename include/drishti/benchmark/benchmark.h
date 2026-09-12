#ifndef DRISHTI_BENCHMARK_BENCHMARK_H
#define DRISHTI_BENCHMARK_BENCHMARK_H

// Phase 16: Generalization Benchmark Suite for Drishti.
//
// End-to-end benchmark suite for representative GPU workload classes:
// 1. GEMM (Matrix Multiplication - Compute-bound)
// 2. Reduction (Sum Reduction - Memory-bound / Atomic vs Tree)
// 3. Elementwise Fusion (Multi-op Fusion - Memory traffic & launch overhead)
// 4. Transpose (2D Matrix Transpose - Memory layout / strided access bound)
// 5. Attention-like Workload (Scaled Dot-Product Attention - Multi-pass vs Fused Softmax)
//
// Full pipeline per workload:
// MLIR/provenance -> GPU lowering -> profiling -> diagnosis -> cost prediction
// -> optimization experiment -> correctness -> measured performance.

#include <cstddef>
#include <string>
#include <vector>

#include "drishti/diagnosis/root_cause.h"
#include "drishti/optimizer/cost_model.h"
#include "drishti/profiling/gpu_metrics.h"
#include "drishti/provenance/provenance.h"

namespace drishti::benchmark {

enum class WorkloadType {
    GEMM,
    Reduction,
    Fusion,
    Transpose,
    Attention,
    All
};

struct BenchmarkCandidateResult {
    std::string candidate_id;    // e.g. "tiled_shared_mem"
    std::string title;           // human-readable description
    std::string transformation;  // e.g. "block-tiling-shared-mem"

    // Analytical Cost Prediction
    optimizer::CostModelFeatures features;
    optimizer::CostModelEstimate cost_estimate;
    std::size_t predicted_rank = 0;  // 1-indexed

    // Empirical GPU Profiling Results
    profiling::GpuProfileMetrics metrics;
    double measured_ms = 0.0;
    double measured_speedup_percent = 0.0;  // relative to baseline candidate
    std::size_t measured_rank = 0;   // 1-indexed

    // Validation against prediction
    optimizer::PredictionValidation validation;
};

struct BenchmarkWorkloadResult {
    bool ok = false;
    std::string error;

    std::string name;             // e.g. "GEMM"
    std::string description;      // e.g. "Dense Matrix Multiplication (C = A x B)"
    std::string regime;           // e.g. "compute-bound"
    std::size_t num_elements = 0; // problem size

    // Provenance Summary
    std::size_t mlir_ops = 0;
    std::size_t mlir_funcs = 0;
    std::size_t pass_count = 0;

    // Bottleneck Diagnosis
    diagnosis::DiagnosisReport diagnosis;

    // Baseline & Candidate Experiments
    std::vector<BenchmarkCandidateResult> candidates;

    // Winners & Validation Summary
    std::string baseline_id;
    std::string predicted_winner_id;
    std::string actual_winner_id;
    bool rank_matched = false;
    bool correct = false;

    double baseline_ms = 0.0;
    double best_ms = 0.0;
    double speedup_factor = 1.0;
    double prediction_error_pct = 0.0;
};

struct BenchmarkSuiteReport {
    bool ok = false;
    std::string error;

    profiling::GpuDeviceModel device;
    std::size_t total_workloads = 0;
    std::size_t passed_workloads = 0;
    std::size_t rank_matched_workloads = 0;

    double avg_prediction_error_pct = 0.0;

    std::vector<BenchmarkWorkloadResult> workloads;
    std::string summary;
};

struct BenchmarkConfig {
    WorkloadType filter = WorkloadType::All;
    std::size_t num_elements = 262144;  // Default problem size
    int repeats = 5;
};

// Convert string to WorkloadType (e.g. "gemm", "reduction", "fusion", "transpose", "attention", "all")
[[nodiscard]] WorkloadType parse_workload_type(const std::string& s);

// Format WorkloadType to string
[[nodiscard]] std::string workload_type_to_string(WorkloadType type);

// Run the full Generalization Benchmark Suite. Never throws.
[[nodiscard]] BenchmarkSuiteReport run_benchmark_suite(const BenchmarkConfig& cfg = {});

// Machine-readable JSON output for benchmark suite reports.
[[nodiscard]] std::string benchmark_to_json(const BenchmarkSuiteReport& rep);

// Human-readable formatted report for `drishti benchmark`.
[[nodiscard]] std::string format_benchmark_report(const BenchmarkSuiteReport& rep);

}  // namespace drishti::benchmark

#endif  // DRISHTI_BENCHMARK_BENCHMARK_H
