#ifndef DRISHTI_BENCHMARK_JOURNAL_EXPERIMENTS_H
#define DRISHTI_BENCHMARK_JOURNAL_EXPERIMENTS_H

// Journal Research Benchmark Framework & Ablation Suite Header.
// Supports evaluation across:
// 1. GEMM, Attention, Elementwise Fusion, Reduction/Normalization
// 2. Systems: Standard Triton, Triton + Autotune, Full Drishti
// 3. Ablations: Full Drishti, -Vectorization, -Fusion, -IR Analysis, -Hardware Telemetry

#include <cstddef>
#include <string>
#include <vector>

#include "drishti/benchmark/benchmark.h"

namespace drishti::benchmark {

struct JournalExperimentRecord {
    std::string workload_id;           // e.g. "gemm_small", "fused_add_relu"
    std::string system_id;             // e.g. "Standard Triton", "Triton + Autotune", "Full Drishti"
    std::string ablation_id;           // e.g. "none", "Drishti - Vectorization", etc.

    double latency_ms = 0.0;
    double stddev_ms = 0.0;
    double speedup_vs_baseline = 1.0;  // relative to Standard Triton (1.0x)
    double vram_bandwidth_gbps = 0.0;
    double compilation_overhead_ms = 0.0;

    bool correctness = false;
    double max_abs_error = 0.0;
    std::string occupancy = "N/A";
};

struct JournalBenchmarkReport {
    bool ok = false;
    std::string error;
    std::string timestamp;
    std::string hardware_device;

    std::size_t total_experiments = 0;
    std::vector<JournalExperimentRecord> results;
    std::string latex_table;
};

// Execute full reproducible journal experiment suite. Never throws.
[[nodiscard]] JournalBenchmarkReport run_journal_experiments();

// Format report to JSON.
[[nodiscard]] std::string journal_report_to_json(const JournalBenchmarkReport& rep);

// Format human readable terminal output.
[[nodiscard]] std::string format_journal_report(const JournalBenchmarkReport& rep);

}  // namespace drishti::benchmark

#endif  // DRISHTI_BENCHMARK_JOURNAL_EXPERIMENTS_H
