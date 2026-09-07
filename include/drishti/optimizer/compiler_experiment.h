#ifndef DRISHTI_OPTIMIZER_COMPILER_EXPERIMENT_H
#define DRISHTI_OPTIMIZER_COMPILER_EXPERIMENT_H

// Phase 9: Real MLIR-to-GPU transformation experiment.
//
// One narrowly scoped, true compiler experiment: the embedded affine fusion
// workload (d = (a+b)*c as two loops) is compiled twice through the real
// MLIR/LLVM GPU path — once with a baseline pipeline, once with the same
// pipeline plus exactly one transformation (affine-loop-fusion) — with
// provenance recorded for both. Each variant is lowered to PTX, executed on
// the GPU, measured with the same harness, verified for correctness, and
// compared with the shared Phase 8 verdict machinery.
//
// Reported chain per variant:
//   MLIR op -> transformation/pass -> generated kernel -> GPU result

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/optimizer/experiment.h"
#include "drishti/profiling/gpu_metrics.h"

namespace drishti::optimizer {

struct CompilerVariant {
    std::string label;  // "baseline" | "fused"
    std::string pipeline;
    // Provenance summary for the pipeline run.
    std::size_t mlir_ops = 0;
    std::size_t mlir_funcs = 0;
    std::size_t pass_count = 0;
    std::size_t edge_count = 0;
    std::string func_anchor;   // e.g. "func.func (node 3)"
    std::vector<std::string> loop_anchors;  // affine.for nodes
    // Generated code + measurement.
    std::vector<std::string> kernels;  // PTX entry names, emission order
    profiling::GpuProfileMetrics metrics;
    std::vector<backends::cuda::PerKernelTime> per_kernel;
    double bytes_per_element = 0.0;
};

struct CompilerExperimentConfig {
    std::size_t num_elements = 65536;
    int block_size = 256;
    int repeats = 5;
};

struct CompilerExperimentReport {
    bool ok = false;
    std::string error;
    CompilerVariant baseline;
    CompilerVariant candidate;
    ExperimentResult result;  // reused Phase 8 verdict machinery
};

// Run the full baseline-vs-fused compiler experiment. Never throws;
// failures set ok=false with error. Requires N % block == 0.
CompilerExperimentReport run_compiler_experiment(const CompilerExperimentConfig& cfg,
                                                 std::string* err);

// Machine-readable rendering.
[[nodiscard]] std::string compiler_experiment_to_json(const CompilerExperimentReport& r);

// Human-readable full-chain report for `drishti compiler-experiment`.
[[nodiscard]] std::string format_compiler_experiment_report(
    const CompilerExperimentReport& r);

// Provenance wiring.
[[nodiscard]] drishti::provenance::PassInfo compiler_experiment_to_pass_info(
    const CompilerExperimentReport& r);

}  // namespace drishti::optimizer

#endif
