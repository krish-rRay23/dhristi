#ifndef DRISHTI_TRITON_TRITON_INTEGRATION_H
#define DRISHTI_TRITON_TRITON_INTEGRATION_H

// Phase 18: Triton Integration for Drishti.
//
// Traces a real Triton GPU workload through the compiler stack into executable GPU code:
//   Triton Workload -> Triton Compilation -> Intermediate IRs (TTIR, TTGIR, LLVM IR)
//                   -> NVPTX Assembly -> GPU Execution -> Drishti Profiling
//                   -> Provenance -> Diagnosis -> Hardware-Calibrated Cost Model
//
// Exposes:
//  - Capture of intermediate Triton compiler stages (TTIR, TTGIR, LLVM IR, PTX)
//  - Launch and profiling of the compiled Triton kernel on NVIDIA GPU via CUDA driver
//  - End-to-end provenance graph linking Triton AST, MLIR dialects, LLVM IR, PTX, and runtime
//  - Automated root-cause bottleneck diagnosis on Triton workloads
//  - Hardware-calibrated latency prediction and cost model error validation

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "drishti/diagnosis/root_cause.h"
#include "drishti/optimizer/cost_model.h"
#include "drishti/profiling/gpu_metrics.h"
#include "drishti/provenance/provenance.h"

namespace drishti::triton {

struct TritonCompilerArtifacts {
    std::string kernel_name;
    std::string python_source;
    std::string ttir;       // Triton-IR (MLIR `tt` dialect)
    std::string ttgir;      // TritonGPU-IR (MLIR `ttg` dialect)
    std::string llvm_ir;    // Lowered LLVM IR
    std::string ptx;        // Lowered NVPTX assembly
    int num_warps = 4;
    int num_stages = 2;
    std::size_t shared_mem_bytes = 0;
    int register_count = 0;
    std::string target_arch;
};

struct TritonWorkloadConfig {
    std::string workload_name = "fused_add_relu";  // "fused_add_relu", "vector_add", "gemm", "reduction", or custom file
    std::size_t num_elements = 65536;
    int block_size = 256;
    int repeats = 5;
    bool verify_gpu = true;
    std::string custom_script_path;  // optional path to external Triton kernel script
};

struct TritonPipelineReport {
    bool ok = false;
    std::string error;

    std::string workload_name;
    std::size_t num_elements = 0;
    int block_size = 256;
    int grid_size = 0;

    // Compiler Artifacts
    TritonCompilerArtifacts artifacts;

    // Provenance Tracking
    provenance::ProvenanceGraph provenance;

    // GPU Execution & Verification
    bool gpu_executed = false;
    profiling::GpuProfileMetrics gpu_metrics;
    bool gpu_correct = false;
    double measured_kernel_ms = 0.0;
    double measured_tflops = 0.0;

    // White-Box Cost Modeling (Phase 13 & 15 integration)
    optimizer::CostModelEstimate cost_estimate;
    optimizer::PredictionValidation cost_validation;

    // Root-Cause Diagnosis (Phase 6 & 10 integration)
    diagnosis::DiagnosisReport diagnosis;
};

// Run the full Triton -> Drishti analysis and execution pipeline. Never throws.
[[nodiscard]] TritonPipelineReport run_triton_pipeline(
    const TritonWorkloadConfig& cfg = {},
    std::string* err = nullptr);

// Machine-readable JSON rendering for `drishti triton --json-only`.
[[nodiscard]] std::string triton_pipeline_to_json(const TritonPipelineReport& r);

// Human-readable formatted report for `drishti triton`.
[[nodiscard]] std::string format_triton_pipeline_report(
    const TritonPipelineReport& r,
    bool show_ir = false);

// Provenance conversion helper.
[[nodiscard]] provenance::PassInfo triton_pipeline_to_pass_info(
    const TritonPipelineReport& r);

}  // namespace drishti::triton

#endif  // DRISHTI_TRITON_TRITON_INTEGRATION_H
