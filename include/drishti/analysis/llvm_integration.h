#ifndef DRISHTI_ANALYSIS_LLVM_INTEGRATION_H
#define DRISHTI_ANALYSIS_LLVM_INTEGRATION_H

// Phase 17: Deep LLVM Integration for Drishti.
//
// Makes LLVM IR a real, inspectable, and optimizable stage in the compilation chain:
//   MLIR -> LLVM IR -> LLVM optimization passes -> Optimized LLVM IR -> PTX -> GPU Execution
//
// Exposes:
//  - Real MLIR-to-LLVM module lowering
//  - Captured raw and optimized LLVM IR strings
//  - Real LLVM pass execution using modern LLVM PassBuilder / PassManager
//  - Full instruction and basic block statistics tracking
//  - Provenance integration recording LLVM passes and IR transformations
//  - GPU execution and correctness verification of the optimized lowered PTX

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "drishti/profiling/gpu_metrics.h"
#include "drishti/provenance/provenance.h"

namespace drishti::analysis {

struct LlvmModuleStats {
    std::size_t function_count = 0;
    std::size_t basic_block_count = 0;
    std::size_t instruction_count = 0;
    std::size_t arithmetic_inst_count = 0;
    std::size_t memory_inst_count = 0;
    std::size_t control_inst_count = 0;
    std::size_t alloca_count = 0;
    std::size_t phi_count = 0;
};

struct LlvmPassInfo {
    std::string pass_name;
    std::string description;
    std::size_t inst_count_before = 0;
    std::size_t inst_count_after = 0;
    int instruction_delta = 0;  // negative indicates reduction
};

struct LlvmPipelineConfig {
    std::string workload_label = "fusion";
    std::size_t num_elements = 65536;
    std::string mlir_pass_pipeline;  // default if empty
    std::vector<std::string> llvm_passes = {"sroa", "instcombine", "simplifycfg", "dce"};
    int cc_major = 8;
    int cc_minor = 6;
    bool verify_gpu = true;
    int repeats = 5;
};

struct LlvmPipelineReport {
    bool ok = false;
    std::string error;

    std::string workload_label;
    std::size_t num_elements = 0;

    // MLIR Stage
    std::string mlir_source;
    std::size_t mlir_ops_count = 0;
    std::size_t mlir_funcs_count = 0;
    std::string mlir_pipeline_used;

    // Raw LLVM IR Stage (Pre-Optimization)
    std::string raw_llvm_ir;
    LlvmModuleStats raw_stats;

    // LLVM Optimization Passes Stage
    std::vector<LlvmPassInfo> passes_applied;
    std::string optimized_llvm_ir;
    LlvmModuleStats optimized_stats;
    int total_instruction_delta = 0;

    // Codegen Stage
    std::string lowered_ptx;
    std::vector<std::string> kernel_entries;

    // Provenance Tracking
    provenance::ProvenanceGraph provenance;

    // GPU Execution & Verification
    bool gpu_executed = false;
    profiling::GpuProfileMetrics gpu_metrics;
    bool gpu_correct = false;
    double measured_kernel_ms = 0.0;
};

// Run the full Deep LLVM Integration pipeline. Never throws.
[[nodiscard]] LlvmPipelineReport run_llvm_pipeline(
    const LlvmPipelineConfig& cfg = {},
    std::string* err = nullptr);

// Machine-readable JSON rendering for `drishti llvm --json-only`.
[[nodiscard]] std::string llvm_pipeline_to_json(const LlvmPipelineReport& r);

// Human-readable formatted report for `drishti llvm`.
[[nodiscard]] std::string format_llvm_pipeline_report(
    const LlvmPipelineReport& r,
    bool show_ir = false);

// Provenance conversion helper.
[[nodiscard]] provenance::PassInfo llvm_pipeline_to_pass_info(
    const LlvmPipelineReport& r);

}  // namespace drishti::analysis

#endif  // DRISHTI_ANALYSIS_LLVM_INTEGRATION_H
