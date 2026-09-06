#ifndef DRISHTI_CORRELATION_CORRELATION_H
#define DRISHTI_CORRELATION_CORRELATION_H

// Phase 5: Compiler-to-GPU correlation.
//
// Connects the MLIR provenance system with the NVIDIA profiling system for a
// small deterministic workload, recording the chain:
//
//   MLIR operation -> transformation/pass -> kernel -> measured GPU performance
//
// Only relationships that can be established reliably are claimed:
//  - SameComputation: the built-in reference vecadd workload, where the MLIR
//    module provably specifies the same elementwise f32 add of N elements
//    that the profiled kernel executes (func @vecadd + arith.addf anchors
//    found, N identical on both sides, GPU result verified).
//  - CoExecuted: a user-supplied MLIR file analyzed in the same session as a
//    kernel run. Anchors are reported with provenance node IDs, but no
//    computation-equivalence claim is made.
//
// No root-cause diagnosis or automatic optimization is performed here.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "drishti/profiling/gpu_metrics.h"
#include "drishti/provenance/provenance.h"

namespace drishti::correlation {

enum class RelationKind {
    SameComputation,  // reference workload: equivalence verified by construction
    CoExecuted,       // same session, anchors reported, no equivalence claim
};

[[nodiscard]] std::string relation_name(RelationKind r);

// An MLIR-side anchor: a provenance graph node participating in the chain.
struct MlirAnchor {
    std::string role;      // "function" or "compute-op"
    std::string op_name;   // e.g. "func.func", "arith.addf"
    std::string dialect;   // e.g. "func", "arith"
    std::uint64_t node_id = 0;
    std::string location;  // "file:line:col"
};

struct CorrelationConfig {
    std::size_t num_elements = 65536;
    int block_size = 256;
    int repeats = 5;
    std::string pipeline = "canonicalize,cse";
    std::string mlir_file;  // empty => built-in reference vecadd workload
};

struct CorrelationRecord {
    bool ok = false;
    std::string error;

    // Compiler stage.
    std::string mlir_source_label;  // "<embedded-vecadd>" or file path
    std::size_t mlir_ops = 0;
    std::size_t mlir_funcs = 0;
    std::vector<MlirAnchor> anchors;
    std::string pipeline;
    std::uint64_t pipeline_pass_id = 0;
    std::size_t transform_edges = 0;

    // Kernel stage.
    std::string kernel;  // e.g. "vecadd"
    std::size_t num_elements = 0;
    int grid_size = 0;
    int block_size = 0;
    profiling::GpuProfileMetrics gpu;

    // Relationship.
    RelationKind relation = RelationKind::CoExecuted;
    std::vector<std::string> evidence;
};

// Deterministic reference workload: elementwise f32 vecadd over memref<Nxf32>.
// The generated text always defines func @vecadd with one arith.addf.
[[nodiscard]] std::string generate_vecadd_mlir(std::size_t n);

// Run the end-to-end flow. Never throws; failures set ok=false with error.
CorrelationRecord run_correlation(const CorrelationConfig& cfg, std::string* err);

// Machine-readable rendering for later diagnosis/provenance phases.
[[nodiscard]] std::string correlation_to_json(const CorrelationRecord& r);

// Human-readable chain report for `drishti correlate`.
[[nodiscard]] std::string format_correlation_report(const CorrelationRecord& r);

// Provenance wiring: represent the correlation as a provenance pass.
[[nodiscard]] drishti::provenance::PassInfo correlation_to_pass_info(
    const CorrelationRecord& r);

}  // namespace drishti::correlation

#endif
