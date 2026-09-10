#ifndef DRISHTI_DIAGNOSIS_ROOT_CAUSE_H
#define DRISHTI_DIAGNOSIS_ROOT_CAUSE_H

// Phase 6: Root-cause intelligence.
//
// Deterministic, evidence-based diagnosis built on the analysis, provenance,
// profiling, and correlation layers. Each rule is a pure function over a
// CorrelationRecord: fixed thresholds, no randomness, no LLM, no autotuning.
// Output is sorted (severity, then rule id), so repeated runs are identical.
//
// Evidence honesty without CUPTI: no hardware counters are available, so
// every claim is derived from measured event times, byte counts, launch
// geometry, device limits, and static kernel traits. Static estimates
// (notably register counts from PTX) are labeled as such and capped at
// medium confidence.
//
// Extensibility: detectors are free functions collected in default_rules();
// future hardware backends add KernelTraits entries or append new RuleFn
// entries without touching the engine. No optimizations are generated here.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "drishti/correlation/correlation.h"
#include "drishti/diagnosis/diagnosis.h"

namespace drishti::optimizer {
struct CompilerExperimentReport;
}

namespace drishti::diagnosis {

// Static per-kernel properties used by the rules. regs_per_thread_est == 0
// means "unknown": register/occupancy rules abstain instead of guessing.
struct KernelTraits {
    std::string kernel;
    double flops_per_element = 0.0;    // e.g. vecadd: 1 f32 add
    double bytes_per_element_min = 0.0;  // minimal traffic, e.g. 12 (2 ld + 1 st)
    int regs_per_thread_est = 0;         // static upper bound, 0 = unknown
    std::string regs_basis;              // provenance of the estimate
};

// Extra tables (tests, future backends) take precedence over built-ins.
struct DiagnosisConfig {
    std::map<std::string, KernelTraits> extra_traits;
};

struct BottleneckRef {
    std::string rule_id;         // stable machine id, e.g. "memory-traffic"
    std::string title;           // human bottleneck name
    std::string related_op;      // e.g. "arith.addf (node 6)"; "n/a" if none
    std::string related_pass;    // e.g. "canonicalize,cse (pass 1)"; "n/a"
};

struct Finding {
    BottleneckRef bottleneck;
    Severity severity = Severity::Note;
    double confidence = 0.0;  // 0..1, fixed tier per rule outcome
    std::vector<std::string> evidence;
    std::string explanation;
};

struct DiagnosisReport {
    bool ok = false;  // false only when the input record itself failed
    std::string error;
    std::string relation;  // correlation relation name, for context
    std::vector<Finding> findings;
};

using RuleFn = std::function<std::vector<Finding>(const correlation::CorrelationRecord&,
                                                  const DiagnosisConfig&)>;

// The five Phase 6 detectors, in stable order.
[[nodiscard]] std::vector<RuleFn> default_rules();

// Look up traits for a kernel (extra table first, then built-ins).
// Returns nullptr when the kernel is unknown.
[[nodiscard]] const KernelTraits* kernel_traits(const std::string& kernel,
                                                const DiagnosisConfig& cfg);

// Run all rules over a correlated workload. Never throws.
DiagnosisReport diagnose(const correlation::CorrelationRecord& record,
                         const DiagnosisConfig& cfg = {});

// Machine-readable rendering for later phases.
[[nodiscard]] std::string diagnosis_to_json(const DiagnosisReport& r);

// Human-readable report for `drishti diagnose`.
[[nodiscard]] std::string format_diagnosis_report(const DiagnosisReport& r);

// Provenance wiring: represent the diagnosis as a provenance pass.
[[nodiscard]] drishti::provenance::PassInfo diagnosis_to_pass_info(
    const DiagnosisReport& r, const std::string& kernel);

// Feed findings into the existing diagnostic engine.
void emit_to_engine(const DiagnosisReport& r, DiagnosticEngine& engine);

// Phase 10: Hardware-Aware Root-Cause Analysis for Compiler Experiments.
struct CompilerDiagnosisReport {
    bool ok = false;
    std::string error;
    std::string transformation;      // e.g. "affine-loop-fusion"
    std::string device_name;         // e.g. "NVIDIA GeForce RTX 3050 Laptop GPU"
    std::string compute_capability;  // e.g. "8.6"
    std::string primary_cause;       // e.g. "Kernel Fusion & Memory Traffic Elimination"
    double baseline_ms = 0.0;
    double candidate_ms = 0.0;
    double speedup_percent = 0.0;
    std::size_t baseline_launches = 0;
    std::size_t candidate_launches = 0;
    double baseline_bytes_per_elem = 0.0;
    double candidate_bytes_per_elem = 0.0;
    std::size_t vram_bytes_saved = 0;
    std::vector<Finding> findings;
};

// Run hardware-aware root-cause analysis over a compiler experiment result.
// Deterministic, evidence-based, and non-random. Never throws.
CompilerDiagnosisReport diagnose_compiler_experiment(
    const optimizer::CompilerExperimentReport& exp,
    const DiagnosisConfig& cfg = {});

// Machine-readable rendering for Phase 10 diagnosis.
[[nodiscard]] std::string compiler_diagnosis_to_json(const CompilerDiagnosisReport& r);

// Human-readable report for `drishti compiler-experiment`.
[[nodiscard]] std::string format_compiler_diagnosis_report(
    const CompilerDiagnosisReport& r);

}  // namespace drishti::diagnosis

#endif
