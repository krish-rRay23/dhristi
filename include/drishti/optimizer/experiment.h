#ifndef DRISHTI_OPTIMIZER_EXPERIMENT_H
#define DRISHTI_OPTIMIZER_EXPERIMENT_H

// Phase 8: Optimization experimentation.
//
// Turns optimization candidates into controlled experiments: apply the
// transformation/configuration reproducibly, execute on the GPU, measure the
// same metrics as the baseline, verify correctness, compare, and classify the
// verdict as improved / regressed / unchanged / invalid.
//
// Scope is deliberately narrow: exactly two reliably implementable
// experiments exist (no general-purpose source-to-source optimizer):
//  - persist-buffers: recopy-per-launch (naive app pattern) vs persistent
//    device buffers, for the reuse-device-data candidate.
//  - vary-block-size: baseline block size vs a deterministic alternative,
//    for batch-launches / enlarge-grid candidates.
// Any other candidate maps to SkippedUnsupported with a reason, never to a
// fabricated result.
//
// Verdicts compare setup-free metrics only (one-time context/JIT setup is
// excluded): copy+kernel portion for persist-buffers, kernel min time for
// vary-block-size. Thresholds are fixed; classification is a pure function.

#include <string>
#include <vector>

#include "drishti/correlation/correlation.h"
#include "drishti/diagnosis/root_cause.h"
#include "drishti/optimizer/suggest.h"
#include "drishti/profiling/gpu_metrics.h"

namespace drishti::optimizer {

enum class Verdict { Improved, Regressed, Unchanged, Invalid };
enum class ExperimentStatus { Completed, SkippedUnsupported };

[[nodiscard]] std::string verdict_name(Verdict v);

struct ExperimentConfig {
    double improve_threshold = 0.05;  // |rel| <= this => unchanged
};

struct ExperimentResult {
    std::string experiment_id;  // "persist-buffers" | "vary-block-size"
    std::string candidate_id;
    ExperimentStatus status = ExperimentStatus::Completed;
    std::string skip_reason;  // set when SkippedUnsupported
    // Controlled pair, same N/repeats; baseline describes the variant.
    std::string baseline_desc;    // e.g. "recopy-per-launch", "block 256"
    std::string candidate_desc;   // e.g. "persist-buffers", "block 512"
    profiling::GpuProfileMetrics baseline;
    profiling::GpuProfileMetrics candidate;
    std::string primary_metric;  // e.g. "copy-kernel-portion-ms"
    double baseline_value = 0.0;
    double candidate_value = 0.0;
    double rel_improvement = 0.0;  // (base - cand) / base
    bool candidate_correct = false;
    Verdict verdict = Verdict::Invalid;
    std::string verdict_reason;
    double confidence = 0.7;
};

struct ExperimentReport {
    bool ok = false;  // false only on infrastructure failure, not on verdicts
    std::string error;
    std::string relation;
    std::string kernel;
    std::vector<ExperimentResult> experiments;
};

// Pure verdict classification over already-measured values. Never throws.
// Invalid when either side failed, results are unverified, or the baseline
// is non-positive.
[[nodiscard]] Verdict classify(double baseline_value, double candidate_value,
                               bool baseline_ok, bool candidate_ok,
                               bool candidate_correct, double threshold,
                               std::string* reason = nullptr);

// Which experiment (if any) implements a candidate id. Empty = unsupported.
[[nodiscard]] std::string experiment_for_candidate(const std::string& candidate_id);

// Run supported experiments for a suggestion report. The correlated baseline
// metrics anchor N/block/repeats; each experiment executes a fresh controlled
// pair back-to-back. Never throws.
ExperimentReport run_experiments(const correlation::CorrelationRecord& record,
                                 const diagnosis::DiagnosisReport& diagnosis,
                                 const SuggestionReport& suggestions,
                                 const ExperimentConfig& cfg = {});

// Machine-readable rendering for later phases.
[[nodiscard]] std::string experiments_to_json(const ExperimentReport& r);

// Human-readable chain report for `drishti optimize`:
// diagnosis -> candidate -> baseline -> candidate result -> correctness -> verdict.
[[nodiscard]] std::string format_experiments_report(
    const ExperimentReport& r, const SuggestionReport& suggestions,
    const diagnosis::DiagnosisReport& diagnosis);

// Provenance wiring: represent the experiments as a provenance pass.
[[nodiscard]] drishti::provenance::PassInfo experiments_to_pass_info(
    const ExperimentReport& r);

}  // namespace drishti::optimizer

#endif
