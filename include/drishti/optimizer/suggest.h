#ifndef DRISHTI_OPTIMIZER_SUGGEST_H
#define DRISHTI_OPTIMIZER_SUGGEST_H

// Phase 7: Optimization candidate generation.
//
// Explainable, deterministic mapping from diagnosis findings to concrete
// optimization candidates. This layer proposes only; it never applies,
// benchmarks, or autotunes anything (a later phase may).
//
// Conservativeness is structural, not advisory:
//  - Only findings with severity Warning generate candidates. Notes (healthy
//    or abstaining) and Errors (invalid input/config) never do.
//  - A candidate requires a known related op (related_op != "n/a").
//  - Source findings below min_confidence (default 0.5) generate nothing.
//  - Candidate confidence never exceeds the source finding's confidence.
//  - Unknown kernels / unassessed rules therefore yield zero candidates.
//
// Extensibility: mappings are table entries (rule id + finding title ->
// builder); new detectors add rows without touching the engine. Backend
// specifics never appear here: everything flows from Finding evidence.

#include <functional>
#include <string>
#include <vector>

#include "drishti/diagnosis/root_cause.h"

namespace drishti::optimizer {

struct Candidate {
    std::string id;            // stable machine id, e.g. "reuse-device-data"
    std::string title;
    std::string target_op;     // from the finding, e.g. "arith.addf (node 6)"
    std::string target_kernel;  // e.g. "vecadd"
    std::string related_pass;  // e.g. "canonicalize,cse (pass 1)"
    std::string transformation;  // concrete proposed change
    std::string rationale;       // cites the diagnosis evidence verbatim
    std::string expected_effect;
    double confidence = 0.0;  // <= source finding confidence, capped per rule
    std::string source_rule_id;
    double source_confidence = 0.0;
    double estimated_speedup_percent = 0.0;
    std::string cost_model_summary;
};

struct SuggestionReport {
    bool ok = false;  // false only when the input diagnosis itself failed
    std::string error;
    std::string relation;  // correlation relation name, for context
    std::string kernel;
    std::vector<Candidate> candidates;
};

struct SuggestConfig {
    double min_confidence = 0.5;  // findings below this generate nothing
};

using CandidateBuilder = std::function<Candidate(const diagnosis::Finding&,
                                                 const std::string& kernel)>;

// Table-driven mapping; stable order, deterministic output.
struct CandidateMapping {
    std::string rule_id;
    std::string finding_title;  // exact title required; empty matches any
    double confidence_cap;
    CandidateBuilder build;
};

[[nodiscard]] std::vector<CandidateMapping> default_mappings();

// Map a diagnosis to candidates. Never throws; empty candidates with ok=true
// means the evidence was insufficient for any conservative recommendation.
SuggestionReport suggest_for(const diagnosis::DiagnosisReport& report,
                             const std::string& kernel,
                             const SuggestConfig& cfg = {});

// Machine-readable rendering for later phases.
[[nodiscard]] std::string suggestions_to_json(const SuggestionReport& r);

// Human-readable report for `drishti suggest`.
[[nodiscard]] std::string format_suggestions_report(const SuggestionReport& r);

// Provenance wiring: represent the suggestions as a provenance pass.
[[nodiscard]] drishti::provenance::PassInfo suggestions_to_pass_info(
    const SuggestionReport& r);

}  // namespace drishti::optimizer

#endif
