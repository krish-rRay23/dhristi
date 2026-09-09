#include "drishti/optimizer/suggest.h"
#include "drishti/optimizer/cost_model.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace drishti::optimizer {
namespace {

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

std::string fmt(double v, int prec = 2) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(prec);
    oss << v;
    return oss.str();
}

// Rationale quotes the finding's own evidence so the chain
// diagnosis-evidence -> candidate stays auditable.
std::string rationale_for(const diagnosis::Finding& f) {
    std::ostringstream oss;
    oss << "Finding '" << f.bottleneck.title << "' (conf " << fmt(f.confidence)
        << "): ";
    for (std::size_t i = 0; i < f.evidence.size(); ++i) {
        if (i) oss << "; ";
        oss << f.evidence[i];
    }
    return oss.str();
}

Candidate base_candidate(const std::string& id, const std::string& title,
                         const diagnosis::Finding& f, const std::string& kernel,
                         double cap) {
    Candidate c;
    c.id = id;
    c.title = title;
    c.target_op = f.bottleneck.related_op;
    c.target_kernel = kernel;
    c.related_pass = f.bottleneck.related_pass;
    c.rationale = rationale_for(f);
    c.confidence = std::min(cap, f.confidence);
    c.source_rule_id = f.bottleneck.rule_id;
    c.source_confidence = f.confidence;
    return c;
}

}  // namespace

std::vector<CandidateMapping> default_mappings() {
    return {
        {"transfer-share", "transfer-dominated runtime", 0.7,
         [](const diagnosis::Finding& f, const std::string& kernel) {
             Candidate c = base_candidate("reuse-device-data",
                                          "Reuse/persist device data", f, kernel, 0.7);
             c.transformation =
                 "Allocate device buffers once and reuse them across launches; "
                 "transfer host inputs a single time instead of copying per launch.";
             c.expected_effect =
                 "Host-device copy share of wall time falls toward zero; end-to-end "
                 "time approaches kernel time.";
             return c;
         }},
        {"memory-traffic", "excessive memory traffic", 0.7,
         [](const diagnosis::Finding& f, const std::string& kernel) {
             Candidate c = base_candidate("improve-reuse-layout",
                                          "Improve data reuse / layout", f, kernel, 0.7);
             c.transformation =
                 "Restructure accesses for reuse in fast memory (tiling, shared "
                 "memory, coalescing) to approach the algorithmic byte minimum.";
             c.expected_effect =
                 "Bytes moved per element fall toward the minimum; effective "
                 "bandwidth rises for the same kernel time.";
             return c;
         }},
        {"register-pressure", "high register pressure", 0.6,
         [](const diagnosis::Finding& f, const std::string& kernel) {
             Candidate c = base_candidate("reduce-tile-aggressiveness",
                                          "Reduce tile/fusion aggressiveness", f, kernel,
                                          0.6);
             c.transformation =
                 "Reduce threads per block and/or split the fused region so register "
                 "demand fits occupancy targets (basis is a static estimate).";
             c.expected_effect =
                 "Higher theoretical occupancy; better latency hiding if occupancy "
                 "was the limiter. Confirm with hardware counters when available.";
             return c;
         }},
        {"launch-overhead", "launch/synchronization overhead", 0.7,
         [](const diagnosis::Finding& f, const std::string& kernel) {
             Candidate c = base_candidate("batch-launches", "Batch work per launch",
                                          f, kernel, 0.7);
             c.transformation =
                 "Batch more elements into each launch (and fuse surrounding "
                 "launches) so fixed launch/sync costs amortize over more work.";
             c.expected_effect =
                 "Kernel share of wall time rises; fixed per-launch overhead "
                 "becomes negligible.";
             return c;
         }},
        {"launch-occupancy", "under-occupied launch", 0.7,
         [](const diagnosis::Finding& f, const std::string& kernel) {
             Candidate c = base_candidate("enlarge-grid",
                                          "Enlarge grid / problem size", f, kernel, 0.7);
             c.transformation =
                 "Increase the grid (or the problem size driving it) so every SM "
                 "receives thread blocks.";
             c.expected_effect = "Idle SMs are eliminated; throughput scales with "
                                 "the newly utilized hardware.";
             return c;
         }},
        {"fusion", "possible unfused intermediate traffic", 0.6,
         [](const diagnosis::Finding& f, const std::string& kernel) {
             Candidate c =
                 base_candidate("fuse-stages", "Fuse producer/consumer stages", f,
                                kernel, 0.6);
             c.transformation =
                 "Fuse the stages separated by the compiler into a single kernel so "
                 "intermediates stay on-chip instead of round-tripping memory.";
             c.expected_effect =
                 "Intermediate traffic disappears; bytes moved approach the "
                 "algorithmic minimum.";
             return c;
         }},
    };
}

SuggestionReport suggest_for(const diagnosis::DiagnosisReport& report,
                             const std::string& kernel,
                             const SuggestConfig& cfg) {
    SuggestionReport r;
    r.relation = report.relation;
    r.kernel = kernel;
    if (!report.ok) {
        r.ok = false;
        r.error = "diagnosis failed: " + report.error;
        return r;
    }
    r.ok = true;
    const auto mappings = default_mappings();
    for (const auto& f : report.findings) {
        // Conservative gates: warnings only, known op, sufficient confidence.
        if (f.severity != diagnosis::Severity::Warning) continue;
        if (f.bottleneck.related_op == "n/a") continue;
        if (f.confidence < cfg.min_confidence) continue;
        if (f.evidence.empty()) continue;
        for (const auto& m : mappings) {
            if (m.rule_id != f.bottleneck.rule_id) continue;
            if (!m.finding_title.empty() && m.finding_title != f.bottleneck.title)
                continue;
            Candidate c = m.build(f, kernel);
            // Calculate analytical cost model estimate for candidate
            CostModelFeatures base_feat;
            base_feat.kernel_label = kernel;
            base_feat.launch_count = (c.id == "fuse-stages") ? 2 : 1;
            base_feat.bytes_per_element = (c.id == "fuse-stages") ? 24.0 : 12.0;

            CostModelFeatures cand_feat = base_feat;
            if (c.id == "fuse-stages") {
                cand_feat.launch_count = 1;
                cand_feat.bytes_per_element = 16.0;
            } else if (c.id == "reduce-tile-aggressiveness") {
                cand_feat.regs_per_thread_est = 12;
            }

            const auto est_base = estimate_kernel_cost(base_feat);
            const auto est_cand = estimate_kernel_cost(cand_feat);

            if (est_base.predicted_total_ms > 0.0) {
                c.estimated_speedup_percent = (est_base.predicted_total_ms - est_cand.predicted_total_ms) / est_base.predicted_total_ms * 100.0;
            }
            c.cost_model_summary = "Predicted latency: " + fmt(est_cand.predicted_total_ms, 4) + " ms (baseline " +
                                  fmt(est_base.predicted_total_ms, 4) + " ms, speedup " +
                                  (c.estimated_speedup_percent >= 0 ? "+" : "") + fmt(c.estimated_speedup_percent, 1) + "%)";

            r.candidates.push_back(std::move(c));
        }
    }
    return r;
}

std::string suggestions_to_json(const SuggestionReport& r) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"schema\": \"drishti.suggestions/v1\",\n"
        << "  \"ok\": " << (r.ok ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(r.error) << "\",\n"
        << "  \"relation\": \"" << json_escape(r.relation) << "\",\n"
        << "  \"kernel\": \"" << json_escape(r.kernel) << "\",\n"
        << "  \"candidates\": [";
    for (std::size_t i = 0; i < r.candidates.size(); ++i) {
        const auto& c = r.candidates[i];
        oss << (i ? "," : "") << "\n    {\"id\": \"" << json_escape(c.id)
            << "\", \"title\": \"" << json_escape(c.title)
            << "\", \"target_op\": \"" << json_escape(c.target_op)
            << "\", \"target_kernel\": \"" << json_escape(c.target_kernel)
            << "\", \"related_pass\": \"" << json_escape(c.related_pass)
            << "\", \"transformation\": \"" << json_escape(c.transformation)
            << "\", \"rationale\": \"" << json_escape(c.rationale)
            << "\", \"expected_effect\": \"" << json_escape(c.expected_effect)
            << "\", \"confidence\": " << fmt(c.confidence)
            << ", \"source_rule_id\": \"" << json_escape(c.source_rule_id)
            << "\", \"source_confidence\": " << fmt(c.source_confidence) << "}";
    }
    oss << (r.candidates.empty() ? "]" : "\n  ]") << "\n}";
    return oss.str();
}

std::string format_suggestions_report(const SuggestionReport& r) {
    std::ostringstream oss;
    oss << "Drishti Suggestions Report\n"
        << "  Relation : " << r.relation << "\n"
        << "  Kernel   : " << (r.kernel.empty() ? "(unknown)" : r.kernel) << "\n";
    if (!r.ok) {
        oss << "  Status   : FAILED: " << r.error << "\n";
        return oss.str();
    }
    oss << "  Candidates: " << r.candidates.size() << "\n";
    if (r.candidates.empty()) {
        oss << "  No candidates: evidence insufficient for a conservative "
               "recommendation.\n";
        return oss.str();
    }
    for (const auto& c : r.candidates) {
        oss << "\n  [" << c.id << "] " << c.title << " (conf " << fmt(c.confidence)
            << ")\n"
            << "    target   : " << c.target_op << " in kernel " << c.target_kernel
            << " @ " << c.related_pass << "\n"
            << "    change   : " << c.transformation << "\n"
            << "    rationale: " << c.rationale << "\n"
            << "    effect   : " << c.expected_effect << "\n";
        if (!c.cost_model_summary.empty()) {
            oss << "    est cost : " << c.cost_model_summary << "\n";
        }
        oss << "    source   : rule " << c.source_rule_id << " (conf "
            << fmt(c.source_confidence) << ")\n";
    }
    oss << "\n  Note: candidates are proposed only, not applied or benchmarked.\n";
    return oss.str();
}

drishti::provenance::PassInfo suggestions_to_pass_info(const SuggestionReport& r) {
    drishti::provenance::PassInfo p;
    p.name = "suggest:" + (r.kernel.empty() ? "unknown" : r.kernel);
    std::string top = r.candidates.empty() ? "none" : r.candidates.front().id;
    p.description = std::to_string(r.candidates.size()) + " candidates, top: " + top;
    p.enabled_by_default = true;
    return p;
}

}  // namespace drishti::optimizer
