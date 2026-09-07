#include "drishti/optimizer/experiment.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/core/config.h"

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

std::string fmt(double v, int prec = 4) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(prec);
    oss << v;
    return oss.str();
}

// Setup-free comparison metric: copies + kernel work, excluding one-time
// context/JIT setup captured in wall_ms.
double copy_kernel_portion(const profiling::GpuProfileMetrics& m) {
    return m.h2d_ms + m.d2h_ms +
           m.kernel_ms_avg * static_cast<double>(std::max(1, m.repeats));
}

// Deterministic alternative block size: first of {512, 256, 128, 1024}
// that differs from the baseline.
int alt_block(int baseline) {
    constexpr int kAlts[] = {512, 256, 128, 1024};
    for (int b : kAlts) {
        if (b != baseline) return b;
    }
    return baseline;  // unreachable
}

#if DRISHTI_HAVE_CUDA
bool run_variant(std::size_t n, int block, int repeats, bool recopy,
                 profiling::GpuProfileMetrics& out, std::string* err) {
    backends::cuda::VecaddVariantConfig cfg;
    cfg.num_elements = n;
    cfg.block_size = block;
    cfg.repeats = repeats;
    cfg.recopy_per_launch = recopy;
    return backends::cuda::run_vecadd_variant(cfg, out, err);
}
#endif

}  // namespace

std::string verdict_name(Verdict v) {
    switch (v) {
        case Verdict::Improved: return "improved";
        case Verdict::Regressed: return "regressed";
        case Verdict::Unchanged: return "unchanged";
        case Verdict::Invalid: return "invalid";
    }
    return "unknown";
}

Verdict classify(double baseline_value, double candidate_value, bool baseline_ok,
                 bool candidate_ok, bool candidate_correct, double threshold,
                 std::string* reason) {
    const auto set_reason = [&](const std::string& s) {
        if (reason) *reason = s;
    };
    if (!baseline_ok || !candidate_ok) {
        set_reason("measurement failed on at least one side; no comparison made");
        return Verdict::Invalid;
    }
    if (!candidate_correct) {
        set_reason("candidate output failed host verification; result unusable");
        return Verdict::Invalid;
    }
    if (!(baseline_value > 0.0)) {
        set_reason("non-positive baseline value; relative change undefined");
        return Verdict::Invalid;
    }
    const double rel = (baseline_value - candidate_value) / baseline_value;
    if (reason) {
        *reason = "relative change " + fmt(rel * 100.0, 1) + "% vs threshold " +
                  fmt(threshold * 100.0, 1) + "%";
    }
    if (rel > threshold) return Verdict::Improved;
    if (rel < -threshold) return Verdict::Regressed;
    return Verdict::Unchanged;
}

std::string experiment_for_candidate(const std::string& candidate_id) {
    if (candidate_id == "reuse-device-data") return "persist-buffers";
    if (candidate_id == "batch-launches") return "vary-block-size";
    if (candidate_id == "enlarge-grid") return "vary-block-size";
    return "";
}

ExperimentReport run_experiments(const correlation::CorrelationRecord& record,
                                 const diagnosis::DiagnosisReport&,
                                 const SuggestionReport& suggestions,
                                 const ExperimentConfig& cfg) {
    ExperimentReport r;
    r.relation = suggestions.relation;
    r.kernel = suggestions.kernel;
    if (!suggestions.ok) {
        r.ok = false;
        r.error = "suggestions failed: " + suggestions.error;
        return r;
    }
    r.ok = true;
    const std::size_t n = record.num_elements;
    const int block = record.block_size;
    const int repeats = record.gpu.repeats > 0 ? record.gpu.repeats : 1;
    for (const auto& c : suggestions.candidates) {
        ExperimentResult e;
        e.candidate_id = c.id;
        e.experiment_id = experiment_for_candidate(c.id);
        if (e.experiment_id.empty()) {
            e.status = ExperimentStatus::SkippedUnsupported;
            e.skip_reason =
                "no reliable implementation yet for candidate '" + c.id + "'";
            e.verdict = Verdict::Invalid;
            e.verdict_reason = "skipped: " + e.skip_reason;
            r.experiments.push_back(std::move(e));
            continue;
        }
#if !DRISHTI_HAVE_CUDA
        e.status = ExperimentStatus::SkippedUnsupported;
        e.skip_reason = "CUDA backend unavailable in this build";
        e.verdict = Verdict::Invalid;
        e.verdict_reason = "skipped: " + e.skip_reason;
        r.experiments.push_back(std::move(e));
        continue;
#else
        std::string err;
        if (e.experiment_id == "persist-buffers") {
            e.baseline_desc = "recopy-per-launch (naive app pattern)";
            e.candidate_desc = "persist-buffers (copy once, reuse)";
            e.primary_metric = "copy-kernel-portion-ms";
            if (!run_variant(n, block, repeats, true, e.baseline, &err)) {
                e.verdict = Verdict::Invalid;
                e.verdict_reason = "baseline run failed: " + err;
                e.candidate_correct = false;
                r.experiments.push_back(std::move(e));
                continue;
            }
            if (!run_variant(n, block, repeats, false, e.candidate, &err)) {
                e.verdict = Verdict::Invalid;
                e.verdict_reason = "candidate run failed: " + err;
                e.candidate_correct = false;
                r.experiments.push_back(std::move(e));
                continue;
            }
            e.candidate_correct = e.candidate.correct;
            e.baseline_value = copy_kernel_portion(e.baseline);
            e.candidate_value = copy_kernel_portion(e.candidate);
        } else {  // vary-block-size
            const int alt = alt_block(block);
            e.baseline_desc = "block " + std::to_string(block);
            e.candidate_desc = "block " + std::to_string(alt);
            e.primary_metric = "kernel-min-ms";
            if (!run_variant(n, block, repeats, false, e.baseline, &err)) {
                e.verdict = Verdict::Invalid;
                e.verdict_reason = "baseline run failed: " + err;
                e.candidate_correct = false;
                r.experiments.push_back(std::move(e));
                continue;
            }
            if (!run_variant(n, alt, repeats, false, e.candidate, &err)) {
                e.verdict = Verdict::Invalid;
                e.verdict_reason = "candidate run failed: " + err;
                e.candidate_correct = false;
                r.experiments.push_back(std::move(e));
                continue;
            }
            e.candidate_correct = e.candidate.correct;
            e.baseline_value = e.baseline.kernel_ms_min;
            e.candidate_value = e.candidate.kernel_ms_min;
        }
        e.rel_improvement = e.baseline_value > 0.0
                                ? (e.baseline_value - e.candidate_value) / e.baseline_value
                                : 0.0;
        e.verdict = classify(e.baseline_value, e.candidate_value, e.baseline.ok,
                             e.candidate.ok, e.candidate_correct,
                             cfg.improve_threshold, &e.verdict_reason);
        r.experiments.push_back(std::move(e));
#endif
    }
    return r;
}

std::string experiments_to_json(const ExperimentReport& r) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"schema\": \"drishti.experiments/v1\",\n"
        << "  \"ok\": " << (r.ok ? "true" : "false") << ",\n"
        << "  \"error\": \"" << json_escape(r.error) << "\",\n"
        << "  \"relation\": \"" << json_escape(r.relation) << "\",\n"
        << "  \"kernel\": \"" << json_escape(r.kernel) << "\",\n"
        << "  \"experiments\": [";
    for (std::size_t i = 0; i < r.experiments.size(); ++i) {
        const auto& e = r.experiments[i];
        oss << (i ? "," : "") << "\n    {\"experiment_id\": \""
            << json_escape(e.experiment_id) << "\", \"candidate_id\": \""
            << json_escape(e.candidate_id) << "\", \"status\": \""
            << (e.status == ExperimentStatus::Completed ? "completed"
                                                        : "skipped-unsupported")
            << "\", \"skip_reason\": \"" << json_escape(e.skip_reason)
            << "\", \"baseline_desc\": \"" << json_escape(e.baseline_desc)
            << "\", \"candidate_desc\": \"" << json_escape(e.candidate_desc)
            << "\", \"primary_metric\": \"" << json_escape(e.primary_metric)
            << "\", \"baseline_value\": " << fmt(e.baseline_value)
            << ", \"candidate_value\": " << fmt(e.candidate_value)
            << ", \"rel_improvement\": " << fmt(e.rel_improvement)
            << ", \"candidate_correct\": " << (e.candidate_correct ? "true" : "false")
            << ", \"verdict\": \"" << verdict_name(e.verdict)
            << "\", \"verdict_reason\": \"" << json_escape(e.verdict_reason) << "\"}";
    }
    oss << (r.experiments.empty() ? "]" : "\n  ]") << "\n}";
    return oss.str();
}

std::string format_experiments_report(
    const ExperimentReport& r, const SuggestionReport& suggestions,
    const diagnosis::DiagnosisReport& diagnosis) {
    std::ostringstream oss;
    oss << "Drishti Optimization Experiments\n"
        << "  Relation : " << r.relation << "\n"
        << "  Kernel   : " << (r.kernel.empty() ? "(unknown)" : r.kernel) << "\n";
    if (!r.ok) {
        oss << "  Status   : FAILED: " << r.error << "\n";
        return oss.str();
    }
    auto finding_for = [&](const std::string& rule) -> const diagnosis::Finding* {
        for (const auto& f : diagnosis.findings) {
            if (f.bottleneck.rule_id == rule) return &f;
        }
        return nullptr;
    };
    auto candidate_for = [&](const std::string& id) -> const Candidate* {
        for (const auto& c : suggestions.candidates) {
            if (c.id == id) return &c;
        }
        return nullptr;
    };
    oss << "  Experiments: " << r.experiments.size() << "\n";
    for (const auto& e : r.experiments) {
        oss << "\n  [experiment " << (e.experiment_id.empty() ? "(none)" : e.experiment_id)
            << "] for candidate " << e.candidate_id << "\n";
        if (const auto* c = candidate_for(e.candidate_id)) {
            oss << "    candidate : " << c->title << " (conf " << fmt(c->confidence)
                << ")\n";
            oss << "    change    : " << c->transformation << "\n";
            if (const auto* f = finding_for(c->source_rule_id)) {
                oss << "    diagnosis : [" << f->bottleneck.rule_id << "] "
                    << f->bottleneck.title << " on " << f->bottleneck.related_op
                    << "\n";
            }
        }
        if (e.status == ExperimentStatus::SkippedUnsupported) {
            oss << "    status    : SKIPPED (" << e.skip_reason << ")\n";
            continue;
        }
        oss << "    baseline  : " << e.baseline_desc << " -> " << e.primary_metric
            << " " << fmt(e.baseline_value) << " ms\n"
            << "    candidate : " << e.candidate_desc << " -> " << e.primary_metric
            << " " << fmt(e.candidate_value) << " ms\n"
            << "    correctness: " << (e.candidate_correct ? "PASS" : "FAIL")
            << " (host reference)\n"
            << "    verdict   : " << verdict_name(e.verdict) << " (" << e.verdict_reason
            << ")\n";
    }
    oss << "\n  Note: experiment-only by default; nothing was applied to source.\n";
    return oss.str();
}

drishti::provenance::PassInfo experiments_to_pass_info(const ExperimentReport& r) {
    drishti::provenance::PassInfo p;
    p.name = "optimize:" + (r.kernel.empty() ? "unknown" : r.kernel);
    std::size_t improved = 0, completed = 0;
    std::string top = "none";
    for (const auto& e : r.experiments) {
        if (e.status != ExperimentStatus::Completed) continue;
        ++completed;
        if (e.verdict == Verdict::Improved && top == "none") top = e.candidate_id;
        if (e.verdict == Verdict::Improved) ++improved;
    }
    p.description = std::to_string(completed) + " experiments, " +
                    std::to_string(improved) + " improved, top: " + top;
    p.enabled_by_default = true;
    return p;
}

}  // namespace drishti::optimizer
