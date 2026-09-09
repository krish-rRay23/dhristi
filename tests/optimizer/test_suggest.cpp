// Focused Phase 7 tests: deterministic diagnosis -> candidate mapping.
//
// Synthetic diagnosis reports exercise every mapping in firing and abstaining
// states; one live test runs correlate -> diagnose -> suggest on the RTX 3050
// and checks mapping consistency (not specific findings, which vary with the
// shared GPU). No optimizations are applied anywhere here.

#include "drishti/optimizer/suggest.h"

#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/core/version.h"
#include "drishti/correlation/correlation.h"
#include "drishti/diagnosis/root_cause.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using drishti::correlation::CorrelationConfig;
using drishti::correlation::CorrelationRecord;
using drishti::correlation::MlirAnchor;
using drishti::correlation::RelationKind;
using drishti::correlation::run_correlation;
using drishti::diagnosis::diagnose;
using drishti::diagnosis::DiagnosisConfig;
using drishti::diagnosis::DiagnosisReport;
using drishti::diagnosis::KernelTraits;
using drishti::diagnosis::Severity;
using drishti::optimizer::format_suggestions_report;
using drishti::optimizer::suggest_for;
using drishti::optimizer::SuggestionReport;
using drishti::optimizer::SuggestConfig;
using drishti::optimizer::suggestions_to_json;
using drishti::optimizer::suggestions_to_pass_info;

// Healthy synthetic vecadd record (mirrors the Phase 6 fixture).
CorrelationRecord healthy_record() {
    CorrelationRecord r;
    r.ok = true;
    r.mlir_source_label = "<embedded-vecadd>";
    r.mlir_ops = 12;
    r.mlir_funcs = 1;
    r.pipeline = "canonicalize,cse";
    r.pipeline_pass_id = 1;
    MlirAnchor fn;
    fn.role = "function";
    fn.op_name = "func.func";
    fn.dialect = "func";
    fn.node_id = 11;
    fn.location = "<input>:2:3";
    MlirAnchor op;
    op.role = "compute-op";
    op.op_name = "arith.addf";
    op.dialect = "arith";
    op.node_id = 6;
    op.location = "<input>:9:12";
    r.anchors = {fn, op};
    r.kernel = "vecadd";
    r.num_elements = 65536;
    r.block_size = 256;
    r.grid_size = 256;
    r.relation = RelationKind::SameComputation;
    r.gpu.ok = true;
    r.gpu.kernel = "vecadd";
    r.gpu.num_elements = 65536;
    r.gpu.block_size = 256;
    r.gpu.grid_size = 256;
    r.gpu.repeats = 5;
    r.gpu.kernel_ms_avg = 0.02;
    r.gpu.kernel_ms_min = 0.018;
    r.gpu.h2d_ms = 0.03;
    r.gpu.d2h_ms = 0.02;
    r.gpu.wall_ms = 0.8;
    r.gpu.gbps_effective = 78.0;
    r.gpu.bytes_moved = 786432;
    r.gpu.correct = true;
    r.gpu.timing_source = "cuda-events";
    r.gpu.device.present = true;
    r.gpu.device.backend = "cuda";
    r.gpu.device.name = "Synthetic GPU";
    r.gpu.device.compute_major = 8;
    r.gpu.device.compute_minor = 6;
    r.gpu.device.sm_count = 16;
    r.gpu.device.max_threads_per_block = 1024;
    r.gpu.device.clock_mhz = 1500;
    return r;
}

const drishti::optimizer::Candidate* find_candidate(const SuggestionReport& r,
                                                    const std::string& id) {
    for (const auto& c : r.candidates) {
        if (c.id == id) return &c;
    }
    return nullptr;
}

SuggestionReport suggest_healthy() {
    return suggest_for(diagnose(healthy_record()), "vecadd");
}

TEST(SuggestMapping, TransferWarningMapsToReuse) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.h2d_ms = 0.09;  // copies 0.1 vs kernel 0.02 -> transfer warning
    rec.gpu.d2h_ms = 0.01;
    const SuggestionReport r = suggest_for(diagnose(rec), "vecadd");
    ASSERT_TRUE(r.ok);
    const auto* c = find_candidate(r, "reuse-device-data");
    ASSERT_NE(c, nullptr);
    EXPECT_NE(c->target_op.find("arith.addf"), std::string::npos);
    EXPECT_EQ(c->target_kernel, "vecadd");
    EXPECT_FALSE(c->transformation.empty());
    EXPECT_FALSE(c->expected_effect.empty());
    EXPECT_NE(c->rationale.find("transfer-dominated runtime"), std::string::npos);
    EXPECT_LE(c->confidence, c->source_confidence);
    EXPECT_EQ(c->source_rule_id, "transfer-share");
}

TEST(SuggestMapping, TrafficWarningMapsToReuseLayout) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.bytes_moved = 3 * 786432u;
    const SuggestionReport r = suggest_for(diagnose(rec), "vecadd");
    ASSERT_TRUE(r.ok);
    const auto* c = find_candidate(r, "improve-reuse-layout");
    ASSERT_NE(c, nullptr);
    EXPECT_LE(c->confidence, c->source_confidence);
    EXPECT_NE(c->rationale.find("excessive memory traffic"), std::string::npos);
}

TEST(SuggestMapping, RegisterWarningMapsWithCap) {
    DiagnosisConfig cfg;
    KernelTraits synth;
    synth.kernel = "synth-wide";
    synth.flops_per_element = 4.0;
    synth.bytes_per_element_min = 16.0;
    synth.regs_per_thread_est = 128;
    synth.regs_basis = "synthetic test trait";
    cfg.extra_traits["synth-wide"] = synth;
    CorrelationRecord rec = healthy_record();
    rec.kernel = "synth-wide";
    rec.gpu.kernel = "synth-wide";
    rec.gpu.bytes_moved = 16u * 65536u;
    const SuggestionReport r = suggest_for(diagnose(rec, cfg), "synth-wide");
    ASSERT_TRUE(r.ok);
    const auto* c = find_candidate(r, "reduce-tile-aggressiveness");
    ASSERT_NE(c, nullptr);
    // Static-estimate basis: candidate confidence capped at 0.6.
    EXPECT_LE(c->confidence, 0.6);
    EXPECT_NE(c->transformation.find("threads per block"), std::string::npos);
}

TEST(SuggestMapping, LaunchWarningsMap) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.wall_ms = 50.0;  // launch overhead
    rec.grid_size = 2;       // under-occupied
    rec.gpu.grid_size = 2;
    const SuggestionReport r = suggest_for(diagnose(rec), "vecadd");
    ASSERT_TRUE(r.ok);
    EXPECT_NE(find_candidate(r, "batch-launches"), nullptr);
    const auto* g = find_candidate(r, "enlarge-grid");
    ASSERT_NE(g, nullptr);
    EXPECT_NE(g->expected_effect.find("SM"), std::string::npos);
}

TEST(SuggestMapping, UnfusedTrafficMapsToFuse) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.bytes_moved = 3 * 786432u;  // fires traffic + fusion warnings
    const SuggestionReport r = suggest_for(diagnose(rec), "vecadd");
    ASSERT_TRUE(r.ok);
    const auto* c = find_candidate(r, "fuse-stages");
    ASSERT_NE(c, nullptr);
    EXPECT_LE(c->confidence, 0.6);
}

TEST(SuggestConservative, HealthyYieldsNone) {
    const SuggestionReport r = suggest_healthy();
    ASSERT_TRUE(r.ok);
    EXPECT_TRUE(r.candidates.empty());
}

TEST(SuggestConservative, UnknownKernelYieldsNone) {
    CorrelationRecord rec = healthy_record();
    rec.kernel = "mystery";
    rec.gpu.kernel = "mystery";
    const SuggestionReport r = suggest_for(diagnose(rec), "mystery");
    ASSERT_TRUE(r.ok);
    EXPECT_TRUE(r.candidates.empty());
}

TEST(SuggestConservative, InvalidConfigYieldsNone) {
    CorrelationRecord rec = healthy_record();
    // > device max 1024 (Error) but still occupiable, so the *only* finding
    // is the Error, which must never source a candidate.
    rec.block_size = 1200;
    rec.gpu.block_size = 1200;
    const SuggestionReport r = suggest_for(diagnose(rec), "vecadd");
    ASSERT_TRUE(r.ok);
    EXPECT_TRUE(r.candidates.empty());
}

TEST(SuggestConservative, FailedDiagnosisYieldsNoneAndNotOk) {
    CorrelationRecord rec;
    rec.ok = false;
    rec.error = "GPU profiling failed: no device";
    const SuggestionReport r = suggest_for(diagnose(rec), "vecadd");
    EXPECT_FALSE(r.ok);
    EXPECT_TRUE(r.candidates.empty());
    EXPECT_FALSE(r.error.empty());
}

TEST(SuggestConservative, ConfidenceFloorIsEnforced) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.h2d_ms = 0.09;
    rec.gpu.d2h_ms = 0.01;  // transfer warning at conf 0.7
    SuggestConfig cfg;
    cfg.min_confidence = 0.9;  // above every finding tier
    const SuggestionReport r = suggest_for(diagnose(rec), "vecadd", cfg);
    ASSERT_TRUE(r.ok);
    EXPECT_TRUE(r.candidates.empty());
}

TEST(SuggestConservative, CandidateNeverExceedsSource) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.bytes_moved = 3 * 786432u;
    rec.gpu.wall_ms = 50.0;
    const SuggestionReport r = suggest_for(diagnose(rec), "vecadd");
    ASSERT_TRUE(r.ok);
    ASSERT_FALSE(r.candidates.empty());
    for (const auto& c : r.candidates) {
        EXPECT_LE(c.confidence, c.source_confidence);
        EXPECT_GE(c.confidence, 0.0);
        EXPECT_LE(c.confidence, 1.0);
        EXPECT_FALSE(c.target_op.empty());
        EXPECT_FALSE(c.transformation.empty());
        EXPECT_FALSE(c.rationale.empty());
    }
}

TEST(SuggestModel, DeterministicAcrossRuns) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.bytes_moved = 3 * 786432u;
    const auto a = suggest_for(diagnose(rec), "vecadd");
    const auto b = suggest_for(diagnose(rec), "vecadd");
    EXPECT_EQ(suggestions_to_json(a), suggestions_to_json(b));
}

TEST(SuggestModel, JsonAndReportShape) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.h2d_ms = 0.09;
    rec.gpu.d2h_ms = 0.01;
    const SuggestionReport r = suggest_for(diagnose(rec), "vecadd");
    const std::string j = suggestions_to_json(r);
    for (const char* key : {"drishti.suggestions/v1", "reuse-device-data",
                            "target_op", "transformation", "rationale",
                            "expected_effect", "source_rule_id"}) {
        EXPECT_NE(j.find(key), std::string::npos) << "missing key: " << key;
    }
    const std::string rep = format_suggestions_report(r);
    EXPECT_NE(rep.find("Drishti Suggestions Report"), std::string::npos);
    EXPECT_NE(rep.find("[reuse-device-data]"), std::string::npos);
    EXPECT_NE(rep.find("not applied or benchmarked"), std::string::npos);

    const SuggestionReport empty = suggest_healthy();
    EXPECT_NE(format_suggestions_report(empty).find("evidence insufficient"),
              std::string::npos);
    const auto pass = suggestions_to_pass_info(r);
    EXPECT_EQ(pass.name, "suggest:vecadd");
    EXPECT_NE(pass.description.find("reuse-device-data"), std::string::npos);
}

TEST(SuggestFlow, LiveRtx3050MappingConsistent) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    if (!drishti::core::have_cuda()) GTEST_SKIP() << "CUDA backend unavailable";
    if (!drishti::backends::cuda::device_present())
        GTEST_SKIP() << "no CUDA device/driver present";
    CorrelationConfig cfg;
    cfg.num_elements = 4096;
    cfg.repeats = 2;
    std::string err;
    const CorrelationRecord rec = run_correlation(cfg, &err);
    ASSERT_TRUE(rec.ok) << err;
    const auto diag = diagnose(rec);
    ASSERT_TRUE(diag.ok);
    const SuggestionReport r = suggest_for(diag, rec.kernel);
    EXPECT_TRUE(r.ok);
    // Every candidate traces to a Warning finding ...
    for (const auto& c : r.candidates) {
        bool found = false;
        for (const auto& f : diag.findings) {
            if (f.bottleneck.rule_id == c.source_rule_id) {
                EXPECT_EQ(f.severity, Severity::Warning);
                EXPECT_LE(c.confidence, f.confidence);
                found = true;
            }
        }
        EXPECT_TRUE(found) << "candidate without source finding: " << c.id;
        EXPECT_FALSE(c.transformation.empty());
        EXPECT_FALSE(c.rationale.empty());
    }
    // ... and every mapped Warning finding produced its candidate.
    for (const auto& f : diag.findings) {
        if (f.severity != Severity::Warning) continue;
        if (f.bottleneck.related_op == "n/a") continue;
        bool mapped = false;
        for (const auto& m : drishti::optimizer::default_mappings()) {
            if (m.rule_id != f.bottleneck.rule_id) continue;
            if (!m.finding_title.empty() && m.finding_title != f.bottleneck.title)
                continue;
            if (f.confidence < SuggestConfig{}.min_confidence) continue;
            mapped = true;
        }
        if (!mapped) continue;  // unmapped warning kinds need no candidate
        bool present = false;
        for (const auto& c : r.candidates) {
            if (c.source_rule_id == f.bottleneck.rule_id) present = true;
        }
        EXPECT_TRUE(present) << "warning without candidate: " << f.bottleneck.rule_id;
    }
}

}  // namespace
