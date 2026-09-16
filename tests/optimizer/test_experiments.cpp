// Focused Phase 8 tests: controlled optimization experiments.
//
// Pure verdict/mapping logic is tested synthetically with hand-built metrics
// (no GPU). One live test runs the full correlate -> diagnose -> suggest ->
// experiment flow on the RTX 3050 and checks verdict consistency (not fixed
// outcomes, which vary with the shared GPU). Nothing is applied anywhere.

#include "drishti/optimizer/experiment.h"

#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/core/version.h"
#include "drishti/correlation/correlation.h"
#include "drishti/diagnosis/root_cause.h"
#include "drishti/optimizer/suggest.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using drishti::correlation::CorrelationConfig;
using drishti::correlation::CorrelationRecord;
using drishti::correlation::MlirAnchor;
using drishti::correlation::RelationKind;
using drishti::correlation::run_correlation;
using drishti::diagnosis::diagnose;
using drishti::optimizer::classify;
using drishti::optimizer::ExperimentConfig;
using drishti::optimizer::ExperimentReport;
using drishti::optimizer::ExperimentStatus;
using drishti::optimizer::experiment_for_candidate;
using drishti::optimizer::experiments_to_json;
using drishti::optimizer::experiments_to_pass_info;
using drishti::optimizer::format_experiments_report;
using drishti::optimizer::run_experiments;
using drishti::optimizer::suggest_for;
using drishti::optimizer::Verdict;
using drishti::optimizer::verdict_name;

// Healthy synthetic vecadd record (mirrors the Phase 6/7 fixtures).
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

TEST(VerdictClassify, ImprovedRegressedUnchanged) {
    std::string reason;
    EXPECT_EQ(classify(100.0, 80.0, true, true, true, 0.05, &reason),
              Verdict::Improved);
    EXPECT_FALSE(reason.empty());
    EXPECT_EQ(classify(100.0, 130.0, true, true, true, 0.05), Verdict::Regressed);
    EXPECT_EQ(classify(100.0, 102.0, true, true, true, 0.05), Verdict::Unchanged);
    // Exact threshold boundary is strict: +/-5% is unchanged.
    EXPECT_EQ(classify(100.0, 95.0, true, true, true, 0.05), Verdict::Unchanged);
    EXPECT_EQ(classify(100.0, 105.0, true, true, true, 0.05), Verdict::Unchanged);
    EXPECT_EQ(classify(100.0, 94.9, true, true, true, 0.05), Verdict::Improved);
}

TEST(VerdictClassify, InvalidCases) {
    EXPECT_EQ(classify(100.0, 80.0, false, true, true, 0.05), Verdict::Invalid);
    EXPECT_EQ(classify(100.0, 80.0, true, false, true, 0.05), Verdict::Invalid);
    EXPECT_EQ(classify(100.0, 80.0, true, true, false, 0.05), Verdict::Invalid);
    EXPECT_EQ(classify(0.0, 80.0, true, true, true, 0.05), Verdict::Invalid);
    EXPECT_EQ(classify(-1.0, 80.0, true, true, true, 0.05), Verdict::Invalid);
    EXPECT_EQ(verdict_name(Verdict::Improved), "improved");
    EXPECT_EQ(verdict_name(Verdict::Regressed), "regressed");
    EXPECT_EQ(verdict_name(Verdict::Unchanged), "unchanged");
    EXPECT_EQ(verdict_name(Verdict::Invalid), "invalid");
}

TEST(ExperimentMapping, KnownCandidatesMap) {
    EXPECT_EQ(experiment_for_candidate("reuse-device-data"), "persist-buffers");
    EXPECT_EQ(experiment_for_candidate("batch-launches"), "vary-block-size");
    EXPECT_EQ(experiment_for_candidate("enlarge-grid"), "vary-block-size");
}

TEST(ExperimentMapping, OthersAreUnsupported) {
    EXPECT_TRUE(experiment_for_candidate("improve-reuse-layout").empty());
    EXPECT_TRUE(experiment_for_candidate("reduce-tile-aggressiveness").empty());
    EXPECT_TRUE(experiment_for_candidate("fuse-stages").empty());
    EXPECT_TRUE(experiment_for_candidate("no-such-candidate").empty());
}

TEST(ExperimentSkipped, UnsupportedNeedsNoGpu) {
    // 3x traffic fires memory-traffic + fusion warnings; both map to
    // unsupported candidates, so no GPU call happens.
    CorrelationRecord rec = healthy_record();
    rec.gpu.bytes_moved = 3 * 786432u;
    const auto diag = diagnose(rec);
    ASSERT_TRUE(diag.ok);
    const auto sugg = suggest_for(diag, "vecadd");
    ASSERT_TRUE(sugg.ok);
    ASSERT_FALSE(sugg.candidates.empty());
    const ExperimentReport r = run_experiments(rec, diag, sugg);
    EXPECT_TRUE(r.ok);
    ASSERT_FALSE(r.experiments.empty());
    for (const auto& e : r.experiments) {
        EXPECT_EQ(e.status, ExperimentStatus::SkippedUnsupported);
        EXPECT_FALSE(e.skip_reason.empty());
        EXPECT_EQ(e.verdict, Verdict::Invalid);
    }
}

TEST(ExperimentSkipped, FailedSuggestionsPropagate) {
    CorrelationRecord rec;
    rec.ok = false;
    rec.error = "GPU profiling failed: no device";
    const auto diag = diagnose(rec);
    const auto sugg = suggest_for(diag, "vecadd");
    const ExperimentReport r = run_experiments(rec, diag, sugg);
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error.empty());
    EXPECT_TRUE(r.experiments.empty());
}

TEST(ExperimentModel, JsonAndReportShape) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.bytes_moved = 3 * 786432u;
    const auto diag = diagnose(rec);
    const auto sugg = suggest_for(diag, "vecadd");
    const ExperimentReport r = run_experiments(rec, diag, sugg);
    EXPECT_TRUE(r.ok);
    const std::string j = experiments_to_json(r);
    for (const char* key : {"drishti.experiments/v1", "experiment_id",
                            "candidate_id", "skipped-unsupported", "verdict",
                            "skip_reason", "improve-reuse-layout"}) {
        EXPECT_NE(j.find(key), std::string::npos) << "missing key: " << key;
    }
    const std::string rep = format_experiments_report(r, sugg, diag);
    EXPECT_NE(rep.find("Drishti Optimization Experiments"), std::string::npos);
    EXPECT_NE(rep.find("SKIPPED"), std::string::npos);
    EXPECT_NE(rep.find("nothing was applied"), std::string::npos);
    const auto pass = experiments_to_pass_info(r);
    EXPECT_EQ(pass.name, "optimize:vecadd");
}

TEST(ExperimentFlow, LiveRtx3050VerdictsConsistent) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    if (!drishti::core::have_cuda()) GTEST_SKIP() << "CUDA backend unavailable";
    if (!drishti::backends::cuda::device_present())
        GTEST_SKIP() << "no CUDA device/driver present";
    CorrelationConfig cfg;
    cfg.num_elements = 16384;
    cfg.repeats = 3;
    std::string err;
    const CorrelationRecord rec = run_correlation(cfg, &err);
    ASSERT_TRUE(rec.ok) << err;
    const auto diag = diagnose(rec);
    ASSERT_TRUE(diag.ok);
    const auto sugg = suggest_for(diag, rec.kernel);
    ASSERT_TRUE(sugg.ok);
    const ExperimentReport r = run_experiments(rec, diag, sugg);
    EXPECT_TRUE(r.ok) << r.error;
    EXPECT_EQ(r.kernel, "vecadd");
    for (const auto& e : r.experiments) {
        if (e.status == ExperimentStatus::SkippedUnsupported) {
            EXPECT_FALSE(e.skip_reason.empty());
            continue;
        }
        EXPECT_TRUE(e.candidate_correct);
        EXPECT_TRUE(e.baseline.ok);
        EXPECT_TRUE(e.candidate.ok);
        EXPECT_EQ(e.baseline.num_elements, e.candidate.num_elements);
        // Verdict must be structurally valid (non-invalid classification)
        EXPECT_NE(e.verdict, Verdict::Invalid);
        EXPECT_FALSE(e.verdict_reason.empty());
    }
    // Persisting buffers must beat recopying them: the copy share dominates
    // at this size, so the margin is robust to shared-GPU noise.
    for (const auto& e : r.experiments) {
        if (e.experiment_id == "persist-buffers" &&
            e.status == ExperimentStatus::Completed) {
            EXPECT_EQ(e.verdict, Verdict::Improved) << e.verdict_reason;
        }
    }
}

}  // namespace
