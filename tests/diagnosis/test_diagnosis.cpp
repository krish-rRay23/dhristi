// Focused Phase 6 tests: deterministic root-cause rules.
//
// Synthetic records exercise every rule in both firing and non-firing states;
// one live test runs the full correlate->diagnose flow on the RTX 3050.
// Hardware tests skip cleanly when the MLIR or CUDA backends are unavailable.

#include "drishti/diagnosis/root_cause.h"
#include "drishti/optimizer/compiler_experiment.h"

#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/correlation/correlation.h"
#include "drishti/core/version.h"

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
using drishti::diagnosis::diagnosis_to_json;
using drishti::diagnosis::diagnosis_to_pass_info;
using drishti::diagnosis::emit_to_engine;
using drishti::diagnosis::format_diagnosis_report;
using drishti::diagnosis::KernelTraits;
using drishti::diagnosis::Severity;

// A healthy synthetic vecadd record: minimal traffic, sane geometry.
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
    r.gpu.bytes_moved = 786432;  // 12 * 65536: exactly minimal
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

const drishti::diagnosis::Finding* find_rule(const DiagnosisReport& r,
                                             const std::string& id) {
    for (const auto& f : r.findings) {
        if (f.bottleneck.rule_id == id) return &f;
    }
    return nullptr;
}

bool any_severity(const DiagnosisReport& r, Severity s) {
    for (const auto& f : r.findings) {
        if (f.severity == s) return true;
    }
    return false;
}

TEST(RulesEngine, HealthyRecordHasNoWarningsOrErrors) {
    const DiagnosisReport r = diagnose(healthy_record());
    ASSERT_TRUE(r.ok);
    EXPECT_FALSE(any_severity(r, Severity::Warning));
    EXPECT_FALSE(any_severity(r, Severity::Error));
    const auto* roof = find_rule(r, "roofline-balance");
    ASSERT_NE(roof, nullptr);
    EXPECT_EQ(roof->bottleneck.title, "memory-bound workload");
    const auto* occ = find_rule(r, "register-pressure");
    ASSERT_NE(occ, nullptr);
    EXPECT_EQ(occ->severity, Severity::Note);
    const auto* fusion = find_rule(r, "fusion");
    ASSERT_NE(fusion, nullptr);
    EXPECT_EQ(fusion->bottleneck.title, "fusion adequate");
}

TEST(RulesEngine, DeterministicAcrossRuns) {
    const CorrelationRecord rec = healthy_record();
    EXPECT_EQ(diagnosis_to_json(diagnose(rec)), diagnosis_to_json(diagnose(rec)));
}

TEST(RulesEngine, ExcessTrafficFires) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.bytes_moved = 3 * 786432u;  // 3x minimal
    const DiagnosisReport r = diagnose(rec);
    ASSERT_TRUE(r.ok);
    const auto* f = find_rule(r, "memory-traffic");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->severity, Severity::Warning);
    EXPECT_GT(f->confidence, 0.8);
    EXPECT_NE(f->bottleneck.related_op.find("arith.addf"), std::string::npos);
    EXPECT_NE(f->explanation.find("redundant"), std::string::npos);
}

TEST(RulesEngine, TransferDominatedFires) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.h2d_ms = 0.09;  // copies 0.1 vs kernel 0.02 -> ratio 5
    rec.gpu.d2h_ms = 0.01;
    const DiagnosisReport r = diagnose(rec);
    const auto* f = find_rule(r, "transfer-share");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->severity, Severity::Warning);
}

TEST(RulesEngine, UnderOccupiedLaunchFires) {
    CorrelationRecord rec = healthy_record();
    rec.grid_size = 2;  // < 16 SMs
    rec.gpu.grid_size = 2;
    const DiagnosisReport r = diagnose(rec);
    const auto* f = find_rule(r, "launch-occupancy");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->severity, Severity::Warning);
    EXPECT_EQ(f->bottleneck.title, "under-occupied launch");
}

TEST(RulesEngine, InvalidBlockIsError) {
    CorrelationRecord rec = healthy_record();
    rec.block_size = 2048;  // > device max 1024
    const DiagnosisReport r = diagnose(rec);
    const auto* f = find_rule(r, "launch-config");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->severity, Severity::Error);
    EXPECT_DOUBLE_EQ(f->confidence, 1.0);
}

TEST(RulesEngine, LaunchOverheadFires) {
    CorrelationRecord rec = healthy_record();
    rec.gpu.wall_ms = 50.0;  // kernel total 0.1 of 50 ms wall
    const DiagnosisReport r = diagnose(rec);
    const auto* f = find_rule(r, "launch-overhead");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->severity, Severity::Warning);
}

TEST(RulesEngine, HighRegisterSyntheticFires) {
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
    const DiagnosisReport r = diagnose(rec, cfg);
    ASSERT_TRUE(r.ok);
    const auto* f = find_rule(r, "register-pressure");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->severity, Severity::Warning);
    EXPECT_DOUBLE_EQ(f->confidence, 0.6);
    EXPECT_NE(f->explanation.find("Static analysis"), std::string::npos);
}

TEST(RulesEngine, UnknownKernelAbstains) {
    CorrelationRecord rec = healthy_record();
    rec.kernel = "mystery";
    rec.gpu.kernel = "mystery";
    const DiagnosisReport r = diagnose(rec);
    ASSERT_TRUE(r.ok);
    const auto* roof = find_rule(r, "roofline-balance");
    ASSERT_NE(roof, nullptr);
    EXPECT_EQ(roof->bottleneck.title, "roofline unclassified");
    EXPECT_DOUBLE_EQ(roof->confidence, 0.0);
    const auto* occ = find_rule(r, "register-pressure");
    ASSERT_NE(occ, nullptr);
    EXPECT_EQ(occ->severity, Severity::Note);
    // No warnings from rules that need traits.
    for (const auto& f : r.findings) {
        if (f.bottleneck.rule_id == "memory-traffic" ||
            f.bottleneck.rule_id == "register-pressure") {
            EXPECT_EQ(f.severity, Severity::Note);
        }
    }
}

TEST(RulesEngine, FailedRecordYieldsSingleError) {
    CorrelationRecord rec;
    rec.ok = false;
    rec.error = "GPU profiling failed: no device";
    const DiagnosisReport r = diagnose(rec);
    EXPECT_FALSE(r.ok);
    ASSERT_EQ(r.findings.size(), 1u);
    EXPECT_EQ(r.findings[0].severity, Severity::Error);
    EXPECT_FALSE(r.findings[0].explanation.empty());
}

TEST(RulesEngine, CoExecutedFusionNotAssessable) {
    CorrelationRecord rec = healthy_record();
    rec.relation = RelationKind::CoExecuted;
    const DiagnosisReport r = diagnose(rec);
    const auto* f = find_rule(r, "fusion");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->bottleneck.title, "fusion not assessable");
}

TEST(RulesEngine, JsonAndReportShape) {
    const DiagnosisReport r = diagnose(healthy_record());
    const std::string j = diagnosis_to_json(r);
    for (const char* key : {"drishti.diagnosis/v1", "rule_id", "severity",
                            "confidence", "related_op", "related_pass",
                            "explanation", "memory-bound workload"}) {
        EXPECT_NE(j.find(key), std::string::npos) << "missing key: " << key;
    }
    const std::string rep = format_diagnosis_report(r);
    EXPECT_NE(rep.find("Drishti Diagnosis Report"), std::string::npos);
    EXPECT_NE(rep.find("register-pressure"), std::string::npos);
    EXPECT_NE(rep.find("arith.addf (node 6)"), std::string::npos);
}

TEST(RulesEngine, EmitToEngine) {
    const DiagnosisReport r = diagnose(healthy_record());
    drishti::diagnosis::DiagnosticEngine engine;
    emit_to_engine(r, engine);
    EXPECT_EQ(engine.all().size(), r.findings.size());
    EXPECT_FALSE(engine.has_errors());
}

TEST(RulesEngine, LiveRtx3050Flow) {
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
    const DiagnosisReport r = diagnose(rec);
    EXPECT_TRUE(r.ok);
    EXPECT_FALSE(r.findings.empty());
    const auto* roof = find_rule(r, "roofline-balance");
    ASSERT_NE(roof, nullptr);
    EXPECT_EQ(roof->bottleneck.title, "memory-bound workload");
    EXPECT_NE(roof->bottleneck.related_op.find("arith.addf"), std::string::npos);
    for (const auto& f : r.findings) {
        EXPECT_FALSE(f.explanation.empty());
        EXPECT_FALSE(f.evidence.empty());
        EXPECT_GE(f.confidence, 0.0);
        EXPECT_LE(f.confidence, 1.0);
    }
    const auto pass = diagnosis_to_pass_info(r, rec.kernel);
    EXPECT_EQ(pass.name, "diagnose:vecadd");
}

TEST(CompilerDiagnosis, IdentifiesKernelFusionAndMemoryTrafficReduction) {
    drishti::optimizer::CompilerExperimentReport exp;
    exp.ok = true;
    exp.baseline.label = "baseline";
    exp.baseline.kernels = {"_fusedemo_kernel_0", "_fusedemo_kernel_1"};
    exp.baseline.bytes_per_element = 24.0;
    exp.baseline.metrics.num_elements = 65536;
    exp.baseline.metrics.bytes_moved = 1572864;
    exp.baseline.metrics.kernel_ms_min = 0.0174;
    exp.baseline.metrics.device.compute_major = 8;
    exp.baseline.metrics.device.compute_minor = 6;
    exp.baseline.metrics.device.sm_count = 16;

    exp.candidate.label = "fused";
    exp.candidate.kernels = {"_fusedemo_kernel_fused"};
    exp.candidate.bytes_per_element = 16.0;
    exp.candidate.metrics.num_elements = 65536;
    exp.candidate.metrics.bytes_moved = 1048576;
    exp.candidate.metrics.kernel_ms_min = 0.0123;
    exp.candidate.metrics.device.name = "NVIDIA GeForce RTX 3050 Laptop GPU";
    exp.candidate.metrics.device.compute_major = 8;
    exp.candidate.metrics.device.compute_minor = 6;
    exp.candidate.metrics.device.sm_count = 16;

    const auto r = drishti::diagnosis::diagnose_compiler_experiment(exp);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.transformation, "affine-loop-fusion");
    EXPECT_EQ(r.baseline_launches, 2u);
    EXPECT_EQ(r.candidate_launches, 1u);
    EXPECT_EQ(r.vram_bytes_saved, 524288u);
    EXPECT_GT(r.speedup_percent, 20.0);
    EXPECT_FALSE(r.findings.empty());

    const std::string formatted = drishti::diagnosis::format_compiler_diagnosis_report(r);
    EXPECT_NE(formatted.find("Phase 10"), std::string::npos);
    EXPECT_NE(formatted.find("Kernel Fusion & Launch Overhead Elimination"), std::string::npos);
    EXPECT_NE(formatted.find("Eliminated Intermediate Global-Memory Round-Trip"), std::string::npos);
    EXPECT_NE(formatted.find("524288 bytes saved"), std::string::npos);

    const std::string json = drishti::diagnosis::compiler_diagnosis_to_json(r);
    EXPECT_NE(json.find("drishti.compiler_diagnosis/v1"), std::string::npos);
    EXPECT_NE(json.find("kernel-fusion"), std::string::npos);
}

TEST(CompilerDiagnosis, LiveRtx3050ExperimentDiagnosis) {
    if (!drishti::core::have_cuda()) GTEST_SKIP() << "CUDA backend unavailable";
    if (!drishti::backends::cuda::device_present())
        GTEST_SKIP() << "no CUDA device/driver present";

    drishti::optimizer::CompilerExperimentConfig cfg;
    cfg.num_elements = 65536;
    cfg.block_size = 256;
    cfg.repeats = 3;
    std::string err;
    const auto exp = drishti::optimizer::run_compiler_experiment(cfg, &err);
    ASSERT_TRUE(exp.ok) << err;

    const auto r = drishti::diagnosis::diagnose_compiler_experiment(exp);
    EXPECT_TRUE(r.ok);
    EXPECT_EQ(r.baseline_launches, 2u);
    EXPECT_EQ(r.candidate_launches, 1u);
    EXPECT_EQ(r.vram_bytes_saved, 524288u);
    EXPECT_FALSE(r.findings.empty());

    bool found_fusion = false, found_mem = false;
    for (const auto& f : r.findings) {
        if (f.bottleneck.rule_id == "kernel-fusion") found_fusion = true;
        if (f.bottleneck.rule_id == "memory-traffic-elimination") found_mem = true;
    }
    EXPECT_TRUE(found_fusion);
    EXPECT_TRUE(found_mem);
}

}  // namespace
