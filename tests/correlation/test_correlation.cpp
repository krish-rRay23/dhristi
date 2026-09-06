// Focused Phase 5 tests: compiler-to-GPU correlation model + live flow.
//
// Hardware tests skip cleanly when the MLIR or CUDA backends are unavailable;
// the model tests always run.

#include "drishti/correlation/correlation.h"

#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/core/version.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using drishti::correlation::CorrelationConfig;
using drishti::correlation::CorrelationRecord;
using drishti::correlation::correlation_to_json;
using drishti::correlation::correlation_to_pass_info;
using drishti::correlation::format_correlation_report;
using drishti::correlation::generate_vecadd_mlir;
using drishti::correlation::relation_name;
using drishti::correlation::RelationKind;
using drishti::correlation::run_correlation;

TEST(CorrelationModel, GeneratedWorkloadHasAnchors) {
    const std::string src = generate_vecadd_mlir(1024);
    EXPECT_NE(src.find("func.func @vecadd"), std::string::npos);
    EXPECT_NE(src.find("arith.addf"), std::string::npos);
    EXPECT_NE(src.find("memref<1024xf32>"), std::string::npos);
    EXPECT_NE(src.find("scf.for"), std::string::npos);
}

TEST(CorrelationModel, RelationNames) {
    EXPECT_EQ(relation_name(RelationKind::SameComputation), "same-computation");
    EXPECT_EQ(relation_name(RelationKind::CoExecuted), "co-executed");
}

TEST(CorrelationModel, JsonContainsChain) {
    CorrelationRecord r;
    r.ok = true;
    r.mlir_source_label = "<embedded-vecadd>";
    r.mlir_ops = 12;
    r.mlir_funcs = 1;
    r.pipeline = "canonicalize,cse";
    r.pipeline_pass_id = 1;
    r.transform_edges = 2;
    r.kernel = "vecadd";
    r.num_elements = 1024;
    r.relation = RelationKind::SameComputation;
    r.evidence.push_back("profile N == mlir element count N (1024)");
    drishti::correlation::MlirAnchor a;
    a.role = "compute-op";
    a.op_name = "arith.addf";
    a.dialect = "arith";
    a.node_id = 7;
    a.location = "<input>:9:12";
    r.anchors.push_back(a);
    r.gpu.ok = true;
    r.gpu.kernel = "vecadd";
    r.gpu.correct = true;

    const std::string j = correlation_to_json(r);
    for (const char* key : {"drishti.correlation/v1", "same-computation",
                            "arith.addf", "transform_edges", "node_id",
                            "drishti.gpu_profile/v1", "evidence"}) {
        EXPECT_NE(j.find(key), std::string::npos) << "missing key: " << key;
    }
}

TEST(CorrelationModel, ReportShowsChain) {
    CorrelationRecord r;
    r.ok = true;
    r.mlir_source_label = "<embedded-vecadd>";
    r.kernel = "vecadd";
    r.num_elements = 512;
    r.relation = RelationKind::SameComputation;
    drishti::correlation::MlirAnchor a;
    a.role = "compute-op";
    a.op_name = "arith.addf";
    a.node_id = 7;
    a.location = "<input>:9:12";
    r.anchors.push_back(a);
    r.gpu.kernel_ms_avg = 0.05;
    r.gpu.gbps_effective = 100.0;
    r.gpu.correct = true;

    const std::string rep = format_correlation_report(r);
    EXPECT_NE(rep.find("Drishti Correlation Report"), std::string::npos);
    EXPECT_NE(rep.find("arith.addf (#7)"), std::string::npos);
    EXPECT_NE(rep.find("kernel:vecadd"), std::string::npos);
    EXPECT_NE(rep.find("same-computation"), std::string::npos);

    const auto pass = correlation_to_pass_info(r);
    EXPECT_EQ(pass.name, "correlate:vecadd");
    EXPECT_NE(pass.description.find("same-computation"), std::string::npos);
}

TEST(CorrelationModel, FailureReportMentionsError) {
    CorrelationRecord r;
    r.ok = false;
    r.error = "GPU profiling failed: no CUDA-capable device found";
    const std::string rep = format_correlation_report(r);
    EXPECT_NE(rep.find("FAILED"), std::string::npos);
    EXPECT_NE(rep.find("no CUDA-capable device"), std::string::npos);
}

TEST(CorrelationFlow, EmbeddedWorkloadEndToEnd) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    if (!drishti::core::have_cuda()) GTEST_SKIP() << "CUDA backend unavailable";
    if (!drishti::backends::cuda::device_present())
        GTEST_SKIP() << "no CUDA device/driver present";

    CorrelationConfig cfg;
    cfg.num_elements = 4096;
    cfg.block_size = 256;
    cfg.repeats = 2;
    std::string err;
    const CorrelationRecord r = run_correlation(cfg, &err);
    ASSERT_TRUE(r.ok) << err;
    EXPECT_EQ(r.relation, RelationKind::SameComputation);
    EXPECT_EQ(r.kernel, "vecadd");
    EXPECT_EQ(r.num_elements, 4096u);
    EXPECT_FALSE(r.anchors.empty());
    bool saw_compute = false;
    for (const auto& a : r.anchors) {
        EXPECT_GT(a.node_id, 0u);
        if (a.role == "compute-op") {
            saw_compute = true;
            EXPECT_EQ(a.op_name, "arith.addf");
        }
    }
    EXPECT_TRUE(saw_compute);
    EXPECT_TRUE(r.gpu.correct);
    EXPECT_GT(r.gpu.kernel_ms_avg, 0.0);
    EXPECT_FALSE(r.evidence.empty());
}

TEST(CorrelationFlow, UserFileCoExecuted) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    if (!drishti::core::have_cuda()) GTEST_SKIP() << "CUDA backend unavailable";
    if (!drishti::backends::cuda::device_present())
        GTEST_SKIP() << "no CUDA device/driver present";

    CorrelationConfig cfg;
    cfg.num_elements = 4096;
    cfg.repeats = 2;
    cfg.mlir_file = "C:/Users/krish/Dhristi/samples/sample.mlir";
    std::string err;
    const CorrelationRecord r = run_correlation(cfg, &err);
    ASSERT_TRUE(r.ok) << err;
    // sample.mlir is not the reference vecadd workload: no equivalence claim.
    EXPECT_EQ(r.relation, RelationKind::CoExecuted);
    EXPECT_GT(r.mlir_ops, 0u);
    EXPECT_TRUE(r.gpu.correct);
}

TEST(CorrelationFlow, MissingFileFailsCleanly) {
    if (!drishti::core::have_mlir()) GTEST_SKIP() << "MLIR backend unavailable";
    if (!drishti::core::have_cuda()) GTEST_SKIP() << "CUDA backend unavailable";
    CorrelationConfig cfg;
    cfg.mlir_file = "C:/nonexistent/path.mlir";
    std::string err;
    const CorrelationRecord r = run_correlation(cfg, &err);
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(err.empty());
}

}  // namespace
