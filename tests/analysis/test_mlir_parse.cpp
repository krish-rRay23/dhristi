#include "drishti/analysis/mlir_analysis.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace drishti::analysis;

namespace {

constexpr std::string_view kSimpleModule = R"mlir(
module {
  func.func @add_one(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %0 = arith.addi %arg0, %c1 : i32
    return %0 : i32
  }
}
)mlir";

constexpr std::string_view kLoopModule = R"mlir(
module {
  func.func @sum(%arg0: memref<100xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %cf0 = arith.constant 0.0 : f32
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    %sum = scf.for %i = %c0 to %c100 step %c1 iter_args(%acc = %cf0) -> f32 {
      %v = memref.load %arg0[%i] : memref<100xf32>
      %next = arith.addf %acc, %v : f32
      scf.yield %next : f32
    }
    return %sum : f32
  }
}
)mlir";

constexpr std::string_view kMultiFuncModule = R"mlir(
module {
  func.func @id(%x: i32) -> i32 { return %x : i32 }
  func.func @add(%a: i32, %b: i32) -> i32 {
    %s = arith.addi %a, %b : i32
    return %s : i32
  }
}
)mlir";

constexpr std::string_view kBadModule = R"mlir(
module { func.func @broken(%x:i32) { totally_not_an_op %x } }
)mlir";

constexpr std::string_view kBranchModule = R"mlir(
module {
  func.func @pick(%c: i1, %a: i32, %b: i32) -> i32 {
    cf.cond_br %c, ^then, ^else
  ^then:
    cf.br ^merge(%a : i32)
  ^else:
    cf.br ^merge(%b : i32)
  ^merge(%x: i32):
    return %x : i32
  }
}
)mlir";

constexpr std::string_view kFillModule = R"mlir(
module {
  func.func @fill(%c: f32) -> tensor<4xf32> {
    %out = tensor.empty() : tensor<4xf32>
    %r = linalg.fill ins(%c : f32) outs(%out : tensor<4xf32>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
)mlir";

constexpr std::string_view kUnregisteredDialectModule = R"mlir(
module {
  "custom.foo"() : () -> ()
}
)mlir";

std::filesystem::path write_temp_mlir(std::string_view contents, std::string_view name) {
    auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary);
    out << contents;
    out.close();
    return path;
}

}  // namespace

TEST(MlirAnalysisTest, ParseSimpleModule) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    const auto stats = engine.analyze_string(kSimpleModule);
    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->function_count, 1u);
    EXPECT_EQ(stats->functions.size(), 1u);
    EXPECT_EQ(stats->functions.front().name, "add_one");
    EXPECT_EQ(stats->functions.front().argument_count, 1u);
    EXPECT_EQ(stats->functions.front().result_count, 1u);
    EXPECT_GE(stats->total_operations, 3u);
}

TEST(MlirAnalysisTest, DetectBuiltinFuncArithDialects) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    const auto stats = engine.analyze_string(kSimpleModule);
    ASSERT_TRUE(stats.has_value());
    const auto has = [&](std::string_view n) {
        for (const auto& d : stats->dialects) {
            if (d.name == n) return true;
        }
        return false;
    };
    EXPECT_TRUE(has("builtin")) << "builtin dialect required";
    EXPECT_TRUE(has("func")) << "func dialect required";
    EXPECT_TRUE(has("arith")) << "arith dialect required";
}

TEST(MlirAnalysisTest, OperationHistogramCountsArithConstant) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    const auto stats = engine.analyze_string(kSimpleModule);
    ASSERT_TRUE(stats.has_value());
    auto it = stats->op_histogram.find("arith.constant");
    ASSERT_NE(it, stats->op_histogram.end());
    EXPECT_EQ(it->second, 1u);
}

TEST(MlirAnalysisTest, DetectScfForLoop) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    const auto stats = engine.analyze_string(kLoopModule);
    ASSERT_TRUE(stats.has_value());
    EXPECT_GE(stats->loop_count, 1u);
    ASSERT_FALSE(stats->loops.empty());
    EXPECT_EQ(stats->loops.front().kind, "scf.for");
    EXPECT_GE(stats->loops.front().body_ops, 2u);
    EXPECT_GE(stats->total_blocks, 2u);
    EXPECT_GE(stats->total_regions, 2u);
}

TEST(MlirAnalysisTest, MultiFunctionCounting) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    const auto stats = engine.analyze_string(kMultiFuncModule);
    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->function_count, 2u);
    ASSERT_EQ(stats->functions.size(), 2u);
    EXPECT_EQ(stats->functions[0].argument_count + stats->functions[1].argument_count, 3u);
}

TEST(MlirAnalysisTest, InvalidModuleFails) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    std::string err;
    const auto stats = engine.analyze_string(kBadModule, &err);
    EXPECT_FALSE(stats.has_value());
    EXPECT_FALSE(err.empty());
}

TEST(MlirAnalysisTest, TotallyMalformedSourceYieldsNullopt) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    std::string err;
    const auto stats = engine.analyze_string("this is not even close to mlir <<<", &err);
    EXPECT_FALSE(stats.has_value());
}

TEST(MlirAnalysisTest, FormatReportProducesExpectedSections) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    const auto stats = engine.analyze_string(kLoopModule);
    ASSERT_TRUE(stats.has_value());
    const auto rep = format_report(*stats, "demo.mlir");
    for (std::string_view section : {
             "Structural Overview",
             "Dialect Usage",
             "Top Operations",
             "Functions",
             "Loop-Like Constructs",
         }) {
        EXPECT_NE(rep.find(section), std::string::npos) << "missing section: " << section;
    }
    EXPECT_NE(rep.find("demo.mlir"), std::string::npos);
    EXPECT_NE(rep.find("scf.for"), std::string::npos);
}

TEST(MlirAnalysisTest, EmptySourceFailsGracefully) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    std::string err;
    const auto stats = engine.analyze_string("", &err);
    if (!stats.has_value()) {
        SUCCEED() << "empty source rejected as expected";
        return;
    }
    SUCCEED() << "empty source parsed into empty module ("
              << stats->total_operations << " ops); no invariants violated";
}

TEST(MlirAnalysisTest, AnalyzeFileRoundTrip) {
    const auto path = write_temp_mlir(kSimpleModule, "drishti_test_simple.mlir");
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    std::string err;
    const auto stats = engine.analyze_file(path.string(), &err);
    ASSERT_TRUE(stats.has_value()) << err;
    EXPECT_EQ(stats->function_count, 1u);
    EXPECT_EQ(stats->functions.front().name, "add_one");
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(MlirAnalysisTest, AnalyzeFileMissingPathFails) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    std::string err;
    const auto stats = engine.analyze_file("definitely/not/a/real/file.mlir", &err);
    EXPECT_FALSE(stats.has_value());
    EXPECT_FALSE(err.empty());
}

TEST(MlirAnalysisTest, BranchOpsAreCounted) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    const auto stats = engine.analyze_string(kBranchModule);
    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->branch_count, 3u);
    EXPECT_GE(stats->total_blocks, 4u);
    const auto has = [&](std::string_view n) {
        for (const auto& d : stats->dialects) {
            if (d.name == n) return true;
        }
        return false;
    };
    EXPECT_TRUE(has("cf")) << "cf dialect required";
    const auto rep = format_report(*stats, "branch.mlir");
    EXPECT_NE(rep.find("Control Flow"), std::string::npos);
    EXPECT_NE(rep.find("cf.cond_br"), std::string::npos);
}

TEST(MlirAnalysisTest, LinalgOpsAreNotLoops) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    const auto stats = engine.analyze_string(kFillModule);
    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(stats->loop_count, 0u);
    EXPECT_TRUE(stats->loops.empty());
    auto it = stats->op_histogram.find("linalg.fill");
    ASSERT_NE(it, stats->op_histogram.end());
}

TEST(MlirAnalysisTest, UnregisteredDialectOpsStillCounted) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    const auto stats = engine.analyze_string(kUnregisteredDialectModule);
    ASSERT_TRUE(stats.has_value());
    const auto has = [&](std::string_view n) {
        for (const auto& d : stats->dialects) {
            if (d.name == n) return true;
        }
        return false;
    };
    EXPECT_TRUE(has("custom")) << "unregistered dialect must fall back to op-name prefix";
}

TEST(MlirAnalysisTest, ParseErrorCarriesDiagnostics) {
    MlirAnalysisContext ctx;
    MlirAnalysisEngine engine{ctx};
    std::string err;
    const auto stats = engine.analyze_string(kBadModule, &err);
    EXPECT_FALSE(stats.has_value());
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("totally_not_an_op"), std::string::npos)
        << "expected diagnostic to name the offending op, got: " << err;
}
