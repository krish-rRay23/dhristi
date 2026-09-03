#include "drishti/provenance/provenance.h"

#include <algorithm>
#include <gtest/gtest.h>

#include <string>

using namespace drishti::provenance;

namespace {

TEST(ProvenanceGraphTest, AddNodeReturnsUniqueIds) {
    ProvenanceGraph graph;
    OperationNode n1; n1.op_name = "op1";
    OperationNode n2; n2.op_name = "op2";
    OperationNode n3; n3.op_name = "op3";

    auto id1 = graph.add_node(std::move(n1));
    auto id2 = graph.add_node(std::move(n2));
    auto id3 = graph.add_node(std::move(n3));

    EXPECT_EQ(id1, 1u);
    EXPECT_EQ(id2, 2u);
    EXPECT_EQ(id3, 3u);
    EXPECT_EQ(graph.node_count(), 3u);
}

TEST(ProvenanceGraphTest, GetNodeReturnsCorrectNode) {
    ProvenanceGraph graph;
    OperationNode n; n.op_name = "test.op";
    n.dialect = "test";
    const auto id = graph.add_node(std::move(n));

    const auto* node = graph.get_node(id);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_name, "test.op");
    EXPECT_EQ(node->dialect, "test");
    EXPECT_EQ(node->id, id);
}

TEST(ProvenanceGraphTest, GetNodeReturnsNullForInvalidId) {
    ProvenanceGraph graph;
    EXPECT_EQ(graph.get_node(999), nullptr);
}

TEST(ProvenanceGraphTest, AddEdgeRecordsTransformation) {
    ProvenanceGraph graph;
    OperationNode n1; n1.op_name = "parent";
    OperationNode n2; n2.op_name = "child";
    const auto id1 = graph.add_node(std::move(n1));
    const auto id2 = graph.add_node(std::move(n2));

    TransformationEdge edge;
    edge.from_id = id1;
    edge.to_id = id2;
    edge.kind = TransformationKind::Created;
    edge.pass_id = 1;
    graph.add_edge(edge);

    EXPECT_EQ(graph.edge_count(), 1u);
    const auto& edges = graph.edges();
    EXPECT_EQ(edges[0].from_id, id1);
    EXPECT_EQ(edges[0].to_id, id2);
    EXPECT_EQ(edges[0].kind, TransformationKind::Created);
}

TEST(ProvenanceGraphTest, RecordPassStoresPassInfo) {
    ProvenanceGraph graph;
    PassInfo pass; pass.name = "test-pass"; pass.description = "A test pass";
    graph.record_pass(pass);

    EXPECT_EQ(graph.pass_count(), 1u);
    EXPECT_EQ(graph.passes()[0].name, "test-pass");
    EXPECT_EQ(graph.passes()[0].description, "A test pass");
}

TEST(ProvenanceGraphTest, GetAncestorsReturnsParentChain) {
    ProvenanceGraph graph;
    OperationNode n1; n1.op_name = "root";
    OperationNode n2; n2.op_name = "child"; n2.parent_ids = {1};
    OperationNode n3; n3.op_name = "grandchild"; n3.parent_ids = {2};
    (void)graph.add_node(std::move(n1));
    (void)graph.add_node(std::move(n2));
    (void)graph.add_node(std::move(n3));

    auto ancestors = graph.get_ancestors(3);
    EXPECT_EQ(ancestors.size(), 2u);
    EXPECT_TRUE(std::find(ancestors.begin(), ancestors.end(), 1u) != ancestors.end());
    EXPECT_TRUE(std::find(ancestors.begin(), ancestors.end(), 2u) != ancestors.end());
}

TEST(ProvenanceGraphTest, GetDescendantsReturnsChildChain) {
    ProvenanceGraph graph;
    OperationNode n1; n1.op_name = "root"; n1.derived_ids = {2};
    OperationNode n2; n2.op_name = "child"; n2.derived_ids = {3};
    OperationNode n3; n3.op_name = "grandchild";
    (void)graph.add_node(std::move(n1));
    (void)graph.add_node(std::move(n2));
    (void)graph.add_node(std::move(n3));

    auto descendants = graph.get_descendants(1);
    EXPECT_EQ(descendants.size(), 2u);
    EXPECT_TRUE(std::find(descendants.begin(), descendants.end(), 2u) != descendants.end());
    EXPECT_TRUE(std::find(descendants.begin(), descendants.end(), 3u) != descendants.end());
}

TEST(ProvenanceGraphTest, GetRootsReturnsNodesWithoutParents) {
    ProvenanceGraph graph;
    OperationNode n1; n1.op_name = "root1";
    OperationNode n2; n2.op_name = "root2";
    OperationNode n3; n3.op_name = "child"; n3.parent_ids = {1};
    (void)graph.add_node(std::move(n1));
    (void)graph.add_node(std::move(n2));
    (void)graph.add_node(std::move(n3));

    auto roots = graph.get_roots();
    EXPECT_EQ(roots.size(), 2u);
    EXPECT_TRUE(std::find(roots.begin(), roots.end(), 1u) != roots.end());
    EXPECT_TRUE(std::find(roots.begin(), roots.end(), 2u) != roots.end());
}

TEST(ProvenanceGraphTest, ClearResetsGraph) {
    ProvenanceGraph graph;
    OperationNode n; n.op_name = "op";
    (void)graph.add_node(std::move(n));
    graph.add_edge({1, 2, 1, TransformationKind::Created, ""});
    graph.record_pass({"pass", "desc"});

    graph.clear();

    EXPECT_EQ(graph.node_count(), 0u);
    EXPECT_EQ(graph.edge_count(), 0u);
    EXPECT_EQ(graph.pass_count(), 0u);
}

TEST(ProvenanceTrackerTest, RecordOperationCreatesNode) {
    ProvenanceTracker tracker;
    SourceLocation loc; loc.file = "test.mlir"; loc.line = 10; loc.column = 5;

    auto id = tracker.record_operation("arith.addi", "arith", loc);

    EXPECT_EQ(id, 1u);
    EXPECT_EQ(tracker.graph().node_count(), 1u);
    const auto* node = tracker.graph().get_node(id);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->op_name, "arith.addi");
    EXPECT_EQ(node->dialect, "arith");
    EXPECT_EQ(node->location.file, "test.mlir");
    EXPECT_EQ(node->location.line, 10u);
    EXPECT_TRUE(node->is_original);
}

TEST(ProvenanceTrackerTest, RecordDerivedOperationLinksToParent) {
    ProvenanceTracker tracker;
    SourceLocation loc; loc.file = "test.mlir"; loc.line = 10;

    auto parent_id = tracker.record_operation("arith.constant", "arith", loc);
    auto child_id = tracker.record_derived_operation(parent_id, "arith.addi", "arith", loc,
                                                      TransformationKind::Created);

    EXPECT_EQ(child_id, 2u);
    EXPECT_EQ(tracker.graph().node_count(), 2u);

    const auto* parent = tracker.graph().get_node(parent_id);
    const auto* child = tracker.graph().get_node(child_id);
    ASSERT_NE(parent, nullptr);
    ASSERT_NE(child, nullptr);

    EXPECT_EQ(parent->derived_ids.size(), 1u);
    EXPECT_EQ(parent->derived_ids[0], child_id);
    EXPECT_EQ(child->parent_ids.size(), 1u);
    EXPECT_EQ(child->parent_ids[0], parent_id);
    EXPECT_FALSE(child->is_original);
    EXPECT_EQ(child->kind, TransformationKind::Created);
    EXPECT_EQ(tracker.graph().edge_count(), 1u);
}

TEST(ProvenanceTrackerTest, BeginEndPassManagesPassStack) {
    ProvenanceTracker tracker;

    auto pass_id = tracker.begin_pass("canonicalize", "Canonicalize operations");
    EXPECT_EQ(tracker.graph().pass_count(), 1u);
    EXPECT_EQ(pass_id, 1u);

    auto pass_id2 = tracker.begin_pass("cse", "Common subexpression elimination");
    EXPECT_EQ(tracker.graph().pass_count(), 2u);
    EXPECT_EQ(pass_id2, 2u);

    tracker.end_pass(pass_id);
    tracker.end_pass(pass_id2);
}

TEST(ProvenanceTrackerTest, OperationsRecordedDuringPassGetPassId) {
    ProvenanceTracker tracker;
    SourceLocation loc;

    auto pass_id = tracker.begin_pass("test-pass");
    auto op_id = tracker.record_operation("test.op", "test", loc);
    tracker.end_pass(pass_id);

    const auto* node = tracker.graph().get_node(op_id);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->pass_id, pass_id);
}

TEST(ProvenanceTrackerTest, FormatReportProducesOutput) {
    ProvenanceTracker tracker;
    SourceLocation loc; loc.file = "test.mlir"; loc.line = 1; loc.column = 1;

    (void)tracker.record_operation("func.func", "func", loc);
    (void)tracker.record_operation("arith.constant", "arith", loc);
    auto parent = tracker.record_operation("arith.addi", "arith", loc);
    (void)tracker.record_derived_operation(parent, "arith.muli", "arith", loc,
                                      TransformationKind::Canonicalized);

    auto pass_id = tracker.begin_pass("canonicalize");
    (void)tracker.record_operation("arith.subi", "arith", loc);
    tracker.end_pass(pass_id);

    const auto report = tracker.format_report("test.mlir");

    EXPECT_NE(report.find("Drishti Provenance Report"), std::string::npos);
    EXPECT_NE(report.find("test.mlir"), std::string::npos);
    EXPECT_NE(report.find("func.func"), std::string::npos);
    EXPECT_NE(report.find("arith.addi"), std::string::npos);
    EXPECT_NE(report.find("canonicalized"), std::string::npos);
    EXPECT_NE(report.find("canonicalize"), std::string::npos);
    EXPECT_NE(report.find("Nodes"), std::string::npos);
    EXPECT_NE(report.find("Edges"), std::string::npos);
    EXPECT_NE(report.find("Passes"), std::string::npos);
}

TEST(ProvenanceTrackerTest, DeterministicOutput) {
    ProvenanceTracker tracker1;
    ProvenanceTracker tracker2;
    SourceLocation loc; loc.file = "test.mlir"; loc.line = 1; loc.column = 1;

    for (auto& t : {&tracker1, &tracker2}) {
        (void)t->record_operation("func.func", "func", loc);
        (void)t->record_operation("arith.constant", "arith", loc);
        auto p = t->record_operation("arith.addi", "arith", loc);
        (void)t->record_derived_operation(p, "arith.muli", "arith", loc,
                                         TransformationKind::Canonicalized);
    }

    const auto report1 = tracker1.format_report("test.mlir");
    const auto report2 = tracker2.format_report("test.mlir");

    EXPECT_EQ(report1, report2);
}

TEST(ProvenanceTrackerTest, OperandAndResultIdsTracked) {
    ProvenanceTracker tracker;
    SourceLocation loc;

    auto c1 = tracker.record_operation("arith.constant", "arith", loc, {}, {100});
    auto c2 = tracker.record_operation("arith.constant", "arith", loc, {}, {200});
    auto add = tracker.record_operation("arith.addi", "arith", loc, {c1, c2}, {300});

    const auto* add_node = tracker.graph().get_node(add);
    ASSERT_NE(add_node, nullptr);
    EXPECT_EQ(add_node->operand_ids.size(), 2u);
    EXPECT_TRUE(std::find(add_node->operand_ids.begin(), add_node->operand_ids.end(), c1) != add_node->operand_ids.end());
    EXPECT_TRUE(std::find(add_node->operand_ids.begin(), add_node->operand_ids.end(), c2) != add_node->operand_ids.end());
    EXPECT_EQ(add_node->result_ids.size(), 1u);
    EXPECT_EQ(add_node->result_ids[0], 300u);
}

}  // namespace