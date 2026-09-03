#ifndef DRISHTI_PROVENANCE_PROVENANCE_H
#define DRISHTI_PROVENANCE_PROVENANCE_H

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace drishti::provenance {

struct SourceLocation {
    std::string file;
    unsigned line = 0;
    unsigned column = 0;
};

enum class TransformationKind : uint8_t {
    Unknown = 0,
    Created,
    Cloned,
    Replaced,
    Erased,
    Moved,
    Inlined,
    Outlined,
    Fused,
    Split,
    Canonicalized,
    Simplified,
    Legalized,
    Lowered,
};

struct PassInfo {
    std::string name;
    std::string description;
    bool enabled_by_default = true;
};

struct OperationNode {
    std::uint64_t id = 0;
    std::uint64_t uid = 0;
    std::string op_name;
    std::string dialect;
    SourceLocation location;
    std::vector<std::uint64_t> operand_ids;
    std::vector<std::uint64_t> result_ids;
    std::vector<std::uint64_t> parent_ids;
    std::vector<std::uint64_t> derived_ids;
    std::uint64_t pass_id = 0;
    TransformationKind kind = TransformationKind::Unknown;
    bool is_original = false;
};

struct TransformationEdge {
    std::uint64_t from_id = 0;
    std::uint64_t to_id = 0;
    std::uint64_t pass_id = 0;
    TransformationKind kind = TransformationKind::Unknown;
    std::string details;
};

class ProvenanceGraph {
public:
    ProvenanceGraph() = default;
    ~ProvenanceGraph() = default;

    ProvenanceGraph(const ProvenanceGraph&) = delete;
    ProvenanceGraph& operator=(const ProvenanceGraph&) = delete;
    ProvenanceGraph(ProvenanceGraph&&) = default;
    ProvenanceGraph& operator=(ProvenanceGraph&&) = default;

    [[nodiscard]] std::uint64_t add_node(OperationNode node);
    [[nodiscard]] std::optional<std::uint64_t> find_node_by_original_id(std::uint64_t original_id) const;
    void add_edge(const TransformationEdge& edge);
    void record_pass(const PassInfo& pass);
    [[nodiscard]] std::uint64_t next_pass_id();

    [[nodiscard]] const OperationNode* get_node(std::uint64_t id) const noexcept;
    [[nodiscard]] const std::vector<OperationNode>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] const std::vector<TransformationEdge>& edges() const noexcept { return edges_; }
    [[nodiscard]] const std::vector<PassInfo>& passes() const noexcept { return passes_; }
    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t edge_count() const noexcept { return edges_.size(); }
    [[nodiscard]] std::size_t pass_count() const noexcept { return passes_.size(); }

    [[nodiscard]] std::vector<std::uint64_t> get_ancestors(std::uint64_t id) const;
    [[nodiscard]] std::vector<std::uint64_t> get_descendants(std::uint64_t id) const;
    [[nodiscard]] std::vector<std::uint64_t> get_roots() const;

    void clear();

private:
    std::vector<OperationNode> nodes_;
    std::vector<TransformationEdge> edges_;
    std::vector<PassInfo> passes_;
    std::map<std::uint64_t, std::size_t> id_to_index_;
    std::uint64_t next_id_ = 1;
    std::uint64_t next_pass_id_ = 1;
};

struct ProvenanceConfig {
    bool track_operands = true;
    bool track_results = true;
    bool track_location = true;
    bool track_derived_only = false;
};

struct PassTransformInfo {
    std::uint64_t pass_id = 0;
    std::vector<std::uint64_t> before_ops;
    std::vector<std::uint64_t> after_ops;
    std::vector<TransformationEdge> created_edges;
    std::vector<TransformationEdge> erased_edges;
    std::vector<TransformationEdge> preserved_edges;
};

class ProvenanceTracker {
public:
    explicit ProvenanceTracker(ProvenanceConfig config = {});
    ~ProvenanceTracker() = default;

    ProvenanceTracker(const ProvenanceTracker&) = delete;
    ProvenanceTracker& operator=(const ProvenanceTracker&) = delete;
    ProvenanceTracker(ProvenanceTracker&&) = default;
    ProvenanceTracker& operator=(ProvenanceTracker&&) = default;

    [[nodiscard]] std::uint64_t record_operation(const std::string& op_name,
                                                   const std::string& dialect,
                                                   const SourceLocation& loc,
                                                   const std::vector<std::uint64_t>& operand_ids = {},
                                                   const std::vector<std::uint64_t>& result_ids = {},
                                                   std::uint64_t uid = 0);

    [[nodiscard]] std::uint64_t record_derived_operation(std::uint64_t parent_id,
                                                           const std::string& op_name,
                                                           const std::string& dialect,
                                                           const SourceLocation& loc,
                                                           TransformationKind kind,
                                                           const std::vector<std::uint64_t>& operand_ids = {},
                                                           const std::vector<std::uint64_t>& result_ids = {});

    [[nodiscard]] std::uint64_t begin_pass(const std::string& name, const std::string& description = "");
    void end_pass(std::uint64_t pass_id);

    void record_pass_transform(PassTransformInfo info);

    [[nodiscard]] const ProvenanceGraph& graph() const noexcept { return graph_; }
    [[nodiscard]] ProvenanceGraph& graph() noexcept { return graph_; }

    [[nodiscard]] std::string format_report(std::string_view source_name = "<input>") const;

private:
    void compute_and_record_transformations(std::uint64_t pass_id);

    ProvenanceGraph graph_;
    ProvenanceConfig config_;
    std::vector<std::uint64_t> pass_stack_;
    std::uint64_t last_pass_id_ = 0;
    std::map<std::uint64_t, std::vector<std::uint64_t>> before_ops_by_pass_;
    std::map<std::uint64_t, std::vector<std::uint64_t>> after_ops_by_pass_;
    std::map<std::uint64_t, std::uint64_t> uid_to_node_id_;
};

}  // namespace drishti::provenance

namespace drishti::provenance {
[[nodiscard]] std::string format_report(const ProvenanceGraph& graph, std::string_view source_name = "<input>");
}

#endif