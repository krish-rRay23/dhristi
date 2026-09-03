#include "drishti/provenance/provenance.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace drishti::provenance {

std::uint64_t ProvenanceGraph::add_node(OperationNode node) {
    const std::uint64_t id = next_id_++;
    node.id = id;
    const std::size_t index = nodes_.size();
    nodes_.push_back(std::move(node));
    id_to_index_[id] = index;
    return id;
}

std::optional<std::uint64_t> ProvenanceGraph::find_node_by_original_id(std::uint64_t original_id) const {
    auto it = id_to_index_.find(original_id);
    if (it != id_to_index_.end()) {
        return nodes_[it->second].id;
    }
    return std::nullopt;
}

void ProvenanceGraph::add_edge(const TransformationEdge& edge) {
    edges_.push_back(edge);
}

void ProvenanceGraph::record_pass(const PassInfo& pass) {
    passes_.push_back(pass);
}

const OperationNode* ProvenanceGraph::get_node(std::uint64_t id) const noexcept {
    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end()) {
        return &nodes_[it->second];
    }
    return nullptr;
}

std::vector<std::uint64_t> ProvenanceGraph::get_ancestors(std::uint64_t id) const {
    std::vector<std::uint64_t> result;
    const OperationNode* node = get_node(id);
    if (!node) return result;

    std::vector<std::uint64_t> stack = node->parent_ids;
    while (!stack.empty()) {
        const std::uint64_t current = stack.back();
        stack.pop_back();
        if (std::find(result.begin(), result.end(), current) == result.end()) {
            result.push_back(current);
            const OperationNode* parent = get_node(current);
            if (parent) {
                stack.insert(stack.end(), parent->parent_ids.begin(), parent->parent_ids.end());
            }
        }
    }
    return result;
}

std::vector<std::uint64_t> ProvenanceGraph::get_descendants(std::uint64_t id) const {
    std::vector<std::uint64_t> result;
    const OperationNode* node = get_node(id);
    if (!node) return result;

    std::vector<std::uint64_t> stack = node->derived_ids;
    while (!stack.empty()) {
        const std::uint64_t current = stack.back();
        stack.pop_back();
        if (std::find(result.begin(), result.end(), current) == result.end()) {
            result.push_back(current);
            const OperationNode* child = get_node(current);
            if (child) {
                stack.insert(stack.end(), child->derived_ids.begin(), child->derived_ids.end());
            }
        }
    }
    return result;
}

std::vector<std::uint64_t> ProvenanceGraph::get_roots() const {
    std::vector<std::uint64_t> roots;
    for (const auto& node : nodes_) {
        if (node.parent_ids.empty()) {
            roots.push_back(node.id);
        }
    }
    return roots;
}

void ProvenanceGraph::clear() {
    nodes_.clear();
    edges_.clear();
    passes_.clear();
    id_to_index_.clear();
    next_id_ = 1;
    next_pass_id_ = 1;
}

std::uint64_t ProvenanceGraph::next_pass_id() {
    return next_pass_id_++;
}

ProvenanceTracker::ProvenanceTracker(ProvenanceConfig config)
    : config_(std::move(config)) {}

std::uint64_t ProvenanceTracker::record_operation(const std::string& op_name,
                                                   const std::string& dialect,
                                                   const SourceLocation& loc,
                                                   const std::vector<std::uint64_t>& operand_ids,
                                                   const std::vector<std::uint64_t>& result_ids,
                                                   std::uint64_t uid) {
    OperationNode node;
    node.op_name = op_name;
    node.dialect = dialect;
    node.location = loc;
    node.operand_ids = operand_ids;
    node.result_ids = result_ids;
    node.uid = uid;
    node.is_original = true;
    node.pass_id = pass_stack_.empty() ? 0 : pass_stack_.back();

    const std::uint64_t id = graph_.add_node(std::move(node));

    if (uid != 0) {
        uid_to_node_id_[uid] = id;
    }

    return id;
}

std::uint64_t ProvenanceTracker::record_derived_operation(std::uint64_t parent_id,
                                                           const std::string& op_name,
                                                           const std::string& dialect,
                                                           const SourceLocation& loc,
                                                           TransformationKind kind,
                                                           const std::vector<std::uint64_t>& operand_ids,
                                                           const std::vector<std::uint64_t>& result_ids) {
    OperationNode node;
    node.op_name = op_name;
    node.dialect = dialect;
    node.location = loc;
    node.operand_ids = operand_ids;
    node.result_ids = result_ids;
    node.parent_ids.push_back(parent_id);
    node.kind = kind;
    node.is_original = false;
    node.pass_id = pass_stack_.empty() ? 0 : pass_stack_.back();

    const std::uint64_t id = graph_.add_node(std::move(node));

    OperationNode* parent = const_cast<OperationNode*>(graph_.get_node(parent_id));
    if (parent) {
        parent->derived_ids.push_back(id);
    }

    TransformationEdge edge;
    edge.from_id = parent_id;
    edge.to_id = id;
    edge.pass_id = node.pass_id;
    edge.kind = kind;
    graph_.add_edge(edge);

    return id;
}

std::uint64_t ProvenanceTracker::begin_pass(const std::string& name, const std::string& description) {
    PassInfo pass;
    pass.name = name;
    pass.description = description;
    pass.enabled_by_default = true;

    const std::uint64_t pass_id = graph_.next_pass_id();
    pass_stack_.push_back(pass_id);

    // Snapshot operation IDs before the pass begins
    std::vector<std::uint64_t> before_ops;
    for (const auto& node : graph_.nodes()) {
        before_ops.push_back(node.id);
    }
    before_ops_by_pass_[pass_id] = before_ops;

    graph_.record_pass(std::move(pass));
    return pass_id;
}

void ProvenanceTracker::end_pass(std::uint64_t pass_id) {
    // Erase from pass stack first
    auto it = std::find(pass_stack_.begin(), pass_stack_.end(), pass_id);
    if (it != pass_stack_.end()) {
        pass_stack_.erase(it);
    }

    // Snapshot operation IDs after the pass ends
    std::vector<std::uint64_t> after_ops;
    for (const auto& node : graph_.nodes()) {
        after_ops.push_back(node.id);
    }
    after_ops_by_pass_[pass_id] = after_ops;

    // Compute and record transformations
    compute_and_record_transformations(pass_id);
}

void ProvenanceTracker::compute_and_record_transformations(std::uint64_t pass_id) {
    const auto before_it = before_ops_by_pass_.find(pass_id);
    const auto after_it = after_ops_by_pass_.find(pass_id);
    if (before_it == before_ops_by_pass_.end() || after_it == after_ops_by_pass_.end()) {
        return;
    }

    const auto& before_ops = before_it->second;
    const auto& after_ops = after_it->second;

    // Build lookup sets
    std::unordered_set<std::uint64_t> before_set(before_ops.begin(), before_ops.end());
    std::unordered_set<std::uint64_t> after_set(after_ops.begin(), after_ops.end());

    // Find erased operations (in before, not in after)
    std::vector<std::uint64_t> erased_ops;
    for (auto id : before_ops) {
        if (after_set.find(id) == after_set.end()) {
            erased_ops.push_back(id);
        }
    }

    // Find created operations (in after, not in before)
    std::vector<std::uint64_t> created_ops;
    for (auto id : after_ops) {
        if (before_set.find(id) == before_set.end()) {
            created_ops.push_back(id);
        }
    }

    // Find preserved operations (in both before and after)
    std::vector<std::uint64_t> preserved_ops;
    for (auto id : before_ops) {
        if (after_set.find(id) != after_set.end()) {
            preserved_ops.push_back(id);
        }
    }

    // Create erased edges: from erased op to a sentinel, recording the erasure
    for (auto erased_id : erased_ops) {
        // Find the node to get details
        const OperationNode* node = graph_.get_node(erased_id);
        std::string details = "erased";
        if (node) {
            details = node->op_name + " (" + node->dialect + ") erased";
        }
        TransformationEdge edge;
        edge.from_id = erased_id;
        edge.to_id = 0;
        edge.pass_id = pass_id;
        edge.kind = TransformationKind::Erased;
        edge.details = details;
        graph_.add_edge(edge);
    }

    // Create created edges: from erased parent to new op, or from first preserved parent
    for (auto created_id : created_ops) {
        // Try to find a parent among erased ops with matching op_name
        const OperationNode* new_node = graph_.get_node(created_id);
        std::string new_op_name = new_node ? new_node->op_name : "unknown";
        std::string new_dialect = new_node ? new_node->dialect : "unknown";

        // Look for an erased op with the same op_name to serve as parent
        std::uint64_t parent_id = 0;
        TransformationKind parent_kind = TransformationKind::Unknown;

        for (auto erased_id : erased_ops) {
            const OperationNode* erased_node = graph_.get_node(erased_id);
            if (erased_node && erased_node->op_name == new_op_name) {
                parent_id = erased_id;
                parent_kind = TransformationKind::Created;
                break;
            }
        }

        // If no matching erased op, try preserved ops as potential parents
        // (the pass may have transformed rather than erased+created)
        if (parent_id == 0 && !preserved_ops.empty()) {
            parent_id = preserved_ops[0];
            parent_kind = TransformationKind::Canonicalized;
        }

        TransformationEdge edge;
        edge.from_id = parent_id;
        edge.to_id = created_id;
        edge.pass_id = pass_id;
        edge.kind = parent_kind;
        if (parent_id != 0) {
            const OperationNode* parent_node = graph_.get_node(parent_id);
            if (parent_node) {
                edge.details = std::string("created from ") + parent_node->op_name + " (" + parent_node->dialect + ")";
            } else {
                edge.details = "created";
            }
        } else {
            edge.details = "created";
        }
        graph_.add_edge(edge);
    }

    // Record preserved edges: from old op to itself (same ID, just preserved)
    // No new edges needed for preserved; their pass_id is already set during recording
}

void ProvenanceTracker::record_pass_transform(PassTransformInfo info) {
    for (const auto& edge : info.created_edges) {
        graph_.add_edge(edge);
    }
    for (const auto& edge : info.erased_edges) {
        graph_.add_edge(edge);
    }
    for (const auto& edge : info.preserved_edges) {
        graph_.add_edge(edge);
    }
}

std::string ProvenanceTracker::format_report(std::string_view source_name) const {
    std::ostringstream o;
    o << "Drishti Provenance Report\n";
    o << "  Source : " << source_name << "\n";
    o << "  Nodes  : " << graph_.node_count() << "\n";
    o << "  Edges  : " << graph_.edge_count() << "\n";
    o << "  Passes : " << graph_.pass_count() << "\n\n";

    o << "--- Passes ---\n";
    if (graph_.passes().empty()) {
        o << "  (no passes recorded)\n";
    } else {
        for (std::size_t i = 0; i < graph_.passes().size(); ++i) {
            const auto& p = graph_.passes()[i];
            o << "  " << std::setw(2) << (i + 1) << ". " << p.name;
            if (!p.description.empty()) {
                o << " - " << p.description;
            }
            o << "\n";
        }
    }
    o << "\n";

    o << "--- Operations (Topological) ---\n";
    const auto roots = graph_.get_roots();
    if (roots.empty()) {
        o << "  (no operations)\n";
    } else {
        std::vector<std::uint64_t> visited;
        std::vector<std::uint64_t> stack = roots;

        while (!stack.empty()) {
            const std::uint64_t current = stack.back();
            stack.pop_back();

            if (std::find(visited.begin(), visited.end(), current) != visited.end()) {
                continue;
            }
            visited.push_back(current);

            const OperationNode* node = graph_.get_node(current);
            if (!node) continue;

            o << "  " << std::setw(4) << current << ": " << node->op_name;
            if (!node->dialect.empty() && node->dialect != "builtin") {
                o << " (" << node->dialect << ")";
            }
            if (!node->location.file.empty()) {
                o << " @" << node->location.file << ":" << node->location.line << ":" << node->location.column;
            }
            if (!node->is_original) {
                o << " [derived: ";
                switch (node->kind) {
                    case TransformationKind::Created: o << "created"; break;
                    case TransformationKind::Cloned: o << "cloned"; break;
                    case TransformationKind::Replaced: o << "replaced"; break;
                    case TransformationKind::Erased: o << "erased"; break;
                    case TransformationKind::Moved: o << "moved"; break;
                    case TransformationKind::Inlined: o << "inlined"; break;
                    case TransformationKind::Outlined: o << "outlined"; break;
                    case TransformationKind::Fused: o << "fused"; break;
                    case TransformationKind::Split: o << "split"; break;
                    case TransformationKind::Canonicalized: o << "canonicalized"; break;
                    case TransformationKind::Simplified: o << "simplified"; break;
                    case TransformationKind::Legalized: o << "legalized"; break;
                    case TransformationKind::Lowered: o << "lowered"; break;
                    default: o << "unknown"; break;
                }
                o << "]";
            }
            if (!node->parent_ids.empty()) {
                o << " <- parents:";
                for (const auto pid : node->parent_ids) {
                    o << " " << pid;
                }
            }
            if (!node->derived_ids.empty()) {
                o << " -> children:";
                for (const auto did : node->derived_ids) {
                    o << " " << did;
                }
            }
            o << "\n";

            for (auto it = node->derived_ids.rbegin(); it != node->derived_ids.rend(); ++it) {
                stack.push_back(*it);
            }
        }
    }
    o << "\n";

    o << "--- Transformation Edges ---\n";
    if (graph_.edges().empty()) {
        o << "  (no edges)\n";
    } else {
        for (const auto& edge : graph_.edges()) {
            o << "  " << edge.from_id << " -> " << edge.to_id;
            o << " [pass=" << edge.pass_id << "]";
            switch (edge.kind) {
                case TransformationKind::Created: o << " created"; break;
                case TransformationKind::Cloned: o << " cloned"; break;
                case TransformationKind::Replaced: o << " replaced"; break;
                case TransformationKind::Erased: o << " erased"; break;
                case TransformationKind::Moved: o << " moved"; break;
                case TransformationKind::Inlined: o << " inlined"; break;
                case TransformationKind::Outlined: o << " outlined"; break;
                case TransformationKind::Fused: o << " fused"; break;
                case TransformationKind::Split: o << " split"; break;
                case TransformationKind::Canonicalized: o << " canonicalized"; break;
                case TransformationKind::Simplified: o << "simplified"; break;
                case TransformationKind::Legalized: o << "legalized"; break;
                case TransformationKind::Lowered: o << "lowered"; break;
                default: o << " unknown"; break;
            }
            if (!edge.details.empty()) {
                o << " (" << edge.details << ")";
            }
            o << "\n";
        }
    }

    return o.str();
}

std::string format_report(const ProvenanceGraph& graph, std::string_view source_name) {
    std::ostringstream o;
    o << "Drishti Provenance Report\n";
    o << "  Source : " << source_name << "\n";
    o << "  Nodes  : " << graph.node_count() << "\n";
    o << "  Edges  : " << graph.edge_count() << "\n";
    o << "  Passes : " << graph.pass_count() << "\n\n";

    o << "--- Passes ---\n";
    if (graph.passes().empty()) {
        o << "  (no passes recorded)\n";
    } else {
        for (std::size_t i = 0; i < graph.passes().size(); ++i) {
            const auto& p = graph.passes()[i];
            o << "  " << std::setw(2) << (i + 1) << ". " << p.name;
            if (!p.description.empty()) {
                o << " - " << p.description;
            }
            o << "\n";
        }
    }
    o << "\n";

    o << "--- Operations (Topological) ---\n";
    const auto roots = graph.get_roots();
    if (roots.empty()) {
        o << "  (no operations)\n";
    } else {
        std::vector<std::uint64_t> visited;
        std::vector<std::uint64_t> stack = roots;

        while (!stack.empty()) {
            const std::uint64_t current = stack.back();
            stack.pop_back();

            if (std::find(visited.begin(), visited.end(), current) != visited.end()) {
                continue;
            }
            visited.push_back(current);

            const OperationNode* node = graph.get_node(current);
            if (!node) continue;

            o << "  " << std::setw(4) << current << ": " << node->op_name;
            if (!node->dialect.empty() && node->dialect != "builtin") {
                o << " (" << node->dialect << ")";
            }
            if (!node->location.file.empty()) {
                o << " @" << node->location.file << ":" << node->location.line << ":" << node->location.column;
            }
            if (!node->is_original) {
                o << " [derived: ";
                switch (node->kind) {
                    case TransformationKind::Created: o << "created"; break;
                    case TransformationKind::Cloned: o << "cloned"; break;
                    case TransformationKind::Replaced: o << "replaced"; break;
                    case TransformationKind::Erased: o << "erased"; break;
                    case TransformationKind::Moved: o << "moved"; break;
                    case TransformationKind::Inlined: o << "inlined"; break;
                    case TransformationKind::Outlined: o << "outlined"; break;
                    case TransformationKind::Fused: o << "fused"; break;
                    case TransformationKind::Split: o << "split"; break;
                    case TransformationKind::Canonicalized: o << "canonicalized"; break;
                    case TransformationKind::Simplified: o << "simplified"; break;
                    case TransformationKind::Legalized: o << "legalized"; break;
                    case TransformationKind::Lowered: o << "lowered"; break;
                    default: o << "unknown"; break;
                }
                o << "]";
            }
            if (!node->parent_ids.empty()) {
                o << " <- parents:";
                for (const auto pid : node->parent_ids) {
                    o << " " << pid;
                }
            }
            if (!node->derived_ids.empty()) {
                o << " -> children:";
                for (const auto did : node->derived_ids) {
                    o << " " << did;
                }
            }
            o << "\n";

            for (auto it = node->derived_ids.rbegin(); it != node->derived_ids.rend(); ++it) {
                stack.push_back(*it);
            }
        }
    }
    o << "\n";

    o << "--- Transformation Edges ---\n";
    if (graph.edges().empty()) {
        o << "  (no edges)\n";
    } else {
        for (const auto& edge : graph.edges()) {
            o << "  " << edge.from_id << " -> " << edge.to_id;
            o << " [pass=" << edge.pass_id << "]";
            switch (edge.kind) {
                case TransformationKind::Created: o << " created"; break;
                case TransformationKind::Cloned: o << " cloned"; break;
                case TransformationKind::Replaced: o << " replaced"; break;
                case TransformationKind::Erased: o << " erased"; break;
                case TransformationKind::Moved: o << " moved"; break;
                case TransformationKind::Inlined: o << " inlined"; break;
                case TransformationKind::Outlined: o << " outlined"; break;
                case TransformationKind::Fused: o << " fused"; break;
                case TransformationKind::Split: o << " split"; break;
                case TransformationKind::Canonicalized: o << " canonicalized"; break;
                case TransformationKind::Simplified: o << " simplified"; break;
                case TransformationKind::Legalized: o << " legalized"; break;
                case TransformationKind::Lowered: o << " lowered"; break;
                default: o << " unknown"; break;
            }
            if (!edge.details.empty()) {
                o << " (" << edge.details << ")";
            }
            o << "\n";
        }
    }

    return o.str();
}

}  // namespace drishti::provenance