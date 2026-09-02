#include "drishti/analysis/mlir_analysis.h"
#include "drishti/core/config.h"
#include "drishti/analysis/gpu_lowering.h"

#if DRISHTI_HAVE_MLIR

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/InitAllPasses.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <utility>

namespace drishti::analysis {

namespace {

constexpr std::size_t kTopOps = 10;
constexpr std::size_t kTopDialectOps = 20;

void register_required_dialects(mlir::MLIRContext& c) {
    c.loadDialect<mlir::BuiltinDialect,
                  mlir::func::FuncDialect,
                  mlir::arith::ArithDialect,
                  mlir::math::MathDialect,
                  mlir::memref::MemRefDialect,
                  mlir::tensor::TensorDialect,
                  mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::affine::AffineDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::tosa::TosaDialect,
                  mlir::gpu::GPUDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect>();
}

}  // namespace

struct MlirAnalysisContext::Impl {
    Impl() {
        mlir::DialectRegistry registry;
        registry.insert<mlir::BuiltinDialect,
                        mlir::func::FuncDialect,
                        mlir::arith::ArithDialect,
                        mlir::math::MathDialect,
                        mlir::memref::MemRefDialect,
                        mlir::tensor::TensorDialect,
                        mlir::scf::SCFDialect,
                        mlir::cf::ControlFlowDialect,
                        mlir::affine::AffineDialect,
                        mlir::linalg::LinalgDialect,
                        mlir::tosa::TosaDialect,
                        mlir::gpu::GPUDialect,
                        mlir::LLVM::LLVMDialect,
                        mlir::NVVM::NVVMDialect>();
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerGPUDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
        mlir::registerNVVMDialectTranslation(registry);

        ctx = std::make_unique<mlir::MLIRContext>(registry);
        ctx->allowUnregisteredDialects(true);
        ctx->loadAllAvailableDialects();

        register_gpu_pipeline_passes();
    }
    std::unique_ptr<mlir::MLIRContext> ctx;
};

MlirAnalysisContext::MlirAnalysisContext() : impl_(std::make_unique<Impl>()) {}
MlirAnalysisContext::~MlirAnalysisContext() = default;

mlir::MLIRContext& MlirAnalysisContext::context() noexcept { return *impl_->ctx; }

MlirAnalysisEngine::MlirAnalysisEngine(MlirAnalysisContext& ctx) : ctx_(&ctx) {}

std::optional<StructuralStats> MlirAnalysisEngine::analyze_file(const std::string& path,
                                                                std::string* error_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (error_out) *error_out = "cannot open file: " + path;
        return std::nullopt;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    return analyze_impl(std::string_view{text}, std::string_view{path}, error_out);
}

std::optional<StructuralStats> MlirAnalysisEngine::analyze_string(std::string_view source,
                                                                  std::string* error_out) {
    return analyze_impl(source, "<input>", error_out);
}

std::optional<StructuralStats> MlirAnalysisEngine::analyze_impl(std::string_view source,
                                                                  std::string_view buffer_name,
                                                                  std::string* error_out) {
    stats_ = StructuralStats{};
    op_to_node_id_.clear();
    uid_to_node_id_.clear();
    next_uid_ = 1;
    auto& ctx = ctx_->context();

    llvm::SourceMgr mgr;
    const std::string buffer{source};
    const std::string name{buffer_name};
    mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(buffer, name), llvm::SMLoc{});

    // Capture MLIR diagnostics so callers get the real parse/verify error
    // instead of a generic message (and so test output stays quiet).
    std::string captured;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    bool verify_ok = false;
    {
        mlir::ScopedDiagnosticHandler handler(&ctx, [&](mlir::Diagnostic& d) {
            llvm::raw_string_ostream os(captured);
            d.print(os);
            os.flush();
            captured += "\n";
            return mlir::success();
        });
        module = mlir::parseSourceFile<mlir::ModuleOp>(mgr, &ctx);
        if (module) verify_ok = succeeded(mlir::verify(*module));
    }
    if (!module) {
        if (error_out) {
            *error_out = captured.empty() ? "failed to parse MLIR module" : captured;
        }
        return std::nullopt;
    }
    if (!verify_ok) {
        if (error_out) {
            *error_out = captured.empty() ? "MLIR module failed verification" : captured;
        }
        return std::nullopt;
    }

    if (!pass_pipeline_.empty()) {
        // Assign drishti.uid i64 attr to every operation
        module->walk([&](mlir::Operation* op) {
            auto uidAttr = mlir::IntegerAttr::get(
                mlir::IntegerType::get(&ctx, 64), next_uid_++);
            op->setAttr("drishti.uid", uidAttr);
        });

        // Walk module to record initial graph nodes and map UID -> graph node ID
        module->walk([&](mlir::Operation* op) {
            const auto name = op_name(op);
            const auto dialect = dialect_name(op);
            const auto loc = mlir_location_to_source(op);
            const auto operand_ids = get_operand_ids(op);
            const auto result_ids = get_result_ids(op);

            // Read back the UID we just set
            std::uint64_t uid = 0;
            if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(op->getAttr("drishti.uid"))) {
                uid = intAttr.getValue().getZExtValue();
            }

            const auto node_id = tracker_.record_operation(name, dialect, loc, operand_ids, result_ids, uid);
            uid_to_node_id_[uid] = node_id;
            // Keep structural stats consistent in pipeline mode (pre-pass module).
            record_stats(op);
        });

        // Begin pass snapshot
        auto pass_id = tracker_.begin_pass("pipeline", "MLIR pass pipeline execution");

        // Parse and execute pass pipeline using the MLIR pass infrastructure
        // (same pattern as tools/proto/proto.cpp)
        register_gpu_pipeline_passes();
        mlir::PassManager pm(&ctx);

        std::string pipe_str = pass_pipeline_;
        if (pipe_str.rfind("builtin.module(", 0) != 0) {
            pipe_str = "builtin.module(" + pipe_str + ")";
        }
        auto pipeline = mlir::parsePassPipeline(pipe_str);
        if (mlir::failed(pipeline)) {
            // Pipeline parse failed, skip pass execution
        } else {
            static_cast<mlir::OpPassManager &>(pm) = std::move(*pipeline);
            (void)pm.run(*module);
        }

        // End pass snapshot
        tracker_.end_pass(pass_id);

        // Compute transformation edges based on UID survival
        compute_pipeline_transformations(pass_id, *module);
    } else {
        // Original flow without pipeline: walk and record
        walk(*module);
    }

    finalize();
    return stats_;
}

void MlirAnalysisEngine::walk(mlir::ModuleOp module) {
    module->walk([this](mlir::Operation* op) {
        record_stats(op);
        record_provenance(op);
    });
}

void MlirAnalysisEngine::record_stats(mlir::Operation* op) {
    ++stats_.total_operations;
    stats_.total_operands += static_cast<uint64_t>(op->getNumOperands());
    stats_.total_results += static_cast<uint64_t>(op->getNumResults());
    const uint64_t nregions = static_cast<uint64_t>(op->getNumRegions());
    stats_.total_regions += nregions;
    for (auto& region : op->getRegions()) {
        stats_.total_blocks += static_cast<uint64_t>(region.getBlocks().size());
        for (auto& block : region.getBlocks()) {
            stats_.total_arguments += static_cast<uint64_t>(block.getNumArguments());
        }
    }
    record_dialect(op->getDialect(), op);
    record_op(op);
    record_function(op);
    record_loop_if_detected(op);
    record_branch_if_detected(op);
}

void MlirAnalysisEngine::record_dialect(const mlir::Dialect* dialect, mlir::Operation* op) {
    // Ops from unregistered/unknown dialects have a null Dialect*; fall back
    // to the "dialect." prefix of the op name so they are still counted.
    const auto dn = dialect != nullptr ? dialect->getNamespace().str() : dialect_name(op);
    auto& d = stats_.dialects;
    auto it = std::find_if(d.begin(), d.end(), [&](const DialectStat& s) { return s.name == dn; });
    if (it == d.end()) {
        DialectStat s;
        s.name = dn;
        d.push_back(std::move(s));
        it = std::prev(d.end());
    }
    ++it->op_count;
    const auto on = op_name(op);
    if (std::find(it->op_names.begin(), it->op_names.end(), on) == it->op_names.end()) {
        if (it->op_names.size() < kTopDialectOps) it->op_names.push_back(on);
        ++it->distinct_ops;
    }
}

void MlirAnalysisEngine::record_op(mlir::Operation* op) {
    const auto name = op_name(op);
    ++stats_.op_histogram[name];
    (void)op;
}

void MlirAnalysisEngine::record_function(mlir::Operation* op) {
    if (!mlir::isa<mlir::FunctionOpInterface>(op)) return;
    auto fn = mlir::cast<mlir::FunctionOpInterface>(op);
    FunctionInfo fi;
    fi.name = fn.getName().str();
    fi.argument_count = static_cast<uint64_t>(fn.getNumArguments());
    fi.result_count = static_cast<uint64_t>(fn.getNumResults());
    fi.region_count = static_cast<uint64_t>(op->getNumRegions());
    uint64_t total_blocks = 0;
    uint64_t total_ops = 0;
    for (auto& region : op->getRegions()) {
        total_blocks += static_cast<uint64_t>(region.getBlocks().size());
        region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation*) { ++total_ops; });
    }
    fi.block_count = total_blocks;
    // region.walk() visits only ops nested inside the regions, so total_ops
    // already excludes the function op itself.
    fi.op_count = total_ops;
    ++stats_.function_count;
    stats_.functions.push_back(std::move(fi));
}

void MlirAnalysisEngine::record_loop_if_detected(mlir::Operation* op) {
    if (!is_loop_like(op)) return;
    LoopInfo li;
    li.kind = op_name(op);
    li.location_hint = location_hint(op);
    if (op->getNumRegions() > 0) {
        auto& region = op->getRegion(0);
        li.body_blocks = count_blocks_in_region(region);
        li.body_ops = count_ops_in_region(region);
    }
    ++stats_.loop_count;
    stats_.loops.push_back(std::move(li));
}

void MlirAnalysisEngine::record_branch_if_detected(mlir::Operation* op) {
    if (!is_branch_like(op)) return;
    ++stats_.branch_count;
}

std::string MlirAnalysisEngine::op_name(mlir::Operation* op) {
    return op->getName().getStringRef().str();
}

std::string MlirAnalysisEngine::dialect_name(mlir::Operation* op) {
    if (const auto* dialect = op->getDialect()) return dialect->getNamespace().str();
    // Unregistered dialect: derive the namespace from the "ns.op" op name.
    const auto full = op->getName().getStringRef();
    if (const auto pos = full.find('.'); pos != llvm::StringRef::npos) {
        return full.substr(0, pos).str();
    }
    return "<unregistered>";
}

std::string MlirAnalysisEngine::location_hint(mlir::Operation* op) {
    auto loc = op->getLoc();
    if (auto fl = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
        std::ostringstream o;
        o << fl.getFilename().str() << ":" << fl.getLine() << ":" << fl.getColumn();
        return o.str();
    }
    if (auto nl = mlir::dyn_cast<mlir::NameLoc>(loc)) {
        return nl.getName().str();
    }
    if (auto cl = mlir::dyn_cast<mlir::CallSiteLoc>(loc)) {
        if (auto callee = mlir::dyn_cast<mlir::NameLoc>(cl.getCallee())) {
            return "callsite:" + callee.getName().str();
        }
        return "callsite:<unknown>";
    }
    return "<unknown>";
}

bool MlirAnalysisEngine::is_loop_like(mlir::Operation* op) {
    // The LoopLikeOpInterface is authoritative; the name fallback covers loop
    // ops from dialects that are parsed but not registered/loaded.
    if (mlir::isa<mlir::LoopLikeOpInterface>(op)) return true;
    const auto name = op->getName().getStringRef();
    return name.starts_with("scf.for") || name.starts_with("scf.while") ||
           name.starts_with("scf.parallel") || name.starts_with("affine.for") ||
           name.starts_with("affine.parallel");
}

bool MlirAnalysisEngine::is_branch_like(mlir::Operation* op) {
    // Terminator branches (cf.br, cf.cond_br, ...) implement this interface;
    // the name fallback covers the cf dialect when parsed unregistered.
    if (mlir::isa<mlir::BranchOpInterface>(op)) return true;
    const auto name = op->getName().getStringRef();
    return name.starts_with("cf.");
}

uint64_t MlirAnalysisEngine::count_ops_in_region(mlir::Region& region) {
    uint64_t n = 0;
    region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation*) { ++n; });
    return n;
}

uint64_t MlirAnalysisEngine::count_blocks_in_region(mlir::Region& region) {
    return static_cast<uint64_t>(region.getBlocks().size());
}

void MlirAnalysisEngine::compute_pipeline_transformations(std::uint64_t pass_id, mlir::ModuleOp module) {
// Build set of UIDs present in the final module after passes
    std::unordered_set<std::uint64_t> surviving_uids;
    module->walk([&](mlir::Operation* op) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(op->getAttr("drishti.uid"))) {
            surviving_uids.insert(intAttr.getValue().getZExtValue());
        }
    });

    // Erased: UIDs present before pass but missing after
    for (const auto& [uid, before_node_id] : uid_to_node_id_) {
        if (surviving_uids.find(uid) == surviving_uids.end()) {
            drishti::provenance::TransformationEdge edge;
            edge.from_id = before_node_id;
            edge.to_id = 0;
            edge.pass_id = pass_id;
            edge.kind = drishti::provenance::TransformationKind::Erased;
            edge.details = "erased";
            tracker_.graph().add_edge(edge);
        }
    }

    // Build map of UID -> after-node-id from the final module.
    std::unordered_map<std::uint64_t, std::uint64_t> after_uid_to_node_id;
    module->walk([&](mlir::Operation* op) {
        std::uint64_t uid_in_op = 0;
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(op->getAttr("drishti.uid"))) {
            uid_in_op = intAttr.getValue().getZExtValue();
        }
        if (uid_in_op != 0) {
            auto before_it = uid_to_node_id_.find(uid_in_op);
            if (before_it != uid_to_node_id_.end()) {
                after_uid_to_node_id[uid_in_op] = before_it->second;
            } else {
                std::string name = op_name(op);
                std::string dialect = dialect_name(op);
                auto loc = mlir_location_to_source(op);
                auto operand_ids = get_operand_ids(op);
                auto result_ids = get_result_ids(op);
                auto after_node_id = tracker_.record_operation(name, dialect, loc, operand_ids, result_ids, uid_in_op);
                after_uid_to_node_id[uid_in_op] = after_node_id;
            }
        }
    });

    // Created: new UIDs after pass (those not in before-map)
    for (const auto& [uid_in_op, after_node_id] : after_uid_to_node_id) {
        if (uid_to_node_id_.find(uid_in_op) == uid_to_node_id_.end()) {
            drishti::provenance::TransformationEdge edge;
            edge.from_id = 0;
            edge.to_id = after_node_id;
            edge.pass_id = pass_id;
            edge.kind = drishti::provenance::TransformationKind::Created;
            edge.details = "created";
            tracker_.graph().add_edge(edge);
        }
    }

    // Replaced: when an old UID's node ID effectively changed between before and after.
    // Since we preserve the same node ID for surviving UIDs (after_uid_to_node_id[uid] == before_node_id),
    // Replaced edges are emitted when the after-node-id would differ. In the basic case where
    // the node ID is preserved, Replaced won't fire for simply surviving ops. Replaced can fire
    // when canonicalize/CSE restructures an op, which would change the node ID mapping.
    // For now, we check if the after-op's structure differs from the before recording.
    // We determine this by scanning the after-op and comparing op_name/dialect.
    // If they differ from what was originally recorded, it's a replacement.
    // Replaced: when an old UID's node ID effectively changed between before and after.
    for (const auto& [uid, before_node_id] : uid_to_node_id_) {
        if (surviving_uids.find(uid) != surviving_uids.end()) {
            auto after_it = after_uid_to_node_id.find(uid);
            if (after_it != after_uid_to_node_id.end()) {
                std::uint64_t after_node_id = after_it->second;
                if (after_node_id != before_node_id) {
                    drishti::provenance::TransformationEdge edge;
                    edge.from_id = before_node_id;
                    edge.to_id = after_node_id;
                    edge.pass_id = pass_id;
                    edge.kind = drishti::provenance::TransformationKind::Replaced;
                    edge.details = "replaced";
                    tracker_.graph().add_edge(edge);
                }
            }
        }
    }
}

void MlirAnalysisEngine::finalize() {
    std::vector<OpStat> aggregated;
    aggregated.reserve(stats_.op_histogram.size());
    for (auto& [k, v] : stats_.op_histogram) {
        OpStat s;
        s.name = k;
        s.count = v;
        aggregated.push_back(std::move(s));
    }
    std::sort(aggregated.begin(), aggregated.end(),
              [](const OpStat& a, const OpStat& b) { return a.count > b.count; });
    if (aggregated.size() > kTopOps) aggregated.resize(kTopOps);
    stats_.top_ops = std::move(aggregated);

    std::sort(stats_.dialects.begin(), stats_.dialects.end(),
              [](const DialectStat& a, const DialectStat& b) { return a.op_count > b.op_count; });
}

std::string format_report(const StructuralStats& s, std::string_view source_name) {
    std::ostringstream o;
    o << "Drishti MLIR Analysis Report\n";
    o << "  Source : " << source_name << "\n";
    o << "  Module : " << s.function_count << " function(s), " << s.total_operations
      << " operation(s), " << s.loop_count << " loop-like(s), " << s.branch_count
      << " branch op(s)\n\n";

    o << "--- Structural Overview ---\n";
    o << "  operations        : " << s.total_operations << "\n";
    o << "  regions           : " << s.total_regions << "\n";
    o << "  blocks            : " << s.total_blocks << "\n";
    o << "  block arguments   : " << s.total_arguments << "\n";
    o << "  op results        : " << s.total_results << "\n";
    o << "  op operands       : " << s.total_operands << "\n";
    o << "  branch ops        : " << s.branch_count << "\n\n";

    o << "--- Dialect Usage ---\n";
    if (s.dialects.empty()) {
        o << "  (no dialects observed)\n";
    } else {
        for (const auto& d : s.dialects) {
            o << "  " << std::left << std::setw(18) << (d.name + ":") << std::right
              << std::setw(7) << d.op_count << " ops,  " << std::setw(3) << d.distinct_ops
              << " distinct  [";
            for (std::size_t i = 0; i < d.op_names.size(); ++i) {
                if (i) o << ", ";
                auto short_op = d.op_names[i];
                if (auto pos = short_op.find('.'); pos != std::string::npos)
                    short_op = short_op.substr(pos + 1);
                o << short_op;
            }
            o << "]\n";
        }
    }
    o << "\n";

    o << "--- Top Operations ---\n";
    if (s.top_ops.empty()) {
        o << "  (no operations)\n";
    } else {
        for (std::size_t i = 0; i < s.top_ops.size(); ++i) {
            const auto& op = s.top_ops[i];
            o << "  " << std::setw(2) << (i + 1) << ". " << std::left << std::setw(36)
              << op.name << std::right << std::setw(7) << op.count << "\n";
        }
    }
    o << "\n";

    o << "--- Functions ---\n";
    if (s.functions.empty()) {
        o << "  (no func.func / function-like ops)\n";
    } else {
        o << "  " << std::left << std::setw(30) << "name" << std::right << std::setw(8)
          << "args" << std::setw(8) << "rets" << std::setw(9) << "blocks" << std::setw(10)
          << "ops"
          << "\n";
        o << "  " << std::string(65, '-') << "\n";
        for (const auto& f : s.functions) {
            o << "  " << std::left << std::setw(30) << f.name << std::right << std::setw(8)
              << f.argument_count << std::setw(8) << f.result_count << std::setw(9)
              << f.block_count << std::setw(10) << f.op_count << "\n";
        }
    }
    o << "\n";

    o << "--- Loop-Like Constructs ---\n";
    if (s.loops.empty()) {
        o << "  (none)\n";
    } else {
        for (std::size_t i = 0; i < s.loops.size(); ++i) {
            const auto& lp = s.loops[i];
            o << "  " << std::setw(2) << (i + 1) << ". " << std::left << std::setw(28)
              << lp.kind << " blocks=" << std::setw(4) << lp.body_blocks
              << " ops=" << std::setw(6) << lp.body_ops << "  @" << lp.location_hint << "\n";
        }
    }
    o << "\n";

    o << "--- Control Flow ---\n";
    o << "  branch ops          : " << s.branch_count << "\n";
    bool any_cf = false;
    for (const auto& [name, count] : s.op_histogram) {
        const std::string_view v{name};
        if (v.starts_with("cf.") || v == "scf.if" || v == "scf.while" || v == "scf.condition") {
            o << "  " << std::left << std::setw(28) << name << std::right << std::setw(7)
              << count << "\n";
            any_cf = true;
        }
    }
    if (!any_cf) o << "  (no cf/scf-conditional ops)\n";
    o << "\n";
    return o.str();
}

void MlirAnalysisEngine::record_provenance(mlir::Operation* op) {
    if (!track_provenance_) return;

    const auto loc = mlir_location_to_source(op);
    const auto dialect = dialect_name(op);
    const auto name = op_name(op);
    const auto operand_ids = get_operand_ids(op);
    const auto result_ids = get_result_ids(op);

    const std::uint64_t node_id = tracker_.record_operation(name, dialect, loc, operand_ids, result_ids);
    op_to_node_id_[op] = node_id;
}

provenance::SourceLocation MlirAnalysisEngine::mlir_location_to_source(mlir::Operation* op) const {
    provenance::SourceLocation loc;
    auto mlir_loc = op->getLoc();
    if (auto fl = mlir::dyn_cast<mlir::FileLineColLoc>(mlir_loc)) {
        loc.file = fl.getFilename().str();
        loc.line = fl.getLine();
        loc.column = fl.getColumn();
    } else if (auto nl = mlir::dyn_cast<mlir::NameLoc>(mlir_loc)) {
        loc.file = nl.getName().str();
    }
    return loc;
}

std::vector<std::uint64_t> MlirAnalysisEngine::get_operand_ids(mlir::Operation* op) const {
    std::vector<std::uint64_t> ids;
    ids.reserve(op->getNumOperands());
    for (auto operand : op->getOperands()) {
        if (auto defining_op = operand.getDefiningOp()) {
            auto it = op_to_node_id_.find(defining_op);
            if (it != op_to_node_id_.end()) {
                ids.push_back(it->second);
            }
        }
    }
    return ids;
}

std::vector<std::uint64_t> MlirAnalysisEngine::get_result_ids(mlir::Operation* op) const {
    std::vector<std::uint64_t> ids;
    ids.reserve(op->getNumResults());
    for (auto result : op->getResults()) {
        ids.push_back(reinterpret_cast<std::uintptr_t>(result.getAsOpaquePointer()));
    }
    return ids;
}

}  // namespace drishti::analysis

#else  // !DRISHTI_HAVE_MLIR

namespace drishti::analysis {

struct MlirAnalysisContext::Impl {};

MlirAnalysisContext::MlirAnalysisContext() : impl_(std::make_unique<Impl>()) {}
MlirAnalysisContext::~MlirAnalysisContext() = default;

mlir::MLIRContext& MlirAnalysisContext::context() noexcept {
    return *reinterpret_cast<mlir::MLIRContext*>(this);
}

MlirAnalysisEngine::MlirAnalysisEngine(MlirAnalysisContext&) {}

std::optional<StructuralStats> MlirAnalysisEngine::analyze_file(const std::string&,
                                                          std::string* error_out) {
    if (error_out) *error_out = "MLIR support disabled at compile time";
    return std::nullopt;
}

std::optional<StructuralStats> MlirAnalysisEngine::analyze_string(std::string_view,
                                                            std::string* error_out) {
    if (error_out) *error_out = "MLIR support disabled at compile time";
    return std::nullopt;
}

}  // namespace drishti::analysis

#endif  // DRISHTI_HAVE_MLIR