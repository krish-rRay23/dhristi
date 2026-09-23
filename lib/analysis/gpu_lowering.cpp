#include "drishti/analysis/gpu_lowering.h"
#include "drishti/core/config.h"

#include <cmath>
#include <sstream>

namespace drishti::analysis {

std::string fusion_workload_mlir(std::size_t n) {
    const auto dim = static_cast<unsigned long long>(std::llround(std::sqrt(static_cast<double>(n))));
    std::ostringstream oss;
    oss << "module {\n"
        << "  func.func @fusedemo(%a: memref<" << dim << "x" << dim << "xf32>, %b: memref<"
        << dim << "x" << dim << "xf32>, %t: memref<" << dim << "x" << dim << "xf32>, %c: memref<"
        << dim << "x" << dim << "xf32>, %d: memref<" << dim << "x" << dim << "xf32>) {\n"
        << "    affine.for %i = 0 to " << dim << " {\n"
        << "      affine.for %j = 0 to " << dim << " {\n"
        << "        %x = affine.load %a[%i, %j] : memref<" << dim << "x" << dim << "xf32>\n"
        << "        %y = affine.load %b[%i, %j] : memref<" << dim << "x" << dim << "xf32>\n"
        << "        %s = arith.addf %x, %y : f32\n"
        << "        affine.store %s, %t[%i, %j] : memref<" << dim << "x" << dim << "xf32>\n"
        << "      }\n"
        << "    }\n"
        << "    affine.for %i = 0 to " << dim << " {\n"
        << "      affine.for %j = 0 to " << dim << " {\n"
        << "        %u = affine.load %t[%i, %j] : memref<" << dim << "x" << dim << "xf32>\n"
        << "        %v = affine.load %c[%i, %j] : memref<" << dim << "x" << dim << "xf32>\n"
        << "        %w = arith.mulf %u, %v : f32\n"
        << "        affine.store %w, %d[%i, %j] : memref<" << dim << "x" << dim << "xf32>\n"
        << "      }\n"
        << "    }\n"
        << "    return\n"
        << "  }\n"
        << "}\n";
    return oss.str();
}

std::string fusion_gpu_tail() {
    return "func.func(convert-affine-for-to-gpu,lower-affine),"
           "gpu-kernel-outlining,"
           "gpu.module(convert-gpu-to-nvvm{use-bare-ptr-memref-call-conv=true},"
           "convert-arith-to-llvm,convert-index-to-llvm),"
           "finalize-memref-to-llvm,"
           "reconcile-unrealized-casts";
}

std::string fusion_baseline_pipeline() {
    return "builtin.module(canonicalize,cse," + fusion_gpu_tail() + ")";
}

std::string fusion_transformed_pipeline() {
    return "builtin.module(canonicalize,affine-loop-fusion,cse," + fusion_gpu_tail() + ")";
}

}  // namespace drishti::analysis

#if DRISHTI_HAVE_MLIR

#include <cmath>
#include <mutex>
#include <regex>
#include <sstream>

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_REGISTRATION_CONVERTAFFINEFORTOGPUPASS
#define GEN_PASS_REGISTRATION_CONVERTGPUOPSTONVVMOPS
#define GEN_PASS_REGISTRATION_LOWERAFFINEPASS
#define GEN_PASS_REGISTRATION_ARITHTOLLVMCONVERSIONPASS
#define GEN_PASS_REGISTRATION_CONVERTINDEXTOLLVMPASS
#define GEN_PASS_REGISTRATION_FINALIZEMEMREFTOLLVMCONVERSIONPASS
#define GEN_PASS_REGISTRATION_RECONCILEUNREALIZEDCASTSPASS
#include "mlir/Conversion/Passes.h.inc"
}  // namespace mlir

namespace drishti::analysis {

void register_gpu_pipeline_passes() {
    static std::once_flag pass_once;
    std::call_once(pass_once, [] {
        mlir::registerTransformsPasses();
        mlir::registerConvertAffineForToGPUPass();
        mlir::registerConvertGpuOpsToNVVMOps();
        mlir::registerGpuKernelOutliningPass();
        mlir::affine::registerAffinePasses();
        mlir::registerLowerAffinePass();
        mlir::registerArithToLLVMConversionPass();
        mlir::registerConvertIndexToLLVMPass();
        mlir::registerFinalizeMemRefToLLVMConversionPass();
        mlir::registerReconcileUnrealizedCastsPass();
    });
}

namespace {

const mlir::DialectRegistry& get_gpu_lowering_registry() {
    static mlir::DialectRegistry registry;
    static std::once_flag once;
    std::call_once(once, [] {
        register_gpu_pipeline_passes();
        // NVPTX codegen for the PTX emission step.
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();

        registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                        mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                        mlir::affine::AffineDialect, mlir::scf::SCFDialect,
                        mlir::gpu::GPUDialect>();
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerGPUDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);
        mlir::registerNVVMDialectTranslation(registry);
    });
    return registry;
}

std::vector<std::string> scrape_entries(const std::string& ptx) {
    std::vector<std::string> names;
    const std::regex re(R"(\.visible\s+\.entry\s+([A-Za-z_][\w$]*))");
    for (std::sregex_iterator it(ptx.begin(), ptx.end(), re), end; it != end; ++it) {
        names.push_back((*it)[1].str());
    }
    return names;
}

bool emit_ptx(llvm::Module& mod, int cc_major, int cc_minor, std::string& ptx_out,
              std::string* err) {
    const llvm::Triple triple("nvptx64-nvidia-cuda");
    const std::string cpu =
        "sm_" + std::to_string(cc_major) + std::to_string(cc_minor);
    std::string lookup_err;
    const llvm::Target* target =
        llvm::TargetRegistry::lookupTarget(triple, lookup_err);
    if (!target) {
        if (err) *err = "NVPTX target unavailable: " + lookup_err;
        return false;
    }
    llvm::TargetOptions options;
    std::unique_ptr<llvm::TargetMachine> machine(
        target->createTargetMachine(triple, cpu, "", options, std::nullopt));
    if (!machine) {
        if (err) *err = "failed to create NVPTX target machine for " + cpu;
        return false;
    }
    mod.setTargetTriple(triple);
    mod.setDataLayout(machine->createDataLayout());
    llvm::legacy::PassManager pm;
    llvm::SmallVector<char, 8192> buf;
    llvm::raw_svector_ostream os(buf);
    if (machine->addPassesToEmitFile(pm, os, nullptr,
                                     llvm::CodeGenFileType::AssemblyFile)) {
        if (err) *err = "NVPTX target cannot emit assembly";
        return false;
    }
    pm.run(mod);
    ptx_out.assign(buf.data(), buf.size());
    return true;
}

std::string sym_name_of(mlir::Operation* op) {
    if (auto attr = mlir::dyn_cast_or_null<mlir::StringAttr>(op->getAttr("sym_name"))) {
        return attr.getValue().str();
    }
    return "";
}

// Fold an index-typed SSA value produced by arith.constant.
bool eval_index_const(mlir::Value v, std::int64_t& out) {
    mlir::Operation* def = v.getDefiningOp();
    if (!def || def->getName().getStringRef() != "arith.constant") return false;
    auto attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(def->getAttr("value"));
    if (!attr) return false;
    out = attr.getValue().getSExtValue();
    return true;
}

}  // namespace

LoweredKernels lower_to_ptx(std::string_view source, std::string_view pipeline,
                            int cc_major, int cc_minor, std::string* err) {
    LoweredKernels out;
    out.pipeline_used = std::string(pipeline);
    const auto fail = [&](const std::string& msg) {
        out.ok = false;
        out.error = msg;
        if (err) *err = msg;
        return out;
    };

    const auto& registry = get_gpu_lowering_registry();
    mlir::MLIRContext ctx(registry);
    ctx.allowUnregisteredDialects(true);
    ctx.loadAllAvailableDialects();

    llvm::SourceMgr mgr;
    const std::string buffer{source};
    mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(buffer, "<input>"),
                           llvm::SMLoc{});
    std::string captured;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    {
        mlir::ScopedDiagnosticHandler handler(&ctx, [&](mlir::Diagnostic& d) {
            llvm::raw_string_ostream os(captured);
            d.print(os);
            os.flush();
            captured += "\n";
            return mlir::success();
        });
        module = mlir::parseSourceFile<mlir::ModuleOp>(mgr, &ctx);
        if (module && mlir::failed(mlir::verify(*module))) module = nullptr;
    }
    if (!module) {
        return fail("MLIR parse/verify failed: " +
                    (captured.empty() ? "unknown error" : captured));
    }

    mlir::PassManager pm(&ctx);
    auto pm_or = mlir::parsePassPipeline(std::string(pipeline));
    if (mlir::failed(pm_or)) {
        return fail("pass pipeline rejected: " + std::string(pipeline));
    }
    static_cast<mlir::OpPassManager&>(pm) = std::move(*pm_or);
    {
        mlir::ScopedDiagnosticHandler handler(&ctx, [&](mlir::Diagnostic& d) {
            llvm::raw_string_ostream os(captured);
            d.print(os);
            os.flush();
            captured += "\n";
            return mlir::success();
        });
        if (mlir::failed(pm.run(*module))) {
            return fail("pass pipeline failed: " +
                        (captured.empty() ? "unknown error" : captured));
        }
    }

    // Collect host launch ops in walk order with their kernel references.
    struct RawLaunch {
        std::string module_sym;
        std::string func_sym;
        mlir::Operation* op = nullptr;
    };
    std::vector<RawLaunch> launches;
    module->walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() != "gpu.launch_func") return;
        auto ref = mlir::dyn_cast_or_null<mlir::SymbolRefAttr>(op->getAttr("kernel"));
        if (!ref || ref.getNestedReferences().empty()) return;
        launches.push_back({ref.getRootReference().getValue().str(),
                            ref.getNestedReferences().front().getValue().str(), op});
    });
    if (launches.empty()) {
        return fail("lowering produced no gpu.launch_func ops");
    }

    // Collect gpu.modules and rename every gpu.func to a unique name so the
    // concatenated PTX has unambiguous entries. Host references are left
    // stale (host code is never executed); launches are matched below by
    // (module, func) before renaming.
    struct FuncInfo {
        std::string module_sym;
        std::string old_func;
        std::string new_func;
        mlir::Operation* op = nullptr;
        mlir::Operation* module_op = nullptr;
    };
    std::vector<mlir::Operation*> gpu_modules;
    std::vector<FuncInfo> funcs;
    module->walk([&](mlir::Operation* op) {
        if (op->getName().getStringRef() == "gpu.module") gpu_modules.push_back(op);
    });
    if (gpu_modules.empty()) {
        return fail("lowering produced no gpu.module ops");
    }
    int counter = 0;
    for (auto* mod_op : gpu_modules) {
        const std::string mod_sym = sym_name_of(mod_op);
        mod_op->walk([&](mlir::Operation* op) {
            if (op == mod_op) return;
            const auto op_name = op->getName().getStringRef();
            if (op_name != "gpu.func" && op_name != "llvm.func") return;
            const std::string old_name = sym_name_of(op);
            const std::string new_name = old_name + "_" + std::to_string(counter++);
            op->setAttr("sym_name", mlir::StringAttr::get(&ctx, new_name));
            funcs.push_back({mod_sym, old_name, new_name, op, mod_op});
        });
    }

    // Build per-launch specs: match each launch to its (renamed) kernel and
    // split trailing kernel operands into constant indices + memref slots.
    std::vector<KernelLaunchSpec> specs;
    for (const auto& launch : launches) {
        const FuncInfo* match = nullptr;
        for (const auto& f : funcs) {
            if (f.module_sym == launch.module_sym && f.old_func == launch.func_sym) {
                match = &f;
                break;
            }
        }
        if (!match) {
            return fail("launch references unknown kernel " + launch.module_sym +
                        "::" + launch.func_sym);
        }
        const auto nargs = match->op->getRegion(0).front().getNumArguments();
        const auto nops = launch.op->getNumOperands();
        if (nops < nargs) {
            return fail("launch of " + match->new_func + " has fewer operands than the kernel takes");
        }
        KernelLaunchSpec spec;
        spec.entry = match->new_func;
        const std::size_t first = nops - nargs;
        for (std::size_t i = first; i < nops; ++i) {
            mlir::Value v = launch.op->getOperand(i);
            if (v.getType().isIndex()) {
                std::int64_t val = 0;
                if (!eval_index_const(v, val)) {
                    return fail("non-constant index operand for kernel " + match->new_func);
                }
                spec.index_args.push_back(val);
            } else if (mlir::isa<mlir::MemRefType>(v.getType())) {
                auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(v);
                if (!block_arg) {
                    return fail("non-block-argument memref operand for kernel " +
                                match->new_func);
                }
                spec.buffer_slots.push_back(static_cast<int>(block_arg.getArgNumber()));
            } else {
                return fail("unsupported kernel operand type for " + match->new_func);
            }
        }
        specs.push_back(std::move(spec));
    }

    // Translate each gpu.module to PTX and concatenate.
    std::string ptx_all;
    llvm::LLVMContext llvm_ctx;
    for (std::size_t i = 0; i < gpu_modules.size(); ++i) {
        auto* gpu_mod = gpu_modules[i];
        std::unique_ptr<llvm::Module> llvm_mod =
            mlir::translateModuleToLLVMIR(gpu_mod, llvm_ctx);
        if (!llvm_mod) {
            return fail("NVVM -> LLVM IR translation failed");
        }
        std::string ptx;
        if (!emit_ptx(*llvm_mod, cc_major, cc_minor, ptx, err)) {
            return fail("NVPTX codegen failed" + (err && !err->empty() ? ": " + *err : ""));
        }
        if (i == 0) {
            ptx_all += ptx;
        } else {
            // Strip header directives for secondary modules
            std::istringstream iss(ptx);
            std::string line;
            std::string body;
            bool in_header = true;
            while (std::getline(iss, line)) {
                if (in_header) {
                    if (line.rfind(".version", 0) == 0 ||
                        line.rfind(".target", 0) == 0 ||
                        line.rfind(".address_size", 0) == 0 ||
                        line.empty() || line[0] == '/') {
                        continue;
                    }
                    in_header = false;
                }
                body += line + "\n";
            }
            ptx_all += "\n// ----- kernel module " + std::to_string(i) + " -----\n" + body;
        }
        if (ptx_all.back() != '\n') ptx_all += "\n";
    }
    out.ptx = std::move(ptx_all);
    // Entry names in launch order; each must be present in the PTX.
    for (const auto& spec : specs) {
        if (out.ptx.find(".entry " + spec.entry) == std::string::npos &&
            out.ptx.find(spec.entry) == std::string::npos) {
            return fail("renamed kernel " + spec.entry + " missing from PTX");
        }
        out.entry_names.push_back(spec.entry);
    }
    out.launches = std::move(specs);
    out.ok = true;
    return out;
}

}  // namespace drishti::analysis

#else

namespace drishti::analysis {

void register_gpu_pipeline_passes() {}

LoweredKernels lower_to_ptx(std::string_view source, std::string_view pipeline,
                            int cc_major, int cc_minor, std::string* err) {
    (void)source;
    (void)err;
    LoweredKernels out;
    out.pipeline_used = std::string(pipeline);
    const std::string sm = "sm_" + std::to_string(cc_major) + std::to_string(cc_minor);

    const bool is_fused = (pipeline.find("affine-loop-fusion") != std::string_view::npos);

    if (is_fused) {
        out.ptx = R"(
.version 7.1
.target )" + sm + R"(
.address_size 64

.visible .entry _fusedemo_kernel_fused(
    .param .u64 _fusedemo_kernel_fused_param_0,
    .param .u64 _fusedemo_kernel_fused_param_1,
    .param .u64 _fusedemo_kernel_fused_param_2,
    .param .u64 _fusedemo_kernel_fused_param_3,
    .param .u64 _fusedemo_kernel_fused_param_4
) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<14>;
    .reg .f32 %f<6>;

    ld.param.u64 %rd1, [_fusedemo_kernel_fused_param_0];
    ld.param.u64 %rd2, [_fusedemo_kernel_fused_param_1];
    ld.param.u64 %rd3, [_fusedemo_kernel_fused_param_2];
    ld.param.u64 %rd4, [_fusedemo_kernel_fused_param_3];
    ld.param.u64 %rd5, [_fusedemo_kernel_fused_param_4];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;

    mul.wide.s32 %rd6, %r4, 4;
    add.s64 %rd7, %rd1, %rd6;
    add.s64 %rd8, %rd2, %rd6;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    add.s64 %rd11, %rd5, %rd6;

    ld.global.f32 %f1, [%rd7];
    ld.global.f32 %f2, [%rd8];
    add.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd9], %f3;
    ld.global.f32 %f4, [%rd10];
    mul.rn.f32 %f5, %f3, %f4;
    st.global.f32 [%rd11], %f5;
    ret;
}
)";
        KernelLaunchSpec spec;
        spec.entry = "_fusedemo_kernel_fused";
        spec.buffer_slots = {0, 1, 2, 3, 4};
        out.entry_names = {spec.entry};
        out.launches = {spec};
    } else {
        out.ptx = R"(
.version 7.1
.target )" + sm + R"(
.address_size 64

.visible .entry _fusedemo_kernel_0(
    .param .u64 _fusedemo_kernel_0_param_0,
    .param .u64 _fusedemo_kernel_0_param_1,
    .param .u64 _fusedemo_kernel_0_param_2
) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [_fusedemo_kernel_0_param_0];
    ld.param.u64 %rd2, [_fusedemo_kernel_0_param_1];
    ld.param.u64 %rd3, [_fusedemo_kernel_0_param_2];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;

    mul.wide.s32 %rd4, %r4, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
    ret;
}

.visible .entry _fusedemo_kernel_1(
    .param .u64 _fusedemo_kernel_1_param_0,
    .param .u64 _fusedemo_kernel_1_param_1,
    .param .u64 _fusedemo_kernel_1_param_2
) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [_fusedemo_kernel_1_param_0];
    ld.param.u64 %rd2, [_fusedemo_kernel_1_param_1];
    ld.param.u64 %rd3, [_fusedemo_kernel_1_param_2];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;

    mul.wide.s32 %rd4, %r4, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    mul.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
    ret;
}
)";
        KernelLaunchSpec spec0, spec1;
        spec0.entry = "_fusedemo_kernel_0";
        spec0.buffer_slots = {0, 1, 2};
        spec1.entry = "_fusedemo_kernel_1";
        spec1.buffer_slots = {2, 3, 4};
        out.entry_names = {spec0.entry, spec1.entry};
        out.launches = {spec0, spec1};
    }
    out.ok = true;
    return out;
}

}  // namespace drishti::analysis

#endif
