#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

static const char *kSrc = R"mlir(
module {
  func.func @f(%arg0: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %s = arith.addi %c2, %c3 : i32
    %a = arith.addi %arg0, %s : i32
    %b = arith.addi %arg0, %s : i32
    %r = arith.addi %a, %b : i32
    return %r : i32
  }
}
)mlir";

struct Instr : mlir::PassInstrumentation {
  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    llvm::errs() << "BEFORE pass=" << pass->getName()
                 << " arg=" << pass->getArgument()
                 << " target=" << op->getName().getStringRef() << "\n";
  }
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
    llvm::errs() << "AFTER  pass=" << pass->getArgument()
                 << " target=" << op->getName().getStringRef() << "\n";
  }
};

int main() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  ctx.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  ctx.disableMultithreading();
  mlir::registerTransformsPasses();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(kSrc, &ctx);
  if (!module) { llvm::errs() << "parse failed\n"; return 1; }

  // tag every op with a uid attribute
  mlir::Builder b(&ctx);
  uint64_t uid = 1;
  module->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    op->setAttr("drishti.uid", b.getI64IntegerAttr((int64_t)uid++));
  });

  llvm::errs() << "--- tagged IR ---\n";
  module->print(llvm::errs());
  llvm::errs() << "\n";

  auto pipeline = mlir::parsePassPipeline("builtin.module(canonicalize,cse)");
  if (mlir::failed(pipeline)) { llvm::errs() << "pipeline parse failed\n"; return 1; }

  mlir::PassManager pm(&ctx);
  pm.addInstrumentation(std::make_unique<Instr>());
  static_cast<mlir::OpPassManager &>(pm) = std::move(*pipeline);
  pm.addInstrumentation(std::make_unique<Instr>());

  if (mlir::failed(pm.run(*module))) { llvm::errs() << "run failed\n"; return 1; }

  llvm::errs() << "--- after IR (uids preserved?) ---\n";
  module->print(llvm::errs());
  llvm::errs() << "\n";
  return 0;
}
