#include "drishti/optimizer/optimizer.h"

namespace drishti::optimizer {

OptimizerPipeline::OptimizerPipeline(OptimizationLevel level) : level_(level) {
    if (level_ != OptimizationLevel::O0) {
        passes_.push_back({"canonicalize", true});
    }
    if (level_ == OptimizationLevel::O2 || level_ == OptimizationLevel::O3) {
        passes_.push_back({"cse", true});
        passes_.push_back({"licm", true});
    }
    if (level_ == OptimizationLevel::O3) {
        passes_.push_back({"vectorize", true});
    }
}

std::string_view OptimizerPipeline::level_name(OptimizationLevel lv) noexcept {
    switch (lv) {
        case OptimizationLevel::O0: return "O0";
        case OptimizationLevel::O1: return "O1";
        case OptimizationLevel::O2: return "O2";
        case OptimizationLevel::O3: return "O3";
    }
    return "unknown";
}

}
