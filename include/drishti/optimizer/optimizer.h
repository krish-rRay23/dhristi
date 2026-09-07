#ifndef DRISHTI_OPTIMIZER_OPTIMIZER_H
#define DRISHTI_OPTIMIZER_OPTIMIZER_H

#include <string>
#include <string_view>
#include <vector>

namespace drishti::optimizer {

enum class OptimizationLevel { O0, O1, O2, O3 };

struct PassOption {
    std::string name;
    bool enabled = true;
};

class OptimizerPipeline {
public:
    explicit OptimizerPipeline(OptimizationLevel level = OptimizationLevel::O2);

    void add_pass(PassOption opt) { passes_.push_back(std::move(opt)); }
    [[nodiscard]] const std::vector<PassOption>& passes() const noexcept { return passes_; }
    [[nodiscard]] OptimizationLevel level() const noexcept { return level_; }

    [[nodiscard]] static std::string_view level_name(OptimizationLevel lv) noexcept;

private:
    OptimizationLevel level_;
    std::vector<PassOption> passes_;
};

}  // namespace drishti::optimizer

#endif
