#ifndef DRISHTI_ANALYSIS_ANALYSIS_H
#define DRISHTI_ANALYSIS_ANALYSIS_H

#include <cstdint>
#include <string>
#include <string_view>

namespace drishti::analysis {

struct PassInfo {
    std::string name;
    std::string description;
    bool enabled_by_default = true;
};

class AnalysisPass {
public:
    virtual ~AnalysisPass() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual std::string_view description() const noexcept = 0;
};

enum class OpLoc : uint8_t { Unknown = 0, FileLineCol, Name, CallSite };

}  // namespace drishti::analysis

#endif
