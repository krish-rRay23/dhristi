#ifndef DRISHTI_BACKENDS_BACKENDS_H
#define DRISHTI_BACKENDS_BACKENDS_H

#include <string>
#include <string_view>
#include <vector>

namespace drishti::backends {

enum class BackendKind {
    Host,
    LLVM,
    CUDA,
    ROCm,
    Vulkan,
};

struct BackendInfo {
    BackendKind kind;
    std::string name;
    bool available;
    std::string description;
};

class BackendRegistry {
public:
    BackendRegistry();

    [[nodiscard]] const std::vector<BackendInfo>& list() const noexcept { return backends_; }
    [[nodiscard]] static std::string_view kind_name(BackendKind k) noexcept;

private:
    std::vector<BackendInfo> backends_;
};

}  // namespace drishti::backends

#endif
