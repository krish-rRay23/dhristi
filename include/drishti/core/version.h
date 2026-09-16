#ifndef DRISHTI_CORE_VERSION_H
#define DRISHTI_CORE_VERSION_H

#include <string>
#include <string_view>

#include "drishti/core/config.h"

namespace drishti::core {

struct Version {
    int major = DRISHTI_VERSION_MAJOR;
    int minor = DRISHTI_VERSION_MINOR;
    int patch = DRISHTI_VERSION_PATCH;

    [[nodiscard]] constexpr std::string_view str() const noexcept {
        return DRISHTI_VERSION_STRING;
    }
};

[[nodiscard]] constexpr Version version() noexcept { return {}; }

[[nodiscard]] std::string banner();

[[nodiscard]] constexpr bool have_llvm() noexcept { return DRISHTI_HAVE_LLVM ? true : false; }
[[nodiscard]] constexpr bool have_mlir() noexcept { return DRISHTI_HAVE_MLIR ? true : false; }
[[nodiscard]] constexpr bool have_cuda() noexcept { return DRISHTI_HAVE_CUDA ? true : false; }

}  // namespace drishti::core

#endif
