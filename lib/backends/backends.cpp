#include "drishti/backends/backends.h"

#include "drishti/backends/cuda/cuda_backend.h"
#include "drishti/backends/rocm/rocm_backend.h"
#include "drishti/core/config.h"

namespace drishti::backends {

BackendRegistry::BackendRegistry() {
    backends_.push_back({BackendKind::Host, "host", true, "Native CPU (reference)"});
    const bool cuda_available =
#if DRISHTI_HAVE_CUDA
        cuda::device_present();
#else
        false;
#endif

    const bool rocm_available = rocm::device_present();

#ifdef DRISHTI_HAS_LLVM
    backends_.push_back({BackendKind::LLVM, "llvm", true, "LLVM IR / JIT backend"});
#else
    backends_.push_back({BackendKind::LLVM, "llvm", false, "LLVM IR / JIT backend (not built)"});
#endif

    backends_.push_back({BackendKind::CUDA, "cuda", cuda_available,
                         cuda_available ? "NVIDIA CUDA backend (driver-loaded)"
                                        : "NVIDIA CUDA backend (no CUDA device/driver)"});
    backends_.push_back({BackendKind::ROCm, "rocm", rocm_available,
                         rocm_available ? "AMD ROCm backend (driver/HIP loaded)"
                                        : "AMD ROCm backend (no ROCm/HIP device/driver)"});
    backends_.push_back({BackendKind::Vulkan, "vulkan", false, "Vulkan backend (not enabled)"});
}

std::string_view BackendRegistry::kind_name(BackendKind k) noexcept {
    switch (k) {
        case BackendKind::Host:   return "host";
        case BackendKind::LLVM:   return "llvm";
        case BackendKind::CUDA:   return "cuda";
        case BackendKind::ROCm:   return "rocm";
        case BackendKind::Vulkan: return "vulkan";
    }
    return "unknown";
}

}
