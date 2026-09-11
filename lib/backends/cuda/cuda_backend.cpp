// NVIDIA CUDA backend (Phase 4): driver-API profiling with embedded PTX.
//
// Design notes:
//  - No CUDA toolkit required at build or run time. The CUDA driver library
//    (nvcuda.dll / libcuda.so.1) is loaded dynamically and every driver entry
//    point is resolved via GetProcAddress/dlsym.
//  - The representative kernel (single-precision vecadd) is embedded as PTX
//    text and JIT-compiled on the device by the driver at module load.
//  - Timing uses CUDA events (device side) plus host timers for transfers.
//  - CUPTI is probed opportunistically and reported; HW counters are NOT
//    required for the base metrics.

#include "drishti/backends/cuda/cuda_backend.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace drishti::backends::cuda {
namespace {

// ---- Minimal CUDA driver-API types (no cuda.h needed) ----
struct CtxSt;
struct ModSt;
struct FnSt;
struct EvSt;
using Ctx = CtxSt*;
using Mod = ModSt*;
using Fn = FnSt*;
using Ev = EvSt*;
using Dev = int;
using DevPtr = unsigned long long;
constexpr int kSuccess = 0;

// CUdevice_attribute values used here (stable ABI).
constexpr int kAttrMaxThreadsPerBlock = 1;
constexpr int kAttrClockRate = 13;            // kHz
constexpr int kAttrMultiProcessorCount = 16;
constexpr int kAttrMemBusWidth = 34;          // bits
constexpr int kAttrMemClockRate = 36;         // kHz
constexpr int kAttrL2CacheSize = 38;          // bytes
constexpr int kAttrMaxThreadsPerSM = 39;
constexpr int kAttrCCMajor = 75;
constexpr int kAttrCCMinor = 76;

// On Windows, nvcuda.dll uses the Microsoft x64 ABI.
// MinGW clang++ defaults to the SystemV AMD64 ABI for function pointer calls,
// which mismatches the Windows ABI for functions with >4 integer/pointer args
// (cuLaunchKernel has 11).  __attribute__((ms_abi)) forces the correct ABI.
#if defined(_WIN32) && defined(__GNUC__)
#  define CUDA_API __attribute__((ms_abi))
#else
#  define CUDA_API
#endif

struct DriverApi {
    void* lib = nullptr;
    int (CUDA_API *init)(unsigned) = nullptr;
    int (CUDA_API *driverVersion)(int*) = nullptr;
    int (CUDA_API *devCount)(int*) = nullptr;
    int (CUDA_API *devGet)(Dev*, int) = nullptr;
    int (CUDA_API *devName)(char*, int, Dev) = nullptr;
    int (CUDA_API *devAttr)(int*, int, Dev) = nullptr;
    int (CUDA_API *devTotalMem)(std::size_t*, Dev) = nullptr;
    int (CUDA_API *ctxCreate)(Ctx*, unsigned, Dev) = nullptr;
    int (CUDA_API *ctxDestroy)(Ctx) = nullptr;
    int (CUDA_API *ctxSync)() = nullptr;
    int (CUDA_API *modLoadData)(Mod*, const void*, unsigned, void*, void**) = nullptr;
    int (CUDA_API *modGetFunc)(Fn*, Mod, const char*) = nullptr;
    int (CUDA_API *modUnload)(Mod) = nullptr;
    int (CUDA_API *memAlloc)(DevPtr*, std::size_t) = nullptr;
    int (CUDA_API *memFree)(DevPtr) = nullptr;
    int (CUDA_API *cpyHtoD)(DevPtr, const void*, std::size_t) = nullptr;
    int (CUDA_API *cpyDtoH)(void*, DevPtr, std::size_t) = nullptr;
    int (CUDA_API *evCreate)(Ev*, unsigned) = nullptr;
    int (CUDA_API *evRecord)(Ev, void*) = nullptr;
    int (CUDA_API *evSync)(Ev) = nullptr;
    int (CUDA_API *evElapsed)(float*, Ev, Ev) = nullptr;
    int (CUDA_API *evDestroy)(Ev) = nullptr;
    int (CUDA_API *launch)(Fn, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                          unsigned, void*, void**, void**) = nullptr;
    int (CUDA_API *errString)(int, const char**) = nullptr;

    bool ok = false;
    std::string load_error;
};

void* open_driver() {
#if defined(_WIN32)
    return static_cast<void*>(::LoadLibraryA("nvcuda.dll"));
#else
    void* h = ::dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) h = ::dlopen("libcuda.so", RTLD_NOW);
    return h;
#endif
}

void* sym(void* h, const char* primary, const char* fallback = nullptr) {
#if defined(_WIN32)
    void* p = reinterpret_cast<void*>(::GetProcAddress(static_cast<HMODULE>(h), primary));
    if (!p && fallback) {
        p = reinterpret_cast<void*>(::GetProcAddress(static_cast<HMODULE>(h), fallback));
    }
    return p;
#else
    void* p = ::dlsym(h, primary);
    if (!p && fallback) p = ::dlsym(h, fallback);
    return p;
#endif
}

DriverApi& driver() {
    static DriverApi d;
    static std::once_flag once;
    std::call_once(once, [] {
        d.lib = open_driver();
        if (!d.lib) {
            d.load_error = "driver library not found (nvcuda.dll / libcuda.so.1)";
            return;
        }
#define LOAD(field, p, f) d.field = reinterpret_cast<decltype(d.field)>(sym(d.lib, p, f))
        LOAD(init, "cuInit", nullptr);
        LOAD(driverVersion, "cuDriverGetVersion", nullptr);
        LOAD(devCount, "cuDeviceGetCount", nullptr);
        LOAD(devGet, "cuDeviceGet", nullptr);
        LOAD(devName, "cuDeviceGetName", nullptr);
        LOAD(devAttr, "cuDeviceGetAttribute", nullptr);
        LOAD(devTotalMem, "cuDeviceTotalMem_v2", "cuDeviceTotalMem");
        LOAD(ctxCreate, "cuCtxCreate_v2", "cuCtxCreate");
        LOAD(ctxDestroy, "cuCtxDestroy_v2", "cuCtxDestroy");
        // Unversioned cuMemFree / cuEventDestroy are the CUDA 3.1 32-bit
        // compatibility exports.  The CUDA headers #define those names to the
        // _v2 64-bit Driver ABI; GetProcAddress must resolve _v2 explicitly.
        LOAD(ctxSync, "cuCtxSynchronize", "cuCtxSynchronize_v2");
        LOAD(modLoadData, "cuModuleLoadDataEx", nullptr);
        LOAD(modGetFunc, "cuModuleGetFunction", nullptr);
        LOAD(modUnload, "cuModuleUnload", nullptr);
        LOAD(memAlloc, "cuMemAlloc_v2", "cuMemAlloc");
        LOAD(memFree, "cuMemFree_v2", "cuMemFree");
        LOAD(cpyHtoD, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
        LOAD(cpyDtoH, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
        LOAD(evCreate, "cuEventCreate", nullptr);
        LOAD(evRecord, "cuEventRecord", nullptr);
        LOAD(evSync, "cuEventSynchronize", nullptr);
        LOAD(evElapsed, "cuEventElapsedTime", nullptr);
        LOAD(evDestroy, "cuEventDestroy_v2", "cuEventDestroy");
        LOAD(launch, "cuLaunchKernel", nullptr);
        LOAD(errString, "cuGetErrorString", nullptr);
#undef LOAD
        d.ok = d.init && d.devCount && d.devGet && d.devName && d.devAttr &&
               d.devTotalMem && d.ctxCreate && d.ctxDestroy && d.ctxSync &&
               d.modLoadData && d.modGetFunc && d.modUnload && d.memAlloc &&
               d.memFree && d.cpyHtoD && d.cpyDtoH && d.evCreate && d.evRecord &&
               d.evSync && d.evElapsed && d.evDestroy && d.launch;
        if (!d.ok) d.load_error = "driver library present but entry points missing";
        // errString/driverVersion are optional (used only for messages).
    });
    return d;
}

#if defined(DRISHTI_CUDA_TRACE)
#include <cstdio>
#define TR_BEGIN(name) \
    do {               \
        std::fprintf(stderr, "[trace] %s ...\n", name); \
        std::fflush(stderr);                            \
    } while (0)
#define TR_END(name, rc)                          \
    do {                                          \
        std::fprintf(stderr, "[trace] %s rc=%d\n", name, (int)(rc)); \
        std::fflush(stderr);                      \
    } while (0)
#else
#define TR_BEGIN(name) ((void)0)
#define TR_END(name, rc) ((void)0)
#endif

std::string err_text(DriverApi& d, int code) {
    if (d.errString) {
        const char* s = nullptr;
        if (d.errString(code, &s) == kSuccess && s) return s;
    }
    return "CUDA driver error " + std::to_string(code);
}

// Representative kernel: single-precision vector add, PTX for SM 8.x
// (JIT-compiled by the driver for the actual device).
constexpr const char* kVecaddPtx = R"ptx(
.version 7.8
.target sm_86
.address_size 64

.visible .entry vecadd(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
)
{
    .reg .pred %p;
    .reg .u32 %mytid, %myntid, %myctaid, %idx, %n;
    .reg .u64 %a, %b, %c, %off, %tmp;
    .reg .f32 %x, %y, %z;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %c, [c_ptr];
    ld.param.u32 %n, [n];

    mov.u32 %mytid, %tid.x;
    mov.u32 %myntid, %ntid.x;
    mov.u32 %myctaid, %ctaid.x;
    mad.lo.u32 %idx, %myctaid, %myntid, %mytid;
    setp.ge.u32 %p, %idx, %n;
    @%p ret;

    mul.wide.u32 %off, %idx, 4;
    add.u64 %tmp, %a, %off;
    ld.global.f32 %x, [%tmp];
    add.u64 %tmp, %b, %off;
    ld.global.f32 %y, [%tmp];
    add.f32 %z, %x, %y;
    add.u64 %tmp, %c, %off;
    st.global.f32 [%tmp], %z;
    ret;
}
)ptx";

struct Guard {
    DriverApi& d;
    Ctx ctx = nullptr;
    Mod mod = nullptr;
    Ev start = nullptr;
    Ev stop = nullptr;
    DevPtr a = 0, b = 0, c = 0;
    ~Guard() {
        if (start) d.evDestroy(start);
        if (stop) d.evDestroy(stop);
        if (a) d.memFree(a);
        if (b) d.memFree(b);
        if (c) d.memFree(c);
        if (mod) d.modUnload(mod);
        if (ctx) d.ctxDestroy(ctx);
    }
};

}  // namespace

bool device_present() noexcept {
    DriverApi& d = driver();
    if (!d.ok) return false;
    if (d.init(0) != kSuccess) return false;
    int n = 0;
    if (d.devCount(&n) != kSuccess) return false;
    return n > 0;
}

bool query_device(profiling::GpuDeviceModel& out) {
    out = profiling::GpuDeviceModel{};
    out.backend = "cuda";
    DriverApi& d = driver();
    if (!d.ok) {
        out.error = d.load_error;
        return false;
    }
    int rc = d.init(0);
    if (rc != kSuccess) {
        out.error = "cuInit failed: " + err_text(d, rc);
        return false;
    }
    int n = 0;
    rc = d.devCount(&n);
    if (rc != kSuccess || n <= 0) {
        out.error = n <= 0 ? "no CUDA-capable device found" : "cuDeviceGetCount failed: " + err_text(d, rc);
        return false;
    }
    Dev dev = 0;
    rc = d.devGet(&dev, 0);
    if (rc != kSuccess) {
        out.error = "cuDeviceGet failed: " + err_text(d, rc);
        return false;
    }
    char name[256] = {};
    auto attr = [&](int id, int fallback = 0) {
        int v = fallback;
        return (d.devAttr(&v, id, dev) == kSuccess) ? v : fallback;
    };
    if (d.devName(name, sizeof(name), dev) != kSuccess) {
        out.error = "cuDeviceGetName failed";
        return false;
    }
    std::size_t bytes = 0;
    if (d.devTotalMem(&bytes, dev) != kSuccess) {
        out.error = "cuDeviceTotalMem failed";
        return false;
    }
    int drv = 0;
    std::string drv_str = "unknown";
    if (d.driverVersion && d.driverVersion(&drv) == kSuccess && drv > 0) {
        drv_str = std::to_string(drv / 1000) + "." + std::to_string((drv % 100) / 10);
    }
    out.present = true;
    out.name = name;
    out.compute_major = attr(kAttrCCMajor);
    out.compute_minor = attr(kAttrCCMinor);
    out.sm_count = attr(kAttrMultiProcessorCount);
    out.max_threads_per_block = attr(kAttrMaxThreadsPerBlock, 1024);
    out.clock_mhz = attr(kAttrClockRate) / 1000;
    out.mem_clock_mhz = attr(kAttrMemClockRate) / 1000;
    out.mem_bus_width_bits = attr(kAttrMemBusWidth);
    if (out.mem_clock_mhz > 0 && out.mem_bus_width_bits > 0) {
        // DDR: 2 transfers per clock.
        out.mem_theoretical_gbps =
            2.0 * static_cast<double>(out.mem_clock_mhz) * 1e6 *
            static_cast<double>(out.mem_bus_width_bits) / 8.0 / 1e9;
    }
    out.total_mem_bytes = static_cast<unsigned long long>(bytes);
    out.driver_version = drv_str;
    (void)drv;
    return true;
}

void cupti_status(bool& available, std::string& version) {
    available = false;
    version = "no CUDA toolkit / CUPTI library found";
#if defined(_WIN32)
    constexpr const char* kCandidates[] = {
        "cupti64_131.dll", "cupti64_130.dll", "cupti64_124.dll",
        "cupti64_123.dll", "cupti64_120.dll", "cupti.dll",
    };
    for (const char* cand : kCandidates) {
        HMODULE h = ::LoadLibraryA(cand);
        if (!h) continue;
        using CuptiVer = int (*)(std::uint32_t*);
        FARPROC proc = ::GetProcAddress(h, "cuptiGetVersion");
        CuptiVer fn = nullptr;
        if (proc) std::memcpy(&fn, &proc, sizeof(fn));
        if (fn) {
            std::uint32_t v = 0;
            if (fn(&v) == 0 && v > 0) {
                available = true;
                version = std::to_string(v / 1000) + "." + std::to_string(v % 1000);
                ::FreeLibrary(h);
                return;
            }
        }
        ::FreeLibrary(h);
    }
#else
    constexpr const char* kCandidates[] = {
        "libcupti.so.13", "libcupti.so.12", "libcupti.so",
    };
    for (const char* cand : kCandidates) {
        void* h = ::dlopen(cand, RTLD_NOW);
        if (!h) continue;
        using CuptiVer = int (*)(std::uint32_t*);
        auto fn = reinterpret_cast<CuptiVer>(::dlsym(h, "cuptiGetVersion"));
        if (fn) {
            std::uint32_t v = 0;
            if (fn(&v) == 0 && v > 0) {
                available = true;
                version = std::to_string(v / 1000) + "." + std::to_string(v % 1000);
                ::dlclose(h);
                return;
            }
        }
        ::dlclose(h);
    }
#endif
}

static bool run_vecadd_impl(const VecaddConfig& cfg, bool recopy,
                              profiling::GpuProfileMetrics& out, std::string* err) {
    out = profiling::GpuProfileMetrics{};
    out.kernel = "vecadd";
    out.timing_source = "cuda-events";
    const auto fail = [&](const std::string& msg) {
        out.ok = false;
        out.error = msg;
        if (err) *err = msg;
        return false;
    };

    if (cfg.num_elements == 0 || cfg.num_elements > (1u << 28))
        return fail("num_elements out of range (1 .. 268435456)");
    if (cfg.block_size <= 0 || cfg.block_size > 1024)
        return fail("block_size out of range (1 .. 1024)");
    if (cfg.repeats <= 0 || cfg.repeats > 1000) return fail("repeats out of range (1 .. 1000)");

    DriverApi& d = driver();
    if (!d.ok) return fail("CUDA unavailable: " + d.load_error);

    profiling::GpuDeviceModel dev;
    if (!query_device(dev)) return fail("CUDA unavailable: " + dev.error);
    out.device = dev;

    const std::size_t n = cfg.num_elements;
    const std::size_t words = n * sizeof(float);
    const int block = cfg.block_size;
    const int grid = static_cast<int>((n + static_cast<std::size_t>(block) - 1) /
                                      static_cast<std::size_t>(block));
    out.num_elements = n;
    out.block_size = block;
    out.grid_size = grid;
    out.repeats = cfg.repeats;

    const auto wall0 = std::chrono::steady_clock::now();
    Guard g{d};

    int rc = kSuccess;
    TR_BEGIN("cuInit");
    rc = d.init(0);
    TR_END("cuInit", rc);
    if (rc != kSuccess) return fail("cuInit failed: " + err_text(d, rc));
    Dev handle = 0;
    TR_BEGIN("cuDeviceGet");
    rc = d.devGet(&handle, 0);
    TR_END("cuDeviceGet", rc);
    if (rc != kSuccess) return fail("cuDeviceGet failed: " + err_text(d, rc));
    TR_BEGIN("cuCtxCreate");
    rc = d.ctxCreate(&g.ctx, 0, handle);
    TR_END("cuCtxCreate", rc);
    if (rc != kSuccess) return fail("cuCtxCreate failed: " + err_text(d, rc));

    // Capture the driver JIT log so PTX errors are diagnosable.
    // CUjit_option ids: ERROR_LOG_BUFFER = 5, ERROR_LOG_BUFFER_SIZE_BYTES = 6.
    char jit_log[8192] = {};
    unsigned jit_log_size = static_cast<unsigned>(sizeof(jit_log));
    int jit_options[2] = {5, 6};  // CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    void* jit_values[2] = {jit_log, &jit_log_size};
    TR_BEGIN("cuModuleLoadDataEx");
    rc = d.modLoadData(&g.mod, kVecaddPtx, 2, jit_options, jit_values);
    TR_END("cuModuleLoadDataEx", rc);
    if (rc != kSuccess) {
        std::string msg = "PTX JIT failed: " + err_text(d, rc);
        if (jit_log[0] != '\0') {
            msg += "\nJIT log:\n";
            msg += jit_log;
        }
        return fail(msg);
    }
    Fn func = nullptr;
    TR_BEGIN("cuModuleGetFunction");
    rc = d.modGetFunc(&func, g.mod, "vecadd");
    TR_END("cuModuleGetFunction", rc);
    if (rc != kSuccess) return fail("cuModuleGetFunction(vecadd) failed: " + err_text(d, rc));

    TR_BEGIN("cuMemAlloc");
    const int arc = d.memAlloc(&g.a, words);
    const int brc = d.memAlloc(&g.b, words);
    const int crc = d.memAlloc(&g.c, words);
    TR_END("cuMemAlloc", arc == 0 && brc == 0 && crc == 0 ? 0 : -1);
    if (arc != kSuccess || brc != kSuccess || crc != kSuccess)
        return fail("cuMemAlloc failed (need 3 x " + std::to_string(words) + " bytes)");

    std::vector<float> ha(n, 1.0f), hb(n, 2.0f), hc(n, 0.0f);

    TR_BEGIN("cuEventCreate");
    rc = d.evCreate(&g.start, 0);
    if (rc == kSuccess) rc = d.evCreate(&g.stop, 0);
    TR_END("cuEventCreate", rc);
    if (rc != kSuccess) return fail("cuEventCreate failed: " + err_text(d, rc));

    std::uint32_t n32 = static_cast<std::uint32_t>(n);
    void* args[] = {&g.a, &g.b, &g.c, &n32};
    int launch_count = 0;
    const auto launch_once = [&]() {
        TR_BEGIN("cuLaunchKernel");
        const int lrc = d.launch(func, static_cast<unsigned>(grid), 1, 1,
                                 static_cast<unsigned>(block), 1, 1, 0, nullptr,
                                 args, nullptr);
        ++launch_count;
        TR_END("cuLaunchKernel", lrc);
        return lrc;
    };
    const auto copy_in = [&]() {
        const auto c0 = std::chrono::steady_clock::now();
        int crc = d.cpyHtoD(g.a, ha.data(), words);
        if (crc == kSuccess) crc = d.cpyHtoD(g.b, hb.data(), words);
        const auto c1 = std::chrono::steady_clock::now();
        return std::make_pair(crc, std::chrono::duration<double, std::milli>(c1 - c0).count());
    };
    const auto copy_out = [&]() {
        const auto c0 = std::chrono::steady_clock::now();
        const int crc = d.cpyDtoH(hc.data(), g.c, words);
        const auto c1 = std::chrono::steady_clock::now();
        return std::make_pair(crc, std::chrono::duration<double, std::milli>(c1 - c0).count());
    };
    const auto launch_timed = [&](double& ms_out) {
        int lrc = d.evRecord(g.start, nullptr);
        if (lrc != kSuccess) return lrc;
        lrc = launch_once();
        if (lrc != kSuccess) return lrc;
        lrc = d.evRecord(g.stop, nullptr);
        if (lrc != kSuccess) return lrc;
        lrc = d.evSync(g.stop);
        if (lrc != kSuccess) return lrc;
        float ms = 0.0f;
        lrc = d.evElapsed(&ms, g.start, g.stop);
        ms_out = ms;
        return lrc;
    };
    const auto check_vals = [&]() {
        for (std::size_t i = 0; i < n; ++i) {
            if (hc[i] != 3.0f) return false;
        }
        return true;
    };

    // Untimed warmup (covers WDDM first-launch cost) in both modes so the
    // timed regions compare fairly. Synchronize via events, keeping a single
    // device-sync path.
    TR_BEGIN("warmup");
    {
        auto [wcrc, wms] = copy_in();
        (void)wms;
        rc = wcrc;
    }
    if (rc != kSuccess) return fail("HtoD failed: " + err_text(d, rc));
    {
        double wms = 0.0;
        rc = launch_timed(wms);
    }
    TR_END("warmup", rc);
    if (rc != kSuccess) return fail("warmup failed: " + err_text(d, rc));

    double sum_ms = 0.0;
    double min_ms = 0.0;
    double h2d_ms = 0.0;
    double d2h_ms = 0.0;
    bool ok_vals = true;
    for (int i = 0; i < cfg.repeats; ++i) {
        TR_BEGIN("iter");
        if (recopy) {
            auto [crc, cms] = copy_in();
            rc = crc;
            h2d_ms += cms;
            if (rc != kSuccess) return fail("HtoD failed: " + err_text(d, rc));
        } else if (i == 0) {
            const auto t0 = std::chrono::steady_clock::now();
            TR_BEGIN("cuMemcpyHtoD");
            rc = d.cpyHtoD(g.a, ha.data(), words);
            if (rc == kSuccess) rc = d.cpyHtoD(g.b, hb.data(), words);
            TR_END("cuMemcpyHtoD", rc);
            if (rc != kSuccess) return fail("HtoD failed: " + err_text(d, rc));
            const auto t1 = std::chrono::steady_clock::now();
            h2d_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        double ms = 0.0;
        rc = launch_timed(ms);
        TR_END("iter", rc);
        if (rc != kSuccess) return fail("kernel launch failed: " + err_text(d, rc));
        sum_ms += ms;
        if (i == 0 || ms < min_ms) min_ms = ms;
        if (recopy) {
            auto [crc, cms] = copy_out();
            rc = crc;
            d2h_ms += cms;
            if (rc != kSuccess) return fail("DtoH(c) failed: " + err_text(d, rc));
            if (i == 0 || i == cfg.repeats - 1) ok_vals = check_vals() && ok_vals;
        }
    }
    (void)launch_count;

    const auto wall1 = std::chrono::steady_clock::now();
    if (!recopy) {
        const auto t2 = std::chrono::steady_clock::now();
        TR_BEGIN("cuMemcpyDtoH");
        rc = d.cpyDtoH(hc.data(), g.c, words);
        TR_END("cuMemcpyDtoH", rc);
        if (rc != kSuccess) return fail("DtoH(c) failed: " + err_text(d, rc));
        const auto t3 = std::chrono::steady_clock::now();
        d2h_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        ok_vals = check_vals();
    }

    const auto ms_of = [](std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    out.kernel_ms_avg = sum_ms / static_cast<double>(cfg.repeats);
    out.kernel_ms_min = min_ms;
    out.h2d_ms = h2d_ms;
    out.d2h_ms = d2h_ms;
    out.wall_ms = ms_of(wall0, wall1);
    if (recopy) {
        out.bytes_moved = static_cast<unsigned long long>(words) * 3u *
                          static_cast<unsigned long long>(cfg.repeats);
        if (sum_ms > 0.0) {
            // Total traffic over total kernel time.
            out.gbps_effective =
                static_cast<double>(out.bytes_moved) / (sum_ms / 1000.0) / 1e9;
        }
    } else {
        // Historical formula, preserved exactly: per-launch traffic over
        // average per-launch kernel time.
        out.bytes_moved = static_cast<unsigned long long>(words) * 3u;
        if (out.kernel_ms_avg > 0.0) {
            const double bytes_total = 3.0 * static_cast<double>(words);  // read a,b + write c
            out.gbps_effective = bytes_total / (out.kernel_ms_avg / 1000.0) / 1e9;
        }
    }
    out.correct = ok_vals;
    cupti_status(out.cupti_available, out.cupti_version);
    out.notes = "warmup excluded; WDDM first-launch cost absorbed in warmup";
    if (recopy) {
        out.notes += "; recopy-per-launch variant (naive app pattern: copies every round)";
    }
    if (dev.mem_bus_width_bits <= 0) {
        out.notes += "; theoretical bandwidth unknown (driver reports bus width 0)";
    }
    if (!ok_vals) out.notes = "RESULT MISMATCH: device output failed host verification";
    out.ok = ok_vals;
    if (!ok_vals) {
        if (err) *err = out.notes;
        return false;
    }
    return true;
}

bool run_vecadd_profile(const VecaddConfig& cfg, profiling::GpuProfileMetrics& out,
                        std::string* err) {
    // Persistent buffers: identical behavior to the historical implementation.
    return run_vecadd_impl(cfg, false, out, err);
}

bool run_vecadd_variant(const VecaddVariantConfig& vcfg,
                        profiling::GpuProfileMetrics& out, std::string* err) {
    VecaddConfig cfg;
    cfg.num_elements = vcfg.num_elements;
    cfg.block_size = vcfg.block_size;
    cfg.repeats = vcfg.repeats;
    return run_vecadd_impl(cfg, vcfg.recopy_per_launch, out, err);
}

// RAII for a variable number of device buffers + events + module + context.
struct PtxGuard {
    DriverApi& d;
    Ctx ctx = nullptr;
    Mod mod = nullptr;
    Ev start = nullptr;
    Ev stop = nullptr;
    std::vector<DevPtr> bufs{};
    ~PtxGuard() {
        if (start) d.evDestroy(start);
        if (stop) d.evDestroy(stop);
        for (DevPtr p : bufs) {
            if (p) d.memFree(p);
        }
        if (mod) d.modUnload(mod);
        if (ctx) d.ctxDestroy(ctx);
    }
};

bool run_ptx_kernel(const PtxLaunchConfig& cfg, PtxRunResult& out,
                    std::string* err) {
    out = PtxRunResult{};
    auto& m = out.metrics;
    m.kernel = cfg.kernel_label.empty() ? "ptx-kernel" : cfg.kernel_label;
    m.timing_source = "cuda-events";
    const auto fail = [&](const std::string& msg) {
        m.ok = false;
        m.error = msg;
        if (err) *err = msg;
        return false;
    };

    if (cfg.ptx_text.empty()) return fail("empty PTX text");
    if (cfg.launches.empty()) return fail("no kernel launches specified");
    if (cfg.num_elements == 0 || cfg.num_elements > (1u << 28))
        return fail("num_elements out of range (1 .. 268435456)");
    if (cfg.grid_size <= 0 || cfg.grid_size > (1 << 16))
        return fail("grid_size out of range");
    if (cfg.block_size <= 0 || cfg.block_size > 1024)
        return fail("block_size out of range (1 .. 1024)");
    if (cfg.repeats <= 0 || cfg.repeats > 1000) return fail("repeats out of range (1 .. 1000)");
    if (cfg.num_buffers <= 0) return fail("num_buffers must be positive");
    if (cfg.output_slot < 0 || cfg.output_slot >= cfg.num_buffers)
        return fail("output_slot out of range");
    if (static_cast<std::size_t>(cfg.num_buffers) != cfg.input_fills.size())
        return fail("input_fills size must equal num_buffers");
    if (cfg.bytes_per_element <= 0.0) return fail("bytes_per_element must be positive");
    for (const auto& l : cfg.launches) {
        if (l.entry.empty()) return fail("launch with empty entry name");
        for (int s : l.buffer_slots) {
            if (s < 0 || s >= cfg.num_buffers)
                return fail("launch buffer slot out of range for " + l.entry);
        }
    }

    DriverApi& d = driver();
    if (!d.ok) return fail("CUDA unavailable: " + d.load_error);

    profiling::GpuDeviceModel dev;
    if (!query_device(dev)) return fail("CUDA unavailable: " + dev.error);
    m.device = dev;

    const std::size_t n = cfg.num_elements;
    const std::size_t words = n * sizeof(float);
    const int block = cfg.block_size;
    const int grid = cfg.grid_size;
    const int nbufs = cfg.num_buffers;
    m.num_elements = n;
    m.block_size = block;
    m.grid_size = grid;
    m.repeats = cfg.repeats;

    const auto wall0 = std::chrono::steady_clock::now();
    PtxGuard g{d};

    int rc = d.init(0);
    if (rc != kSuccess) return fail("cuInit failed: " + err_text(d, rc));
    Dev handle = 0;
    rc = d.devGet(&handle, 0);
    if (rc != kSuccess) return fail("cuDeviceGet failed: " + err_text(d, rc));
    rc = d.ctxCreate(&g.ctx, 0, handle);
    if (rc != kSuccess) return fail("cuCtxCreate failed: " + err_text(d, rc));

    char jit_log[8192] = {};
    unsigned jit_log_size = static_cast<unsigned>(sizeof(jit_log));
    int jit_options[2] = {5, 6};
    void* jit_values[2] = {jit_log, &jit_log_size};
    rc = d.modLoadData(&g.mod, cfg.ptx_text.c_str(), 2, jit_options, jit_values);
    if (rc != kSuccess) {
        std::string msg = "PTX JIT failed: " + err_text(d, rc);
        if (jit_log[0] != '\0') {
            msg += "\nJIT log:\n";
            msg += jit_log;
        }
        return fail(msg);
    }
    g.bufs.assign(static_cast<std::size_t>(nbufs), 0);
    for (int i = 0; i < nbufs; ++i) {
        rc = d.memAlloc(&g.bufs[static_cast<std::size_t>(i)], words);
        if (rc != kSuccess)
            return fail("cuMemAlloc failed (need " + std::to_string(nbufs) + " x " +
                        std::to_string(words) + " bytes)");
    }

    struct ResolvedLaunch {
        Fn func = nullptr;
        std::string entry;
        std::vector<std::uint64_t> index_vals;
        std::vector<void*> args;
    };
    std::vector<ResolvedLaunch> launches;
    // Index values must outlive the launches; device pointers are stable.
    std::vector<std::vector<std::uint64_t>> index_store(cfg.launches.size());
    std::size_t li = 0;
    for (const auto& spec : cfg.launches) {
        Fn f = nullptr;
        rc = d.modGetFunc(&f, g.mod, spec.entry.c_str());
        if (rc != kSuccess)
            return fail("cuModuleGetFunction(" + spec.entry + ") failed: " + err_text(d, rc));
        ResolvedLaunch rl;
        rl.func = f;
        rl.entry = spec.entry;
        for (std::int64_t v : spec.index_args) {
            index_store[li].push_back(static_cast<std::uint64_t>(v));
        }
        for (std::size_t k = 0; k < index_store[li].size(); ++k) {
            rl.args.push_back(&index_store[li][k]);
        }
        for (int s : spec.buffer_slots) {
            rl.args.push_back(&g.bufs[static_cast<std::size_t>(s)]);
        }
        launches.push_back(std::move(rl));
        ++li;
    }
    std::vector<std::vector<float>> host(static_cast<std::size_t>(nbufs));
    for (int i = 0; i < nbufs; ++i) {
        host[static_cast<std::size_t>(i)].assign(n, cfg.input_fills[static_cast<std::size_t>(i)]);
    }

    rc = d.evCreate(&g.start, 0);
    if (rc == kSuccess) rc = d.evCreate(&g.stop, 0);
    if (rc != kSuccess) return fail("cuEventCreate failed: " + err_text(d, rc));

    const auto copy_in = [&]() {
        const auto c0 = std::chrono::steady_clock::now();
        int crc = kSuccess;
        for (int i = 0; i < nbufs && crc == kSuccess; ++i) {
            if (i == cfg.output_slot) continue;
            crc = d.cpyHtoD(g.bufs[static_cast<std::size_t>(i)],
                            host[static_cast<std::size_t>(i)].data(), words);
        }
        const auto c1 = std::chrono::steady_clock::now();
        return std::make_pair(crc, std::chrono::duration<double, std::milli>(c1 - c0).count());
    };
    const auto launch_all = [&](bool timed, double* per_launch_ms) {
        for (std::size_t k = 0; k < launches.size(); ++k) {
            const auto& ln = launches[k];
            int lrc = d.evRecord(g.start, nullptr);
            if (lrc != kSuccess) return lrc;
            lrc = d.launch(ln.func, static_cast<unsigned>(grid), 1, 1,
                           static_cast<unsigned>(block), 1, 1, 0, nullptr,
                           const_cast<void**>(ln.args.data()), nullptr);
            if (lrc != kSuccess) return lrc;
            lrc = d.evRecord(g.stop, nullptr);
            if (lrc != kSuccess) return lrc;
            lrc = d.evSync(g.stop);
            if (lrc != kSuccess) return lrc;
            if (timed && per_launch_ms) {
                float ms = 0.0f;
                lrc = d.evElapsed(&ms, g.start, g.stop);
                if (lrc != kSuccess) return lrc;
                per_launch_ms[k] = ms;
            }
        }
        return kSuccess;
    };

    // Untimed warmup so the timed region compares fairly.
    {
        auto [wcrc, wms] = copy_in();
        (void)wms;
        rc = wcrc;
    }
    if (rc != kSuccess) return fail("HtoD failed: " + err_text(d, rc));
    rc = launch_all(false, nullptr);
    if (rc != kSuccess) return fail("warmup failed: " + err_text(d, rc));

    double h2d_ms = 0.0;
    {
        const auto t0 = std::chrono::steady_clock::now();
        auto [crc, cms] = copy_in();
        (void)cms;
        rc = crc;
        const auto t1 = std::chrono::steady_clock::now();
        if (rc != kSuccess) return fail("HtoD failed: " + err_text(d, rc));
        h2d_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::vector<double> sum_ms(launches.size(), 0.0), min_ms(launches.size(), 0.0);
    std::vector<double> iter_ms(launches.size(), 0.0);
    double workload_sum = 0.0;
    double workload_min = 0.0;
    for (int i = 0; i < cfg.repeats; ++i) {
        rc = launch_all(true, iter_ms.data());
        if (rc != kSuccess) return fail("kernel launch failed: " + err_text(d, rc));
        double total = 0.0;
        for (std::size_t k = 0; k < launches.size(); ++k) {
            sum_ms[k] += iter_ms[k];
            if (i == 0 || iter_ms[k] < min_ms[k]) min_ms[k] = iter_ms[k];
            total += iter_ms[k];
        }
        workload_sum += total;
        if (i == 0 || total < workload_min) workload_min = total;
    }
    double d2h_ms = 0.0;
    {
        const auto t2 = std::chrono::steady_clock::now();
        rc = d.cpyDtoH(host[static_cast<std::size_t>(cfg.output_slot)].data(),
                       g.bufs[static_cast<std::size_t>(cfg.output_slot)], words);
        const auto t3 = std::chrono::steady_clock::now();
        if (rc != kSuccess) return fail("DtoH failed: " + err_text(d, rc));
        d2h_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    bool ok_vals = true;
    const auto& hout = host[static_cast<std::size_t>(cfg.output_slot)];
    for (std::size_t i = 0; i < n; ++i) {
        if (hout[i] != cfg.expected) {
            ok_vals = false;
            break;
        }
    }

    const auto wall1 = std::chrono::steady_clock::now();
    for (std::size_t k = 0; k < launches.size(); ++k) {
        PerKernelTime t;
        t.entry = launches[k].entry;
        t.ms_avg = sum_ms[k] / static_cast<double>(cfg.repeats);
        t.ms_min = min_ms[k];
        out.per_kernel.push_back(t);
    }
    m.kernel_ms_avg = workload_sum / static_cast<double>(cfg.repeats);
    m.kernel_ms_min = workload_min;
    m.h2d_ms = h2d_ms;
    m.d2h_ms = d2h_ms;
    m.wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
    m.bytes_moved = static_cast<unsigned long long>(cfg.bytes_per_element *
                                                    static_cast<double>(n));
    if (m.kernel_ms_avg > 0.0) {
        m.gbps_effective =
            static_cast<double>(m.bytes_moved) / (m.kernel_ms_avg / 1000.0) / 1e9;
    }
    m.correct = ok_vals;
    cupti_status(m.cupti_available, m.cupti_version);
    m.notes = "compiler-generated kernel; warmup excluded";
    if (dev.mem_bus_width_bits <= 0) {
        m.notes += "; theoretical bandwidth unknown (driver reports bus width 0)";
    }
    if (!ok_vals) m.notes = "RESULT MISMATCH: device output failed host verification";
    m.ok = ok_vals;
    if (!ok_vals) {
        if (err) *err = m.notes;
        return false;
    }
    return true;
}

bool run_triton_kernel(const TritonLaunchConfig& cfg, TritonRunResult& out,
                       std::string* err) {
    out = TritonRunResult{};
    auto& m = out.metrics;
    m.kernel = cfg.workload_name.empty() ? "triton-kernel" : ("triton:" + cfg.workload_name);
    m.timing_source = "cuda-events";
    const auto fail = [&](const std::string& msg) {
        m.ok = false;
        m.error = msg;
        out.error = msg;
        if (err) *err = msg;
        return false;
    };

    if (cfg.ptx_text.empty()) return fail("empty Triton PTX text");
    if (cfg.entry_name.empty()) return fail("empty Triton entry function name");
    if (cfg.num_elements == 0 || cfg.num_elements > (1u << 28))
        return fail("num_elements out of range");
    if (cfg.grid_size <= 0) return fail("grid_size must be positive");
    if (cfg.block_size <= 0) return fail("block_size must be positive");
    if (cfg.repeats <= 0) return fail("repeats must be positive");

    DriverApi& d = driver();
    if (!d.ok) return fail("CUDA unavailable: " + d.load_error);

    profiling::GpuDeviceModel dev;
    if (!query_device(dev)) return fail("CUDA unavailable: " + dev.error);
    m.device = dev;

    const std::size_t n = cfg.num_elements;
    const std::size_t words = n * sizeof(float);
    m.num_elements = n;
    m.block_size = cfg.block_size;
    m.grid_size = cfg.grid_size;
    m.repeats = cfg.repeats;

    const auto wall0 = std::chrono::steady_clock::now();
    PtxGuard g{d};

    int rc = d.init(0);
    if (rc != kSuccess) return fail("cuInit failed: " + err_text(d, rc));
    Dev handle = 0;
    rc = d.devGet(&handle, 0);
    if (rc != kSuccess) return fail("cuDeviceGet failed: " + err_text(d, rc));
    rc = d.ctxCreate(&g.ctx, 0, handle);
    if (rc != kSuccess) return fail("cuCtxCreate failed: " + err_text(d, rc));

    char jit_log[8192] = {};
    unsigned jit_log_size = static_cast<unsigned>(sizeof(jit_log));
    int jit_options[2] = {5, 6};
    void* jit_values[2] = {jit_log, &jit_log_size};
    rc = d.modLoadData(&g.mod, cfg.ptx_text.c_str(), 2, jit_options, jit_values);
    if (rc != kSuccess) {
        std::string msg = "Triton PTX JIT failed: " + err_text(d, rc);
        if (jit_log[0] != '\0') {
            msg += "\nJIT log:\n";
            msg += jit_log;
        }
        return fail(msg);
    }

    Fn kernel_fn = nullptr;
    rc = d.modGetFunc(&kernel_fn, g.mod, cfg.entry_name.c_str());
    if (rc != kSuccess || !kernel_fn) {
        return fail("cuModuleGetFunction failed for entry '" + cfg.entry_name + "': " + err_text(d, rc));
    }

    const bool is_reduction = (cfg.workload_name == "reduction");
    const std::size_t out_elements = is_reduction ? static_cast<std::size_t>(cfg.grid_size) : n;
    const std::size_t out_words = out_elements * sizeof(float);

    DevPtr dev_a = 0, dev_b = 0, dev_out = 0;
    rc = d.memAlloc(&dev_a, words);
    if (rc != kSuccess) return fail("cuMemAlloc failed for input A");
    g.bufs.push_back(dev_a);

    if (!is_reduction) {
        rc = d.memAlloc(&dev_b, words);
        if (rc != kSuccess) return fail("cuMemAlloc failed for input B");
        g.bufs.push_back(dev_b);
    }

    rc = d.memAlloc(&dev_out, out_words);
    if (rc != kSuccess) return fail("cuMemAlloc failed for output");
    g.bufs.push_back(dev_out);

    std::vector<float> host_a(n);
    std::vector<float> host_b(is_reduction ? 0 : n);
    std::vector<float> host_out(out_elements, 0.0f);
    std::vector<float> host_expected(out_elements, 0.0f);

    for (std::size_t i = 0; i < n; ++i) {
        host_a[i] = 1.0f + 0.05f * static_cast<float>(i % 37);
        if (!is_reduction) {
            host_b[i] = -0.5f + 0.02f * static_cast<float>(i % 23);
        }
    }

    if (cfg.workload_name == "fused_add_relu") {
        for (std::size_t i = 0; i < n; ++i) {
            float sum = host_a[i] + host_b[i];
            host_expected[i] = (sum > 0.0f) ? sum : 0.0f;
        }
    } else if (cfg.workload_name == "vector_add") {
        for (std::size_t i = 0; i < n; ++i) {
            host_expected[i] = host_a[i] + host_b[i];
        }
    } else if (is_reduction) {
        const std::size_t block_sz = 256;
        for (int b = 0; b < cfg.grid_size; ++b) {
            float acc = 0.0f;
            for (std::size_t offset = 0; offset < block_sz; ++offset) {
                std::size_t idx = static_cast<std::size_t>(b) * block_sz + offset;
                if (idx < n) acc += host_a[idx];
            }
            host_expected[static_cast<std::size_t>(b)] = acc;
        }
    } else {
        // Generic fallback verification
        for (std::size_t i = 0; i < n; ++i) {
            float sum = host_a[i] + (host_b.empty() ? 0.0f : host_b[i]);
            host_expected[i] = (sum > 0.0f) ? sum : 0.0f;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    rc = d.cpyHtoD(dev_a, host_a.data(), words);
    if (rc != kSuccess) return fail("HtoD failed for input A: " + err_text(d, rc));
    if (!is_reduction) {
        rc = d.cpyHtoD(dev_b, host_b.data(), words);
        if (rc != kSuccess) return fail("HtoD failed for input B: " + err_text(d, rc));
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double h2d_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::int32_t n_elem_i32 = static_cast<std::int32_t>(n);

    // Triton kernel signatures (verified from PTX output of drishti_triton_compiler.py):
    //   fused_add_relu / vector_add: 6 params
    //     param_0: x_ptr (u64 ptr), param_1: y_ptr (u64 ptr),
    //     param_2: output_ptr (u64 ptr), param_3: n_elements (u32),
    //     param_4: scratch_ptr0 (u64 ptr), param_5: scratch_ptr1 (u64 ptr)
    //   reduction: 3 params
    //     param_0: x_ptr, param_1: output_ptr, param_2: n_elements (u32)
    //
    // param_4/param_5 are declared in PTX but never loaded by the kernel body.
    // Allocate minimal real device buffers (4 bytes each) so the driver
    // receives valid non-null device pointers instead of 0.
    DevPtr dev_scratch0 = 0, dev_scratch1 = 0;
    if (!is_reduction) {
        rc = d.memAlloc(&dev_scratch0, 4);
        if (rc != kSuccess) return fail("cuMemAlloc failed for scratch0");
        g.bufs.push_back(dev_scratch0);
        rc = d.memAlloc(&dev_scratch1, 4);
        if (rc != kSuccess) return fail("cuMemAlloc failed for scratch1");
        g.bufs.push_back(dev_scratch1);
    }

    void* args_reduction[3] = {&dev_a, &dev_out, &n_elem_i32};
    void* args_binary[6]    = {&dev_a, &dev_b, &dev_out, &n_elem_i32, &dev_scratch0, &dev_scratch1};
    void** args = is_reduction ? args_reduction : args_binary;

    // Warmup launch
    rc = d.launch(kernel_fn, static_cast<unsigned>(cfg.grid_size), 1, 1,
                  static_cast<unsigned>(cfg.block_size), 1, 1,
                  static_cast<unsigned>(cfg.shared_mem_bytes), nullptr,
                  args, nullptr);
    if (rc != kSuccess) return fail("Triton warmup launch failed: " + err_text(d, rc));
    rc = d.ctxSync();
    if (rc != kSuccess) return fail("ctxSync after warmup failed: " + err_text(d, rc));

    rc = d.evCreate(&g.start, 0);
    if (rc != kSuccess) return fail("cuEventCreate start failed");
    rc = d.evCreate(&g.stop, 0);
    if (rc != kSuccess) return fail("cuEventCreate stop failed");

    double total_kernel_ms = 0.0;
    double min_kernel_ms = 1e9;
    double max_kernel_ms = 0.0;

    for (int rep = 0; rep < cfg.repeats; ++rep) {
        d.evRecord(g.start, nullptr);
        rc = d.launch(kernel_fn, static_cast<unsigned>(cfg.grid_size), 1, 1,
                      static_cast<unsigned>(cfg.block_size), 1, 1,
                      static_cast<unsigned>(cfg.shared_mem_bytes), nullptr,
                      args, nullptr);
        if (rc != kSuccess) {
            return fail("Triton timed launch failed on iteration " + std::to_string(rep) + ": " + err_text(d, rc));
        }
        d.evRecord(g.stop, nullptr);
        d.evSync(g.stop);
        float elapsed_f = 0.0f;
        d.evElapsed(&elapsed_f, g.start, g.stop);
        double elapsed_ms = static_cast<double>(elapsed_f);
        total_kernel_ms += elapsed_ms;
        if (elapsed_ms < min_kernel_ms) min_kernel_ms = elapsed_ms;
        if (elapsed_ms > max_kernel_ms) max_kernel_ms = elapsed_ms;
    }

    const auto t2 = std::chrono::steady_clock::now();
    rc = d.cpyDtoH(host_out.data(), dev_out, out_words);
    const auto t3 = std::chrono::steady_clock::now();
    if (rc != kSuccess) return fail("DtoH failed: " + err_text(d, rc));
    const double d2h_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    bool ok_vals = true;
    for (std::size_t i = 0; i < out_elements; ++i) {
        float diff = std::abs(host_out[i] - host_expected[i]);
        if (diff > 1e-3f) {
            ok_vals = false;
            break;
        }
    }

    const auto wall1 = std::chrono::steady_clock::now();
    m.kernel_ms_avg = total_kernel_ms / static_cast<double>(cfg.repeats);
    m.kernel_ms_min = min_kernel_ms;
    m.h2d_ms = h2d_ms;
    m.d2h_ms = d2h_ms;
    m.wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count();
    m.bytes_moved = static_cast<unsigned long long>(cfg.bytes_per_element * static_cast<double>(n));

    if (m.kernel_ms_avg > 0.0) {
        m.gbps_effective = static_cast<double>(m.bytes_moved) / (m.kernel_ms_avg / 1000.0) / 1e9;
    }
    m.correct = ok_vals;
    cupti_status(m.cupti_available, m.cupti_version);
    m.notes = "Triton-generated kernel: " + cfg.entry_name;
    m.ok = ok_vals;

    out.verified = ok_vals;
    out.measured_kernel_ms = m.kernel_ms_avg;
    if (!ok_vals) {
        m.notes = "RESULT MISMATCH: Triton device output failed verification";
        if (err) *err = m.notes;
        return false;
    }
    return true;
}

}  // namespace drishti::backends::cuda
