# Dṛṣṭi (v0.1.0)

**Dṛṣṭi** (Sanskrit for *"vision, insight, clear sight"*) is an open-source, research-grade **C++20 ML compiler and GPU performance analysis framework**.

Drishti unifies MLIR compiler pass pipelines, GPU kernel performance modeling (Roofline, theoretical occupancy, effective memory bandwidth), execution provenance graph tracking, live GPU profiling, automated root-cause diagnosis, and deep Triton integration into a modular, embeddable toolkit.

---

## Key Features & Highlights

- **Native LLVM/MLIR Integration (22.1.8):** Real end-to-end lowering pipeline ($\text{MLIR} \rightarrow \text{LLVM IR} \rightarrow \text{Real LLVM Passes} \rightarrow \text{NVPTX Assembly} \rightarrow \text{CUDA Driver API}$).
- **Deep Triton Integration (3.8.0):** Ingests Triton workloads and captures 5 compiler stages ($\text{Triton AST} \rightarrow \text{TTIR} \rightarrow \text{TTGIR} \rightarrow \text{LLVM IR} \rightarrow \text{NVPTX}$) with live GPU execution and verification.
- **Triton Codegen Case Study:** Automated analysis of global memory vectorization codegen ($\text{sizePerThread} = [1]$ scalar $\text{ld.global.b32}$ vs $\text{sizePerThread} = [2]/[4]$ vectorized $\text{ld.global.v4.b32}$), achieving a **32.8% latency reduction** and **48.8% bandwidth increase** on NVIDIA RTX 3050 GPUs.
- **Dynamic CUDA Driver API Backend:** Hardware-profiling backend loading `nvcuda.dll` / `libcuda.so` dynamically without hard toolkit dependencies, measuring device-side CUDA event timings, effective VRAM bandwidth, and TFLOPS throughput.
- **Analytical & Calibrated Cost Modeling:** White-box Roofline model combined with empirical calibration to predict memory-bound vs compute-bound bottleneck regimes.
- **Provenance Graph Infrastructure:** Tracks multi-stage IR transformations and associates generated PTX assembly with hardware execution metrics.

---

## Architecture

```
include/drishti/          # Public headers
  ├── core/               # Version, config, macros, project identity
  ├── analysis/           # MLIR structural analysis, Roofline, AI modeling
  ├── provenance/         # Multi-stage transformation graph tracking
  ├── profiling/          # Live GPU event timers, sample buffers
  ├── diagnosis/          # Automated root-cause rules engine & findings
  ├── optimizer/          # Cost model, pass pipelines, compiler experiments
  ├── triton/             # Deep Triton JIT pipeline & codegen analysis
  └── backends/           # Backend registry (CUDA Driver API, ROCm, Host)

lib/                      # Modular STATIC libraries
  ├── core/ analysis/ provenance/ profiling/ diagnosis/ optimizer/ triton/ backends/

tools/
  └── drishti/            # `drishti` CLI

tests/                    # GoogleTest suites (44 tests, discovered via CTest)
```

---

## Prerequisites & Installation

| Tool / Dependency | Minimum Version | Notes |
| :--- | :--- | :--- |
| **CMake** | $\ge 3.24$ | Ninja generator recommended |
| **C++ Compiler** | C++20 | Clang $\ge 16$, GCC $\ge 12$, MSVC 2022 |
| **LLVM / MLIR** | $\ge 22.0$ | Optional; enables native LLVM/MLIR lowering |
| **Python / Triton** | $\ge 3.8.0$ | Optional; enables Triton JIT compilation pipeline |
| **NVIDIA Driver** | CUDA $\ge 11.0$ | `nvcuda.dll` / `libcuda.so.1` dynamically loaded |

### Build & Run Test Suite

```bash
# Configure and build with Ninja
cmake -S . -B build-clang -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build-clang

# Execute test suite (100% pass rate)
ctest --test-dir build-clang --output-on-failure
```

---

## CLI Usage

```text
drishti --help                             Full CLI command list
drishti --version                          Print version string
drishti --info                             Build configuration summary (C++ std, LLVM, MLIR, CUDA)
drishti --list-backends                    List registered backends (CUDA, ROCm, Host)

# MLIR Analysis
drishti analyze <file.mlir>                Parse MLIR and output structural dialect report
drishti validate <file.mlir>               Verify MLIR syntax and dialect semantics

# Native LLVM Lowering & Execution
drishti llvm --show-ir                     Lower MLIR to LLVM IR -> NVPTX -> execute on GPU

# Deep Triton Integration & Case Study
drishti triton --workload=fused_add_relu --verify-gpu --show-ir
drishti triton --workload=vector_add_scalar --verify-gpu
drishti triton --workload=vector_add_vectorized --verify-gpu --show-ir

# Benchmark & Optimization
drishti benchmark --full                   Execute full GPU benchmark suite across workloads
drishti diagnose --kernel=<name>           Run automated root-cause bottleneck diagnosis
drishti suggest --kernel=<name>            Output target-aware optimization recommendations
```

---

## Benchmark Methodology

1. **Isolation of GPU Kernel Execution:** Hardware timings are recorded strictly using device-side CUDA driver events (`cuEventRecord`, `cuEventSynchronize`, `cuEventElapsedTime`) over $N=5$ iterations after a warmup launch.
2. **Host Transfer Overhead Separation:** Host-to-device (`cuMemcpyHtoD`) and device-to-host (`cuMemcpyDtoH`) copy latencies are measured separately via high-resolution host steady clocks to isolate data transfer overheads from GPU execution.
3. **Effective Memory Bandwidth Calculation:** Calculated as $\text{GB/s} = \frac{\text{Bytes Moved}}{\text{Kernel Avg Latency (s)} \times 10^9}$ based on the algorithmic minimum bytes moved for the specific tensor shape.

---

## Project Limitations

- **Platform-Specific Calling Conventions:** On Windows MinGW `clang++` builds, CUDA Driver API function pointers require explicit `__attribute__((ms_abi))` annotations to avoid calling convention mismatches for functions with $>4$ parameters (`cuLaunchKernel`).
- **Triton Pointer Alignment Semantics:** Raw pointer arguments in Triton JIT kernels default to 4-byte alignment, requiring explicit `tl.max_contiguous` or `tl.multiple_of` annotations in Python source to generate 128-bit vector memory instructions (`ld.global.v4.b32`).
- **Python Execution Dependency for Triton:** Triton pipeline ingestion invokes `drishti_triton_compiler.py` out-of-process via Python 3.11.

---

## License

Apache-2.0 © The Dṛṣṭi Authors. See [LICENSE](LICENSE).
