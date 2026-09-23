# Dṛṣṭi (Drishti): Hardware-Aware ML Compiler Performance Intelligence

[![CI](https://github.com/krish-rRay23/Drishti/actions/workflows/ci.yml/badge.svg)](https://github.com/krish-rRay23/Drishti/actions/workflows/ci.yml)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![LLVM](https://img.shields.io/badge/LLVM-22.0+-yellow.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

**Dṛṣṭi** (Sanskrit: *sight / insight*) is a high-performance C++20 hardware-aware ML compiler intelligence framework. It provides end-to-end transformation graph tracing, LLVM IR optimization analysis, live GPU profiler telemetry, and roofline bottleneck diagnosis across **Triton 3.8.0**, **MLIR**, and **LLVM / NVPTX** target pipelines.

Designed to bridge static compiler optimization decisions with empirical GPU performance, Drishti correlates compiler IR attributes (e.g., TTGIR layout vectorization `sizePerThread`) against low-level hardware metrics (kernel execution latency, effective VRAM memory bandwidth, and theoretical compute occupancy).

---

##  Key Empirical Highlights & Benchmark Results

All results are measured on live hardware (**NVIDIA GeForce RTX 3050 Laptop GPU**, `sm_86`, 16 SMs, Ampere Architecture) and validated against a **44/44 passing CTest suite (100% success rate)**.

* **55.4% LLVM-IR Instruction Reduction:** Standardized pass chains (`sroa` -> `instcombine` -> `simplifycfg` -> `dce`) eliminate dead code and stack allocations across lowered MLIR modules.
* **32.8% Triton Kernel Latency Reduction:** Vectorizing Triton 1D pointer access patterns (128-bit `ld.global.v4.b32` vs scalar `ld.global.b32`) reduces kernel execution time from **$0.1438\text{ ms}$ to $0.0967\text{ ms}$** ($1.49\times$ speedup).
* **48.8% Bandwidth Utilization Growth:** Vectorized memory coalescing increases effective VRAM throughput from **$16.20\text{ GB/s}$ to $24.11\text{ GB/s}$**.
* **29.4% Kernel Fusion Speedup:** Fusing elementwise `add` and `relu` passes reduces launch overhead and memory roundtrips compared to separate kernels.
* **Full Multi-Stage Pipeline Tracing:** Direct inspection of **Triton Python -> TTIR -> TTGIR -> LLVM IR -> NVPTX Assembly**.

---

## 📊 Triton 3.8.0 Vectorization Codegen Case Study

When compiling 1D runtime pointers (`*fp32`), Triton conservatively assumes 4-byte pointer alignment, forcing `AxisInfoAnalysis` to assign layout `sizePerThread = [1]`. Adding explicit alignment hints (`tl.max_contiguous`, `tl.multiple_of`) allows `CoalescePass` to assign `sizePerThread = [4]`, unlocking 128-bit vector memory instructions (`ld.global.v4.b32`).

| Metric / Parameter | Unvectorized (`vector_add_scalar`) | Vectorized (`vector_add_vectorized`) | Impact / Speedup |
| :--- | :---: | :---: | :---: |
| **TTGIR Layout Attribute** | `sizePerThread = [1]` | `sizePerThread = [4]` | **Vectorized Layout** |
| **Generated PTX Instruction** | `ld.global.b32` (Scalar 32-bit) | `ld.global.v4.b32` (Vector 128-bit) | **$4\times$ wider transaction** |
| **Memory Ops / Warp Iteration** | 32 instructions | 8 instructions | **$4\times$ fewer memory ops** |
| **Kernel Latency ($\\text{ms}$)** | **$0.1438\\text{ ms}$** | **$0.0967\\text{ ms}$** | **$32.8\%$ Latency Reduction ($1.49\\times$)** |
| **Effective VRAM Bandwidth** | **$16.20\\text{ GB/s}$** | **$24.11\\text{ GB/s}$** | **$+48.8\%$ Bandwidth Increase** |
| **Kernel Output Verification** | `PASS` | `PASS` | **$100\%$ Bit-exact Match** |

---

## 📈 Full GPU Workload Benchmark Matrix

Empirical benchmarks collected via `drishti benchmark --full` on **RTX 3050 GPU hardware**:

| Workload ID | Tensor Shape / Size | Latency ($\\text{ms}$) | Effective VRAM BW ($\\text{GB/s}$) | Compute ($\\text{GFLOPS}$) | Bottleneck Classification | Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `fused_add_relu` | $256 \\times 256$ ($65,536$ el) | **$0.0812\\text{ ms}$** | **$28.94\\text{ GB/s}$** | N/A | Memory-Bound (VRAM Coalesced) | `PASS` |
| `vector_add_vectorized` | $65,536$ elements | **$0.0967\\text{ ms}$** | **$24.11\\text{ GB/s}$** | N/A | Memory-Bound (Vectorized 128-bit) | `PASS` |
| `vector_add_scalar` | $65,536$ elements | **$0.1438\\text{ ms}$** | **$16.20\\text{ GB/s}$** | N/A | Memory-Bound (Scalar Access) | `PASS` |
| `matmul_tiled_16x16` | $256 \times 256 \times 256$ | **$0.1945\text{ ms}$** | **$12.40\text{ GB/s}$** | **$108.5\text{ GFLOPS}$** | Balanced / Shared Memory Bandwidth | `PASS` |
| `conv2d_nchw_3x3` | $1 \times 32 \times 64 \times 64$ | **$0.3412\text{ ms}$** | **$9.85\text{ GB/s}$** | **$142.1\text{ GFLOPS}$** | Compute-Bound (Tensor Core Eligible) | `PASS` |

---

## 🧪 Journal Study: NVIDIA Tesla T4 (`sm_75`, Google Colab) Benchmarks

Empirical benchmark execution results collected directly from Google Colab on an **NVIDIA Tesla T4 GPU** (`sm_75`, 40 SMs, Turing Architecture) via [`notebooks/Runned_Drishti_Journal_Benchmark_Suite.ipynb`](file:///C:/Users/krish/Dhristi/notebooks/Runned_Drishti_Journal_Benchmark_Suite.ipynb):

| Workload Domain | System / Evaluation Target | Latency ($\text{ms}$) | Speedup vs Baseline | VRAM Bandwidth ($\text{GB/s}$) | Correctness |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`gemm_small`** ($256^3$) | Standard Triton | $0.0912\text{ ms}$ | $1.00\times$ | $8.62\text{ GB/s}$ | `PASS` |
| | Triton + Autotune | $0.2630\text{ ms}$ | $0.35\times$ | $2.99\text{ GB/s}$ | `PASS` |
| | Full Drishti | $0.2625\text{ ms}$ | $0.35\times$ | $3.00\text{ GB/s}$ | `PASS` |
| | **Drishti - Vectorization** | **$0.0611\text{ ms}$** | **$1.49\times$** | **$12.90\text{ GB/s}$** | `PASS` |
| **`gemm_medium`** ($1024^3$) | Standard Triton | $0.8349\text{ ms}$ | $1.00\times$ | $15.07\text{ GB/s}$ | `PASS` |
| | Triton + Autotune | $0.7684\text{ ms}$ | $1.09\times$ | $16.37\text{ GB/s}$ | `PASS` |
| | Full Drishti | $0.7668\text{ ms}$ | $1.09\times$ | $16.41\text{ GB/s}$ | `PASS` |
| | **Drishti - IR Analysis** | **$0.6086\text{ ms}$** | **$1.37\times$** | **$20.70\text{ GB/s}$** | `PASS` |
| **`gemm_large`** ($2048^3$) | Standard Triton | $5.1804\text{ ms}$ | $1.00\times$ | $9.72\text{ GB/s}$ | `PASS` |
| | Triton + Autotune | $4.8421\text{ ms}$ | $1.07\times$ | $10.39\text{ GB/s}$ | `PASS` |
| | Full Drishti | $4.8853\text{ ms}$ | $1.06\times$ | $10.30\text{ GB/s}$ | `PASS` |
| **`fused_add_relu`** ($262K$) | Standard Triton | $0.0433\text{ ms}$ | $1.00\times$ | $72.71\text{ GB/s}$ | `PASS` |
| | **Drishti - IR Analysis** | **$0.0405\text{ ms}$** | **$1.07\times$** | **$77.60\text{ GB/s}$** | `PASS` |
| **`reduction`** ($1M$) | Standard Triton | $0.0403\text{ ms}$ | $1.00\times$ | $104.12\text{ GB/s}$ | `PASS` |
| | Triton + Autotune | $0.0347\text{ ms}$ | $1.16\times$ | $120.87\text{ GB/s}$ | `PASS` |
| | **Full Drishti** | **$0.0316\text{ ms}$** | **$1.28\times$** | **$132.50\text{ GB/s}$** | `PASS` |
| **`layernorm`** ($128 \times 2048$) | Standard Triton | $0.0417\text{ ms}$ | $1.00\times$ | $50.33\text{ GB/s}$ | `PASS` |
| | Triton + Autotune | $0.0381\text{ ms}$ | $1.09\times$ | $55.03\text{ GB/s}$ | `PASS` |
| | Full Drishti | $0.0382\text{ ms}$ | $1.09\times$ | $54.87\text{ GB/s}$ | `PASS` |

---

## 📐 Architecture Overview

Drishti connects static compiler analysis with dynamic GPU profiling:

```mermaid
flowchart TD
    subgraph Frontend ["Frontends & Workloads"]
        A1["Triton Python Source (JIT)"]
        A2["MLIR Source (.mlir / C++ API)"]
    end

    subgraph CompilerEngine ["Compiler & Pass Engine"]
        B1["Triton Compiler Bridge\n(TTIR -> TTGIR -> LLVM -> PTX)"]
        B2["MLIR Dialect Analyzer\n(Operation Histogram & CFG)"]
        B3["LLVM Optimization Manager\n(SROA, InstCombine, DCE)"]
    end

    subgraph PerformanceIntelligence ["Performance & Telemetry"]
        C1["Provenance Graph Tracker\n(Multi-Stage Stage Mapping)"]
        C2["CUDA / ROCm Event Profiler\n(Hardware Timers & BW Metrics)"]
        C3["Roofline Bottleneck Engine\n(Memory vs Compute Classification)"]
    end

    A1 --> B1
    A2 --> B2
    B2 --> B3
    B1 --> C1
    B3 --> C1
    C2 --> C3
    C1 --> C3
```

---

## 🛠️ Quick Start & Build Instructions

### Prerequisites
* **CMake:** $\\ge 3.24$
* **C++ Compiler:** C++20 compliant (GCC $\\ge 12$, Clang $\\ge 16$, MSVC 2022)
* **Build System:** Ninja (recommended) or MSBuild
* **Python (Optional):** Python $\\ge 3.8$ with Triton $\\ge 3.8.0$ for Triton JIT interop
* **NVIDIA Driver:** CUDA $\\ge 11.0$ (`nvcuda.dll` / `libcuda.so.1` loaded dynamically at runtime; CUDA Toolkit not required for compilation)

### Building & Running Tests

```bash
# Clone the repository
git clone https://github.com/krish-rRay23/Drishti.git
cd Drishti

# Configure CMake with Ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build all binaries and test targets
ninja -C build

# Execute full test suite (44/44 tests pass)
ctest --test-dir build --output-on-failure
```

---

## 💻 CLI Usage Examples

```bash
# Display system configuration & detected GPU devices
drishti --info

# Inspect Triton pipeline transformation (TTIR -> TTGIR -> LLVM -> PTX)
drishti triton --workload=fused_add_relu --show-ir

# Reproduce the Triton 128-bit vectorization case study
drishti triton --workload=vector_add_vectorized --verify-gpu --show-ir

# Execute the full benchmark suite across all registered workloads
drishti benchmark --full

# Run root-cause performance diagnosis on a target kernel
drishti diagnose --kernel=fused_add_relu
```

---

## 📁 Repository Structure

```
Drishti/
├── .github/workflows/ci.yml       # Multi-platform CI pipeline (Linux/Windows)
├── CMakeLists.txt                 # C++20 CMake build configuration
├── README.md                      # Framework documentation & benchmarks
├── include/drishti/               # Modular public C++ header interfaces
│   ├── analysis/                  # MLIR parser & LLVM IR lowering passes
│   ├── backends/                  # CUDA Driver API & ROCm/HIP dynamic backends
│   ├── benchmark/                 # Workload benchmark suite engine
│   ├── core/                      # Core configuration, version, and export macros
│   ├── correlation/               # IR attribute to hardware metric correlation
│   ├── diagnosis/                 # Automated roofline bottleneck diagnosis engine
│   ├── optimizer/                 # Analytical cost model & optimization explorer
│   ├── profiling/                 # Live GPU CUDA event profiler
│   ├── provenance/                # Transformation provenance graph tracking
│   └── triton/                    # Triton 3.8.0 compiler interop bridge
├── lib/                           # Component implementation files
├── tests/                         # 44 GoogleTest unit and integration tests
└── tools/                         # Executable CLI applications
```

---

## ⚠️ Technical Scope & Limitations

1. **Win64 Driver ABI Calling Convention:** Dynamic loading of CUDA Driver API functions with $>4$ parameters (`cuLaunchKernel`) uses explicit `__attribute__((ms_abi))` annotations under MinGW `clang++` to maintain MS x64 ABI stack frame compatibility.
2. **Triton Alignment Annotations:** Triton 3.8.0 conservatively defaults unannotated 1D pointers to scalar 32-bit access. Unlocking 128-bit vectorization requires explicit `tl.max_contiguous` / `tl.multiple_of` pointer alignment hints in Triton source.
3. **ROCm Backend Scope:** ROCm/HIP interfaces are fully defined in headers; full ROCm driver execution requires a Linux host environment with ROCm drivers.

---

## 📄 License

Apache-2.0 © The Dṛṣṭi Authors. See [LICENSE](LICENSE).
