import inspect
import json
import os
import sys
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def fused_add_relu_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    added = x + y
    relu = tl.maximum(added, 0.0)
    tl.store(output_ptr + offsets, relu, mask=mask)

@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

@triton.jit
def vector_add_scalar_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(output_ptr + offsets, x + y)

@triton.jit
def vector_add_vectorized_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_ptr_c = tl.max_contiguous(tl.multiple_of(x_ptr + offsets, 16), 16)
    y_ptr_c = tl.max_contiguous(tl.multiple_of(y_ptr + offsets, 16), 16)
    out_ptr_c = tl.max_contiguous(tl.multiple_of(output_ptr + offsets, 16), 16)
    x = tl.load(x_ptr_c)
    y = tl.load(y_ptr_c)
    tl.store(out_ptr_c, x + y)

@triton.jit
def reduction_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    acc = tl.sum(x, axis=0)
    tl.store(output_ptr + pid, acc)

KERNELS = {
    "fused_add_relu": (
        fused_add_relu_kernel,
        {"BLOCK_SIZE": 256},
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
    ),
    "vector_add": (
        vector_add_kernel,
        {"BLOCK_SIZE": 256},
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
    ),
    "vector_add_scalar": (
        vector_add_scalar_kernel,
        {"BLOCK_SIZE": 1024},
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
    ),
    "vector_add_vectorized": (
        vector_add_vectorized_kernel,
        {"BLOCK_SIZE": 1024},
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
    ),
    "reduction": (
        reduction_kernel,
        {"BLOCK_SIZE": 256},
        {"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
    ),
}

def compile_workload(workload_name="fused_add_relu", block_size=256, num_warps=4, num_stages=2, arch=86):
    if workload_name not in KERNELS:
        raise ValueError(f"Unknown workload: {workload_name}. Available: {list(KERNELS.keys())}")
    
    fn, constexprs, signature = KERNELS[workload_name]
    constexprs = dict(constexprs)
    constexprs["BLOCK_SIZE"] = block_size

    target = GPUTarget("cuda", arch, 32)

    src = triton.compiler.ASTSource(
        fn=fn,
        signature=signature,
        constexprs=constexprs,
    )
    
    # Compile with Triton compiler
    compiled = triton.compile(
        src,
        target=target,
        options={"num_warps": num_warps, "num_stages": num_stages}
    )

    asm_dict = compiled.asm
    metadata = compiled.metadata

    raw_src = inspect.getsource(fn.fn if hasattr(fn, "fn") else fn)

    ttir_str = str(asm_dict.get("ttir", ""))
    ttgir_str = str(asm_dict.get("ttgir", ""))
    llir_str = str(asm_dict.get("llir", ""))
    ptx_str = str(asm_dict.get("ptx", ""))

    result = {
        "workload": workload_name,
        "kernel_name": compiled.name,
        "python_source": raw_src,
        "ttir": ttir_str,
        "ttgir": ttgir_str,
        "llir": llir_str,
        "ptx": ptx_str,
        "num_warps": getattr(metadata, "num_warps", num_warps),
        "num_stages": getattr(metadata, "num_stages", num_stages),
        "shared_mem_bytes": getattr(metadata, "shared", 0) or 0,
        "register_count": getattr(metadata, "num_regs", 0) or 0,
        "target_arch": f"sm_{arch}",
    }
    return result

if __name__ == "__main__":
    workload = sys.argv[1] if len(sys.argv) > 1 else "fused_add_relu"
    block_sz = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    arch_val = int(sys.argv[3]) if len(sys.argv) > 3 else 86
    out_file = sys.argv[4] if len(sys.argv) > 4 else None

    res = compile_workload(workload, block_sz, arch=arch_val)
    json_data = json.dumps(res, indent=2)

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(json_data)
    
    print("\n__DRISHTI_TRITON_JSON_START__")
    print(json_data)
    print("__DRISHTI_TRITON_JSON_END__")
