"""
Fused Add + ReLU Triton Kernel Example for Dṛṣṭi
================================================
Demonstrates elementwise operation fusion with Triton JIT.
"""

import triton
import triton.language as tl


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


if __name__ == "__main__":
    print("Dṛṣṭi sample kernel: fused_add_relu_kernel loaded successfully.")
