"""
Vector Add Triton Kernel Example for Dṛṣṭi
==========================================
Demonstrates memory streaming and layout vectorization.
"""

import triton
import triton.language as tl


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


if __name__ == "__main__":
    print("Dṛṣṭi sample kernel: vector_add_kernel loaded successfully.")
