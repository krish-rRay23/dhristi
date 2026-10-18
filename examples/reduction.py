"""
Reduction Triton Kernel Example for Dṛṣṭi
=========================================
Demonstrates memory reduction and block-level accumulation.
"""

import triton
import triton.language as tl


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


if __name__ == "__main__":
    print("Dṛṣṭi sample kernel: reduction_kernel loaded successfully.")
