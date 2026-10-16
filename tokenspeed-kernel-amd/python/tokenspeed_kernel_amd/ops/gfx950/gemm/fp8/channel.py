# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Channel-scaled FP8 storage with exact BF16 operands and FP32 accumulation."""

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton


def _mm_metadata(grid, kernel, args):
    m, n, k = args["M"], args["N"], args["K"]
    return {
        "name": kernel.name,
        "flops16": 2 * m * n * k,
        "bytes": (m + n) * k + m * n * args["SPLITS"] * args["c_ptr"].element_size(),
    }


def _reduce_metadata(grid, kernel, args):
    values = args["M"] * args["N"]
    return {
        "name": kernel.name,
        "flops32": values * (args["SPLITS"] + 1),
        "bytes": values * (4 * args["SPLITS"] + args["c_ptr"].element_size()),
    }


@gluon.jit
def _scale(acc, row_scale, column_scale):
    # Keep both FP32 rounding points even when the final store is FP16/BF16.
    scaled = gl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        constraints="=v,v,v",
        args=[acc, row_scale],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    return gl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        constraints="=v,v,v",
        args=[scaled, column_scale],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit(launch_metadata=_mm_metadata, do_not_specialize=("M",))
def gluon_mm_fp8_channel_gfx950(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_as,
    stride_bs,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLITS: gl.constexpr,
):
    tile = gl.program_id(0)
    split = gl.program_id(1)
    pid_m = tile // gl.cdiv(N, BLOCK_N)
    pid_n = tile % gl.cdiv(N, BLOCK_N)
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 4] if BLOCK_M == 16 else [2, 2],
    )
    load_a: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    load_b: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
    rows = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_a))
    cols = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, load_b))
    ka = gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_a))
    kb = gl.arange(0, BLOCK_K, gl.SliceLayout(1, load_b))
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)
    TILES: gl.constexpr = gl.cdiv(K, BLOCK_K * SPLITS)
    for step in range(TILES):
        start = (split * TILES + step) * BLOCK_K
        a = gl.load(
            a_ptr
            + rows[:, None].to(gl.int64) * stride_am
            + (start + ka[None, :]).to(gl.int64) * stride_ak,
            (rows < M)[:, None] & (start + ka < K)[None, :],
            other=0.0,
        ).to(gl.bfloat16)
        b = gl.load(
            b_ptr
            + (start + kb[:, None]).to(gl.int64) * stride_bk
            + cols[None, :].to(gl.int64) * stride_bn,
            (start + kb < K)[:, None] & (cols < N)[None, :],
            other=0.0,
        ).to(gl.bfloat16)
        # Every finite E4M3 value is exact in BF16. This avoids the reduced
        # internal product accumulation of gfx950's native FP8 MFMA.
        acc = gl.amd.cdna3.mfma(
            gl.convert_layout(a, gl.DotOperandLayout(0, mfma, 8)),
            gl.convert_layout(b, gl.DotOperandLayout(1, mfma, 8)),
            acc,
        )
    m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    if SPLITS == 1:
        acc = _scale(
            acc,
            gl.load(a_scale_ptr + m * stride_as, m < M, other=0)[:, None],
            gl.load(b_scale_ptr + n * stride_bs, n < N, other=0)[None, :],
        )
    gl.store(
        c_ptr
        + split.to(gl.int64) * M * N
        + m[:, None].to(gl.int64) * stride_cm
        + n[None, :].to(gl.int64) * stride_cn,
        acc.to(c_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        (m < M)[:, None] & (n < N)[None, :],
    )


@gluon.jit(launch_metadata=_reduce_metadata, do_not_specialize=("M",))
def gluon_mm_fp8_channel_reduce_gfx950(
    partial_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    M,
    N: gl.constexpr,
    stride_as,
    stride_bs,
    stride_cm,
    stride_cn,
    SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    index = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout)
    mask = index < M * N
    acc = gl.full((BLOCK,), 0, gl.float32, layout)
    for split in gl.static_range(SPLITS):
        acc += gl.load(partial_ptr + split * M * N + index, mask, other=0)
    m, n = index // N, index % N
    acc = _scale(
        acc,
        gl.load(a_scale_ptr + m * stride_as, mask, other=0),
        gl.load(b_scale_ptr + n * stride_bs, mask, other=0),
    )
    gl.store(
        c_ptr + m.to(gl.int64) * stride_cm + n.to(gl.int64) * stride_cn,
        acc.to(c_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask,
    )


def _config(m: int, n: int, k: int) -> tuple[int, int, int]:
    block_m = 16 if m <= 16 else 32 if m <= 256 else 64
    block_n = 64 if m <= 256 else 128
    tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    wanted = max(1, 512 // tiles)
    splits = next(
        split
        for split in (8, 4, 2, 1)
        if split <= wanted and triton.cdiv(k, 128) % split == 0
    )
    return block_m, block_n, splits


def supports_gluon_mm_fp8_channel_gfx950(m: int, n: int, k: int) -> bool:
    """Limit automatic selection to measured projection geometries."""
    return 0 <= m <= 8192 and (n, k) in {
        (1792, 5120),
        (5120, 768),
        (5120, 5120),
        (5120, 2560),
    }


def launch_gluon_mm_fp8_channel_gfx950(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute channel-scaled E4M3 ``a @ b`` with BF16 MFMA operands.

    Args:
        a: E4M3 activations ``[M, K]``.
        b: E4M3 weights ``[K, N]``, including transposed weight views.
        a_scales: FP32 per-row ``[M, 1]`` or scalar dequantization scales.
        b_scales: FP32 per-column ``[N, 1]`` or scalar dequantization scales.
        out_dtype: BF16, FP16 or FP32 output dtype.
        out: Optional caller-owned ``[M, N]`` destination.

    Returns:
        The destination, scaled in FP32 before its final dtype conversion.
        Split-K partials sum in fixed split order, not atomic arrival order.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("channel FP8 MM requires compatible rank-two operands")
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise TypeError("channel FP8 MM requires E4M3 operands")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("channel FP8 MM requires BF16, FP16 or FP32 output")
    if not a.is_cuda or any(t.device != a.device for t in (b, a_scales, b_scales)):
        raise ValueError("channel FP8 MM requires colocated GPU inputs")
    m, k = a.shape
    n = b.shape[1]
    if n < 1 or k < 1:
        raise ValueError("channel FP8 MM requires positive N and K")
    for scales, rows in ((a_scales, m), (b_scales, n)):
        if scales.dtype != torch.float32 or scales.ndim not in (1, 2):
            raise TypeError("channel FP8 MM requires FP32 scale vectors")
        if scales.numel() not in (1, rows) or (
            scales.ndim == 2 and scales.shape[1] != 1
        ):
            raise ValueError("channel FP8 MM requires one scale per row or one scalar")
    output = (
        torch.empty((m, n), dtype=out_dtype, device=a.device) if out is None else out
    )
    if output.shape != (m, n) or output.dtype != out_dtype or output.device != a.device:
        raise ValueError("channel FP8 MM output must match shape, dtype and device")
    if output.stride(1) != 1 or (m > 1 and output.stride(0) < n):
        raise ValueError(
            "channel FP8 MM output requires nonoverlapping contiguous rows"
        )
    if m == 0:
        return output
    block_m, block_n, splits = _config(m, n, k)
    partials = (
        torch.empty((splits, m, n), dtype=torch.float32, device=a.device)
        if splits > 1
        else output
    )
    stride_as = a_scales.stride(0) if a_scales.numel() == m else 0
    stride_bs = b_scales.stride(0) if b_scales.numel() == n else 0
    gluon_mm_fp8_channel_gfx950[
        (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), splits)
    ](
        a,
        b,
        a_scales,
        b_scales,
        partials,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        stride_as,
        stride_bs,
        *partials.stride()[-2:],
        block_m,
        block_n,
        128,
        splits,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if splits > 1:
        gluon_mm_fp8_channel_reduce_gfx950[(triton.cdiv(m * n, 256),)](
            partials,
            a_scales,
            b_scales,
            output,
            m,
            n,
            stride_as,
            stride_bs,
            *output.stride(),
            splits,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
