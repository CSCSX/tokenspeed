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


"""Compact asynchronous channel-FP8 operands with BF16 MFMA on gfx950."""

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16._grid import get_pids


def _launch_metadata(grid, kernel, args):
    return {
        "name": kernel.name,
        "flops16": 4
        * args["TOKENS"]
        * args["TOPK"]
        * args["HIDDEN"]
        * args["INTERMEDIATE"],
    }


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("TOKENS", "EM", "GRID"))
def gluon_fp8_channel_moe_stage1_gfx950(
    a_ptr,
    b_ptr,
    row_scale_ptr,
    column_scale_ptr,
    sorted_ids_ptr,
    sorted_experts_ptr,
    valid_ptr,
    out_ptr,
    TOKENS,
    EM,
    GRID,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    WARPS: gl.constexpr,
):
    pid_m, pid_n = get_pids(
        gl.cdiv(EM, BM),
        gl.cdiv(INTERMEDIATE, BN),
        GRID,
        1 if BM == 16 else 8,
        4 if BM == 16 else 8,
    )
    if pid_m * BM >= gl.load(valid_ptr):
        return
    expert = gl.load(sorted_experts_ptr + pid_m)
    if expert == -1:
        return
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, WARPS] if BM == 16 or WARPS == 1 else [2, WARPS // 2],
    )
    al: gl.constexpr = gl.BlockedLayout(
        [1, 16], [1024 // BK, BK // 16], [WARPS, 1], [1, 0]
    )
    bl: gl.constexpr = gl.BlockedLayout(
        [16, 1], [BK // 16, 1024 // BK], [1, WARPS], [0, 1]
    )
    ashared: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[1, 0])
    bshared: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[0, 1])
    adot: gl.constexpr = gl.DotOperandLayout(0, mfma, 8)
    bdot: gl.constexpr = gl.DotOperandLayout(1, mfma, 8)
    sm_a = gl.allocate_shared_memory(gl.uint8, [2, BM, BK], ashared)
    sm_g = gl.allocate_shared_memory(gl.uint8, [2, BK, BN], bshared)
    sm_u = gl.allocate_shared_memory(gl.uint8, [2, BK, BN], bshared)
    row = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    packed = gl.load(sorted_ids_ptr + row, row < EM, other=TOKENS)
    token = packed & 0xFFFFFF
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    ao = token[:, None] * HIDDEN + ak[None, :]
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    bn = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    go = bk[:, None] + bn[None, :] * HIDDEN
    uo = go + INTERMEDIATE * HIDDEN
    bp = b_ptr + expert.to(gl.int64) * (2 * INTERMEDIATE * HIDDEN)
    gp = bp
    ap = a_ptr
    cdna4_async_copy.buffer_load_to_shared(
        sm_a.index(0), ap, ao, mask=(token < TOKENS)[:, None]
    )
    cdna4_async_copy.buffer_load_to_shared(sm_g.index(0), gp, go)
    cdna4_async_copy.buffer_load_to_shared(sm_u.index(0), gp, uo)
    cdna4_async_copy.commit_group()
    ap += BK
    gp += BK
    gate_acc = gl.zeros((BM, BN), gl.float32, mfma)
    up_acc = gl.zeros((BM, BN), gl.float32, mfma)
    for step in range(HIDDEN // BK - 1):
        current = step % 2
        following = 1 - current
        cdna4_async_copy.buffer_load_to_shared(
            sm_a.index(following), ap, ao, mask=(token < TOKENS)[:, None]
        )
        cdna4_async_copy.buffer_load_to_shared(sm_g.index(following), gp, go)
        cdna4_async_copy.buffer_load_to_shared(sm_u.index(following), gp, uo)
        cdna4_async_copy.commit_group()
        cdna4_async_copy.wait_group(1)
        a = (
            sm_a.index(current)
            .load(adot)
            .to(gl.float8e4nv, bitcast=True)
            .to(gl.bfloat16)
        )
        g = (
            sm_g.index(current)
            .load(bdot)
            .to(gl.float8e4nv, bitcast=True)
            .to(gl.bfloat16)
        )
        u = (
            sm_u.index(current)
            .load(bdot)
            .to(gl.float8e4nv, bitcast=True)
            .to(gl.bfloat16)
        )
        gate_acc = gl.amd.cdna3.mfma(a, g, gate_acc)
        up_acc = gl.amd.cdna3.mfma(a, u, up_acc)
        ap += BK
        gp += BK
    cdna4_async_copy.wait_group(0)
    last = (HIDDEN // BK - 1) % 2
    a = sm_a.index(last).load(adot).to(gl.float8e4nv, bitcast=True).to(gl.bfloat16)
    g = sm_g.index(last).load(bdot).to(gl.float8e4nv, bitcast=True).to(gl.bfloat16)
    u = sm_u.index(last).load(bdot).to(gl.float8e4nv, bitcast=True).to(gl.bfloat16)
    gate_acc = gl.amd.cdna3.mfma(a, g, gate_acc)
    up_acc = gl.amd.cdna3.mfma(a, u, up_acc)
    m = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, mfma))
    p = gl.load(sorted_ids_ptr + m, m < EM, other=TOKENS)
    t = p & 0xFFFFFF
    slot = p >> 24
    n = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, mfma))
    rs = gl.load(row_scale_ptr + t, t < TOKENS, other=0)
    cs = column_scale_ptr + expert.to(gl.int64) * (2 * INTERMEDIATE) + n
    gs, us = gl.load(cs), gl.load(cs + INTERMEDIATE)
    gate_acc = (gate_acc * rs[:, None]) * gs[None, :]
    up_acc = (up_acc * rs[:, None]) * us[None, :]
    result = gate_acc * (1.0 / (1.0 + gl.exp(-gate_acc))) * up_acc
    offset = (t.to(gl.int64) * TOPK + slot)[:, None] * INTERMEDIATE + n[None, :]
    gl.store(out_ptr + offset, result, (t < TOKENS)[:, None])


def launch_gluon_fp8_channel_moe_stage1_gfx950(
    codes,
    weights,
    row_scales,
    column_scales,
    sorted_ids,
    sorted_experts,
    valid,
    output,
    topk,
    *,
    block_m,
):
    """Run grouped gate/up and SwiGLU from contiguous E4M3 codes.

    Inputs are token codes [M, H], expert weights [E, 2*I, H], FP32
    token/channel scales, and the existing packed-route buffers. Write
    BF16 activations [M*topk, I] in original token-slot order.

    ``sorted_ids`` packs ``(slot << 24) | token``; ``sorted_experts`` names
    each tile's expert and ``valid`` holds the padded route length on device.
    ``block_m`` must match the routing alignment. Return the compiled kernel.
    """
    tokens, hidden = codes.shape
    experts, twice_intermediate, weight_hidden = weights.shape
    intermediate = twice_intermediate // 2
    # Narrow one-wave tiles expose more independent work for sparse experts.
    block_n, block_k, warps = (16, 512, 1) if block_m == 16 else (64, 128, 4)
    assert codes.dtype == weights.dtype == torch.float8_e4m3fn
    assert row_scales.dtype == column_scales.dtype == torch.float32
    assert block_m in (16, 32, 64) and hidden == weight_hidden
    assert hidden % block_k == 0 and intermediate % block_n == 0
    assert row_scales.shape == (tokens, 1)
    assert column_scales.shape == (experts, twice_intermediate)
    assert output.shape == (tokens * topk, intermediate)
    assert output.dtype == torch.bfloat16
    assert all(
        x.is_contiguous() for x in (codes, weights, row_scales, column_scales, output)
    )
    grid = triton.cdiv(sorted_ids.numel(), block_m) * triton.cdiv(intermediate, block_n)
    return gluon_fp8_channel_moe_stage1_gfx950[(grid,)](
        codes.view(torch.uint8),
        weights.view(torch.uint8),
        row_scales,
        column_scales,
        sorted_ids,
        sorted_experts,
        valid,
        output,
        tokens,
        sorted_ids.numel(),
        grid,
        hidden,
        intermediate,
        topk,
        block_m,
        block_n,
        block_k,
        warps,
        num_warps=warps,
        enable_fp_fusion=False,
    )
