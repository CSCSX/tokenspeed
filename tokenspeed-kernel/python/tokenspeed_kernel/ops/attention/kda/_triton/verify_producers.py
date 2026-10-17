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

"""Convolution and gate projection for record-producing KDA verification."""

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit(do_not_specialize=["rows", "gate_programs"])
def _verify_conv_gate_kernel(
    raw,
    fa,
    fb,
    weights,
    pool,
    reads,
    convolved,
    gate,
    replay,
    rows,
    gate_programs,
    stride_raw,
    stride_fa,
    stride_fb,
    stride_pool,
    stride_reads,
    stride_replay,
    P: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    BC: tl.constexpr,
):
    program = tl.program_id(0)
    # Separate CTAs keep both producers parallel without a stream fork/join.
    if program < gate_programs:
        pm = program // triton.cdiv(P, BN)
        pn = program % triton.cdiv(P, BN)
        rr = (pm * BM + tl.arange(0, BM)).to(tl.int64)
        cc = pn * BN + tl.arange(0, BN)
        dd = tl.arange(0, BD)
        aa = tl.load(
            fa + rr[:, None] * stride_fa + dd[None, :],
            mask=(rr[:, None] < rows) & (dd[None, :] < D),
            other=0.0,
        )
        bb = tl.load(
            fb + cc[None, :] * stride_fb + dd[:, None],
            mask=(cc[None, :] < P) & (dd[:, None] < D),
            other=0.0,
        )
        # BF16 operands, FP32 accumulation, then one output rounding.
        projected = tl.dot(aa, bb)
        tl.store(
            gate + rr[:, None] * P + cc[None, :],
            projected,
            mask=(rr[:, None] < rows) & (cc[None, :] < P),
        )
    else:
        conv_program = program - gate_programs
        row = (conv_program // triton.cdiv(3 * P, BC)).to(tl.int64)
        block = conv_program % triton.cdiv(3 * P, BC)
        channels = block * BC + tl.arange(0, BC)
        mask = channels < 3 * P
        request, token = row // T, row % T
        page = tl.load(reads + request * stride_reads).to(tl.int64)
        safe_page = tl.maximum(page, 0)
        acc = tl.zeros((BC,), tl.float32)
        for tap in range(4):
            window = token + tap
            from_raw = window >= 3
            old = tl.load(
                pool + safe_page * stride_pool + channels * 3 + window,
                mask=mask & ~from_raw & (page >= 0),
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                raw + (request * T + window - 3) * stride_raw + channels,
                mask=mask & from_raw,
                other=0.0,
            )
            weight = tl.load(
                weights + channels * 4 + tap,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += tl.where(from_raw, value.to(tl.float32), old) * weight
            if replay is not None:
                if tap == 3:
                    tl.store(replay + row * stride_replay + channels, value, mask=mask)
        acc = acc * tl.sigmoid(acc)
        tl.store(convolved + row * (3 * P) + channels, acc, mask=mask)


def verify_conv_gate(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    read_indices: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    *,
    draft_token_num: int,
    replay_mixed_qkv: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = mixed_qkv.shape[0]
    if f_a_out.ndim != 2 or f_b_weight.ndim != 2:
        raise ValueError("gate projection operands must be matrices")
    channels, rank = f_b_weight.shape
    if channels <= 0 or rank <= 0:
        raise ValueError("gate projection dimensions must be positive")
    if draft_token_num <= 0 or rows % draft_token_num:
        raise ValueError("draft_token_num must evenly divide the projection rows")
    batch = rows // draft_token_num
    matrices = (
        (mixed_qkv, (rows, 3 * channels)),
        (f_a_out, (rows, rank)),
        (f_b_weight, (channels, rank)),
    )
    for tensor, shape in matrices:
        if tensor.shape != shape or tensor.stride(-1) != 1:
            raise ValueError(f"projection must have shape {shape} and last stride 1")
        if tensor.stride(0) < shape[1]:
            raise ValueError("projection rows must not overlap")
    if conv_weights.shape != (3 * channels, 4) or not conv_weights.is_contiguous():
        raise ValueError(f"conv_weights must be contiguous [{3 * channels}, 4]")
    if (
        conv_states.ndim != 3
        or conv_states.shape[1] < 3 * channels
        or conv_states.shape[2] != 3
        or conv_states.stride(1) != 3
        or conv_states.stride(2) != 1
    ):
        raise ValueError("conv_states must contain contiguous channel/tap planes")
    if read_indices.shape != (batch,) or read_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            f"read_indices must contain {batch} int32 or int64 page indices"
        )
    tensors = (mixed_qkv, conv_weights, conv_states, f_a_out, f_b_weight)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("KDA verify producer inputs must be BF16")
    if not mixed_qkv.is_cuda or any(
        t.device != mixed_qkv.device for t in (*tensors, read_indices)
    ):
        raise ValueError("KDA verify producer inputs must share a CUDA device")
    if replay_mixed_qkv is not None:
        if (
            replay_mixed_qkv.shape != mixed_qkv.shape
            or replay_mixed_qkv.stride(-1) != 1
            or replay_mixed_qkv.stride(0) < 3 * channels
            or replay_mixed_qkv.dtype != mixed_qkv.dtype
            or replay_mixed_qkv.device != mixed_qkv.device
        ):
            raise ValueError(
                "replay_mixed_qkv must match the input rows, dtype and device"
            )
        if (
            replay_mixed_qkv.data_ptr() == mixed_qkv.data_ptr()
            and replay_mixed_qkv.stride() == mixed_qkv.stride()
        ):
            # Even identical stores would race with other CTAs reading the window.
            replay_mixed_qkv = None
        elif rows:
            begin = replay_mixed_qkv.data_ptr()
            end = (
                begin
                + ((rows - 1) * replay_mixed_qkv.stride(0) + 3 * channels)
                * replay_mixed_qkv.element_size()
            )
            for tensor in (*tensors, read_indices):
                source_begin = tensor.data_ptr()
                source_end = (
                    source_begin
                    + (
                        1
                        + sum(
                            (size - 1) * stride
                            for size, stride in zip(
                                tensor.shape, tensor.stride(), strict=True
                            )
                        )
                    )
                    * tensor.element_size()
                )
                if begin < source_end and source_begin < end:
                    raise ValueError(
                        "replay_mixed_qkv must not overlap producer inputs"
                    )
    convolved = torch.empty(
        (rows, 3 * channels), dtype=mixed_qkv.dtype, device=mixed_qkv.device
    )
    gate = torch.empty((rows, channels), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    if rows:
        gate_programs = triton.cdiv(rows, 16) * triton.cdiv(channels, 32)
        _verify_conv_gate_kernel[
            (gate_programs + rows * triton.cdiv(3 * channels, 256),)
        ](
            mixed_qkv,
            f_a_out,
            f_b_weight,
            conv_weights,
            conv_states,
            read_indices,
            convolved,
            gate,
            replay_mixed_qkv,
            rows,
            gate_programs,
            mixed_qkv.stride(0),
            f_a_out.stride(0),
            f_b_weight.stride(0),
            conv_states.stride(0),
            read_indices.stride(0),
            0 if replay_mixed_qkv is None else replay_mixed_qkv.stride(0),
            P=channels,
            D=rank,
            T=draft_token_num,
            BM=16,
            BN=32,
            BD=max(16, triton.next_power_of_2(rank)),
            BC=256,
            num_warps=4,
            num_stages=1,
        )
    return convolved, gate
