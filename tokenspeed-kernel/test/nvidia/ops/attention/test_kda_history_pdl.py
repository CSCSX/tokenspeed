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

"""Numerical coverage for PDL producer/RecoverSSM/quantization dependencies."""

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kda import KdaReplayHistory, KdaReplayRecords
from tokenspeed_kernel.ops.attention.kda._triton.nvidia_history_decode import (
    fused_history_decode,
)
from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
    vllm_triton_kda_recoverssm_verify,
)
from tokenspeed_kernel.ops.quantization.triton import triton_gated_fp8_quantize
from tokenspeed_kernel.platform import current_platform, pdl_enabled


@triton.jit
def _publish_input(source, dest, count, ENABLE_PDL: tl.constexpr, BLOCK: tl.constexpr):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + offsets, offsets < count, other=0).to(tl.float32)
    # Keep producer work after the trigger so a premature consumer read sees
    # stale input. Both PDL settings perform exactly the same arithmetic.
    for _ in tl.range(256, loop_unroll_factor=1):
        values = values * 1.0001 + 0.0001
    tl.store(dest + offsets, values, offsets < count)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("history_length", [0, 8])
def test_history_pdl_chain(width: int, captured: bool, history_length: int) -> None:
    if not current_platform().is_hopper_plus:
        pytest.skip("PDL requires Hopper or newer NVIDIA hardware")
    torch.manual_seed(90737)
    batch, heads, dim, capacity, pages = 8, 12, 128, 8, 5
    channels = heads * dim
    source = torch.randn(batch * width, 3 * channels, device="cuda").bfloat16()
    weights = torch.randn(3 * channels, 4, device="cuda").bfloat16() * 0.2
    fa = torch.randn(batch * width, dim, device="cuda").bfloat16() * 0.2
    fb = torch.randn(channels, dim, device="cuda").bfloat16() * 0.05
    beta = torch.randn(1, batch * width, heads, device="cuda").bfloat16()
    gate = torch.randn(1, batch * width, heads, dim, device="cuda").bfloat16()
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    initial_state = torch.randn(pages, heads, dim, dim, device="cuda") * 0.01
    initial_conv = torch.randn(pages, 3 * channels, 3, device="cuda").bfloat16() * 0.1
    initial_u = torch.randn(pages, heads, capacity, dim, device="cuda") * 0.01
    initial_key = torch.randn(pages, heads, capacity, 2 * dim, device="cuda") * 0.01
    initial_decay = torch.full_like(initial_u, 0.98)
    lengths = torch.tensor(
        [0, history_length, 0, 0, 0], device="cuda", dtype=torch.int32
    )
    reads = torch.tensor([1, 2, 0, 0, 0, 0, 0, 0], device="cuda", dtype=torch.int32)
    writes = torch.tensor([3, 2, 0, 0, 0, 0, 0, 0], device="cuda", dtype=torch.int32)
    # Two active rows, one null/invalid row, then five empty rows. T=4 writes
    # zero output for the null row; fused T=1 skips its invalid record slot.
    qsl = (
        torch.tensor([0, 1, 2, 3, 3, 3, 3, 3, 3], device="cuda", dtype=torch.int32)
        * width
    )
    metadata = torch.stack(
        (
            torch.arange(batch, device="cuda"),
            torch.full((batch,), 7, device="cuda"),
            torch.full((batch,), 13, device="cuda"),
        ),
        dim=1,
    )
    metadata[2, 0] = -1
    output_rows = 2 if width == 1 else 3 * width
    output_gate = torch.randn(output_rows, channels, device="cuda").bfloat16()
    norm = torch.randn(dim, device="cuda").bfloat16()

    def make_chain(enable_pdl: bool):
        mixed = torch.zeros_like(source)
        state, conv = initial_state.clone(), initial_conv.clone()
        hu, hk, hd = initial_u.clone(), initial_key.clone(), initial_decay.clone()
        hl = lengths.clone()
        offsets = torch.full((batch,), -1, device="cuda", dtype=torch.int32)
        history = KdaReplayHistory(hu, hk, hd, hl, writes, offsets, False)
        u = torch.full((batch, heads, width, dim), -3.0, device="cuda")
        kg = torch.full(
            (batch, heads, width, 2 * dim),
            -4.0,
            device="cuda",
            dtype=torch.float32 if width == 1 else torch.bfloat16,
        )
        stamps = torch.full((batch, width, 2), -1, device="cuda", dtype=torch.int64)
        raw = torch.full_like(source, -5.0)
        out = torch.zeros_like(gate)
        saved = dict(
            state=state,
            conv=conv,
            hu=hu,
            hk=hk,
            hd=hd,
            hl=hl,
            offsets=offsets,
            correction=u,
            key_gate=kg,
            stamps=stamps,
            payload=raw,
            input=mixed,
        )

        def forward():
            pdl_enabled(enable_pdl)
            _publish_input[(triton.cdiv(source.numel(), 256),)](
                source,
                mixed,
                source.numel(),
                ENABLE_PDL=enable_pdl,
                BLOCK=256,
                launch_pdl=enable_pdl,
            )
            if width == 1:
                records = KdaReplayRecords(
                    correction=u,
                    key_gate=kg,
                    key_gate_format="normalized_key_decay",
                    metadata=metadata,
                    stamps=stamps,
                    query_start_loc=qsl,
                    history=history,
                )
                result = fused_history_decode(
                    mixed,
                    weights,
                    conv,
                    fa,
                    fb,
                    beta.view(batch, heads),
                    a,
                    bias,
                    state_pool=state,
                    read_indices=reads,
                    records=records,
                    replay_mixed_qkv=raw,
                    lower_bound=-5.0,
                    enable_pdl=enable_pdl,
                )
            else:
                q, k, v = (
                    part.view(1, batch * width, heads, dim)
                    for part in mixed.chunk(3, dim=-1)
                )
                result = vllm_triton_kda_recoverssm_verify(
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    a,
                    bias,
                    checkpoint_state=state,
                    correction_cache=u,
                    kg_cache=kg,
                    query_start_loc=qsl,
                    state_indices=reads,
                    spec_query_len=width,
                    lower_bound=-5.0,
                    out=out,
                    record_metadata=metadata,
                    record_stamps=stamps,
                    history=history,
                )
            active = result.view(batch * width, channels)[:output_rows]
            quant, scales = triton_gated_fp8_quantize(
                active,
                output_gate,
                norm_weight=norm,
                eps=1e-6,
                enable_pdl=enable_pdl,
            )
            saved.update(output=active, quant=quant, scales=scales)

        forward()
        forward()
        if captured:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            return graph.replay, saved
        return forward, saved

    previous_pdl = pdl_enabled()
    try:
        control, expected = make_chain(False)
        candidate, actual = make_chain(True)
        for step in range(16):
            source.copy_(torch.randn_like(source) * (0.1 + step * 0.05))
            control()
            candidate()
            for name in expected:
                left, right = expected[name], actual[name]
                if name == "quant":
                    left, right = left.view(torch.uint8), right.view(torch.uint8)
                torch.testing.assert_close(right, left, rtol=0, atol=0, msg=name)
    finally:
        pdl_enabled(previous_pdl)
