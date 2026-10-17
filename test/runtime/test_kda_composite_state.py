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

"""Behavioral cache-copy and reuse tests for composite KDA endpoints."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import kda_state_fields
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec


@pytest.mark.parametrize("capacity", [4, 8, 32])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_composite_endpoint_survives_cache_copy_and_reuse(
    capacity: int, use_tensor_cores: bool, captured: bool
) -> None:
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    heads, dim = 2, 128
    full = CacheGroupSpec(
        group_id="full_attention",
        attn_kind="full_history",
        rows_per_page=128,
        entry_stride_tokens=1,
        replayable=False,
    )
    state = CacheGroupSpec(
        group_id="linear_attention", attn_kind="state", replayable=False
    )
    groups = (
        (
            full,
            (CacheFieldSpec("layer.0.latent_kv", "slot.0", (128, 1, 576), "bfloat16"),),
        ),
        (
            state,
            kda_state_fields(
                1,
                "slot.0",
                conv_shape=(3 * heads * dim, 3),
                recurrent_shape=(heads, dim, dim),
                history_capacity=capacity,
            ),
        ),
    )
    plan = pack(
        groups,
        prefix_granularity=128,
        cache_blocks_per_lcm_block={"full_attention": 4, "linear_attention": 1},
        max_padding_fraction=float("inf"),
    ).bind(4)
    arena = CacheArena(plan, device, cache_group_specs=(full, state))
    pool = HybridKDATokenToKVPool(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        quant_method=None,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=2,
        rank=0,
        layer_types=("full_attention", "linear_attention"),
    )
    names = (
        "conv_state",
        "recurrent_state",
        "replay_correction",
        "replay_weighted_key",
        "replay_decay",
        "replay_length",
    )
    fields = {name: pool.get_component(1, name) for name in names}
    torch.manual_seed(8123)
    # State blocks 1..3 own parents 0..2. Full-history blocks 13..16 own
    # parent 3; groups must never write through aliases of an occupied parent.
    for name, tensor in fields.items():
        if name == "replay_length":
            tensor[1:4].copy_(
                torch.tensor([2, 3, 1], dtype=tensor.dtype, device=device)[:, None]
            )
        else:
            tensor[1:4].copy_(
                torch.randn(tensor[1:4].shape, device=device, dtype=tensor.dtype) * 0.1
            )
    latent = arena.compute_field("layer.0.latent_kv")[13:17]
    latent.fill_(0.25)
    before = {name: tensor.clone() for name, tensor in fields.items()}

    def endpoint(block: int) -> torch.Tensor:
        current = fields["recurrent_state"][block].clone()
        length = int(fields["replay_length"][block, 0].item())
        if length:
            current *= fields["replay_decay"][block, :, None, :]
            for token in range(length):
                current += (
                    fields["replay_correction"][block, :, token, :, None]
                    * fields["replay_weighted_key"][block, :, token, None, :]
                )
        return current

    first, second = endpoint(1), endpoint(2)
    from tokenspeed_kernel.ops.attention.kda import (
        kda_clear_history,
        kda_materialize_history,
    )

    indices = torch.tensor([1, 2, 0, -1], dtype=torch.int32, device=device)
    materialized = torch.empty((4, heads, dim, dim), device=device, dtype=torch.float32)
    lengths = fields["replay_length"].squeeze(-1)

    def materialize() -> None:
        kda_materialize_history(
            fields["recurrent_state"],
            fields["replay_correction"],
            fields["replay_weighted_key"],
            fields["replay_decay"],
            lengths,
            indices,
            out=materialized,
            use_tensor_cores=use_tensor_cores,
        )

    materialize()
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            materialize()
        graph.replay()
    expected = torch.stack(
        (first, second, torch.zeros_like(first), torch.zeros_like(first))
    )
    torch.testing.assert_close(materialized, expected, atol=3e-7, rtol=3e-5)
    # Copying a published source and reusing its old destination in the same
    # plan must preserve both original endpoints, including all factors.
    arena.copy_blocks(
        {"linear_attention": np.asarray([[1, 2], [2, 3]], dtype=np.int64)}
    )
    for name, tensor in fields.items():
        torch.testing.assert_close(tensor[1], before[name][1], atol=0, rtol=0)
        torch.testing.assert_close(tensor[2], before[name][1], atol=0, rtol=0)
        torch.testing.assert_close(tensor[3], before[name][2], atol=0, rtol=0)
    torch.testing.assert_close(endpoint(2), first, atol=0, rtol=0)
    torch.testing.assert_close(endpoint(3), second, atol=0, rtol=0)

    # A restored request may receive new block IDs. Captured materialization
    # must consume those current IDs, rather than a capture-time source.
    indices[:2].copy_(torch.tensor([3, 2], device=device, dtype=torch.int32))
    if captured:
        graph.replay()
    else:
        materialize()
    torch.testing.assert_close(
        materialized,
        torch.stack((second, first, torch.zeros_like(first), torch.zeros_like(first))),
        atol=3e-7,
        rtol=3e-5,
    )

    # Prefill stores a complete dense endpoint, then clears its length. Stale
    # factors must no longer contribute, and padding must not clear a source.
    fields["recurrent_state"][2].copy_(first)
    destinations = torch.tensor([2, 0, -1], device=device, dtype=torch.int32)
    kda_clear_history(lengths, destinations)
    torch.testing.assert_close(endpoint(2), first, atol=0, rtol=0)
    for name in ("replay_correction", "replay_weighted_key", "replay_decay"):
        torch.testing.assert_close(fields[name][2], before[name][1], atol=0, rtol=0)
    if captured:
        graph.replay()
    else:
        materialize()
    torch.testing.assert_close(materialized[1], first, atol=0, rtol=0)

    pool.zero_new_blocks({"linear_attention": np.asarray([2], dtype=np.int64)})
    for name, tensor in fields.items():
        torch.testing.assert_close(
            tensor[2], torch.zeros_like(tensor[2]), atol=0, rtol=0
        )
        torch.testing.assert_close(tensor[1], before[name][1], atol=0, rtol=0)
        torch.testing.assert_close(tensor[3], before[name][2], atol=0, rtol=0)
    torch.testing.assert_close(endpoint(2), torch.zeros_like(first), atol=0, rtol=0)
    torch.testing.assert_close(latent, torch.full_like(latent, 0.25), atol=0, rtol=0)
