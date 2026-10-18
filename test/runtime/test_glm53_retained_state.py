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

"""Numerical GLM checkpoint coverage, including test-only retained layouts."""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_glm53_flash import (
    HybridGlm53FlashTokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.glm53_flash import (
    GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
    Glm53FlashPoolOptions,
    declare_glm53_flash_groups,
    glm53_flash_packing_counts,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import (
    kda_replay_history_capacity,
    kda_state_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)


@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("draft_layers", [0, 1])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("history_capacity", [0, 8])
def test_glm53_retained_endpoint_survives_copy_and_reuse(
    tp_size: int,
    cache_dtype: torch.dtype,
    draft_layers: int,
    captured: bool,
    history_capacity: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    layer_types = (LINEAR_ATTENTION,) * 3 + (FULL_ATTENTION,)
    dsa = DSAConfig(
        backend_name="dsa",
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=256,
        attn_tp_size=tp_size,
        cache_layer_types=layer_types,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        scaling=256**-0.5,
        kv_cache_dim=512,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        index_k_format="fp8_scaled",
        index_kpool=4,
    )
    linear = LinearAttnConfig(
        num_k_heads=64,
        num_v_heads=64,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        layer_ids=(0, 1, 2),
        tp_size=tp_size,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=cache_dtype,
        kv_cache_quant_method="none",
        prefix_granularity=GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
        context_len=4096,
        max_bs=8,
        speculative_num_steps=3 if draft_layers else 0,
        speculative_num_draft_tokens=4 if draft_layers else 1,
        components=(dsa, linear),
    )
    if history_capacity and not kda_replay_history_capacity(config):
        pytest.skip("selected device kernels do not retain KDA history")
    draft_config = (
        replace(
            config,
            is_draft=True,
            components=(replace(dsa, cache_layer_types=(FULL_ATTENTION,)),),
        )
        if draft_layers
        else None
    )
    groups = declare_glm53_flash_groups(
        4,
        attn_config=config,
        draft_attn_config=draft_config,
        draft_layers=draft_layers,
    )
    if history_capacity:
        # Exercise composite packing without enabling retained history in GLM.
        groups = tuple(
            (
                spec,
                (
                    tuple(
                        state_field
                        for field in fields
                        if field.field_id.endswith(".conv_state")
                        for state_field in kda_state_fields(
                            int(field.field_id.split(".")[1]),
                            field.plane_id,
                            conv_shape=linear.conv_state_shape,
                            recurrent_shape=linear.temporal_state_shape,
                            history_capacity=history_capacity,
                        )
                    )
                    if spec.attn_kind == "state"
                    else fields
                ),
            )
            for spec, fields in groups
        )
    packing = glm53_flash_packing_counts(
        tp_size=tp_size, mla_element_size=cache_dtype.itemsize, groups=groups
    )
    plan = pack(
        groups,
        prefix_granularity=GLM53_FLASH_LOGICAL_BLOCK_TOKENS,
        cache_blocks_per_lcm_block=packing,
        alignment=256,
        max_padding_fraction=float("inf"),
    ).bind(4)
    arena = CacheArena(
        plan, "cuda", cache_group_specs=tuple(spec for spec, _ in groups)
    )
    dsa_layers = tuple(range(3, 4 + draft_layers))
    pool = HybridGlm53FlashTokenToKVPool(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=cache_dtype,
        quant_method=None,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        layer_num=4 + draft_layers,
        rank=0,
        layer_types=layer_types + (FULL_ATTENTION,) * draft_layers,
        pool_options=Glm53FlashPoolOptions(
            index_kpool=4,
            tail_extra_slots=0,
            index_head_dim=128,
            num_request_slots=10,
            dsa_layer_ids=dsa_layers,
        ),
    )
    backend = KdaAttnBackend(
        config, dsa, enable_prefill_graph=False, kda_backend="cutedsl_kda"
    )
    backend.set_kv_pool(pool)
    names = ("conv_state", "recurrent_state")
    if history_capacity:
        names += (
            "replay_correction",
            "replay_key",
            "replay_decay",
            "replay_length",
        )
    fields = {name: pool.get_component(0, name) for name in names}
    torch.manual_seed(5317)
    for name, tensor in fields.items():
        if name == "replay_length":
            tensor[0].fill_(7)
            tensor[1:4, 0].copy_(
                torch.tensor([3, 8, 1], dtype=tensor.dtype, device="cuda")
            )
        else:
            tensor[0].fill_(0.0625)
            tensor[1:4].copy_(
                torch.randn(tensor[1:4].shape, dtype=tensor.dtype, device="cuda") * 0.01
            )
    if history_capacity:
        fields["replay_decay"][1:4].add_(0.95)

    # State blocks 1..3 own the first three parents. The fourth belongs to
    # full attention; its latent and index pages must survive state operations.
    full_pages = packing[FULL_ATTENTION]
    guard_rows = slice(1 + 3 * full_pages, 1 + 4 * full_pages)
    guards = []
    for layer_id in dsa_layers:
        for component in ("latent_kv", "index_k"):
            view = arena.compute_field(f"layer.{layer_id}.{component}")[guard_rows]
            view.view(torch.uint8).fill_(0x5A)
            guards.append((view.view(torch.uint8), view.view(torch.uint8).clone()))
        _, tail_k, tail_gate = pool.get_kpool_buffers(layer_id)
        for tail in (tail_k, tail_gate):
            tail.fill_(0.125)
            guards.append((tail, tail.clone()))
    before = {name: tensor.clone() for name, tensor in fields.items()}

    def endpoint(block: int) -> torch.Tensor:
        state = before["recurrent_state"][block].clone()
        length = int(before["replay_length"][block, 0]) if history_capacity else 0
        for token in range(length):
            state *= before["replay_decay"][block, :, token, None, :]
            state += (
                before["replay_correction"][block, :, token, :, None]
                * before["replay_key"][block, :, token, None, :]
            )
        return state

    indices = torch.tensor([1, 2, 0, -1], dtype=torch.int32, device="cuda")
    destinations = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    output = torch.empty(
        (4, *linear.temporal_state_shape), dtype=torch.float32, device="cuda"
    )

    def materialize() -> None:
        nonlocal output
        output, _ = backend._prepare_prefill_state_inputs(
            0, fields["conv_state"], fields["recurrent_state"], indices, destinations
        )

    materialize()
    graph = torch.cuda.CUDAGraph() if captured else None
    if graph is not None:
        with torch.cuda.graph(graph):
            materialize()
        graph.replay()
    first, second = endpoint(1), endpoint(2)
    zeros = torch.zeros_like(first)
    torch.testing.assert_close(
        output, torch.stack((first, second, zeros, zeros)), atol=3e-7, rtol=3e-5
    )
    original = output.clone()
    group_id = plan.field("layer.0.recurrent_state").group_id
    arena.copy_blocks({group_id: np.asarray([[1, 2], [2, 3]], dtype=np.int64)})
    for name, tensor in fields.items():
        torch.testing.assert_close(tensor[0:2], before[name][0:2], atol=0, rtol=0)
        torch.testing.assert_close(tensor[2], before[name][1], atol=0, rtol=0)
        torch.testing.assert_close(tensor[3], before[name][2], atol=0, rtol=0)
    indices[:2].copy_(torch.tensor([3, 2], dtype=torch.int32, device="cuda"))
    destinations.copy_(torch.tensor([3, 2, 1, 4], dtype=torch.int32, device="cuda"))
    if graph is not None:
        graph.replay()
    else:
        materialize()
    torch.testing.assert_close(output, original[[1, 0, 2, 3]], atol=0, rtol=0)

    pool.zero_new_blocks({group_id: np.asarray([2], dtype=np.int64)})
    for name, tensor in fields.items():
        torch.testing.assert_close(
            tensor[2], torch.zeros_like(tensor[2]), atol=0, rtol=0
        )
        torch.testing.assert_close(tensor[0:2], before[name][0:2], atol=0, rtol=0)
        torch.testing.assert_close(tensor[3], before[name][2], atol=0, rtol=0)
    if graph is not None:
        graph.replay()
    else:
        materialize()
    torch.testing.assert_close(
        output, torch.stack((original[1], zeros, zeros, zeros)), atol=0, rtol=0
    )
    for view, saved in guards:
        torch.testing.assert_close(view, saved, atol=0, rtol=0)
