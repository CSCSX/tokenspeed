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

"""Retained KDA verify and accepted commits through real cache/backend objects."""

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

register_cuda_ci(est_time=120, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kda import (
    kda_state_fields,
    kda_verify_workspace_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

H, D, FA = 2, 128, 32
CHANNELS = 3 * H * D
STATE_GROUPS = ("state_a", "state_b")
FIELD_NAMES = (
    "conv_state",
    "recurrent_state",
    "replay_correction",
    "replay_key",
    "replay_decay",
    "replay_length",
)


def _groups(capacity: int):
    full = CacheGroupSpec(
        group_id="full_attention",
        attn_kind="full_history",
        rows_per_page=128,
        entry_stride_tokens=1,
        replayable=False,
    )
    groups = [
        (
            full,
            (CacheFieldSpec("layer.0.latent_kv", "slot.0", (128, 1, 576), "bfloat16"),),
        )
    ]
    for layer, group_id in enumerate(STATE_GROUPS, start=1):
        groups.append(
            (
                CacheGroupSpec(group_id=group_id, attn_kind="state", replayable=False),
                kda_state_fields(
                    layer,
                    "slot.0",
                    conv_shape=(CHANNELS, 3),
                    recurrent_shape=(H, D, D),
                    history_capacity=capacity,
                ),
            )
        )
    return tuple(groups)


def _pool(capacity: int) -> HybridKDATokenToKVPool:
    groups = _groups(capacity)
    plan = pack(
        groups,
        prefix_granularity=128,
        cache_blocks_per_lcm_block={"full_attention": 4, "state_a": 1, "state_b": 1},
        max_padding_fraction=float("inf"),
    ).bind(24)
    arena = CacheArena(
        plan, "cuda", cache_group_specs=tuple(group for group, _ in groups)
    )
    return HybridKDATokenToKVPool(
        arena=arena,
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        quant_method=None,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=3,
        rank=0,
        layer_types=("full_attention", "linear_attention", "linear_attention"),
    )


def _endpoint(pool: HybridKDATokenToKVPool, layer: int, page: int) -> torch.Tensor:
    state = pool.get_component(layer, "recurrent_state")[page].clone()
    length = int(pool.get_component(layer, "replay_length")[page, 0])
    for token in range(length):
        state *= pool.get_component(layer, "replay_decay")[page, :, token, None, :]
        u = pool.get_component(layer, "replay_correction")[page, :, token, :, None]
        key = pool.get_component(layer, "replay_key")[page, :, token, None, :]
        state += u * key
    return state


def _reference(inputs, params, state, conv):
    """Use fused FP32 T1 decode or the independent BF16 multi-token oracle."""
    width = inputs["mixed_qkv"].shape[0]
    if width == 1:
        from tokenspeed_kernel.ops.attention.kda import try_kda_fused_paged_decode

        # Standard decode retains FP32 producer intermediates. A width-one
        # split BF16 verify is a different numerical reference.
        reference_state = state.unsqueeze(0).clone()
        reference_conv = conv.unsqueeze(0).clone()
        page = torch.zeros(1, device="cuda", dtype=torch.int32)
        output = try_kda_fused_paged_decode(
            inputs["mixed_qkv"],
            params["conv_weights"],
            reference_conv,
            inputs["f_a_out"],
            params["f_b_weight"],
            inputs["beta_raw"],
            params["A_log"],
            params["dt_bias"],
            state_pool=reference_state,
            read_indices=page,
            write_indices=page,
            num_heads=H,
            head_dim=D,
            cu_seqlens=torch.arange(2, device="cuda", dtype=torch.int32),
            lower_bound=-5.0,
            recurrent_layout="v_major",
        )
        assert output is not None
        return output.reshape(1, H, D), [reference_state[0]], [reference_conv[0]]
    window = torch.cat((conv.T, inputs["mixed_qkv"]), dim=0)
    projected = []
    for token in range(width):
        accum = torch.zeros(CHANNELS, device="cuda")
        for tap in range(4):
            accum = torch.addcmul(
                accum,
                window[token + tap].float(),
                params["conv_weights"][:, tap].float(),
            )
        projected.append(torch.nn.functional.silu(accum).bfloat16())
    q, key, value = torch.stack(projected).view(width, 3, H, D).unbind(1)
    q = (
        q.float()
        * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        * D**-0.5
    )
    key = key.float() * torch.rsqrt(key.float().square().sum(-1, keepdim=True) + 1e-6)
    gate_raw = torch.nn.functional.linear(inputs["f_a_out"], params["f_b_weight"]).view(
        width, H, D
    )
    decay = (
        -5.0
        * torch.sigmoid(
            params["A_log"].exp()[None, :, None]
            * (gate_raw.float() + params["dt_bias"].view(1, H, D))
        )
    ).exp()
    beta = inputs["beta_raw"].float().sigmoid()
    states, convs, outputs = [], [], []
    state = state.clone()
    for token in range(width):
        state = state * decay[token, :, None, :]
        update = (
            value[token].float() - torch.einsum("hvk,hk->hv", state, key[token])
        ) * beta[token, :, None]
        state += update[:, :, None] * key[token, :, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q[token]))
        states.append(state.clone())
        convs.append(window[token + 1 : token + 4].T.contiguous())
    return torch.stack(outputs).bfloat16(), states, convs


@pytest.mark.parametrize("width", [1, 4, 16])
@pytest.mark.parametrize("capacity", [8, 16, 32])
@pytest.mark.parametrize("captured", [False, True])
def test_retained_backend_rounds_match_recurrence(width, capacity, captured):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(671)
    pool = _pool(capacity)
    spec = MHAConfig(num_attention_heads=H, num_kv_heads=H, head_dim=D, attn_tp_size=1)
    linear = LinearAttnConfig(
        num_k_heads=H,
        num_v_heads=H,
        head_k_dim=D,
        head_v_dim=D,
        conv_kernel_size=4,
        layer_ids=(1, 2),
        tp_size=1,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method=None,
        prefix_granularity=128,
        context_len=4096,
        max_bs=3,
        is_draft=False,
        speculative_num_draft_tokens=width,
        components=(spec, linear),
    )
    backend = KdaAttnBackend(
        config, spec, enable_prefill_graph=False, kda_backend="fla"
    )
    backend.set_kv_pool(pool)
    planned_bytes = kda_verify_workspace_bytes(
        config, _groups(capacity), draft_token_num=width, replay_kda=True
    )
    # Serving reserves this budget before allocating the real record tensors.
    assert backend.preallocate_verify_workspace(3, width) == planned_bytes
    backend.init_cuda_graph_state(3)
    params = {
        layer: {
            "conv_weights": torch.randn(
                CHANNELS, 4, device="cuda", dtype=torch.bfloat16
            )
            * 0.1,
            "f_b_weight": torch.randn(H * D, FA, device="cuda", dtype=torch.bfloat16)
            * 0.1,
            "A_log": torch.randn(H, device="cuda") * 0.1,
            "dt_bias": torch.randn(H * D, device="cuda") * 0.1,
        }
        for layer in (1, 2)
    }
    current = {
        layer: [1 + 12 * (layer - 1), 4 + 12 * (layer - 1), 7 + 12 * (layer - 1)]
        for layer in (1, 2)
    }
    states, convs = {}, {}
    for layer in (1, 2):
        start = 1 + 12 * (layer - 1)
        for name in FIELD_NAMES:
            pool.get_component(layer, name)[start : start + 10].zero_()
        for request, page in enumerate(current[layer]):
            state = torch.randn(H, D, D, device="cuda") * 0.02
            conv = torch.randn(CHANNELS, 3, device="cuda", dtype=torch.bfloat16) * 0.1
            pool.get_component(layer, "recurrent_state")[page].copy_(state)
            pool.get_component(layer, "conv_state")[page].copy_(conv)
            states[layer, request], convs[layer, request] = state, conv
    inputs = {
        "mixed_qkv": torch.empty(
            3 * width, CHANNELS, device="cuda", dtype=torch.bfloat16
        ),
        "f_a_out": torch.empty(3 * width, FA, device="cuda", dtype=torch.bfloat16),
        "beta_raw": torch.empty(3 * width, H, device="cuda", dtype=torch.bfloat16),
    }
    for value in inputs.values():
        value.normal_(std=0.3)
    accepted = torch.ones(3, device="cuda", dtype=torch.int32)
    positions = [37, 19, 83]
    request_slots = torch.tensor([9, 3, 15], dtype=torch.int32, device="cuda")
    seq_lens = torch.empty(3, device="cuda", dtype=torch.int32)

    def forward():
        return [
            backend.forward_decode(
                None,
                None,
                None,
                layer=None,
                token_to_kv_pool=pool,
                bs=3,
                **inputs,
                **params[layer],
                bias=None,
                activation="silu",
                key_dim=H * D,
                value_dim=H * D,
                attention_tp_size=1,
                head_k_dim=D,
                head_v_dim=D,
                lower_bound=-5.0,
                layer_id=layer,
                seq_len=3 * width,
            )
            for layer in (1, 2)
        ]

    def commit():
        backend.commit_verified_state(accepted, accepted_path=None)

    def capture_round():
        # Idle capture still records the commit kernels; every later refresh
        # supplies new page IDs and independently allocated scheduler tables.
        backend.init_forward_metadata_capture_cuda_graph(
            3,
            request_slots,
            seq_lens,
            ForwardMode.DECODE,
        )
        forward()
        commit()
        if not captured:
            return None, None, None
        verify_graph, commit_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(verify_graph):
            outputs = forward()
        with torch.cuda.graph(commit_graph, pool=verify_graph.pool()):
            commit()
        return outputs, verify_graph, commit_graph

    outputs, verify_graph, commit_graph = capture_round()
    old_pool_snapshots = []

    for iteration in range(20):
        order = [2, 0, 1] if iteration % 2 else [0, 1, 2]
        live = 0 if iteration == 13 else (3, 2, 1)[iteration % 3]
        for value in inputs.values():
            value.normal_(std=0.3)
        counts = [1 + (iteration + row) % width for row in range(3)]
        accepted.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        seq_lens.copy_(
            torch.tensor(
                [positions[request] + width for request in order],
                device="cuda",
                dtype=torch.int32,
            )
        )
        tables, destinations, snapshots = {}, {}, []
        for layer, group in enumerate(STATE_GROUPS, start=1):
            pages = []
            destinations[layer] = []
            for row, request in enumerate(order):
                source = current[layer][request]
                rotate = iteration == 0 or (iteration == 7 and request == 0)
                destination = source + 1 if rotate else source
                pages.append([source if rotate else 0, destination])
                destinations[layer].append(destination)
                if rotate and row < live:
                    snapshots.extend(
                        (
                            layer,
                            name,
                            source,
                            pool.get_component(layer, name)[source].clone(),
                        )
                        for name in FIELD_NAMES
                    )
            tables[group] = torch.tensor(pages, device="cuda", dtype=torch.int32)
        backend.refresh_decode_metadata(
            3,
            live,
            request_slots,
            seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=captured,
        )
        if captured:
            verify_graph.replay()
        else:
            outputs = forward()
        expected = {}
        for layer in (1, 2):
            for row, request in enumerate(order[:live]):
                local_inputs = {
                    name: value[row * width : (row + 1) * width]
                    for name, value in inputs.items()
                }
                expected[layer, row] = _reference(
                    local_inputs,
                    params[layer],
                    states[layer, request],
                    convs[layer, request],
                )
                torch.testing.assert_close(
                    outputs[layer - 1].reshape(3, width, H, D)[row],
                    expected[layer, row][0],
                    atol=3e-6,
                    rtol=8e-3,
                )
        if captured:
            commit_graph.replay()
        else:
            commit()
        for layer in (1, 2):
            for row, request in enumerate(order[:live]):
                _, checkpoints, windows = expected[layer, row]
                states[layer, request] = checkpoints[counts[row] - 1]
                convs[layer, request] = windows[counts[row] - 1]
                page = destinations[layer][row]
                current[layer][request] = page
                torch.testing.assert_close(
                    _endpoint(pool, layer, page),
                    states[layer, request],
                    atol=3e-6,
                    rtol=2e-4,
                )
                torch.testing.assert_close(
                    pool.get_component(layer, "conv_state")[page],
                    convs[layer, request],
                    atol=0,
                    rtol=0,
                )
        for layer, name, page, snapshot in snapshots:
            torch.testing.assert_close(
                pool.get_component(layer, name)[page], snapshot, atol=0, rtol=0
            )
        for row, request in enumerate(order[:live]):
            positions[request] += counts[row]

        if iteration == 9:
            # Retraction copies every cache field to a new physical block;
            # subsequent verify sees the endpoint after the request changes rows.
            copy_plan = {}
            for layer, group in enumerate(STATE_GROUPS, start=1):
                destination = 9 + 12 * (layer - 1)
                copy_plan[group] = np.asarray(
                    [[current[layer][0], destination]], dtype=np.int64
                )
                current[layer][0] = destination
            pool.arena.copy_blocks(copy_plan)

        if iteration == 12:
            # Rebinding releases the captures before replacing cache storage.
            # A copied composite must resume identically after all pointer
            # tables and per-bucket physical-page buffers are rebuilt.
            torch.cuda.synchronize()
            del verify_graph, commit_graph, outputs
            previous_pool = pool
            replacement = _pool(capacity)
            for layer in (1, 2):
                start = 1 + 12 * (layer - 1)
                for name in FIELD_NAMES:
                    replacement.get_component(layer, name)[start : start + 10].zero_()
                    for page in current[layer]:
                        value = previous_pool.get_component(layer, name)[page]
                        replacement.get_component(layer, name)[page].copy_(value)
                        old_pool_snapshots.append(
                            (previous_pool, layer, name, page, value.clone())
                        )
            pool = replacement
            backend.set_kv_pool(pool)
            backend.preallocate_verify_workspace(3, width)
            backend.init_cuda_graph_state(3)
            outputs, verify_graph, commit_graph = capture_round()

        if iteration == 14:
            # A new request reuses the row of a finished request. The arena
            # clears its new block, including history length and convolution.
            new_blocks = {}
            for layer, group in enumerate(STATE_GROUPS, start=1):
                page = 6 + 12 * (layer - 1)
                new_blocks[group] = np.asarray([page], dtype=np.int64)
                current[layer][1] = page
                states[layer, 1] = torch.zeros_like(states[layer, 1])
                convs[layer, 1] = torch.zeros_like(convs[layer, 1])
            pool.zero_new_blocks(new_blocks)
            positions[1] = 0

    for previous_pool, layer, name, page, snapshot in old_pool_snapshots:
        torch.testing.assert_close(
            previous_pool.get_component(layer, name)[page], snapshot, atol=0, rtol=0
        )

    # Dense prefill consumes the composite endpoint and clears history only
    # after publishing its final state. The original checkpoint stays intact.
    for layer in (1, 2):
        source, destination = current[layer][0], 10 + 12 * (layer - 1)
        state_pool = pool.get_component(layer, "recurrent_state")
        conv_pool = pool.get_component(layer, "conv_state")
        source_snapshot = [
            pool.get_component(layer, name)[source].clone() for name in FIELD_NAMES
        ]
        reads = torch.tensor([source], device="cuda", dtype=torch.int32)
        writes = torch.tensor([destination], device="cuda", dtype=torch.int32)
        dense, _ = backend._prepare_prefill_state_inputs(
            layer, conv_pool, state_pool, reads, writes
        )
        torch.testing.assert_close(dense[0], states[layer, 0], atol=3e-6, rtol=2e-4)
        final = dense * 0.7
        backend._store_prefill_state(layer, final, state_pool, writes)
        torch.testing.assert_close(
            _endpoint(pool, layer, destination), final[0], atol=0, rtol=0
        )
        for name, snapshot in zip(FIELD_NAMES, source_snapshot, strict=True):
            torch.testing.assert_close(
                pool.get_component(layer, name)[source], snapshot, atol=0, rtol=0
            )
