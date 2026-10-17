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

"""Real prefill scans must resume the composite KDA endpoint."""

from __future__ import annotations

from test.runtime.test_kda_retained_commit import (
    CHANNELS,
    FA,
    FIELD_NAMES,
    STATE_GROUPS,
    D,
    H,
    _endpoint,
    _pool,
    _reference,
)

import pytest
import torch
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig


@pytest.mark.parametrize("kda_backend", ["fla", "cutedsl_kda"])
@pytest.mark.parametrize("shared_source", [False, True])
@pytest.mark.parametrize(
    "mixed, width, accepted, padding",
    [
        (False, 4, 5, 0),
        (True, 1, 1, 0),
        *[(True, 4, count, pad) for count in (1, 2, 3, 4) for pad in (0, 3)],
    ],
)
def test_retained_prefill_matches_dense_endpoint(
    kda_backend, shared_source, mixed, width, accepted, padding
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if kda_backend == "fla":
        pytest.importorskip(
            "fla", reason="optional FLA prefill backend is not installed"
        )
    torch.manual_seed(137)
    retained, dense = _pool(8), _pool(8)
    # State groups share a storage plane, so each group owns distinct pages.
    source_pages = {
        layer: [1 + 12 * (layer - 1), 4 + 12 * (layer - 1)] for layer in (1, 2)
    }
    output_pages = {
        layer: [page + 1 if shared_source else page for page in pages]
        for layer, pages in source_pages.items()
    }
    snapshots = []
    decode_initial = {}
    for layer in (1, 2):
        for page in source_pages[layer]:
            for name in FIELD_NAMES:
                value = retained.get_component(layer, name)[page]
                if name == "replay_length":
                    value.fill_(3)
                elif name == "replay_decay":
                    value.fill_(0.8)
                else:
                    value.normal_(std=0.1)
                snapshots.append((layer, name, page, value.clone()))
            dense.get_component(layer, "recurrent_state")[page].copy_(
                _endpoint(retained, layer, page)
            )
            dense.get_component(layer, "conv_state")[page].copy_(
                retained.get_component(layer, "conv_state")[page]
            )
            dense.get_component(layer, "replay_length")[page].zero_()
        decode_initial[layer] = (
            _endpoint(dense, layer, source_pages[layer][1]),
            dense.get_component(layer, "conv_state")[source_pages[layer][1]].clone(),
        )
    lengths = [7, width if mixed else 5]
    tokens = sum(lengths) + padding
    inputs = {
        "mixed_qkv": torch.randn(tokens, CHANNELS, device="cuda", dtype=torch.bfloat16)
        * 0.3,
        "f_a_out": torch.randn(tokens, FA, device="cuda", dtype=torch.bfloat16) * 0.3,
        "beta_raw": torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16) * 0.3,
        "conv_weights": torch.randn(CHANNELS, 4, device="cuda", dtype=torch.bfloat16)
        * 0.1,
        "f_b_weight": torch.randn(H * D, FA, device="cuda", dtype=torch.bfloat16) * 0.1,
        "A_log": torch.randn(H, device="cuda") * 0.1,
        "dt_bias": torch.randn(H * D, device="cuda") * 0.1,
    }
    spec = MHAConfig(num_attention_heads=H, num_kv_heads=H, head_dim=D, attn_tp_size=1)
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method=None,
        prefix_granularity=128,
        context_len=4096,
        max_bs=2,
        is_draft=False,
        speculative_num_draft_tokens=width,
        components=(spec,),
    )
    outputs = {}
    for pool in (retained, dense):
        reference = pool is dense
        lengths = [7, accepted if reference else width] if mixed else [7, 5]
        tokens = sum(lengths) + (0 if reference else padding)
        num_extends = 1 if mixed and not (reference and width > 1) else 2
        mode = ForwardMode.MIXED if num_extends == 1 else ForwardMode.EXTEND
        lengths_cpu = torch.tensor(lengths[:num_extends], dtype=torch.int32)
        prefixes_cpu = torch.tensor([37, 19][:num_extends], dtype=torch.int32)
        backend = KdaAttnBackend(
            config, spec, enable_prefill_graph=False, kda_backend=kda_backend
        )
        backend.set_kv_pool(pool)
        backend.preallocate_verify_workspace(2, width)
        backend.init_cuda_graph_state(2)
        tables = {
            group: torch.tensor(
                list(zip(source_pages[layer], output_pages[layer])),
                device="cuda",
                dtype=torch.int32,
            )
            for layer, group in enumerate(STATE_GROUPS, start=1)
        }
        backend.init_forward_metadata(
            2,
            num_extends,
            torch.tensor([3, 9], device="cuda", dtype=torch.int32),
            torch.tensor(
                [37 + lengths[0], 19 + lengths[1]], device="cuda", dtype=torch.int32
            ),
            mode,
            block_tables=tables,
            extend_seq_lens=lengths_cpu.cuda(),
            extend_seq_lens_cpu=lengths_cpu,
            extend_prefix_lens=prefixes_cpu.cuda(),
            extend_prefix_lens_cpu=prefixes_cpu,
            extend_replay_lens_cpu=torch.zeros_like(lengths_cpu),
            extend_prompt_lens_cpu=prefixes_cpu + lengths_cpu,
            extend_with_prefix=True,
            query_shard=None,
        )
        for layer in (1, 2):
            outputs[reference, layer] = backend.forward_extend(
                None,
                None,
                None,
                layer=None,
                token_to_kv_pool=pool,
                bs=2,
                forward_mode=mode,
                **{
                    name: (
                        value[:tokens]
                        if name in ("mixed_qkv", "f_a_out", "beta_raw")
                        else value
                    ).clone()
                    for name, value in inputs.items()
                },
                bias=None,
                activation="silu",
                key_dim=H * D,
                value_dim=H * D,
                attention_tp_size=1,
                head_k_dim=D,
                head_v_dim=D,
                lower_bound=-5.0,
                layer_id=layer,
                seq_len=tokens,
            ).clone()
        if reference and mixed and width > 1:
            # Prefill rounds normalized K to BF16. Verify uses FP32 normalized
            # K, so compare its accepted state with the direct FP32 recurrence.
            for layer in (1, 2):
                expected_out, expected_states, expected_convs = _reference(
                    {
                        name: inputs[name][7 : 7 + width]
                        for name in ("mixed_qkv", "f_a_out", "beta_raw")
                    },
                    {
                        name: inputs[name]
                        for name in ("conv_weights", "f_b_weight", "A_log", "dt_bias")
                    },
                    *decode_initial[layer],
                )
                outputs[True, layer][7:] = expected_out[:accepted]
                page = output_pages[layer][1]
                dense.get_component(layer, "recurrent_state")[page].copy_(
                    expected_states[accepted - 1]
                )
                dense.get_component(layer, "conv_state")[page].copy_(
                    expected_convs[accepted - 1]
                )
        if mixed and not reference:
            hybrid = HybridLinearAttnBackend(backend, backend, [])
            hybrid.commit_state_after_acceptance(
                torch.tensor([1, accepted], dtype=torch.int32, device="cuda"),
                num_extends=1,
                accepted_path=None,
            )
    for layer in (1, 2):
        torch.testing.assert_close(
            outputs[False, layer][: 7 + accepted],
            outputs[True, layer],
            atol=1e-5,
            rtol=8e-3,
        )
        for page in output_pages[layer]:
            torch.testing.assert_close(
                _endpoint(retained, layer, page),
                _endpoint(dense, layer, page),
                atol=2e-5,
                rtol=1e-3,
            )
            torch.testing.assert_close(
                retained.get_component(layer, "conv_state")[page],
                dense.get_component(layer, "conv_state")[page],
                atol=0,
                rtol=0,
            )
    if padding:
        for layer in (1, 2):
            tail = outputs[False, layer][7 + width :]
            torch.testing.assert_close(tail, torch.zeros_like(tail), atol=0, rtol=0)
    if shared_source:
        for layer, name, page, expected in snapshots:
            torch.testing.assert_close(
                retained.get_component(layer, name)[page], expected, atol=0, rtol=0
            )
