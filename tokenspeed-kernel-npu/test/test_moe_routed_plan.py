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

"""Host ABI/dataflow checks; device numerics and graph replay are separate."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    path = Path(__file__).parents[1] / "python/tokenspeed_kernel_npu/ops/moe.py"
    spec = importlib.util.spec_from_file_location("moe_plan_host_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bindings(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_ops", SimpleNamespace())
    mm13 = Mock(
        _schema=torch._C.parse_schema(
            "custom::fused_init_routing_mm13_swiglu(Tensor x, Tensor ids, Tensor weight, Tensor weight_scale, int expert_start, Tensor? smooth13=None, Tensor? smooth2=None) -> (Tensor, Tensor, Tensor, Tensor)"
        )
    )
    bf16 = Mock()
    mm2 = Mock(
        _schema=torch._C.parse_schema(
            "custom::fused_mm2_fin_routing(Tensor x, Tensor weight, Tensor? weight_scale, Tensor? x_scale, Tensor counts, Tensor rows, Tensor route_weights, Tensor expert_boundaries, Tensor row_boundaries, Tensor expert_base_boundaries) -> Tensor"
        )
    )
    custom = SimpleNamespace(
        fused_init_routing_mm13_swiglu=SimpleNamespace(default=mm13, bf16=bf16),
        fused_mm2_fin_routing=SimpleNamespace(default=mm2),
    )
    monkeypatch.setattr(torch, "ops", SimpleNamespace(custom=custom))
    return custom, mm13, bf16, mm2


def test_preparation_binds_standard_ops_and_boundary_placeholders(adapter, bindings):
    _, mm13, bf16, mm2 = bindings
    w = SimpleNamespace(w13_weight=torch.empty(1))
    plan = {}
    adapter._prepare_routed_full_moe(plan, w)
    assert plan["routed_mm13"] is mm13
    assert plan["routed_mm13_bf16"] is bf16
    assert plan["routed_mm2"] is mm2
    assert plan["routed_boundary"].shape == (25,)


@pytest.mark.parametrize("fault", ["old_mm2", "old_mm13"])
def test_preparation_rejects_incompatible_contract(
    adapter, bindings, monkeypatch, fault
):
    custom, mm13, _, mm2 = bindings
    if fault == "old_mm2":
        mm2._schema = torch._C.parse_schema(
            "custom::fused_mm2_fin_routing(Tensor x, Tensor weight, Tensor? weight_scale, Tensor? x_scale, Tensor counts, Tensor rows, Tensor route_weights) -> Tensor"
        )
    else:
        mm13._schema = torch._C.parse_schema(
            "custom::fused_init_routing_mm13_swiglu.with_plan(Tensor x, Tensor ids, Tensor weight, Tensor? weight_scale, int expert_start, int cube_cores, Tensor? smooth13, Tensor? smooth2) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        )
    plan = {}
    with pytest.raises(RuntimeError):
        adapter._prepare_routed_full_moe(
            plan, SimpleNamespace(w13_weight=torch.empty(1))
        )
    assert not plan


@pytest.mark.parametrize("quantized", [True, False])
def test_full_apply_passes_standard_mm13_outputs_to_mm2(adapter, bindings, quantized):
    _, mm13, bf16, mm2 = bindings
    w = SimpleNamespace(
        w13_weight=torch.empty(2, 32, 64),
        w2_weight=torch.empty(2, 32, 32),
        w13_weight_scale=torch.ones(2, 64),
        w2_weight_scale=torch.ones(2, 32),
        w13_smooth_scale=None,
        w2_smooth_scale=None,
        ep_rank=1,
        num_local_experts=2,
        _ascend_int8_moe_weights_processed=True,
        _ascend_bf16_moe_weights_processed=True,
    )
    plan = dict(activation="silu")
    adapter._prepare_routed_full_moe(plan, w)
    x = torch.zeros(3, 32, dtype=torch.bfloat16)
    ids = torch.tensor([[2, 3]] * 3, dtype=torch.int64)
    weights = torch.full((3, 2), 0.501953125, dtype=torch.float32)
    apply = (
        adapter.ascend_routed_int8_precomputed_moe_apply
        if quantized
        else adapter.ascend_routed_bf16_precomputed_moe_apply
    )
    for step in range(2):
        outputs = tuple(
            torch.full((4,), step + i) for i in range(4 if quantized else 3)
        )
        producer = mm13 if quantized else bf16
        producer.return_value = outputs
        result = apply(
            plan=plan,
            x=x,
            w=w,
            router_logits=None,
            topk_weights=weights,
            topk_ids=ids,
            num_tokens_global=None,
            max_num_tokens_per_gpu=None,
            do_finalize=True,
            enable_pdl=False,
        )
        assert result is mm2.return_value
        args = producer.call_args.args
        assert args[0] is x and args[1].dtype == torch.int32
        if quantized:
            assert args[3] is w.w13_weight_scale
            assert args[4] == 2
        else:
            assert args[3] == 2
        tail = mm2.call_args.args
        assert tail[0] is outputs[0]
        assert tail[2] is (w.w2_weight_scale if quantized else None)
        assert tail[3] is (outputs[1] if quantized else None)
        offset = 1 if quantized else 0
        assert tail[4] is outputs[1 + offset]
        assert tail[5] is outputs[2 + offset]
        assert tail[6] is weights
        assert tail[7] is tail[8] is tail[9] is plan["routed_boundary"]
    assert producer.call_count == mm2.call_count == 2


@pytest.mark.parametrize("quantized", [True, False])
def test_input_only_keeps_original_producer_and_tail(
    adapter, bindings, monkeypatch, quantized
):
    custom, planned, _, mm2 = bindings
    outputs = tuple(torch.empty(1) for _ in range(4 if quantized else 3))
    producer = Mock(return_value=outputs)
    packet = producer if quantized else SimpleNamespace(bf16=producer)
    custom.fused_init_routing_mm13_swiglu = packet
    tail = Mock()
    monkeypatch.setattr(
        adapter,
        "_ascend_int8_gmm2_finalize" if quantized else "_ascend_bf16_gmm2_finalize",
        tail,
    )
    w = SimpleNamespace(
        w13_weight=torch.empty(1),
        w13_weight_scale=torch.empty(1),
        ep_rank=1,
        num_local_experts=2,
        _ascend_int8_moe_weights_processed=True,
        _ascend_bf16_moe_weights_processed=True,
    )
    apply = (
        adapter.ascend_routed_int8_precomputed_moe_apply
        if quantized
        else adapter.ascend_routed_bf16_precomputed_moe_apply
    )
    weights, ids = torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32)
    result = apply(
        plan=dict(activation="silu"),
        x=torch.zeros(2, 32),
        w=w,
        router_logits=None,
        topk_weights=weights,
        topk_ids=ids,
        num_tokens_global=None,
        max_num_tokens_per_gpu=None,
        do_finalize=True,
        enable_pdl=False,
    )
    assert result is tail.return_value
    assert all(a is b for a, b in zip(tail.call_args.args, outputs))
    assert tail.call_args.args[-2] is weights
    producer.assert_called_once()
    planned.assert_not_called()
    mm2.assert_not_called()


def test_standard_chain_uses_packaged_abi_and_local_finalization(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_ops", SimpleNamespace())
    chain = Mock(
        _schema=torch._C.parse_schema(
            "custom::fused_mm13_mm2(Tensor x, Tensor ids, Tensor w13, Tensor ws13, Tensor w2, Tensor ws2, Tensor route_weights, Tensor expert_boundaries, Tensor row_boundaries, Tensor expert_base_boundaries, int expert_start=0, int quant_bits=8, Tensor? smooth13=None, Tensor? smooth2=None) -> Tensor"
        )
    )
    monkeypatch.setattr(
        torch,
        "ops",
        SimpleNamespace(
            custom=SimpleNamespace(fused_mm13_mm2=SimpleNamespace(default=chain))
        ),
    )
    prepared = Mock()
    monkeypatch.setattr(adapter, "ascend_int8_process_moe_weights", prepared)
    w = SimpleNamespace(
        w13_weight=torch.empty(2, 32, 64, dtype=torch.int8),
        w13_weight_scale=torch.ones(2, 64),
        w2_weight=torch.empty(2, 32, 32, dtype=torch.int8),
        w2_weight_scale=torch.ones(2, 32),
        ep_rank=3,
        num_local_experts=2,
        _ascend_int8_moe_weights_processed=True,
    )
    plan = {"activation": "swiglu"}
    adapter.ascend_routed_chain_int8_process_moe_weights(plan=plan, w=w)
    prepared.assert_called_once_with(plan=plan, w=w)
    assert w.w2_weight_scale.dtype == torch.bfloat16
    assert plan["routed_chain_boundary"].shape == (25,)
    x = torch.zeros(3, 32, dtype=torch.bfloat16)
    ids = torch.tensor([[6, 7]] * 3, dtype=torch.int64)
    weights = torch.full((3, 2), 0.5, dtype=torch.float32)
    result = adapter.ascend_routed_chain_int8_precomputed_moe_apply(
        plan=plan,
        x=x,
        w=w,
        router_logits=None,
        topk_weights=weights,
        topk_ids=ids,
    )
    assert result is chain.return_value
    args = chain.call_args.args
    assert args[0] is x and args[1].dtype == torch.int32
    assert args[6] is weights
    assert args[7] is args[8] is args[9] is plan["routed_chain_boundary"]
    assert args[10:12] == (6, 8)
