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
    mm13 = Mock(
        _schema=torch._C.parse_schema(
            "custom::fused_init_routing_mm13_swiglu.with_plan(Tensor x, Tensor ids, Tensor weight, Tensor? weight_scale, int expert_start, int cube_cores, Tensor? smooth13, Tensor? smooth2) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        )
    )
    mm2 = Mock(
        _schema=torch._C.parse_schema(
            "custom::fused_mm2_fin_routing(Tensor x, Tensor weight, Tensor? weight_scale, Tensor? x_scale, Tensor counts, Tensor rows, Tensor route_weights, Tensor expert_boundaries, Tensor row_boundaries, Tensor expert_base_boundaries) -> Tensor"
        )
    )
    custom = SimpleNamespace(
        fused_init_routing_mm13_swiglu=SimpleNamespace(with_plan=mm13),
        fused_mm2_fin_routing=SimpleNamespace(default=mm2),
    )
    monkeypatch.setattr(torch, "ops", SimpleNamespace(custom=custom))
    limits = Mock(return_value=dict(cube_core_num=24, vector_core_num=40))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(get_device_limit=limits), raising=False
    )
    monkeypatch.delenv("TOKENSPEED_NPU_MM13_AICPU_BALANCE", raising=False)
    return custom, mm13, mm2, limits


def test_preparation_binds_exact_ops_and_core_quota(adapter, bindings):
    _, mm13, mm2, limits = bindings
    w = SimpleNamespace(w13_weight=torch.empty(1))
    plan = {}
    adapter._prepare_routed_full_moe(plan, w)
    assert plan == dict(
        routed_mm13_with_plan=mm13, routed_mm2=mm2, routed_cube_cores=20
    )
    limits.assert_called_once_with(w.w13_weight.device)


@pytest.mark.parametrize(
    "fault", ["old_mm2", "missing_plan", "aicpu", "too_many_cores"]
)
def test_preparation_rejects_incompatible_contract(
    adapter, bindings, monkeypatch, fault
):
    custom, _, mm2, limits = bindings
    if fault == "old_mm2":
        mm2._schema = torch._C.parse_schema(
            "custom::fused_mm2_fin_routing(Tensor x, Tensor weight, Tensor? weight_scale, Tensor? x_scale, Tensor counts, Tensor rows, Tensor route_weights) -> Tensor"
        )
    elif fault == "missing_plan":
        del custom.fused_init_routing_mm13_swiglu.with_plan
    elif fault == "aicpu":
        monkeypatch.setenv("TOKENSPEED_NPU_MM13_AICPU_BALANCE", "1")
    else:
        limits.return_value = dict(cube_core_num=32, vector_core_num=64)
    plan = {}
    with pytest.raises(RuntimeError):
        adapter._prepare_routed_full_moe(
            plan, SimpleNamespace(w13_weight=torch.empty(1))
        )
    assert not plan


@pytest.mark.parametrize("quantized", [True, False])
def test_full_apply_threads_fresh_plan_and_preserves_bf16_fence(
    adapter, bindings, quantized
):
    _, mm13, mm2, limits = bindings
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
        outputs = tuple(torch.full((4,), step + i) for i in range(7))
        mm13.return_value = outputs
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
        args = mm13.call_args.args
        assert args[0] is x and args[1].dtype == torch.int32
        assert args[3] is (w.w13_weight_scale if quantized else None)
        assert args[4:6] == (2, 20)
        tail = mm2.call_args.args
        assert tail[0] is outputs[0]
        assert tail[2] is (w.w2_weight_scale if quantized else None)
        assert tail[3] is (outputs[1] if quantized else None)
        assert tail[4] is outputs[2] and tail[5] is outputs[3]
        torch.testing.assert_close(tail[6], weights.bfloat16(), rtol=0, atol=0)
        assert all(a is b for a, b in zip(tail[7:], outputs[4:]))
    assert mm13.call_count == mm2.call_count == 2
    assert limits.call_count == 1  # No host device query in forward.


@pytest.mark.parametrize("quantized", [True, False])
def test_input_only_keeps_original_producer_and_tail(
    adapter, bindings, monkeypatch, quantized
):
    custom, planned, mm2, limits = bindings
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
    limits.assert_not_called()
