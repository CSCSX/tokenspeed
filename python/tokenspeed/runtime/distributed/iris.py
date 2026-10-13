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

"""Plan the process-wide Iris heap across target and draft models."""

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.comm_backend import Group, get_global_backend
from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass
class IrisBufferPlan:
    all_reduce: dict[Group, tuple[int, int]] = field(default_factory=dict)
    rmsnorm: set[tuple[Group, int, int]] = field(default_factory=set)
    k3: dict[Group, dict] = field(default_factory=dict)
    heap_size: int = field(default=0, init=False)

    def add_model(
        self,
        model,
        mapping,
        *,
        hidden_size: int,
        dtype: torch.dtype,
        max_num_tokens: int,
        fusion_max_tokens: int,
    ) -> None:
        backend = get_global_backend()
        if dtype != torch.bfloat16 or not isinstance(backend, AutoBackend):
            return
        groups = dict.fromkeys(
            (
                mapping.attn.tp_group,
                mapping.attn.head_tp_group,
                mapping.attn.kvp_group,
                mapping.dense.tp_group,
                mapping.moe.tp_group,
                mapping.moe.tp_ep_group,
                mapping.linear_attn.tp_group,
                mapping.vision.tp_group,
                mapping.lm_head.tp_group,
            )
        )
        for group in groups:
            if process_group_manager.has_process_group(
                "nccl", group
            ) and backend.can_prepare_all_reduce_buffers(group, torch.bfloat16):
                self.all_reduce.setdefault(group, backend.iris.default_capacities)
        # Fused layer-boundary norms reduce hidden states, not Q/K projections.
        # The target and draft can have different widths but share equal states.
        if fusion_max_tokens > 0:
            for group in dict.fromkeys(
                (mapping.attn.tp_group, mapping.dense.tp_group, mapping.moe.tp_ep_group)
            ):
                if (
                    len(group) > 1
                    and not backend._group_spans_nodes(group)
                    and process_group_manager.has_process_group("nccl", group)
                ):
                    self.rmsnorm.add((group, fusion_max_tokens, hidden_size))
        requirements = getattr(type(model), "iris_buffer_requirements", None)
        if requirements is not None:
            for group, capacities in requirements(model, max_num_tokens).items():
                previous = self.k3.get(group)
                if previous is not None:
                    capacities = {
                        name: max(value, previous[name])
                        for name, value in capacities.items()
                    }
                self.k3[group] = capacities
                self.all_reduce[group] = (
                    capacities["staged_max_numel"],
                    capacities["producer_direct_max_numel"],
                )

    def heap_bytes(self) -> int:
        if not (self.all_reduce or self.rmsnorm or self.k3):
            return 0
        from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
            k3_communication_heap_bytes,
        )
        from tokenspeed_kernel.ops.communication.iris import (
            IrisAllReduce,
            IrisAllReduceResidualRMSNorm,
            iris_heap_bytes,
        )

        size = sum(
            IrisAllReduce.heap_bytes(staged, producer, len(group), torch.bfloat16)
            for group, (staged, producer) in self.all_reduce.items()
            if group not in self.k3
        )
        size += sum(
            k3_communication_heap_bytes(
                world_size=len(group), dtype=torch.bfloat16, **capacities
            )
            for group, capacities in self.k3.items()
        )
        size += sum(
            IrisAllReduceResidualRMSNorm.heap_bytes(rows, width, torch.bfloat16)
            for _, rows, width in self.rmsnorm
        )
        return iris_heap_bytes(size)

    def configure(self, *, heap_size: int | None) -> None:
        # Iris exchanges heap handles over WORLD, including pipeline stages with
        # different layers or no draft. Every rank must reserve the same heap size.
        size = torch.tensor(self.heap_bytes(), dtype=torch.int64, device="cuda")
        dist.all_reduce(size, op=dist.ReduceOp.MAX)
        required = int(size.item())
        if not required:
            return
        from tokenspeed_kernel.ops.communication.iris import configure_iris

        if heap_size is not None and heap_size < required:
            raise ValueError(
                f"Iris heap needs {required} bytes, but --iris-heap-size={heap_size}"
            )
        self.heap_size = required if heap_size is None else heap_size
        configure_iris(heap_size=self.heap_size, allocator_type=None)
        logger.info(
            f"Iris heap: {self.heap_size} bytes ({required} required by communication buffers)"
        )

    def prepare(self) -> bool:
        """Allocate planned states before cache profiling or any model warmup."""
        if not self.heap_size:
            return False
        from tokenspeed_kernel.ops.communication.iris import (
            prepare_iris_ar_rmsnorm,
            prepare_iris_heap,
        )

        prepare_iris_heap(heap_size=self.heap_size)
        backend = get_global_backend()
        for group, (staged, producer) in self.all_reduce.items():
            backend.prepare_all_reduce_buffers(
                group,
                staged_max_numel=staged,
                producer_direct_max_numel=producer,
                dtype=torch.bfloat16,
            )
        for group, rows, width in sorted(self.rmsnorm):
            process_group = process_group_manager.get_device_process_group(group)
            prepare_iris_ar_rmsnorm(
                group=process_group,
                rank_in_group=group.index(dist.get_rank()),
                max_token_num=rows,
                hidden_dim=width,
                dtype=torch.bfloat16,
            )
        return True


def plan_iris_buffers(
    runners, max_num_tokens: int, *, heap_size: int | None
) -> IrisBufferPlan | None:
    """Size all eligible groups and norm shapes before the first Iris allocation."""
    if (
        not current_platform().is_amd
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return None
    plan = IrisBufferPlan()
    for runner in runners:
        if runner is None:
            continue
        args = runner.server_args
        plan.add_model(
            runner.model,
            runner.mapping,
            hidden_size=runner.model_config.hidden_size,
            dtype=runner.model_config.dtype,
            max_num_tokens=max_num_tokens,
            fusion_max_tokens=(
                args.comm_fusion_max_num_tokens if args.enable_allreduce_fusion else 0
            ),
        )
    plan.configure(heap_size=heap_size)
    return plan
