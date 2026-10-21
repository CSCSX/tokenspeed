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

"""The Qwen4-Exp decode megakernel seam: which forwards take the kernels, and
the per-row operands that a forward's layer launches share. CPU only."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle import (  # noqa: E402
    AcceptedPrefixPublisher,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.models.qwen4_exp_megakernel import (  # noqa: E402
    MegakernelSteps,
    _verify_attention_rows,
    _verify_state_rows,
    runs_fused,
)


class _DraftBackend:
    """The draft model's attention backend: QSA only, no recurrent state."""

    @property
    def linear_attn_backend(self):
        raise RuntimeError("the draft backend has no linear attention")

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        del seq_lens


def _decode_ctx(bs: int, rows: int, narrowing) -> ForwardContext:
    return ForwardContext(
        attn_backend=narrowing.attn_backend if narrowing else _DraftBackend(),
        token_to_kv_pool=None,
        bs=bs,
        num_extends=0,
        input_num_tokens=rows,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, bs, 1),
        draft_narrowing=narrowing,
    )


def test_draft_steps_take_the_kernels() -> None:
    """Every draft step of up to 4 rows runs fused: the first step (the
    verified rows of a request, a chain) and the one-row steps after it. The
    draft backend has no recurrent state, and the rule never asks for it."""
    backend = _DraftBackend()
    first = AcceptedPrefixPublisher(backend, torch.tensor([7]))
    assert runs_fused(torch.empty(4, 8), _decode_ctx(1, 4, first))
    assert runs_fused(torch.empty(1, 8), _decode_ctx(1, 1, None))
    two = AcceptedPrefixPublisher(backend, torch.tensor([7, 9]))
    assert not runs_fused(torch.empty(8, 8), _decode_ctx(2, 8, two))
    assert not runs_fused(torch.empty(5, 8), _decode_ctx(5, 5, None))


def test_shared_operands_build_once_per_forward() -> None:
    steps = MegakernelSteps(torch.device("cpu"), scratch_pairs=8)
    builds = []

    def build():
        builds.append(len(builds))
        return (torch.tensor([len(builds)]),)

    first = steps.shared(("rows", 1), build)
    assert steps.shared(("rows", 1), build) is first
    assert len(builds) == 1
    steps.shared(("rows", 2), build)
    assert len(builds) == 2
    steps.advance()
    assert steps.shared(("rows", 1), build) is not first
    assert len(builds) == 3


def test_verify_row_layout() -> None:
    """Token ``t`` of request ``r`` (of ``chain``) is at ``seq_lens[r] -
    chain + t`` and reads its request's tables and input page."""
    seq_len, full, recent = _verify_attention_rows(
        torch.tensor([10, 20], dtype=torch.int32),
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5], [6]])),
        3,
    )
    assert seq_len.tolist() == [8, 9, 10, 18, 19, 20]
    assert full.tolist() == [[1, 2]] * 3 + [[3, 4]] * 3
    assert recent.tolist() == [[5]] * 3 + [[6]] * 3
    state_in, state_out = _verify_state_rows(torch.tensor([5, 7]), 2)
    assert state_in.tolist() == [5, 5, 7, 7]
    assert state_out.tolist() == [-1] * 4
