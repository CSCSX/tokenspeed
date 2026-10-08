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

"""Optional standalone AscendC accepted-prefix ReplaySSM adapter."""

import os
from functools import cache
from pathlib import Path

import torch

_LIBRARY_ENV = "TOKENSPEED_KDA_REPLAY_FUSED_LIBRARY"
_ARGUMENTS = (
    "descriptors",
    "groups",
    "reads",
    "writes",
    "steps",
    "anchor",
    "tokens",
    "heads",
    "dim",
    "fa_dim",
    "qkv_stride",
    "conv_stride",
    "fa_stride",
    "beta_stride",
    "state_stride",
    "gate_stride",
    "payload_layer_stride",
    "gate_layer_stride",
    "qkv_width",
    "conv_width",
    "pages",
    "lower_bound",
)


@cache
def load_fused_kda_replay():
    """Load the explicitly configured library once, before graph capture.

    An unset variable keeps compatibility with older Flash installations.
    A configured but missing, incompatible or unloadable library is fatal;
    it must never silently select the legacy three-kernel implementation.
    The selection is process-lifetime immutable, including across graphs.
    """
    configured = os.environ.get(_LIBRARY_ENV)
    if configured is None:
        return None
    path = Path(configured)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(
            f"{_LIBRARY_ENV} must name an existing absolute library: {configured!r}"
        )
    torch.ops.load_library(str(path.resolve()))
    packet = getattr(torch.ops.flash_replay, "fused", None)
    op = getattr(packet, "default", None)
    schema = getattr(op, "_schema", None)
    arguments = tuple(schema.arguments) if schema is not None else ()
    alias = getattr(arguments[5], "alias_info", None) if len(arguments) == 22 else None
    if (
        getattr(schema, "name", None) != "flash_replay::fused"
        or tuple(argument.name for argument in arguments) != _ARGUMENTS
        or tuple(str(argument.type) for argument in arguments)
        != ("Tensor",) * 6 + ("int",) * 15 + ("float",)
        or alias is None
        or not alias.is_write
        or len(schema.returns) != 1
        or str(schema.returns[0].type) != "Tensor"
    ):
        raise RuntimeError(
            "incompatible fused ReplaySSM schema or missing mutable state anchor"
        )
    return op


def fused_kda_batched_replay_commit(
    *,
    descriptors: torch.Tensor,
    group_indices: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    state_anchor: torch.Tensor,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    f_a_dim: int,
    qkv_stride: int,
    conv_stride: int,
    f_a_stride: int,
    beta_stride: int,
    state_stride: int,
    gate_stride: int,
    payload_layer_stride: int,
    gate_layer_stride: int,
    qkv_width: int,
    conv_width: int,
    num_pages: int,
    lower_bound: float,
) -> None:
    """Commit all layers through one fused kernel using the existing ABI.

    Descriptor and metadata tensors, element strides, geometry and arena
    mutation anchor retain the batched replay contract. The backend owns all
    referenced storage through asynchronous execution and graph lifetime.
    No tensor is copied, reshaped or read back by this adapter. Outputs are
    the recurrent/conv pools referenced by descriptors, not a return tensor.
    """
    op = load_fused_kda_replay()
    if op is None:
        raise RuntimeError("fused ReplaySSM library was not configured before startup")
    op(
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        state_anchor,
        draft_token_num,
        num_heads,
        head_dim,
        f_a_dim,
        qkv_stride,
        conv_stride,
        f_a_stride,
        beta_stride,
        state_stride,
        gate_stride,
        payload_layer_stride,
        gate_layer_stride,
        qkv_width,
        conv_width,
        num_pages,
        lower_bound,
    )
