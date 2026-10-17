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

"""Materialize cache-owned KDA history at dense-state boundaries."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import _recover_weighted_kda_state


@triton.jit
def _materialize_history_kernel(
    state_ptr,
    correction_ptr,
    key_ptr,
    decay_ptr,
    lengths_ptr,
    indices_ptr,
    out_ptr,
    stride_state_page,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_u_page,
    stride_u_head,
    stride_u_token,
    stride_u_dim,
    stride_key_page,
    stride_key_head,
    stride_key_token,
    stride_key_dim,
    stride_decay_page,
    stride_decay_head,
    stride_decay_dim,
    stride_length,
    stride_indices,
    stride_out_row,
    stride_out_head,
    stride_out_v,
    stride_out_k,
    K: tl.constexpr,
    V: tl.constexpr,
    CAPACITY: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
):
    tile, row, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    page = tl.load(indices_ptr + row * stride_indices).to(tl.int64)
    keys = tl.arange(0, BK)
    values = tile * BV + tl.arange(0, BV)
    key_mask, value_mask = keys < K, values < V
    state_mask = value_mask[:, None] & key_mask[None, :]
    result = tl.zeros((BV, BK), tl.float32)
    if page > 0:
        state_addresses = (
            state_ptr
            + page * stride_state_page
            + head * stride_state_head
            + values[:, None] * stride_state_v
            + keys[None, :] * stride_state_k
        )
        length = tl.load(lengths_ptr + page * stride_length)
        tl.device_assert(
            (length >= 0) & (length <= CAPACITY), "invalid retained KDA interval"
        )
        if length > 0:
            result = _recover_weighted_kda_state(
                state_addresses,
                state_mask,
                correction_ptr + page * stride_u_page + head * stride_u_head,
                key_ptr + page * stride_key_page + head * stride_key_head,
                decay_ptr
                + page * stride_decay_page
                + head * stride_decay_head
                + keys * stride_decay_dim,
                keys,
                values,
                key_mask,
                value_mask,
                length,
                stride_u_token,
                stride_u_dim,
                stride_key_token,
                stride_key_dim,
                BT,
                USE_TENSOR_CORES,
            )
        else:
            result = tl.load(state_addresses, mask=state_mask, other=0.0).to(tl.float32)
    output_addresses = (
        out_ptr
        + row * stride_out_row
        + head * stride_out_head
        + values[:, None] * stride_out_v
        + keys[None, :] * stride_out_k
    )
    tl.store(output_addresses, result, mask=state_mask)


def materialize_history(
    base_state: torch.Tensor,
    correction: torch.Tensor,
    weighted_key: torch.Tensor,
    total_decay: torch.Tensor,
    lengths: torch.Tensor,
    read_indices: torch.Tensor,
    *,
    out: torch.Tensor,
    use_tensor_cores: bool,
) -> torch.Tensor:
    rows = read_indices.numel()
    if not rows:
        return out
    heads, value_dim, key_dim = base_state.shape[1:]
    capacity = correction.shape[2]
    block_v = min(16, triton.next_power_of_2(value_dim))
    _materialize_history_kernel[(triton.cdiv(value_dim, block_v), rows, heads)](
        base_state,
        correction,
        weighted_key,
        total_decay,
        lengths,
        read_indices,
        out,
        *base_state.stride(),
        *correction.stride(),
        *weighted_key.stride(),
        *total_decay.stride(),
        lengths.stride(0),
        read_indices.stride(0),
        *out.stride(),
        K=key_dim,
        V=value_dim,
        CAPACITY=capacity,
        BK=triton.next_power_of_2(key_dim),
        BV=block_v,
        BT=max(8, triton.next_power_of_2(capacity)),
        USE_TENSOR_CORES=use_tensor_cores,
        num_warps=4 if use_tensor_cores else 1,
    )
    return out


@triton.jit(do_not_specialize=["rows"])
def _clear_history_kernel(
    lengths_ptr, indices_ptr, rows, stride_length, stride_indices, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    page = tl.load(indices_ptr + row * stride_indices, mask=row < rows, other=-1).to(
        tl.int64
    )
    tl.store(lengths_ptr + page * stride_length, 0, mask=(row < rows) & (page > 0))


def clear_history(lengths: torch.Tensor, write_indices: torch.Tensor) -> None:
    rows = write_indices.numel()
    if rows:
        _clear_history_kernel[(triton.cdiv(rows, 128),)](
            lengths,
            write_indices,
            rows,
            lengths.stride(0),
            write_indices.stride(0),
            BLOCK=128,
        )


@triton.jit(do_not_specialize=["batch", "stride_length"])
def _clear_batched_history_kernel(
    descriptors,
    group_indices,
    write_indices,
    batch,
    stride_length,
    BLOCK: tl.constexpr,
):
    layer = tl.program_id(1)
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    group = tl.load(group_indices + layer).to(tl.int64)
    page = tl.load(write_indices + group * batch + row, mask=row < batch, other=-1).to(
        tl.int64
    )
    lengths = tl.load(descriptors + layer * 4 + 3).to(tl.pointer_type(tl.int32))
    tl.store(lengths + page * stride_length, 0, mask=(row < batch) & (page > 0))


def clear_batched_history(
    descriptors: torch.Tensor,
    group_indices: torch.Tensor,
    write_indices: torch.Tensor,
    *,
    length_page_stride: int,
) -> None:
    """Reset all layer histories after a wide window commits a dense endpoint."""
    batch = write_indices.shape[1]
    _clear_batched_history_kernel[(triton.cdiv(batch, 128), descriptors.shape[0])](
        descriptors,
        group_indices,
        write_indices,
        batch,
        length_page_stride,
        BLOCK=128,
    )
