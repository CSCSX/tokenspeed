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

"""Absorb accepted KDA updates into a bounded weighted history."""

from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import (
    _kda_gate,
    _recoverssm_decay_product,
)


@triton.jit(do_not_specialize=["batch"])
def append_kda_history_kernel(
    correction_ptr,
    kg_ptr,
    metadata_ptr,
    stamps_ptr,
    history_correction_ptr,
    history_key_ptr,
    history_decay_ptr,
    history_lengths_ptr,
    write_indices_ptr,
    append_offsets_ptr,
    accepted_ptr,
    a_ptr,
    bias_ptr,
    descriptors_ptr,
    history_descriptors_ptr,
    group_indices_ptr,
    batch,
    stride_stamps_layer,
    stride_append_layer,
    HEADS: tl.constexpr,
    stride_correction_slot,
    stride_correction_head,
    stride_correction_token,
    stride_correction_dim,
    stride_kg_slot,
    stride_kg_head,
    stride_kg_token,
    stride_kg_dim,
    stride_history_correction_page,
    stride_history_correction_head,
    stride_history_correction_token,
    stride_history_correction_dim,
    stride_history_key_page,
    stride_history_key_head,
    stride_history_key_token,
    stride_history_key_dim,
    stride_history_decay_page,
    stride_history_decay_head,
    stride_history_decay_dim,
    stride_history_length,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
    RECORD_CAPACITY: tl.constexpr,
    HISTORY_CAPACITY: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    layer_head = tl.program_id(1)
    layer = layer_head // HEADS
    head = layer_head % HEADS
    if descriptors_ptr is not None:
        correction_ptr = tl.load(descriptors_ptr + layer * 10 + 3).to(
            tl.pointer_type(tl.float32)
        )
        kg_ptr = tl.load(descriptors_ptr + layer * 10 + 5).to(
            tl.pointer_type(tl.bfloat16)
        )
        a_ptr = tl.load(descriptors_ptr + layer * 10 + 6).to(
            tl.pointer_type(tl.float32)
        )
        bias_ptr = tl.load(descriptors_ptr + layer * 10 + 7).to(
            tl.pointer_type(tl.float32)
        )
        history_correction_ptr = tl.load(history_descriptors_ptr + layer * 4).to(
            tl.pointer_type(tl.float32)
        )
        history_key_ptr = tl.load(history_descriptors_ptr + layer * 4 + 1).to(
            tl.pointer_type(tl.float32)
        )
        history_decay_ptr = tl.load(history_descriptors_ptr + layer * 4 + 2).to(
            tl.pointer_type(tl.float32)
        )
        history_lengths_ptr = tl.load(history_descriptors_ptr + layer * 4 + 3).to(
            tl.pointer_type(tl.int32)
        )
        group = tl.load(group_indices_ptr + layer).to(tl.int64)
        write_indices_ptr += group * batch
        stamps_ptr += layer * stride_stamps_layer
        append_offsets_ptr += layer * stride_append_layer
    page = tl.load(write_indices_ptr + row).to(tl.int64)
    if page <= 0:
        return
    accepted = tl.load(accepted_ptr + row)
    start = tl.load(append_offsets_ptr + row)
    tl.device_assert(
        (accepted >= 0)
        & (accepted <= RECORD_CAPACITY)
        & (start >= 0)
        & (start + accepted <= HISTORY_CAPACITY),
        "invalid accepted KDA history interval",
    )
    slot = tl.load(metadata_ptr + row * 3)
    generation = tl.load(metadata_ptr + row * 3 + 1)
    position = tl.load(metadata_ptr + row * 3 + 2)
    tokens = tl.arange(0, BT)
    dims = tl.arange(0, BD)
    live = tokens < accepted
    ring = (position + tokens) % RECORD_CAPACITY
    stamp = stamps_ptr + (slot * RECORD_CAPACITY + ring) * 2
    stamped_position = tl.load(stamp, mask=live, other=-1)
    stamped_generation = tl.load(stamp + 1, mask=live, other=-1)
    valid = (
        tl.sum(
            (
                live
                & (
                    (stamped_position != position + tokens)
                    | (stamped_generation != generation)
                )
            ).to(tl.int32)
        )
        == 0
    )
    tl.device_assert(valid, "stale candidate KDA record")
    if not valid:
        return
    correction = tl.load(
        correction_ptr
        + slot * stride_correction_slot
        + head * stride_correction_head
        + ring[:, None] * stride_correction_token
        + dims[None, :] * stride_correction_dim,
        mask=live[:, None] & (dims[None, :] < V),
        other=0.0,
    )
    raw_key = tl.load(
        kg_ptr
        + slot * stride_kg_slot
        + head * stride_kg_head
        + ring[:, None] * stride_kg_token
        + dims[None, :] * stride_kg_dim,
        mask=live[:, None] & (dims[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    raw_gate = tl.load(
        kg_ptr
        + slot * stride_kg_slot
        + head * stride_kg_head
        + ring[:, None] * stride_kg_token
        + (K + dims[None, :]) * stride_kg_dim,
        mask=live[:, None] & (dims[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    key = raw_key * tl.rsqrt(tl.sum(raw_key * raw_key, axis=1) + 1e-6)[:, None]
    A = tl.exp(tl.load(a_ptr + head))
    bias = tl.load(bias_ptr + head * K + dims, mask=dims < K, other=0.0)
    gate = _kda_gate(raw_gate, bias[None, :], A, lower_bound, USE_LOWER_BOUND)
    decay = tl.where(live[:, None], tl.exp(gate), 1.0)
    inclusive = tl.associative_scan(
        decay, axis=0, combine_fn=_recoverssm_decay_product, reverse=True
    )
    next_row = tl.broadcast_to(tl.minimum(tokens + 1, BT - 1)[:, None], (BT, BD))
    suffix = tl.gather(inclusive, next_row, axis=0)
    suffix = tl.where((tokens + 1 < accepted)[:, None], suffix, 1.0)
    window_decay = tl.sum(tl.where(tokens[:, None] == 0, inclusive, 0.0), axis=0)

    # One writer per request/head scales the accepted prefix exactly once.
    # Rejected candidate rows never participate in these products.
    old_tokens = tl.arange(0, BH)
    old_key_ptrs = (
        history_key_ptr
        + page * stride_history_key_page
        + head * stride_history_key_head
        + old_tokens[:, None] * stride_history_key_token
        + dims[None, :] * stride_history_key_dim
    )
    old_mask = (old_tokens[:, None] < start) & (dims[None, :] < K)
    old_keys = tl.load(old_key_ptrs, mask=old_mask, other=0.0)
    tl.store(old_key_ptrs, old_keys * window_decay[None, :], mask=old_mask)
    decay_ptrs = (
        history_decay_ptr
        + page * stride_history_decay_page
        + head * stride_history_decay_head
        + dims * stride_history_decay_dim
    )
    old_decay = tl.load(decay_ptrs, mask=(start > 0) & (dims < K), other=1.0)
    tl.store(decay_ptrs, old_decay * window_decay, mask=dims < K)
    output_tokens = start + tokens
    tl.store(
        history_correction_ptr
        + page * stride_history_correction_page
        + head * stride_history_correction_head
        + output_tokens[:, None] * stride_history_correction_token
        + dims[None, :] * stride_history_correction_dim,
        correction,
        mask=live[:, None] & (dims[None, :] < V),
    )
    tl.store(
        history_key_ptr
        + page * stride_history_key_page
        + head * stride_history_key_head
        + output_tokens[:, None] * stride_history_key_token
        + dims[None, :] * stride_history_key_dim,
        key * suffix,
        mask=live[:, None] & (dims[None, :] < K),
    )
    if head == 0:
        tl.store(history_lengths_ptr + page * stride_history_length, start + accepted)
