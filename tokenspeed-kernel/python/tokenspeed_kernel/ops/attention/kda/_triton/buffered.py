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

"""Append accepted KDA updates in token order."""

from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import _kda_gate


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
    NORMALIZED_KEY_DECAY: tl.constexpr,
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
    stride_history_decay_token,
    stride_history_decay_dim,
    stride_history_length,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
    RECORD_CAPACITY: tl.constexpr,
    HISTORY_CAPACITY: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
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
            tl.pointer_type(tl.float32 if NORMALIZED_KEY_DECAY else tl.bfloat16)
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
    slot = tl.load(metadata_ptr + row * 3)
    if slot < 0:
        return
    generation = tl.load(metadata_ptr + row * 3 + 1)
    position = tl.load(metadata_ptr + row * 3 + 2)
    tokens = tl.arange(0, BT)
    dims = tl.arange(0, BD)
    # A current first record distinguishes zero acceptance from stale padding.
    live = tokens < tl.maximum(accepted, 1)
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
    tl.device_assert(valid | (accepted == 0), "stale candidate KDA record")
    if not valid:
        return
    start = tl.load(append_offsets_ptr + row)
    tl.device_assert(
        (accepted >= 0)
        & (accepted <= RECORD_CAPACITY)
        & (start >= 0)
        & (start + accepted <= HISTORY_CAPACITY),
        "invalid accepted KDA history interval",
    )
    # Keep each raw-key reduction one-dimensional, matching dense recovery.
    if not NORMALIZED_KEY_DECAY:
        A = tl.exp(tl.load(a_ptr + head))
        bias = tl.load(bias_ptr + head * K + dims, mask=dims < K, other=0.0)
    for token in range(accepted):
        ring_token = (position + token) % RECORD_CAPACITY
        correction = tl.load(
            correction_ptr
            + slot * stride_correction_slot
            + head * stride_correction_head
            + ring_token * stride_correction_token
            + dims * stride_correction_dim,
            mask=dims < V,
            other=0.0,
        )
        raw_key = tl.load(
            kg_ptr
            + slot * stride_kg_slot
            + head * stride_kg_head
            + ring_token * stride_kg_token
            + dims * stride_kg_dim,
            mask=dims < K,
            other=0.0,
        ).to(tl.float32)
        raw_gate = tl.load(
            kg_ptr
            + slot * stride_kg_slot
            + head * stride_kg_head
            + ring_token * stride_kg_token
            + (K + dims) * stride_kg_dim,
            mask=dims < K,
            other=0.0,
        ).to(tl.float32)
        if NORMALIZED_KEY_DECAY:
            key, decay = raw_key, raw_gate
        else:
            key = raw_key * tl.rsqrt(tl.sum(raw_key * raw_key) + 1e-6)
            decay = tl.exp(_kda_gate(raw_gate, bias, A, lower_bound, USE_LOWER_BOUND))
        output_token = start + token
        tl.store(
            history_correction_ptr
            + page * stride_history_correction_page
            + head * stride_history_correction_head
            + output_token * stride_history_correction_token
            + dims * stride_history_correction_dim,
            correction,
            mask=dims < V,
        )
        tl.store(
            history_key_ptr
            + page * stride_history_key_page
            + head * stride_history_key_head
            + output_token * stride_history_key_token
            + dims * stride_history_key_dim,
            key,
            mask=dims < K,
        )
        tl.store(
            history_decay_ptr
            + page * stride_history_decay_page
            + head * stride_history_decay_head
            + output_token * stride_history_decay_token
            + dims * stride_history_decay_dim,
            decay,
            mask=dims < K,
        )
    if head == 0:
        tl.store(history_lengths_ptr + page * stride_history_length, start + accepted)
