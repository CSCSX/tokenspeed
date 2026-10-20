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

"""GFX950 Gluon KDA decode kernels using V-major recurrent state."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops._kda_records import (
    kda_record_decay,
    kda_record_interval,
    kda_records_launch_metadata,
    validate_kda_records_commit,
    validate_kda_records_verify,
)

cdna4 = gl.amd.cdna4

_CDNA4_NUM_CUS = 256


@gluon.jit
def _kda_conv_step(
    history0, history1, history2, current, weight0, weight1, weight2, weight3
):
    """Apply one four-tap depthwise convolution step and SiLU."""
    value = (
        history0 * weight0 + history1 * weight1 + history2 * weight2 + current * weight3
    ).to(gl.float32)
    return value * (1.0 / (1.0 + gl.exp(-value)))


@gluon.jit
def _kda_decay(
    raw_gate, dt_bias, a_value, HAS_LOWER_BOUND: gl.constexpr, LOWER_BOUND: gl.constexpr
):
    """Convert the projected raw gate to the recurrent multiplicative decay."""
    gate = raw_gate + dt_bias
    if HAS_LOWER_BOUND:
        log_decay = LOWER_BOUND / (1.0 + gl.exp(-(a_value * gate)))
    else:
        softplus = gl.maximum(gate, 0.0) + gl.log(1.0 + gl.exp(-gl.abs(gate)))
        log_decay = -a_value * softplus
    return gl.exp(log_decay)


@gluon.jit
def _kda_recurrent_step(running, key, query, value, beta, key_query):
    """Apply one delta-rule update and return the updated state and output."""
    prediction = gl.sum(running * key[None, :], axis=1)
    prior_output = gl.sum(running * query[None, :], axis=1)
    delta = beta * (value - prediction)
    running += delta[:, None] * key[None, :]
    return running, prior_output + delta * key_query


@gluon.jit
def _kda_qkvd_indices(head_idx, H: gl.constexpr, D: gl.constexpr):
    """Map Q/K/V/decay lanes to their input channels."""
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [4], [0])
    offsets = gl.arange(0, 4 * D, layout=layout)
    slot_id = offsets // D
    local_offset = offsets - slot_id * D
    projection_width: gl.constexpr = H * D
    qkv_channel = (
        gl.where(
            slot_id == 1,
            projection_width,
            gl.where(slot_id == 2, 2 * projection_width, 0),
        )
        + head_idx * D
        + local_offset
    )
    return qkv_channel, slot_id, local_offset, slot_id == 3


@gluon.jit
def _kda_value_panels(vectors, D: gl.constexpr, VALUE_LAYOUT: gl.constexpr):
    """Load the eight 16-wide value panels from a packed Q/K/V/decay row."""
    return (
        vectors.slice(2 * D + 0, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 16, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 32, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 48, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 64, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 80, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 96, 16, dim=0).load(VALUE_LAYOUT),
        vectors.slice(2 * D + 112, 16, dim=0).load(VALUE_LAYOUT),
    )


@gluon.jit
def gluon_kda_paged_decode_gfx950(
    q,
    k,
    v,
    raw_g,
    raw_beta,
    state_pool,
    read_indices,
    write_indices,
    output,
    cu_seqlens,
    a_log,
    dt_bias,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    Q_TOKEN_STRIDE: gl.constexpr,
    K_TOKEN_STRIDE: gl.constexpr,
    V_TOKEN_STRIDE: gl.constexpr,
    G_TOKEN_STRIDE: gl.constexpr,
    BETA_TOKEN_STRIDE: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    STATE_PAGE_STRIDE: gl.constexpr,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
):
    """One-token KDA decode with direct V-major indexed state-pool IO."""
    value_block = gl.program_id(0)
    sequence_head = gl.program_id(1)
    sequence_idx = sequence_head // H
    head_idx = sequence_head % H

    if BK == 128 and BV == 32:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [4, 16],
            [8, 8],
            [1, 1],
            [1, 0],
        )
    elif BK == 128 and BV == 8:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [1, 8],
            [8, 8],
            [1, 1],
            [1, 0],
        )
    else:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1],
            [64, 1],
            [gl.num_warps(), 1],
            [1, 0],
        )
    value_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)

    begin = gl.load(cu_seqlens + sequence_idx)
    end = gl.load(cu_seqlens + sequence_idx + 1)
    value_offsets = value_block * BV + gl.arange(0, BV, layout=value_layout)
    value_mask = value_offsets < V
    if begin == end:
        gl.store(
            output + (sequence_idx * H + head_idx) * V + value_offsets,
            0.0,
            mask=value_mask,
        )
        return

    read_idx = gl.load(read_indices + sequence_idx)
    write_idx = gl.load(write_indices + sequence_idx)
    valid_read = (read_idx >= 0) & (read_idx < NUM_SLOTS)
    if not valid_read:
        gl.store(
            output + (sequence_idx * H + head_idx) * V + value_offsets,
            0.0,
            mask=value_mask,
        )
        return

    key_offsets = gl.arange(0, BK, layout=key_layout)
    key_mask = key_offsets < K
    read_base = read_idx * STATE_PAGE_STRIDE + head_idx * K * V

    token_idx = begin
    q_value = gl.load(
        q + token_idx * Q_TOKEN_STRIDE + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    k_value = gl.load(
        k + token_idx * K_TOKEN_STRIDE + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    gate_value = gl.load(
        raw_g + token_idx * G_TOKEN_STRIDE + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    gate_value += gl.load(
        dt_bias + head_idx * K + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(gl.float32)
    a_value = gl.exp(gl.load(a_log + head_idx).to(gl.float32))
    if HAS_LOWER_BOUND:
        log_decay = LOWER_BOUND / (1.0 + gl.exp(-(a_value * gate_value)))
    else:
        softplus = gl.maximum(gate_value, 0.0) + gl.log(
            1.0 + gl.exp(-gl.abs(gate_value))
        )
        log_decay = -a_value * softplus
    beta_value = gl.load(raw_beta + token_idx * BETA_TOKEN_STRIDE + head_idx).to(
        gl.float32
    )
    beta_value = 1.0 / (1.0 + gl.exp(-beta_value))

    scale: gl.constexpr = K**-0.5
    q_value *= gl.rsqrt(gl.sum(q_value * q_value, axis=0) + 1e-6) * scale
    k_value *= gl.rsqrt(gl.sum(k_value * k_value, axis=0) + 1e-6)
    valid_write = (write_idx >= 0) & (write_idx < NUM_SLOTS)
    safe_write_idx = gl.where(valid_write, write_idx, 0)
    write_base = safe_write_idx * STATE_PAGE_STRIDE + head_idx * K * V
    decay = gl.exp(log_decay)
    state_mask = value_mask[:, None] & key_mask[None, :]
    read_offsets = value_offsets[:, None] * K + key_offsets[None, :]
    running = cdna4.buffer_load(
        state_pool + read_base,
        read_offsets.to(gl.int32),
        mask=state_mask,
        other=0.0,
    ).to(gl.float32)
    v_value = gl.load(
        v + token_idx * V_TOKEN_STRIDE + head_idx * V + value_offsets,
        mask=value_mask,
        other=0.0,
    ).to(gl.float32)
    running *= decay[None, :]
    prediction = gl.sum(running * k_value[None, :], axis=1)
    delta = beta_value * (v_value - prediction)
    running += delta[:, None] * k_value[None, :]
    out_value = gl.sum(running * q_value[None, :], axis=1)
    gl.store(
        output + (sequence_idx * H + head_idx) * V + value_offsets,
        out_value.to(output.dtype.element_ty),
        mask=value_mask,
    )
    write_offsets = value_offsets[:, None] * K + key_offsets[None, :]
    cdna4.buffer_store(
        running,
        state_pool + write_base,
        write_offsets.to(gl.int32),
        mask=valid_write & state_mask,
    )


@gluon.jit
def gluon_kda_fused_paged_decode_vmajor_gfx950(
    mixed_qkv,
    conv_weights,
    conv_states,
    f_a_out,
    f_b_weight,
    beta_logits,
    state_pool,
    read_indices,
    write_indices,
    output,
    cu_seqlens,
    a_log,
    dt_bias,
    H: gl.constexpr,
    D: gl.constexpr,
    F_A: gl.constexpr,
    MIXED_ROW_STRIDE: gl.constexpr,
    CONV_WEIGHT_ROW_STRIDE: gl.constexpr,
    CONV_WEIGHT_COL_STRIDE: gl.constexpr,
    CONV_PAGE_STRIDE: gl.constexpr,
    CONV_CHANNEL_STRIDE: gl.constexpr,
    CONV_HISTORY_STRIDE: gl.constexpr,
    F_A_ROW_STRIDE: gl.constexpr,
    BETA_ROW_STRIDE: gl.constexpr,
    STATE_PAGE_STRIDE: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
    PIPELINE_DEPTH: gl.constexpr,
):
    """Fuse the K3 decode convolution, decay projection, and recurrence."""
    head_idx = gl.program_id(0)
    sequence_idx = gl.program_id(1)

    # Physical state tiles are [V, K].
    state_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [4, 16],
        [4, 1],
        [1, 0],
    )
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)
    value_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    key_offsets = gl.arange(0, D, layout=key_layout)
    compact_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    output_offsets = gl.arange(0, D, layout=compact_layout)
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    # Pack Q, K, and V into one row (the decay lanes go unused); decay and
    # output each move between layouts through their own D-wide allocation.
    shared_vectors = gl.allocate_shared_memory(gl.float32, [1, 4 * D], shared_layout)
    decay_alloc = gl.allocate_shared_memory(gl.float32, [1, D], shared_layout)
    output_alloc = gl.allocate_shared_memory(gl.float32, [1, D], shared_layout)

    # Loads retire in order, so every small load is issued ahead of the state
    # panels, and the per-head parameters ahead of the padding-row checks.
    # The decay projection takes this head's [D, F_A] slice of f_b_weight.
    # Four lanes share a weight row, so the reduction leaves each thread only
    # two decay channels to finish instead of the state tile's eight.
    projection_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [16, 4],
        [4, 1],
        [1, 0],
    )
    gate_rows = gl.arange(0, D, layout=gl.SliceLayout(1, projection_layout))
    f_a_offsets = gl.arange(0, F_A, layout=gl.SliceLayout(0, projection_layout))
    f_b_tile = gl.load(
        f_b_weight + (head_idx * D + gate_rows)[:, None] * F_A + f_a_offsets[None, :]
    )
    gate_bias = gl.load(dt_bias + head_idx * D + gate_rows)
    a_value = gl.load(a_log + head_idx)
    # Process Q/K/V together so they share convolution loads. The fourth
    # (decay) lanes use Q as an in-bounds dummy address and skip state writes.
    qkv_channel, slot_id, _, _ = _kda_qkvd_indices(head_idx, H, D)
    qkv_weight_base = conv_weights + qkv_channel * CONV_WEIGHT_ROW_STRIDE
    qkv_weight0 = gl.load(qkv_weight_base)
    qkv_weight1 = gl.load(qkv_weight_base + CONV_WEIGHT_COL_STRIDE)
    qkv_weight2 = gl.load(qkv_weight_base + 2 * CONV_WEIGHT_COL_STRIDE)
    qkv_weight3 = gl.load(qkv_weight_base + 3 * CONV_WEIGHT_COL_STRIDE)

    begin = gl.load(cu_seqlens + sequence_idx)
    end = gl.load(cu_seqlens + sequence_idx + 1)
    read_idx = gl.load(read_indices + sequence_idx)
    write_idx = gl.load(write_indices + sequence_idx)
    output_base = (sequence_idx * H + head_idx) * D
    if begin == end:
        gl.store(output + output_base + output_offsets, 0.0)
        return

    valid_read = (read_idx >= 0) & (read_idx < NUM_SLOTS)
    if not valid_read:
        gl.store(output + output_base + output_offsets, 0.0)
        return

    valid_write = (write_idx >= 0) & (write_idx < NUM_SLOTS)
    safe_write_idx = gl.where(valid_write, write_idx, 0)
    read_page_offset = read_idx.to(gl.int64)
    write_page_offset = safe_write_idx.to(gl.int64)
    read_base = read_page_offset * STATE_PAGE_STRIDE + head_idx * D * D
    panel_value_offsets = gl.arange(0, 16, layout=value_layout)

    token_idx = begin
    f_a_value = gl.load(f_a_out + token_idx * F_A_ROW_STRIDE + f_a_offsets)
    beta_value = gl.load(beta_logits + token_idx * BETA_ROW_STRIDE + head_idx)
    qkv_input = gl.load(mixed_qkv + token_idx * MIXED_ROW_STRIDE + qkv_channel)
    qkv_read_base = (
        conv_states
        + read_page_offset * CONV_PAGE_STRIDE
        + qkv_channel * CONV_CHANNEL_STRIDE
    )
    qkv_history0 = gl.load(qkv_read_base)
    qkv_history1 = gl.load(qkv_read_base + CONV_HISTORY_STRIDE)
    qkv_history2 = gl.load(qkv_read_base + 2 * CONV_HISTORY_STRIDE)

    # Keep PIPELINE_DEPTH state panels in flight to balance latency and VGPR use.
    # static_range permits the unrolled tuple window to change length.
    off_pipe = ()
    raw_pipe = ()
    for _pd in gl.static_range(PIPELINE_DEPTH):
        _off_pd = (panel_value_offsets[:, None] + _pd * 16) * D + key_offsets[None, :]
        _raw_pd = gl.load(state_pool + read_base + _off_pd).to(gl.float32)
        off_pipe = off_pipe + (_off_pd,)
        raw_pipe = raw_pipe + (_raw_pd,)

    qkv_value = _kda_conv_step(
        qkv_history0.to(gl.float32),
        qkv_history1.to(gl.float32),
        qkv_history2.to(gl.float32),
        qkv_input.to(gl.float32),
        qkv_weight0,
        qkv_weight1,
        qkv_weight2,
        qkv_weight3,
    )
    qkv_write_base = (
        conv_states
        + write_page_offset * CONV_PAGE_STRIDE
        + qkv_channel * CONV_CHANNEL_STRIDE
    )
    write_mask = valid_write & (slot_id != 3)
    gl.store(qkv_write_base, qkv_history1, mask=write_mask)
    gl.store(qkv_write_base + CONV_HISTORY_STRIDE, qkv_history2, mask=write_mask)
    gl.store(qkv_write_base + 2 * CONV_HISTORY_STRIDE, qkv_input, mask=write_mask)

    # Round like the BF16 GEMM that prefill and verify use for the same gate.
    raw_gate = gl.sum(
        f_b_tile.to(gl.float32) * f_a_value.to(gl.float32)[None, :], axis=1
    )
    raw_gate = raw_gate.to(f_a_out.dtype.element_ty).to(gl.float32)
    decay_value = _kda_decay(
        raw_gate,
        gate_bias.to(gl.float32),
        gl.exp(a_value.to(gl.float32)),
        HAS_LOWER_BOUND,
        LOWER_BOUND,
    )
    beta_value = 1.0 / (1.0 + gl.exp(-beta_value.to(gl.float32)))

    shared_vectors.index(0).store(qkv_value)
    decay_alloc.index(0).store(decay_value)
    q_value = shared_vectors.index(0).slice(0, D, dim=0).load(key_layout)
    k_value = shared_vectors.index(0).slice(D, D, dim=0).load(key_layout)
    decay = decay_alloc.index(0).load(key_layout)
    q_square = gl.sum(q_value * q_value, axis=0)
    k_square = gl.sum(k_value * k_value, axis=0)
    raw_key_query = gl.sum(q_value * k_value, axis=0)
    q_scale = gl.rsqrt(q_square + 1e-6) * (D**-0.5)
    k_scale = gl.rsqrt(k_square + 1e-6)
    q_value *= q_scale
    k_value *= k_scale
    key_query = raw_key_query * q_scale * k_scale
    write_base = write_page_offset * STATE_PAGE_STRIDE + head_idx * D * D
    output_shared = output_alloc.index(0)
    value_panels = _kda_value_panels(shared_vectors.index(0), D, value_layout)
    output_panels = (
        output_shared.slice(0, 16, dim=0),
        output_shared.slice(16, 16, dim=0),
        output_shared.slice(32, 16, dim=0),
        output_shared.slice(48, 16, dim=0),
        output_shared.slice(64, 16, dim=0),
        output_shared.slice(80, 16, dim=0),
        output_shared.slice(96, 16, dim=0),
        output_shared.slice(112, 16, dim=0),
    )
    for panel_idx in gl.static_range(8):
        cur_off = off_pipe[0]
        cur_raw = raw_pipe[0]
        running = cur_raw * decay[None, :]
        prediction = gl.sum(running * k_value[None, :], axis=1)
        prior_output = gl.sum(running * q_value[None, :], axis=1)
        delta = beta_value * (value_panels[panel_idx] - prediction)
        running += delta[:, None] * k_value[None, :]
        panel_out = prior_output + delta * key_query
        gl.store(
            state_pool + write_base + cur_off,
            running,
            mask=valid_write,
            cache_modifier=".cs",
        )
        output_panels[panel_idx].store(panel_out)

        next_idx = panel_idx + PIPELINE_DEPTH
        if next_idx < 8:
            next_off = (panel_value_offsets[:, None] + next_idx * 16) * D + key_offsets[
                None, :
            ]
            next_raw = gl.load(state_pool + read_base + next_off).to(gl.float32)
            off_pipe = off_pipe[1:] + (next_off,)
            raw_pipe = raw_pipe[1:] + (next_raw,)
        else:
            off_pipe = off_pipe[1:] + (off_pipe[-1],)
            raw_pipe = raw_pipe[1:] + (raw_pipe[-1],)
    out_value = output_shared.load(compact_layout)
    gl.store(
        output + output_base + output_offsets,
        out_value.to(output.dtype.element_ty),
    )


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "qkv_stride",
        "gate_stride",
        "beta_stride",
        "state_stride",
        "state_head_stride",
        "state_v_stride",
        "state_k_stride",
        "read_stride",
        "query_stride",
        "u_slot_stride",
        "u_head_stride",
        "u_pos_stride",
        "kg_slot_stride",
        "kg_head_stride",
        "kg_pos_stride",
        "record_slots",
        "lower_bound",
    ],
)
def gluon_kda_fused_paged_verify_nostore_vmajor_gfx950(
    conv_qkv,
    raw_g,
    beta,
    a_log,
    dt_bias,
    state_pool,
    reads,
    correction,
    key_gate,
    metadata,
    stamps,
    query_start_loc,
    output,
    qkv_stride,
    gate_stride,
    beta_stride,
    state_stride,
    state_head_stride,
    state_v_stride,
    state_k_stride,
    read_stride,
    query_stride,
    u_slot_stride,
    u_head_stride,
    u_pos_stride,
    kg_slot_stride,
    kg_head_stride,
    kg_pos_stride,
    record_slots,
    lower_bound,
    H: gl.constexpr,
    D: gl.constexpr,
    T: gl.constexpr,
    CAPACITY: gl.constexpr,
    BOUNDED: gl.constexpr,
    BV: gl.constexpr,
):
    """Verify a candidate prefix without writing live state; save U and K/raw-g."""
    pv, row, head = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [4, 1], [1, 0])
    kk = gl.arange(0, D, layout=gl.SliceLayout(0, layout))
    vv = pv * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    bos = gl.load(query_start_loc + row * query_stride).to(gl.int64)
    length = gl.load(query_start_loc + (row + 1) * query_stride).to(gl.int64) - bos
    source = gl.load(reads + row * read_stride).to(gl.int64)
    if source <= 0:
        for t in gl.static_range(T):
            gl.store(
                output + (bos + t) * H * D + head * D + vv,
                0.0,
                mask=(t < length) & (vv < D),
            )
        return
    slot = gl.load(metadata + row * 3)
    generation = gl.load(metadata + row * 3 + 1)
    start = gl.load(metadata + row * 3 + 2)
    valid = (
        (slot >= 0)
        & (slot < record_slots)
        & (generation >= 0)
        & (start >= 0)
        & (length >= 0)
        & (length <= T)
    )
    gl.device_assert(valid, "invalid replay metadata")
    if not valid:
        return
    state = gl.load(
        state_pool
        + source * state_stride
        + head * state_head_stride
        + vv[:, None] * state_v_stride
        + kk[None, :] * state_k_stride,
        mask=vv[:, None] < D,
        other=0,
    ).to(gl.float32)
    a = gl.exp(gl.load(a_log + head).to(gl.float32))
    bias = gl.load(dt_bias + head * D + kk).to(gl.float32)
    for t in gl.static_range(T):
        live = t < length
        token = bos + t
        q = gl.load(
            conv_qkv + token * qkv_stride + head * D + kk, mask=live, other=0
        ).to(gl.float32)
        k = gl.load(
            conv_qkv + token * qkv_stride + H * D + head * D + kk, mask=live, other=0
        ).to(gl.float32)
        v = gl.load(
            conv_qkv + token * qkv_stride + 2 * H * D + head * D + vv,
            mask=live & (vv < D),
            other=0,
        ).to(gl.float32)
        g = gl.load(raw_g + token * gate_stride + head * D + kk, mask=live, other=0).to(
            gl.float32
        )
        b = gl.load(beta + token * beta_stride + head, mask=live, other=0).to(
            gl.float32
        )
        q *= gl.rsqrt(gl.sum(q * q, 0) + 1e-6) * (D**-0.5)
        normalized_k = k * gl.rsqrt(gl.sum(k * k, 0) + 1e-6)
        state *= kda_record_decay(g, bias, a, lower_bound, BOUNDED)[None, :]
        u = v - gl.sum(state * normalized_k[None, :], 1)
        u *= 1.0 / (1.0 + gl.exp(-b))
        state += u[:, None] * normalized_k[None, :]
        out = gl.sum(state * q[None, :], 1)
        gl.store(output + token * H * D + head * D + vv, out, mask=live & (vv < D))
        position = start + t
        ring = position % CAPACITY
        gl.store(
            correction
            + slot * u_slot_stride
            + head * u_head_stride
            + ring * u_pos_stride
            + vv,
            u,
            mask=live & (vv < D),
        )
        if pv == 0:
            kg = (
                key_gate
                + slot * kg_slot_stride
                + head * kg_head_stride
                + ring * kg_pos_stride
            )
            gl.store(kg + kk, k, mask=live)
            gl.store(kg + D + kk, g, mask=live)
            if head == 0:
                stamp = stamps + (slot * CAPACITY + ring) * 2
                gl.store(stamp, position, mask=live)
                gl.store(stamp + 1, generation, mask=live)


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "BATCH",
        "state_stride",
        "u_stride",
        "kg_stride",
        "record_slots",
        "stamp_layer_stride",
        "lower_bound",
    ],
)
def gluon_kda_fused_replay_gfx950(
    descriptors,
    group_indices,
    reads,
    writes,
    accepted,
    metadata,
    stamps,
    activation_ref,
    BATCH,
    state_stride,
    u_stride,
    kg_stride,
    record_slots,
    stamp_layer_stride,
    lower_bound,
    H: gl.constexpr,
    D: gl.constexpr,
    T: gl.constexpr,
    CAPACITY: gl.constexpr,
    BOUNDED: gl.constexpr,
    BV: gl.constexpr,
):
    """Recover accepted states from saved corrections for all layers."""
    pv, row, lh = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    layer, head = lh // H, lh % H
    group = gl.load(group_indices + layer).to(gl.int64)
    source = gl.load(reads + group * BATCH + row).to(gl.int64)
    target = gl.load(writes + group * BATCH + row).to(gl.int64)
    if source <= 0 or target <= 0:
        return
    steps = gl.load(accepted + row)
    slot, start, valid = kda_record_interval(
        metadata,
        stamps,
        row,
        layer,
        steps,
        record_slots,
        CAPACITY,
        stamp_layer_stride,
        T,
        64,
    )
    if not valid:
        return
    desc = descriptors + layer * 10
    state_pool = gl.load(desc + 8).to(gl.pointer_type(gl.float32))
    u_ptr = gl.load(desc + 3).to(gl.pointer_type(gl.float32))
    kg_ptr = gl.load(desc + 5).to(gl.pointer_type(activation_ref.dtype.element_ty))
    a_ptr = gl.load(desc + 6).to(gl.pointer_type(gl.float32))
    bias_ptr = gl.load(desc + 7).to(gl.pointer_type(gl.float32))
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [4, 1], [1, 0])
    kk = gl.arange(0, D, layout=gl.SliceLayout(0, layout))
    vv = pv * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    offsets = head * D * D + vv[:, None] * D + kk[None, :]
    state = gl.load(
        state_pool + source * state_stride + offsets, mask=vv[:, None] < D, other=0
    ).to(gl.float32)
    a = gl.exp(gl.load(a_ptr + head))
    bias = gl.load(bias_ptr + head * D + kk)
    for t in range(steps):
        ring = (start + t) % CAPACITY
        kg = kg_ptr + (slot * CAPACITY + ring) * kg_stride + head * 2 * D
        k = gl.load(kg + kk).to(gl.float32)
        g = gl.load(kg + D + kk).to(gl.float32)
        u = gl.load(
            u_ptr + (slot * CAPACITY + ring) * u_stride + head * D + vv,
            mask=vv < D,
            other=0,
        ).to(gl.float32)
        k *= gl.rsqrt(gl.sum(k * k, 0) + 1e-6)
        state *= kda_record_decay(g, bias, a, lower_bound, BOUNDED)[None, :]
        state += u[:, None] * k[None, :]
    gl.store(state_pool + target * state_stride + offsets, state, mask=vv[:, None] < D)


@gluon.jit(
    launch_metadata=kda_records_launch_metadata,
    do_not_specialize=[
        "BATCH",
        "qkv_stride",
        "conv_stride",
        "record_slots",
        "stamp_layer_stride",
    ],
)
def gluon_kda_commit_conv_history_gfx950(
    descriptors,
    group_indices,
    reads,
    writes,
    accepted,
    metadata,
    stamps,
    activation_ref,
    history_ref,
    BATCH,
    qkv_stride,
    conv_stride,
    record_slots,
    stamp_layer_stride,
    H: gl.constexpr,
    D: gl.constexpr,
    T: gl.constexpr,
    CAPACITY: gl.constexpr,
    W: gl.constexpr,
):
    """Commit original QKV history after recurrent recovery has finished reading."""
    tile, row, layer = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    group = gl.load(group_indices + layer).to(gl.int64)
    source = gl.load(reads + group * BATCH + row).to(gl.int64)
    target = gl.load(writes + group * BATCH + row).to(gl.int64)
    if source <= 0 or target <= 0:
        return
    steps = gl.load(accepted + row)
    _, _, valid = kda_record_interval(
        metadata,
        stamps,
        row,
        layer,
        steps,
        record_slots,
        CAPACITY,
        stamp_layer_stride,
        T,
        64,
    )
    if not valid:
        return
    desc = descriptors + layer * 10
    qkv = gl.load(desc).to(gl.pointer_type(activation_ref.dtype.element_ty))
    conv = gl.load(desc + 2).to(gl.pointer_type(history_ref.dtype.element_ty))
    channels = tile * 256 + gl.arange(
        0, 256, layout=gl.BlockedLayout([1], [64], [4], [0])
    )
    live = channels < 3 * H * D
    # Load all old taps before storing any tap: source and target can alias.
    values = ()
    for tap in gl.static_range(W - 1):
        position = steps + tap
        old = gl.load(
            conv + source * conv_stride + channels * (W - 1) + position,
            mask=live & (position < W - 1),
            other=0,
        )
        new = gl.load(
            qkv + (row.to(gl.int64) * T + position - (W - 1)) * qkv_stride + channels,
            mask=live & (position >= W - 1),
            other=0,
        )
        values += (gl.where(position < W - 1, old.to(gl.float32), new.to(gl.float32)),)
    for tap in gl.static_range(W - 1):
        gl.store(
            conv + target * conv_stride + channels * (W - 1) + tap,
            values[tap],
            mask=live,
        )


def gluon_kda_recurrent_decode_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run one-token indexed KDA decode on a V-major recurrent-state pool.

    Args:
        q, k, g_raw: Packed ``[1, batch, heads, key_dim]`` tensors.
        v: Packed ``[1, batch, heads, value_dim]`` tensor.
        beta_logits: Packed ``[1, batch, heads]`` beta logits.
        A_log: Per-head FP32 decay parameter.
        dt_bias: Per-head, per-key FP32 decay bias.
        state_pool: Persistent FP32 state, physical shape
            ``[pages, heads, value_dim, key_dim]`` (V-major).
        read_indices: Source page per batch row.
        write_indices: Destination page per batch row.
        cu_seqlens: Packed row boundaries. Each active row contains one token.
        lower_bound: Optional safe lower bound for log decay.

    Returns:
        KDA output with the same shape and dtype as ``v``.
    """
    tensors = (q, k, v, g_raw, beta_logits, A_log, dt_bias, state_pool)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx950 Gluon KDA decode requires GPU tensors")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must have shape [1, batch, heads, key_dim]")
    if read_indices.ndim != 1 or write_indices.shape != read_indices.shape:
        raise ValueError("read_indices and write_indices must be matching vectors")
    if q.shape[1] != read_indices.numel():
        raise ValueError("gfx950 Gluon KDA decode requires one token per sequence")
    if q.shape != k.shape or q.shape != g_raw.shape:
        raise ValueError("q, k, and raw_g must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must match q through the head dimension")

    _, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if beta_logits.shape != (1, tokens, heads):
        raise ValueError("beta_logits must have shape [1, batch, heads]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != tokens + 1:
        raise ValueError("cu_seqlens must contain one boundary per decode row")
    expected_tail = (heads, value_dim, key_dim)
    if state_pool.ndim != 4 or state_pool.shape[1:] != expected_tail:
        raise ValueError(
            f"state_pool must have physical shape [pages, H, V, K] with tail {expected_tail}"
        )
    expected_strides = (value_dim * key_dim, key_dim, 1)
    if state_pool.stride()[1:] != expected_strides:
        raise ValueError("state_pool inner [H, V, K] dimensions must be contiguous")
    if state_pool.stride(0) < heads * key_dim * value_dim:
        raise ValueError("state_pool pages must not overlap")
    if A_log.shape != (heads,) or dt_bias.numel() != heads * key_dim:
        raise ValueError("invalid KDA gate parameter shapes")
    expected_inner_strides = (
        (q, key_dim),
        (k, key_dim),
        (g_raw, key_dim),
        (v, value_dim),
    )
    if any(
        tensor.stride(-1) != 1 or tensor.stride(-2) != width
        for tensor, width in expected_inner_strides
    ):
        raise ValueError("KDA inputs must have contiguous head vectors")
    if beta_logits.stride(-1) != 1:
        raise ValueError("KDA beta logits must have contiguous heads")

    A_log = A_log.contiguous()
    dt_bias = dt_bias.view(heads, key_dim).contiguous()
    read_indices = read_indices.to(device=q.device, dtype=torch.int32).contiguous()
    write_indices = write_indices.to(device=q.device, dtype=torch.int32).contiguous()
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()

    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(32, triton.next_power_of_2(value_dim))
    gluon_kda_paged_decode_gfx950[
        (triton.cdiv(value_dim, block_value), tokens * heads)
    ](
        q,
        k,
        v,
        g_raw,
        beta_logits,
        state_pool,
        read_indices,
        write_indices,
        output,
        cu_seqlens,
        A_log,
        dt_bias,
        H=heads,
        K=key_dim,
        V=value_dim,
        Q_TOKEN_STRIDE=q.stride(1),
        K_TOKEN_STRIDE=k.stride(1),
        V_TOKEN_STRIDE=v.stride(1),
        G_TOKEN_STRIDE=g_raw.stride(1),
        BETA_TOKEN_STRIDE=beta_logits.stride(1),
        BK=block_key,
        BV=block_value,
        NUM_SLOTS=state_pool.shape[0],
        STATE_PAGE_STRIDE=state_pool.stride(0),
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        num_warps=1,
        num_stages=2,
    )
    return output


def gluon_kda_fused_decode_gfx950(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run fused single-token K3 decode against paged convolution/KDA state.

    Args:
        mixed_qkv: BF16 pre-convolution Q/K/V rows with shape
            ``[batch, 3 * num_heads * head_dim]``.
        conv_weights: Contiguous four-tap depthwise convolution weights with
            shape ``[3 * num_heads * head_dim, 4]``.
        conv_states: Mutable BF16 convolution state pool with shape
            ``[pages, 3 * num_heads * head_dim, 3]``.
        f_a_out: BF16 low-rank decay-gate activations with shape
            ``[batch, head_dim]``.
        f_b_weight: Contiguous BF16 decay-gate up projection with shape
            ``[num_heads * head_dim, head_dim]``. The kernel computes the raw
            decay gate ``f_a_out @ f_b_weight.T`` with FP32 accumulation and
            rounds it to BF16.
        beta_logits: Per-token, per-head update logits with shape
            ``[batch, num_heads]``.
        A_log: Per-head FP32 decay parameters with shape ``[num_heads]``.
        dt_bias: Per-head, per-channel FP32 decay bias with shape
            ``[num_heads * head_dim]``.
        state_pool: Mutable FP32 recurrent state, physical shape
            ``[pages, num_heads, head_dim(V), head_dim(K)]`` (V-major).
        read_indices: Source state page per batch row. Negative entries mark
            graph-padding rows.
        write_indices: Destination state page per batch row. Negative entries
            suppress state updates.
        num_heads: Number of local KDA heads; this specialization requires 12.
        head_dim: Per-head key/value width; this specialization requires 128.
        cu_seqlens: Packed row boundaries with shape ``[batch + 1]``. Active
            rows contain one token and graph-padding rows contain zero tokens.
        lower_bound: Optional safe lower bound for the log-decay gate.

    Returns:
        Raw KDA output with shape
        ``[1, batch, num_heads, head_dim]`` and ``mixed_qkv`` dtype. The
        convolution and recurrent states at valid ``write_indices`` are
        updated as part of the call.
    """
    tensors = (
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        read_indices,
        write_indices,
        cu_seqlens,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("gfx950 fused KDA decode requires GPU tensors")
    if num_heads != 12 or head_dim != 128:
        raise ValueError("gfx950 fused KDA decode requires 12 heads of width 128")

    tokens = read_indices.numel()
    projection_width = num_heads * head_dim
    if mixed_qkv.ndim != 2 or mixed_qkv.shape != (tokens, 3 * projection_width):
        raise ValueError("mixed_qkv must have shape [batch, 3 * heads * head_dim]")
    if mixed_qkv.stride(1) != 1:
        raise ValueError("mixed_qkv channels must be contiguous")
    if (
        conv_weights.shape != (3 * projection_width, 4)
        or not conv_weights.is_contiguous()
    ):
        raise ValueError("conv_weights must be contiguous [3 * heads * head_dim, 4]")
    if conv_states.ndim != 3 or conv_states.shape[1:] != (3 * projection_width, 3):
        raise ValueError("conv_states must have shape [pages, 3 * heads * head_dim, 3]")
    if conv_states.stride()[1:] != (3, 1):
        raise ValueError(
            "conv_states inner channel/history dimensions must be contiguous"
        )
    if f_a_out.shape != (tokens, head_dim) or f_a_out.stride(1) != 1:
        raise ValueError("f_a_out must have shape [batch, head_dim]")
    if (
        f_b_weight.shape != (projection_width, head_dim)
        or f_b_weight.dtype != f_a_out.dtype
        or not f_b_weight.is_contiguous()
    ):
        raise ValueError(
            "f_b_weight must be contiguous [heads * head_dim, head_dim] in "
            "f_a_out dtype"
        )
    if beta_logits.shape != (tokens, num_heads) or beta_logits.stride(1) != 1:
        raise ValueError("beta_logits must have shape [batch, heads]")
    if A_log.shape != (num_heads,) or not A_log.is_contiguous():
        raise ValueError("A_log must have shape [heads]")
    if dt_bias.shape != (projection_width,) or not dt_bias.is_contiguous():
        raise ValueError("dt_bias must have shape [heads * head_dim]")
    if state_pool.ndim != 4 or state_pool.shape[1:] != (
        num_heads,
        head_dim,
        head_dim,
    ):
        raise ValueError("state_pool must have physical shape [pages, heads, V, K]")
    if state_pool.stride()[1:] != (head_dim * head_dim, head_dim, 1):
        raise ValueError("state_pool inner dimensions must be contiguous")
    if conv_states.shape[0] != state_pool.shape[0]:
        raise ValueError(
            "convolution and recurrent state pools must have equal capacity"
        )
    if read_indices.shape != (tokens,) or write_indices.shape != (tokens,):
        raise ValueError("read_indices and write_indices must match the decode batch")
    if read_indices.dtype != torch.int32 or write_indices.dtype != torch.int32:
        raise ValueError("read_indices and write_indices must be int32")
    if not read_indices.is_contiguous() or not write_indices.is_contiguous():
        raise ValueError("read_indices and write_indices must be contiguous")
    if cu_seqlens.shape != (tokens + 1,) or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be an int32 boundary vector")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")

    output = torch.empty(
        (1, tokens, num_heads, head_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    # Up to one CTA per CU, keep every state panel in flight. Up to two, trade
    # the panel window for occupancy. Beyond that the grid has enough CTAs to
    # keep memory busy, and a four-panel window measured fastest.
    programs = num_heads * tokens
    if programs <= _CDNA4_NUM_CUS:
        pipeline_depth = 8
    elif programs <= 2 * _CDNA4_NUM_CUS:
        pipeline_depth = 1
    else:
        pipeline_depth = 4
    gluon_kda_fused_paged_decode_vmajor_gfx950[(num_heads, tokens)](
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        state_pool,
        read_indices,
        write_indices,
        output,
        cu_seqlens,
        A_log,
        dt_bias,
        H=num_heads,
        D=head_dim,
        F_A=f_a_out.shape[1],
        MIXED_ROW_STRIDE=mixed_qkv.stride(0),
        CONV_WEIGHT_ROW_STRIDE=conv_weights.stride(0),
        CONV_WEIGHT_COL_STRIDE=conv_weights.stride(1),
        CONV_PAGE_STRIDE=conv_states.stride(0),
        CONV_CHANNEL_STRIDE=conv_states.stride(1),
        CONV_HISTORY_STRIDE=conv_states.stride(2),
        F_A_ROW_STRIDE=f_a_out.stride(0),
        BETA_ROW_STRIDE=beta_logits.stride(0),
        STATE_PAGE_STRIDE=state_pool.stride(0),
        NUM_SLOTS=state_pool.shape[0],
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        PIPELINE_DEPTH=pipeline_depth,
        num_warps=4,
        num_stages=2,
    )
    return output


def gluon_kda_fused_verify_gfx950(
    *,
    conv_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    correction: torch.Tensor,
    key_gate: torch.Tensor,
    record_metadata: torch.Tensor,
    record_stamps: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Produce attention output and records without changing committed state.

    conv_qkv contains activation-dtype post-convolution Q/K/V rows; raw_g and
    beta_logits are projected gate inputs. state_pool is FP32 [page, head, V, K].
    read_indices selects checkpoint pages; nonpositive pages are padding.
    correction is FP32 [slot, head, capacity, V]. key_gate has the same layout
    with activation-dtype unnormalized K followed by raw gate in its last axis.
    record_metadata is INT64 [batch, 3] (slot, generation, absolute start), and
    record_stamps is INT64 [slots, capacity, 2] (position, generation).
    query_start_loc supplies packed token boundaries. draft_token_num bounds
    the candidate window; ordinary decode uses one token. lower_bound selects
    the bounded gate, or None selects softplus. Return [1, tokens, heads, dim].
    """
    validate_kda_records_verify(
        conv_qkv,
        raw_g,
        beta_logits,
        state_pool,
        read_indices,
        correction,
        key_gate,
        record_metadata,
        record_stamps,
        query_start_loc,
        A_log,
        dt_bias,
        num_heads,
        head_dim,
        draft_token_num,
    )
    batch = read_indices.numel()
    output = torch.empty(
        (1, conv_qkv.shape[0], num_heads, head_dim),
        device=conv_qkv.device,
        dtype=conv_qkv.dtype,
    )
    if not batch:
        return output
    gluon_kda_fused_paged_verify_nostore_vmajor_gfx950[
        (triton.cdiv(head_dim, 8), batch, num_heads)
    ](
        conv_qkv,
        raw_g,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        read_indices,
        correction,
        key_gate,
        record_metadata,
        record_stamps,
        query_start_loc,
        output,
        conv_qkv.stride(0),
        raw_g.stride(0),
        beta_logits.stride(0),
        state_pool.stride(0),
        state_pool.stride(1),
        state_pool.stride(2),
        state_pool.stride(3),
        read_indices.stride(0),
        query_start_loc.stride(0),
        correction.stride(0),
        correction.stride(1),
        correction.stride(2),
        key_gate.stride(0),
        key_gate.stride(1),
        key_gate.stride(2),
        correction.shape[0],
        lower_bound,
        H=num_heads,
        D=head_dim,
        T=draft_token_num,
        CAPACITY=correction.shape[2],
        BOUNDED=lower_bound is not None,
        BV=8,
        num_warps=4,
        debug=True,
    )
    return output


def launch_gluon_kda_fused_replay_gfx950(
    descriptors: torch.Tensor,
    group_indices: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    *,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    qkv_stride: int,
    conv_stride: int,
    correction_stride: int,
    key_gate_stride: int,
    state_stride: int,
    conv_width: int,
    activation_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    lower_bound: float | None,
) -> None:
    """Recover all layers from saved U/K/raw-g records, then commit conv history.

    Args:
        descriptors: Contiguous uint64 [layers, 10] pointer table. Columns
            0/2/8 hold original QKV, conv history and recurrent state; 3/5
            hold FP32 U and activation-dtype K/raw-g; 6/7 hold A_log/dt_bias.
            Columns 1/4/9 are reserved and are not read.
        group_indices: INT32 cache-group row for each layer.
        read_indices: Contiguous INT32 [groups, batch] checkpoint pages.
        write_indices: Matching destination pages. Nonpositive source or
            destination pages mark padding and leave both pools unchanged.
        accepted_length: Contiguous INT32 [batch] counts in [0, draft_token_num].
            Zero copies the checkpoint and conv history without reading records.
        draft_token_num: Candidate window bound; standard decode uses 1.
        num_heads: Local head count.
        head_dim: Key/value width, currently 128.
        qkv_stride: Original QKV row stride in elements.
        conv_stride: Conv-history page stride in elements.
        correction_stride: FP32 U token stride in elements.
        key_gate_stride: K/raw-g token stride in elements.
        state_stride: FP32 recurrent page stride in elements.
        conv_width: Convolution taps, currently 4.
        activation_dtype: Original QKV and K/raw-g storage dtype.
        conv_state_dtype: Persistent conv-history dtype, independently of inputs.
        record_metadata: INT64 [batch, 3] (slot, generation, absolute start).
        record_stamps: INT64 [layers, slots, capacity, 2] position/generation.
            Both metadata tensors can be None for standalone unstamped probes.
        lower_bound: Bounded log-decay, or None for softplus.

    Returns:
        None. Updates accepted destination states and history. Source and
        destination may alias within a request; writable pages cannot overlap
        between requests. Recovery never recomputes the state-dependent U.
    """
    if head_dim != 128 or conv_width != 4:
        raise ValueError("gfx950 records commit requires D=128 and W=4")
    validate_kda_records_commit(
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        record_metadata,
        record_stamps,
        draft_token_num,
    )
    batch, layers = accepted_length.numel(), descriptors.shape[0]
    if not batch:
        return
    capacity = draft_token_num if record_stamps is None else record_stamps.shape[2]
    slots = batch if record_stamps is None else record_stamps.shape[1]
    stamp_stride = 0 if record_stamps is None else record_stamps.stride(0)
    activation_ref = torch.empty(0, dtype=activation_dtype, device=descriptors.device)
    history_ref = torch.empty(0, dtype=conv_state_dtype, device=descriptors.device)
    gluon_kda_fused_replay_gfx950[
        (triton.cdiv(head_dim, 8), batch, layers * num_heads)
    ](
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        record_metadata,
        record_stamps,
        activation_ref,
        batch,
        state_stride,
        correction_stride,
        key_gate_stride,
        slots,
        stamp_stride,
        lower_bound,
        H=num_heads,
        D=head_dim,
        T=draft_token_num,
        CAPACITY=capacity,
        BOUNDED=lower_bound is not None,
        BV=8,
        num_warps=4,
        debug=True,
    )
    gluon_kda_commit_conv_history_gfx950[
        (triton.cdiv(3 * num_heads * head_dim, 256), batch, layers)
    ](
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        record_metadata,
        record_stamps,
        activation_ref,
        history_ref,
        batch,
        qkv_stride,
        conv_stride,
        slots,
        stamp_stride,
        H=num_heads,
        D=head_dim,
        T=draft_token_num,
        CAPACITY=capacity,
        W=conv_width,
        num_warps=4,
        debug=True,
    )


__all__ = [
    "gluon_kda_fused_decode_gfx950",
    "launch_gluon_kda_fused_replay_gfx950",
    "gluon_kda_fused_verify_gfx950",
    "gluon_kda_recurrent_decode_gfx950",
]
