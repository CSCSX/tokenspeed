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

"""TokenSpeed adapters for the vendored vLLM KDA RecoverSSM kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import (
    NULL_BLOCK_ID,
    _commit_kda_state_kernel,
    _prepare_commit_plan_kernel,
    kda_recoverssm_verify,
)

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.attention.kda import (
        KdaReplayHistory,
        KdaReplayHistoryBatch,
        KdaReplayKeyGateFormat,
        KdaReplayRecords,
    )


_DENSE_BF16_SIGNATURES = format_signatures(("q", "k", "v"), "dense", {torch.bfloat16})


def _use_tensor_cores_for_commit(
    max_commit_tokens: int, key_dim: int, value_dim: int, num_states: int
) -> bool:
    # Use the configured recovery window, not the ring buffer's capacity or
    # a GPU-to-host read of accepted lengths. GB300 measurements favor CUDA
    # cores for short windows and small grids. Keep other shapes on that path.
    return (
        16 <= max_commit_tokens <= 32
        and key_dim == value_dim == 128
        and num_states >= 96
    )


@register_kernel(
    "attention",
    "kda_recoverssm_verify",
    name="vllm_triton_kda_recoverssm_verify",
    solution="vllm_triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_BF16_SIGNATURES,
    priority=Priority.REFERENCE,
    traits={
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def vllm_triton_kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    lower_bound: float | None,
    out: torch.Tensor,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    history: KdaReplayHistory | None = None,
) -> torch.Tensor:
    """Verify candidates and save current-window recovery records.

    q/k/v and raw_g pack tokens as [1, tokens, heads, dim]; raw_beta omits
    the final dim. A_log and dt_bias parameterize the gate. lower_bound
    selects the bounded gate, or None selects the softplus gate.

    checkpoint_state uses [page, head, value, key]. state_indices selects
    request pages, with zero reserved for padding. query_start_loc gives
    packed-token offsets; spec_query_len bounds each request's window.
    correction_cache is FP32 in request/head/token/channel order. kg_cache
    stores unnormalized K and raw gate in the activation dtype. Verify computes
    normalization and gate decay without modifying the checkpoint.

    record_metadata is INT64 [batch, 3] (slot, generation, absolute start).
    record_stamps is INT64 [slots, capacity, 2] (position, generation).
    Records use position % capacity; verify stamps each written candidate.
    Both may be None only for standalone unstamped kernel diagnostics.

    If history is supplied, reconstruct the accepted input endpoint before
    candidates. A capacity flush or page rotation writes that endpoint into
    the private output page; candidate records remain uncommitted.
    Return out after writing verify results.
    """
    return kda_recoverssm_verify(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        checkpoint_state,
        correction_cache,
        kg_cache,
        query_start_loc,
        state_indices,
        spec_query_len,
        out=out,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        history_correction=None if history is None else history.correction,
        history_key=None if history is None else history.key,
        history_decay=None if history is None else history.decay,
        history_lengths=None if history is None else history.lengths,
        history_write_indices=None if history is None else history.write_indices,
        history_append_offsets=None if history is None else history.append_offsets,
        history_use_tensor_cores=False if history is None else history.use_tensor_cores,
        enable_pdl=pdl_enabled(),
    )


@register_kernel(
    "attention",
    "kda_recoverssm_commit",
    name="vllm_triton_kda_recoverssm_commit",
    solution="vllm_triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_BF16_SIGNATURES,
    priority=Priority.REFERENCE,
    traits={
        "attention_only": frozenset({True}),
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def vllm_triton_kda_recoverssm_commit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    commit_lens: torch.Tensor,
    final_state_indices: torch.Tensor,
    boundary_state_indices: torch.Tensor,
    boundary_recovery_lens: torch.Tensor,
    state_base_addrs: torch.Tensor,
    correction_cache_base_addrs: torch.Tensor,
    kg_cache_base_addrs: torch.Tensor,
    spec_query_len: int,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    gate_parameters: tuple[torch.Tensor, torch.Tensor, float | None],
) -> None:
    """Commit one layer's accepted FP32 U and activation-dtype K/raw-gate records.

    q/k/v identify the registry's activation format; recovery does not read
    them. checkpoint_state uses [page, head, value, key]. correction_cache
    and kg_cache use [request, head, token, channel], with K then raw gate in
    the latter. The three base-address tables each contain this layer's
    pool pointer. Tensor strides describe those pools.

    state_indices selects live source/destination pages (> 0); non-positive
    pages are padding. query_start_loc gives each request's token boundaries.
    num_accepted_tokens is clamped to that window and spec_query_len by the
    planner. commit_lens, final_state_indices, boundary_state_indices and
    boundary_recovery_lens are caller-owned int32 planner scratch.

    record_metadata is INT64 [batch, 3] (slot, generation, checkpoint position).
    record_stamps is INT64 [slots, capacity, 2] (position, generation).
    Recovery checks every accepted stamp before reading payloads. Invalid
    records raise a device assertion. Both may be None for unstamped diagnostics.

    gate_parameters supplies (A_log, dt_bias, lower_bound). Weight parameters
    use FP32; lower_bound=None selects softplus instead of the bounded gate.
    The kernel reads these tensors directly so CUDA graph capture reuses the
    caller-owned buffers.
    The configured window and head dimensions select CUDA-core or TF32x3
    Tensor Core recovery. Accepted lengths stay runtime GPU inputs.
    Return None after updating checkpoint_state.
    """
    del q, k, v
    batch = state_indices.shape[0]
    if (record_metadata is None) != (record_stamps is None):
        raise ValueError("record metadata and stamps must be supplied together")
    if record_metadata is not None:
        if record_metadata.shape != (batch, 3) or record_stamps.shape != (
            correction_cache.shape[0],
            correction_cache.shape[2],
            2,
        ):
            raise ValueError("replay metadata does not match the record view")
        if any(
            (
                x.dtype != torch.int64
                or not x.is_contiguous()
                or x.device != checkpoint_state.device
                for x in (record_metadata, record_stamps)
            )
        ):
            raise ValueError(
                "record metadata must be contiguous INT64 on the state device"
            )
    if batch == 0:
        return
    _prepare_commit_plan_kernel[batch,](
        num_accepted_tokens,
        None,
        state_indices,
        query_start_loc,
        None,
        None,
        commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
        NULL_BLOCK_ID,
        1,
        1,
        num_accepted_tokens.stride(0),
        0,
        state_indices.stride(0),
        query_start_loc.stride(0),
        0,
        0,
        0,
        SPEC_QUERY_LEN=spec_query_len,
        num_warps=1,
    )
    _, num_heads, value_dim, key_dim = checkpoint_state.shape
    block_k = triton.next_power_of_2(key_dim)
    grid = lambda meta: (triton.cdiv(meta["V"], meta["BV"]), batch, num_heads)
    a_log, dt_bias, lower_bound = gate_parameters
    _commit_kda_state_kernel[grid](
        checkpoint_state,
        state_base_addrs,
        checkpoint_state.stride(0),
        correction_cache,
        correction_cache_base_addrs,
        correction_cache.stride(0),
        kg_cache,
        kg_cache_base_addrs,
        kg_cache.stride(0),
        state_indices,
        commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
        NULL_BLOCK_ID,
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        kg_cache.stride(1),
        kg_cache.stride(2),
        kg_cache.stride(3),
        state_indices.stride(0),
        None,
        0,
        1,
        record_metadata,
        record_stamps,
        0,
        correction_cache.shape[2],
        correction_cache.shape[0],
        a_ptr=a_log,
        bias_ptr=dt_bias,
        USE_GATE_BASE_ADDRESSES=False,
        lower_bound=lower_bound,
        USE_LOWER_BOUND=lower_bound is not None,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BT=max(8, triton.next_power_of_2(spec_query_len)),
        USE_TENSOR_CORES=_use_tensor_cores_for_commit(
            spec_query_len, key_dim, value_dim, batch * num_heads
        ),
        NUM_HEADS=num_heads,
        ALIGN_MODE=False,
        BATCH=batch,
        LAUNCH_DEPENDENT_KERNELS=False,
        debug=True,
        sanitize_overflow=False,
        num_stages=2,
    )


__all__ = ["vllm_triton_kda_recoverssm_commit", "vllm_triton_kda_recoverssm_verify"]


def batched_recoverssm_commit(
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
    key_gate_format: KdaReplayKeyGateFormat,
    state_stride: int,
    conv_width: int,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    lower_bound: float | None,
    history: KdaReplayHistoryBatch | None = None,
) -> None:
    """Recover all layers from request-major raw records and commit conv.

    Descriptors retain the runtime's ten-column layout. Columns 0, 2 and 8
    hold raw QKV, conv state and recurrent state. Columns 3 and 5 hold
    correction and key/gate records. Raw records also use columns 6 and 7
    for FP32 A_log and dt_bias. Other columns are unused here
    but retained for the shared descriptor ABI. All strides count elements;
    record token strides may include row padding.

    group_indices maps each layer to a row of read_indices/write_indices.
    Those tables are contiguous [groups, batch]; live pages must be positive.
    Padding uses negative source/destination pages and accepted count zero.
    The shared state-page planner must clamp accepted_length to [0, draft_token_num] before this
    call. A zero count copies the source endpoint without reading records.

    State pages use contiguous [head, value, key] FP32 values; conv pages
    use contiguous [channel, conv_width - 1] BF16 values. U is FP32 and
    key_gate_format selects BF16 raw K/gate or FP32 normalized K/decay.
    lower_bound selects the raw-gate transform. Source and
    destination pages may alias within a request, but
    requests must not share writable pages.

    record_metadata is INT64 [batch, 3] (slot, generation, checkpoint position).
    record_stamps is INT64 [layers, slots, capacity, 2] (position, generation).
    Each recurrent layer validates its accepted interval before reading records.
    Both may be None only for standalone unstamped kernel diagnostics.

    For 16--32-token windows with 128-by-128 state and at least 96 independent
    states (layers * batch * heads), recovery uses one TF32x3 Tensor Core
    product. Other shapes use CUDA-core updates. The matrix product multiplies
    FP32 U by normalized keys weighted by subsequent decays, then adds the
    decayed checkpoint. Buffer capacity does not select the implementation;
    accepted lengths remain GPU inputs. Selection is fixed during graph capture.
    With history, verify already handled any checkpoint materialization.
    Commit appends accepted records across all layers in one launch and
    commits convolution state at the same endpoint. A verify window wider
    than the retained capacity commits a dense endpoint and clears its length.
    Return None after record/state and convolution commits on the caller stream.
    """
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        batched_kda_commit_conv_window_kernel,
    )

    if key_gate_format not in ("raw_key_gate", "normalized_key_decay"):
        raise ValueError(f"unknown KDA key/gate format: {key_gate_format}")
    if key_gate_format == "normalized_key_decay" and (
        history is None or draft_token_num > history.capacity
    ):
        raise ValueError("normalized records require retained-history commit")
    if head_dim != 128 or conv_width != 4:
        raise ValueError("batched KDA commit requires head_dim=128 and conv_width=4")
    layers = descriptors.shape[0]
    if descriptors.shape != (layers, 10) or not descriptors.is_contiguous():
        raise ValueError("descriptors must be a contiguous [layers, 10] pointer table")
    if group_indices.shape != (layers,) or not group_indices.is_contiguous():
        raise ValueError("group_indices must be a contiguous per-layer vector")
    batch = accepted_length.numel()
    if (
        accepted_length.shape != (batch,)
        or read_indices.ndim != 2
        or read_indices.shape[1] != batch
        or (write_indices.shape != read_indices.shape)
        or any(
            (
                not tensor.is_contiguous()
                for tensor in (accepted_length, read_indices, write_indices)
            )
        )
    ):
        raise ValueError(
            "commit metadata must be contiguous [groups, batch] and [batch]"
        )
    if not batch:
        return
    if (record_metadata is None) != (record_stamps is None):
        raise ValueError("record metadata and stamps must be supplied together")
    capacity = draft_token_num if record_stamps is None else record_stamps.shape[2]
    slots = batch if record_stamps is None else record_stamps.shape[1]
    if record_metadata is not None:
        if record_metadata.shape != (batch, 3) or record_stamps.shape != (
            layers,
            slots,
            capacity,
            2,
        ):
            raise ValueError(
                "batched replay metadata needs [batch, 3] and stamps [layers, slots, capacity, 2]"
            )
        if capacity < draft_token_num:
            raise ValueError("record capacity must cover the candidate window")
        if any(
            (
                x.dtype != torch.int64
                or not x.is_contiguous()
                or x.device != descriptors.device
                for x in (record_metadata, record_stamps)
            )
        ):
            raise ValueError(
                "record metadata must be contiguous INT64 on the descriptor device"
            )
    if history is not None and (record_metadata is None or record_stamps is None):
        raise ValueError("persistent history requires stamped candidate records")
    if history is not None and draft_token_num <= history.capacity:
        _append_batched_recoverssm_history(
            descriptors,
            group_indices,
            write_indices,
            accepted_length,
            record_metadata,
            record_stamps,
            history,
            num_heads=num_heads,
            head_dim=head_dim,
            correction_stride=correction_stride,
            key_gate_stride=key_gate_stride,
            key_gate_format=key_gate_format,
            lower_bound=lower_bound,
        )
    else:
        # A window wider than history capacity was flushed into its private
        # base before verify. Recover its accepted prefix directly there.
        recurrent_reads = write_indices if history is not None else read_indices
        grid = lambda meta: (
            triton.cdiv(head_dim, meta["BV"]),
            batch,
            layers * num_heads,
        )
        typed = torch.empty(0, dtype=torch.float32, device=descriptors.device)
        key_typed = torch.empty(0, dtype=torch.bfloat16, device=descriptors.device)
        _commit_kda_state_kernel[grid](
            typed,
            descriptors[:, 8],
            state_stride,
            typed,
            descriptors[:, 3],
            capacity * correction_stride,
            key_typed,
            descriptors[:, 5],
            capacity * key_gate_stride,
            recurrent_reads,
            accepted_length,
            write_indices,
            None,
            None,
            NULL_BLOCK_ID,
            head_dim * head_dim,
            head_dim,
            1,
            head_dim,
            correction_stride,
            1,
            2 * head_dim,
            key_gate_stride,
            1,
            recurrent_reads.stride(1),
            group_indices,
            recurrent_reads.stride(0),
            descriptors.stride(0),
            record_metadata,
            record_stamps,
            0 if record_stamps is None else record_stamps.stride(0),
            capacity,
            slots,
            a_ptr=descriptors[:, 6],
            bias_ptr=descriptors[:, 7],
            USE_GATE_BASE_ADDRESSES=True,
            lower_bound=lower_bound,
            USE_LOWER_BOUND=lower_bound is not None,
            K=head_dim,
            V=head_dim,
            BK=triton.next_power_of_2(head_dim),
            BT=max(8, triton.next_power_of_2(draft_token_num)),
            USE_TENSOR_CORES=_use_tensor_cores_for_commit(
                draft_token_num, head_dim, head_dim, batch * layers * num_heads
            ),
            NUM_HEADS=num_heads,
            ALIGN_MODE=False,
            BATCH=batch,
            LAUNCH_DEPENDENT_KERNELS=False,
            debug=True,
            sanitize_overflow=False,
            num_stages=2,
        )
    if history is not None and draft_token_num > history.capacity:
        from tokenspeed_kernel.ops.attention.kda._triton.history_state import (
            clear_batched_history,
        )

        clear_batched_history(
            history.descriptors,
            group_indices,
            write_indices,
            length_page_stride=history.length_page_stride,
        )
    batched_kda_commit_conv_window_kernel[
        layers, batch, triton.cdiv(3 * num_heads * head_dim, 128)
    ](
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        batch,
        T=draft_token_num,
        STRIDE_QKV=qkv_stride,
        STRIDE_CONV=conv_stride,
        CONV_DIM=3 * num_heads * head_dim,
        BLOCK=128,
        num_warps=1,
    )


def _append_batched_recoverssm_history(
    descriptors: torch.Tensor,
    group_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    record_metadata: torch.Tensor,
    record_stamps: torch.Tensor,
    history: KdaReplayHistoryBatch,
    *,
    num_heads: int,
    head_dim: int,
    correction_stride: int,
    key_gate_stride: int,
    key_gate_format: KdaReplayKeyGateFormat,
    lower_bound: float | None,
) -> None:
    from tokenspeed_kernel.ops.attention.kda._triton.buffered import (
        append_kda_history_kernel,
    )

    batch = accepted_length.numel()
    layers = descriptors.shape[0]
    capacity = record_stamps.shape[2]
    append_kda_history_kernel[batch, layers * num_heads](
        None,
        None,
        record_metadata,
        record_stamps,
        None,
        None,
        None,
        None,
        write_indices,
        history.append_offsets,
        accepted_length,
        None,
        None,
        descriptors,
        history.descriptors,
        group_indices,
        batch,
        record_stamps.stride(0),
        history.append_offsets.stride(0),
        num_heads,
        key_gate_format == "normalized_key_decay",
        capacity * correction_stride,
        head_dim,
        correction_stride,
        1,
        capacity * key_gate_stride,
        2 * head_dim,
        key_gate_stride,
        1,
        history.correction_page_stride,
        history.capacity * head_dim,
        head_dim,
        1,
        history.key_page_stride,
        history.capacity * head_dim,
        head_dim,
        1,
        history.decay_page_stride,
        history.capacity * head_dim,
        head_dim,
        1,
        history.length_page_stride,
        lower_bound,
        lower_bound is not None,
        RECORD_CAPACITY=capacity,
        HISTORY_CAPACITY=history.capacity,
        K=head_dim,
        V=head_dim,
        BT=triton.next_power_of_2(capacity),
        BD=triton.next_power_of_2(head_dim),
        num_warps=1,
        debug=True,
        sanitize_overflow=False,
    )


def commit_recoverssm_history(
    records: KdaReplayRecords,
    accepted_length: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> None:
    """Append accepted records to the recurrent history after sampling.

    records.history owns the persistent page-indexed buffers. Verify must
    have populated append_offsets on the caller stream. accepted_length
    contains the stop-clipped count including the target input, with zero
    for padding. A_log, dt_bias and lower_bound must match verify's gate.
    Store normalized keys and each token's decay in FP32, preserving older
    accepted factors. The caller commits convolution state at
    the same endpoint. All buffers must exist before graph capture.
    """
    from tokenspeed_kernel.ops.attention.kda._triton.buffered import (
        append_kda_history_kernel,
    )

    history = records.history
    if history is None:
        raise ValueError("history commit requires persistent record destinations")
    batch = accepted_length.numel()
    if batch == 0:
        return
    _, heads, capacity, value_dim = records.correction.shape
    key_dim = records.key_gate.shape[-1] // 2
    append_kda_history_kernel[batch, heads](
        records.correction,
        records.key_gate,
        records.metadata,
        records.stamps,
        history.correction,
        history.key,
        history.decay,
        history.lengths,
        history.write_indices,
        history.append_offsets,
        accepted_length,
        A_log,
        dt_bias,
        None,
        None,
        None,
        batch,
        0,
        0,
        heads,
        records.key_gate_format == "normalized_key_decay",
        *records.correction.stride(),
        *records.key_gate.stride(),
        *history.correction.stride(),
        *history.key.stride(),
        *history.decay.stride(),
        history.lengths.stride(0),
        lower_bound,
        lower_bound is not None,
        RECORD_CAPACITY=capacity,
        HISTORY_CAPACITY=history.correction.shape[2],
        K=key_dim,
        V=value_dim,
        BT=triton.next_power_of_2(capacity),
        BD=triton.next_power_of_2(max(key_dim, value_dim)),
        num_warps=1,
        debug=True,
        sanitize_overflow=False,
    )


@triton.jit(do_not_specialize=["batch", "generation"])
def _prepare_record_metadata_kernel(
    committed, metadata, batch, generation, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < batch
    position = tl.load(committed + row, mask=live, other=0)
    tl.store(metadata + row * 3, row.to(tl.int64), mask=live)
    tl.store(metadata + row * 3 + 1, generation.to(tl.int64), mask=live)
    tl.store(metadata + row * 3 + 2, position, mask=live)


def prepare_recoverssm_metadata(
    committed: torch.Tensor, metadata: torch.Tensor, *, batch: int, generation: int
) -> None:
    """Refresh the per-round slot, generation and checkpoint-position view.

    committed contains INT64 checkpoint positions. metadata is caller-owned,
    contiguous INT64 [max_batch, 3] storage. Only the first batch rows change.
    generation identifies this round's workspace ownership. Return None.
    """
    if committed.dtype != torch.int64 or metadata.dtype != torch.int64:
        raise ValueError("record metadata requires INT64 positions and storage")
    if metadata.ndim != 2 or metadata.shape[1] != 3 or (not metadata.is_contiguous()):
        raise ValueError("record metadata must be contiguous [max_batch, 3]")
    if batch > min(committed.numel(), metadata.shape[0]) or generation < 0:
        raise ValueError("invalid record metadata batch or generation")
    if batch == 0:
        return
    _prepare_record_metadata_kernel[triton.cdiv(batch, 128),](
        committed, metadata, batch, generation, BLOCK=128
    )
