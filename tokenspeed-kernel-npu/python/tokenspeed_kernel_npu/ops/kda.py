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

"""Ascend implementations of Lite KDA operators."""

from __future__ import annotations

import math
from functools import cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tokenspeed_kernel_npu.ops.kda_replay_fused import (
    fused_kda_batched_replay_commit,
    load_fused_kda_replay,
)
from tokenspeed_kernel_npu.public_kda_ops import is_available, load_public_kda_ops

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult

_PREFILL_CHUNK_SIZE = 64
_PREFILL_MIN_PUBLIC_TOKENS = 32


@cache
def _load_flash_causal_conv_ops():
    try:
        import flash_ops
    except ModuleNotFoundError as error:
        if error.name != "flash_ops":
            raise
        return None
    # Older packages expose the same schema but assume dense slots and cannot
    # write Prefill results to independent destination pages.
    if not getattr(flash_ops, "CAUSAL_CONV1D_STATE_SLOT_STRIDE", False):
        return None
    return torch.ops.custom


@cache
def _load_flash_recurrent_kda():
    try:
        import flash_ops
    except ModuleNotFoundError as error:
        if error.name != "flash_ops":
            raise
        return None
    # Select by public entry point; tiling is owned by the installed package.
    recurrent = getattr(flash_ops, "npu_recurrent_kda", None)
    return recurrent if callable(recurrent) else None


@cache
def _load_flash_recurrent_kda_replay():
    try:
        import flash_ops
    except ModuleNotFoundError as error:
        if error.name != "flash_ops":
            raise
        return None
    replay = getattr(flash_ops, "npu_recurrent_kda_replay", None)
    if replay is None:
        replay = getattr(
            __import__(
                "flash_ops.recurrent_kda", fromlist=["npu_recurrent_kda_replay"]
            ),
            "npu_recurrent_kda_replay",
            None,
        )
    return replay if callable(replay) else None


@cache
def _load_flash_batched_recurrent_kda_replay():
    if load_fused_kda_replay() is not None:
        return fused_kda_batched_replay_commit
    try:
        import flash_ops
    except ModuleNotFoundError as error:
        if error.name != "flash_ops":
            raise
        return None
    replay = getattr(flash_ops, "npu_batched_recurrent_kda_replay", None)
    return replay if callable(replay) else None


@cache
def _load_flash_kda_capture_replay_payload():
    try:
        import flash_ops
    except ModuleNotFoundError as error:
        if error.name != "flash_ops":
            raise
        return None
    capture = getattr(flash_ops, "npu_kda_capture_replay_payload", None)
    return capture if callable(capture) else None


@cache
def _load_flash_kda_verify_ops():
    try:
        import flash_ops
    except ModuleNotFoundError as error:
        if error.name != "flash_ops":
            raise
        return None
    conv = getattr(flash_ops, "npu_kda_verify_conv", None)
    recurrent = getattr(flash_ops, "npu_recurrent_kda_verify", None)
    if not callable(conv) or not callable(recurrent):
        return None
    return conv, recurrent


@cache
def _load_kda_v5_t8_experimental_op():
    try:
        import flash_ops
    except ModuleNotFoundError as error:
        if error.name != "flash_ops":
            raise
        return None
    if not bool(getattr(flash_ops, "KDA_SPEC_VERIFY_MIX_DEV", False)):
        return None
    if not bool(getattr(flash_ops, "KDA_SPEC_VERIFY_MIX_V5_T8", False)):
        return None
    packet = getattr(torch.ops.custom, "npu_kda_spec_verify_mix_dev", None)
    op = getattr(packet, "default", None)
    schema = getattr(op, "_schema", None)
    if schema is None:
        return None
    expected_arguments = (
        "f_a",
        "beta_down",
        "f_weight",
        "beta_weight",
        "mixed_qkv",
        "conv_weight",
        "conv_state",
        "recurrent_state",
        "a_log",
        "dt_bias",
        "read_indices",
        "payload",
        "draft_token_num",
        "projection_schedule",
        "live_rows",
        "layer",
        "lower_bound",
        "aic_block_cap",
    )
    schema_arguments = tuple(schema.arguments)
    arguments = tuple(getattr(argument, "name", None) for argument in schema_arguments)
    payload_alias = (
        getattr(schema_arguments[11], "alias_info", None)
        if len(schema_arguments) > 11
        else None
    )
    if (
        getattr(schema, "name", None) != "custom::npu_kda_spec_verify_mix_dev"
        or arguments != expected_arguments
        or len(schema.returns) != 2
        or payload_alias is None
        or not bool(getattr(payload_alias, "is_write", False))
    ):
        return None
    return op


# CANN discovers custom OPP metadata when the device is first initialized.  The
# NPU registry imports this module before model tensors are allocated, so load
# the optional artifacts here instead of waiting for the first kernel call.
# Keep this order: loading flash_ops first corrupts CANN teardown when both
# vendor registries are present.
load_public_kda_ops()
_load_flash_causal_conv_ops()
_load_flash_recurrent_kda()
_load_flash_recurrent_kda_replay()
_load_flash_batched_recurrent_kda_replay()
_load_flash_kda_capture_replay_payload()
_load_flash_kda_verify_ops()
_load_kda_v5_t8_experimental_op()


def kda_v5_t8_experimental_available() -> bool:
    """Report only the exact dev-package V5/T8 capability."""
    return _load_kda_v5_t8_experimental_op() is not None


def kda_v5_t8_experimental_verify(
    *,
    f_a: torch.Tensor,
    beta_down: torch.Tensor,
    f_weight: torch.Tensor,
    beta_weight: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    read_indices: torch.Tensor,
    payload: torch.Tensor,
    draft_token_num: int,
    projection_schedule: int,
    live_rows: int,
    layer: int,
    lower_bound: float,
    aic_block_cap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the raw development schema with its full, explicit ABI."""
    op = _load_kda_v5_t8_experimental_op()
    if op is None:
        raise RuntimeError("development KDA V5/T8 capability is unavailable")
    return op(
        f_a,
        beta_down,
        f_weight,
        beta_weight,
        mixed_qkv,
        conv_weight,
        conv_state,
        recurrent_state,
        a_log,
        dt_bias,
        read_indices,
        payload,
        draft_token_num,
        projection_schedule,
        live_rows,
        layer,
        lower_bound,
        aic_block_cap,
    )


def flash_kda_capture_replay_payload(**kwargs) -> None:
    """Pack one KDA layer's ReplaySSM inputs into its persistent payload row."""
    capture = _load_flash_kda_capture_replay_payload()
    if capture is None:
        raise RuntimeError("flash_ops.npu_kda_capture_replay_payload is unavailable")
    capture(**kwargs)


def flash_kda_fused_paged_verify_no_store(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Ascend target verify using one Conv and one recurrent device kernel."""
    del conv_scratch, state_scratch, write_indices
    ops = _load_flash_kda_verify_ops()
    if ops is None:
        raise RuntimeError("flash_ops KDA verify operators are unavailable")
    if head_dim != 128 or mixed_qkv.shape[1] != 3 * num_heads * head_dim:
        raise ValueError("flash KDA verify requires matching Hx128 QKV geometry")
    if lower_bound is None:
        raise ValueError("flash KDA verify requires a finite lower_bound")
    conv, recurrent = ops
    convolved_qkv = conv(
        mixed_qkv,
        conv_weights,
        conv_states,
        read_indices,
        draft_token_num,
    )
    g_raw = F.linear(f_a_out, f_b_weight)
    return recurrent(
        convolved_qkv,
        g_raw,
        beta_logits,
        A_log,
        dt_bias.view(num_heads, head_dim),
        state_pool,
        read_indices,
        draft_token_num,
        float(lower_bound),
    )


def flash_kda_batched_replay_commit(**kwargs) -> None:
    """Launch the all-layer Ascend ReplaySSM operator."""
    replay = _load_flash_batched_recurrent_kda_replay()
    if replay is None:
        raise RuntimeError("flash_ops.npu_batched_recurrent_kda_replay is unavailable")
    replay(**kwargs)


def _activate(value: torch.Tensor, activation: str | None) -> torch.Tensor:
    return value if activation is None else F.silu(value)


def _safe_indices(
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices_active = (read_indices >= 0) & (write_indices >= 0)
    active = indices_active if active is None else active & indices_active
    zero = torch.zeros((), dtype=read_indices.dtype, device=read_indices.device)
    return (
        active,
        torch.where(active, read_indices, zero).to(torch.int64),
        torch.where(active, write_indices, zero).to(torch.int64),
    )


def _channel_major_conv_state_view(
    conv_state: torch.Tensor, channels: int
) -> torch.Tensor:
    """Share storage as [slots, channels, history] for stride-aware Torch ops.

    This is not a physical relayout for the packaged kernel. Keeping a view
    ensures index_copy_ publishes updates to the original cache allocation.
    """
    if conv_state.shape[1] == channels:
        return conv_state
    return conv_state.transpose(1, 2)


def _decode(
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    conv_state = _channel_major_conv_state_view(conv_state, weight.shape[0])
    active, safe_reads, safe_writes = _safe_indices(read_indices, write_indices)
    history = conv_state.index_select(0, safe_reads)
    window = torch.cat((history, projected.unsqueeze(-1)), dim=-1)
    output = (window.float() * weight.float()).sum(dim=-1)
    if bias is not None:
        output = output + bias.float()
    output = _activate(output, activation).to(projected.dtype)

    destination = conv_state.index_select(0, safe_writes)
    next_state = torch.where(active[:, None, None], window[:, :, 1:], destination)
    conv_state.index_copy_(0, safe_writes, next_state)
    return torch.where(active[:, None], output, torch.zeros_like(output))


def _prefill(
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    bias: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    conv_state = _channel_major_conv_state_view(conv_state, weight.shape[0])
    boundaries = [int(value) for value in cu_seqlens_cpu.tolist()]
    if (
        not boundaries
        or boundaries[0] != 0
        or boundaries[-1] != projected.shape[0]
        or any(left > right for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("KDA causal-conv boundaries must cover packed input")

    active, safe_reads, safe_writes = _safe_indices(read_indices, write_indices)
    initial = (
        torch.ones_like(active) if has_initial_state is None else has_initial_state
    )
    output = torch.zeros_like(projected)
    for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            continue
        read = safe_reads[row : row + 1]
        write = safe_writes[row : row + 1]
        history = conv_state.index_select(0, read)[0]
        history = torch.where(initial[row], history, torch.zeros_like(history))
        signal = torch.cat((history, projected[start:end].transpose(0, 1)), dim=-1)
        convolved = F.conv1d(
            signal.unsqueeze(0),
            weight.unsqueeze(1),
            bias=bias,
            groups=weight.shape[0],
        )[0].transpose(0, 1)
        output[start:end] = torch.where(
            active[row],
            _activate(convolved, activation),
            torch.zeros_like(convolved),
        )
        destination = conv_state.index_select(0, write)[0]
        next_state = torch.where(active[row], signal[:, -3:], destination)
        conv_state.index_copy_(0, write, next_state.unsqueeze(0))
    return output


def _validate_decode_metadata_cpu(
    cu_seqlens: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    pool_size: int,
) -> None:
    if cu_seqlens.device.type != "cpu":
        return
    boundaries = cu_seqlens.tolist()
    reads = read_indices.tolist()
    writes = write_indices.tolist()
    increments = [right - left for left, right in zip(boundaries, boundaries[1:])]
    if (
        not boundaries
        or boundaries[0] != 0
        or any(step not in (0, 1) for step in increments)
    ):
        raise ValueError("KDA Decode boundaries must contain zero/one-token segments")
    if 0 in increments and any(step for step in increments[increments.index(0) :]):
        raise ValueError("KDA Decode graph padding must be a batch tail")

    active_writes = []
    for step, read, write in zip(increments, reads, writes):
        if step:
            if read < 0 or write < 0:
                raise ValueError("KDA Decode active rows require read/write pages")
            if read >= pool_size or write >= pool_size:
                raise ValueError("KDA Decode state index is outside the state pool")
            active_writes.append(write)
        elif read != -1 or write != -1:
            raise ValueError("KDA Decode padded rows require -1 read/write pages")
    if len(active_writes) != len(set(active_writes)):
        raise ValueError("KDA Decode active rows require distinct write pages")


def torch_kda_paged_decode(
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
    """Run graph-safe single-token KDA Decode against K-major state pages."""
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape or g_raw.shape != q.shape:
        raise ValueError("KDA Decode requires q/k/g shaped [1,batch,heads,key_dim]")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("KDA Decode value must match q through the head dimension")
    batch, heads, key_dim, value_dim = q.shape[1], q.shape[2], q.shape[3], v.shape[3]
    if read_indices.shape != (batch,) or write_indices.shape != (batch,):
        raise ValueError("KDA Decode requires one read/write page per batch row")
    if cu_seqlens.shape != (batch + 1,):
        raise ValueError("KDA Decode requires one more boundary than batch rows")
    if beta_logits.shape not in (q.shape, q.shape[:-1]):
        raise ValueError("KDA beta logits must be scalar-per-head or featurewise")
    if beta_logits.shape == q.shape and value_dim != key_dim:
        raise ValueError("Featurewise beta requires equal KDA key/value dims")
    if state_pool.ndim != 4 or state_pool.shape[1:] != (
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError(
            "KDA Decode requires key-major [pages,heads,key_dim,value_dim] state"
        )
    if state_pool.shape[0] < 1 or state_pool.dtype != torch.float32:
        raise ValueError("KDA Decode recurrent state must be a non-empty FP32 pool")
    if A_log.shape != (heads,) or dt_bias.numel() != heads * key_dim:
        raise ValueError("KDA Decode gate parameters do not match the local head shape")
    if lower_bound is None or not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("KDA Decode requires a finite negative lower bound")
    if read_indices.dtype not in (
        torch.int32,
        torch.int64,
    ) or write_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("KDA Decode state indices must use int32 or int64")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("KDA Decode boundaries must use int32 or int64")
    tensors = (
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        read_indices,
        write_indices,
        cu_seqlens,
    )
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("KDA Decode inputs and state must share one device")
    _validate_decode_metadata_cpu(
        cu_seqlens, read_indices, write_indices, state_pool.shape[0]
    )
    if q.device.type == "npu" and _supports_flash_recurrent_kda(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        read_indices,
        write_indices,
        cu_seqlens,
        lower_bound,
    ):
        flash_recurrent = _load_flash_recurrent_kda()
        if flash_recurrent is not None:
            return flash_recurrent(
                q,
                k,
                v,
                g_raw,
                beta_logits,
                A_log,
                dt_bias.reshape(heads, key_dim),
                state_pool,
                read_indices,
                write_indices,
                cu_seqlens,
                state_v_major=False,
                lower_bound=lower_bound,
            )

    segment_active = cu_seqlens[1:] > cu_seqlens[:-1]
    active, safe_reads, safe_writes = _safe_indices(
        read_indices, write_indices, segment_active
    )
    q_fp32 = F.normalize(q.float(), p=2, dim=-1)
    k_fp32 = F.normalize(k.float(), p=2, dim=-1)
    v_fp32 = v.float()
    scalar_beta = None
    if beta_logits.shape == q.shape:
        beta_scale = torch.sqrt(torch.sigmoid(beta_logits.float()) + 1e-10)
        k_fp32 = k_fp32 * beta_scale
        v_fp32 = v_fp32 * beta_scale
    else:
        scalar_beta = torch.sigmoid(beta_logits.float())

    gate_logits = g_raw.float() + dt_bias.float().reshape(heads, key_dim)
    gate = lower_bound * torch.sigmoid(
        A_log.float().reshape(heads, 1).exp() * gate_logits
    )
    state = state_pool.index_select(0, safe_reads)
    state = state * gate[0].exp().unsqueeze(-1)
    key = k_fp32[0]
    delta = v_fp32[0] - torch.matmul(key.unsqueeze(-2), state).squeeze(-2)
    if scalar_beta is not None:
        delta = delta * scalar_beta[0].unsqueeze(-1)
    next_state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    output = torch.matmul(
        (q_fp32[0] * (key_dim**-0.5)).unsqueeze(-2), next_state
    ).squeeze(-2)

    destination = state_pool.index_select(0, safe_writes)
    published = torch.where(active[:, None, None, None], next_state, destination)
    state_pool.index_copy_(0, safe_writes, published)
    output = torch.where(active[:, None, None], output, torch.zeros_like(output))
    return output.unsqueeze(0).to(v.dtype)


def torch_kda_fused_paged_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Correctness implementation of Lite featurewise KDA target verify.

    Committed convolution/recurrent pages are read-only. Every speculative
    position writes its post-token states to rollback scratch; TokenSpeed
    commits only the row selected by the verifier afterward. This is also the
    stable ABI intended for the native Ascend fusion.
    """
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[0] % draft_token_num:
        raise ValueError("KDA verify requires packed [batch * draft, channels]")
    batch = mixed_qkv.shape[0] // draft_token_num
    if not 1 <= draft_token_num <= 8:
        raise ValueError("KDA verify draft width must be in [1, 8]")
    if head_dim != 128 or num_heads not in (4, 8, 16, 32, 64):
        raise ValueError("Lite KDA verify requires D=128 and a supported head count")
    channels = 3 * num_heads * head_dim
    if mixed_qkv.shape[1] != channels or conv_weights.shape != (channels, 4):
        raise ValueError("KDA verify QKV/conv shape does not match H and D")
    rows = batch * draft_token_num
    feature_shape = (rows, num_heads * head_dim)
    if beta_logits.shape != feature_shape:
        raise ValueError("Lite KDA verify requires featurewise beta logits")
    if f_a_out.ndim != 2 or f_b_weight.shape[0] != num_heads * head_dim:
        raise ValueError("KDA verify gate projection shape is invalid")
    if f_a_out.shape[0] != rows or f_a_out.shape[1] != f_b_weight.shape[1]:
        raise ValueError("KDA verify gate inputs do not align with packed rows")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * head_dim:
        raise ValueError("KDA verify gate parameters do not match H and D")
    if lower_bound is None or not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("KDA verify requires a finite negative lower bound")
    if read_indices.shape != (batch,) or write_indices.shape != (
        batch,
        draft_token_num,
    ):
        raise ValueError("KDA verify state indices have the wrong shape")
    if state_pool.shape[1:] != (num_heads, head_dim, head_dim):
        raise ValueError("KDA verify requires K-major [pages,H,K,V] state")
    if state_scratch is not None:
        if state_scratch.shape[1:] != state_pool.shape[1:]:
            raise ValueError("KDA verify recurrent scratch layout must match the pool")
        if state_scratch.dtype != torch.float32:
            raise ValueError("KDA verify recurrent scratch must be FP32")
    if state_pool.dtype != torch.float32:
        raise ValueError("KDA verify recurrent state must be FP32")

    has_history = read_indices >= 0
    safe_reads = torch.where(
        has_history, read_indices, torch.zeros_like(read_indices)
    ).long()
    flat_writes = write_indices.reshape(-1).long()
    # Device-side bounds checks would synchronize the host and are deliberately
    # left to the backend's scratch-grid construction contract.

    # Four-tap causal convolution. The post-token three-value window for each
    # candidate is materialized into rollback scratch before the recurrence.
    conv_pool = _channel_major_conv_state_view(conv_states, channels)
    conv_out_pool = _channel_major_conv_state_view(conv_scratch, channels)
    history = conv_pool.index_select(0, safe_reads)
    history = torch.where(
        has_history[:, None, None], history, torch.zeros_like(history)
    )
    projected = mixed_qkv.view(batch, draft_token_num, channels)
    signal = torch.cat((history, projected.transpose(1, 2)), dim=-1)
    convolved = F.conv1d(
        signal,
        conv_weights.unsqueeze(1),
        groups=channels,
    ).transpose(1, 2)
    convolved = F.silu(convolved).to(mixed_qkv.dtype)
    conv_windows = signal.unfold(-1, 3, 1)[:, :, 1:].permute(0, 2, 1, 3)
    conv_windows = conv_windows.reshape(rows, channels, 3)
    if state_scratch is not None:
        conv_out_pool.index_copy_(0, flat_writes, conv_windows)

    q, k, v = convolved.view(batch, draft_token_num, 3, num_heads, head_dim).unbind(2)
    g_raw = F.linear(f_a_out, f_b_weight).view(
        batch, draft_token_num, num_heads, head_dim
    )
    beta = beta_logits.view(batch, draft_token_num, num_heads, head_dim)
    state = state_pool.index_select(0, safe_reads).float()
    state = torch.where(
        has_history[:, None, None, None], state, torch.zeros_like(state)
    )
    outputs = []
    written_states = []
    scale = head_dim**-0.5
    a = A_log.float().exp().view(1, num_heads, 1)
    dt = dt_bias.float().view(1, num_heads, head_dim)
    for token in range(draft_token_num):
        q_t = q[:, token].float()
        k_t = k[:, token].float()
        v_t = v[:, token].float()
        q_t = q_t / torch.linalg.vector_norm(q_t, dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        k_t = k_t / torch.linalg.vector_norm(k_t, dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        beta_scale = torch.sqrt(torch.sigmoid(beta[:, token].float()) + 1.0e-10)
        k_t = k_t * beta_scale
        v_t = v_t * beta_scale
        decay = torch.exp(
            float(lower_bound) * torch.sigmoid(a * (g_raw[:, token].float() + dt))
        )
        state = state * decay.unsqueeze(-1)
        prediction = torch.matmul(k_t.unsqueeze(-2), state).squeeze(-2)
        state = state + k_t.unsqueeze(-1) * (v_t - prediction).unsqueeze(-2)
        outputs.append(torch.matmul((q_t * scale).unsqueeze(-2), state).squeeze(-2))
        written_states.append(state)

    if state_scratch is not None:
        states = torch.stack(written_states, dim=1).reshape(
            rows, num_heads, head_dim, head_dim
        )
        state_scratch.index_copy_(0, flat_writes, states)
    output = torch.stack(outputs, dim=1)
    return output.reshape(1, rows, num_heads, head_dim).to(mixed_qkv.dtype)


def torch_kda_replay_commit(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_out: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_out: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    gate_scratch: torch.Tensor | None = None,
) -> None:
    """Replay the accepted Lite KDA prefix and publish only its final state.

    This is the portable correctness spelling for the Ascend ReplaySSM ABI.
    It keeps one live recurrent state per request; unlike rollback scratch it
    never materializes ``batch * draft`` recurrent checkpoints.
    """
    del gate_scratch
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[0] % draft_token_num:
        raise ValueError("KDA replay requires packed [batch * draft, channels]")
    batch = mixed_qkv.shape[0] // draft_token_num
    channels = 3 * num_heads * head_dim
    rows = batch * draft_token_num
    if not 1 <= draft_token_num <= 8:
        raise ValueError("KDA replay draft width must be in [1, 8]")
    if head_dim != 128 or num_heads not in (4, 8, 16, 32, 64):
        raise ValueError("Lite KDA replay requires D=128 and a supported head count")
    if mixed_qkv.shape != (rows, channels) or conv_weights.shape != (channels, 4):
        raise ValueError("KDA replay QKV/conv shape does not match H and D")
    if beta_logits.shape != (rows, num_heads * head_dim):
        raise ValueError("Lite KDA replay requires featurewise beta logits")
    if f_a_out.shape[0] != rows or f_b_weight.shape != (
        num_heads * head_dim,
        f_a_out.shape[1],
    ):
        raise ValueError("KDA replay gate projection shape is invalid")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * head_dim:
        raise ValueError("KDA replay gate parameters do not match H and D")
    if lower_bound is None or not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError("KDA replay requires a finite negative lower bound")
    if any(
        value.shape != (batch,)
        for value in (read_indices, write_indices, accepted_length)
    ):
        raise ValueError("KDA replay indices and accepted lengths must match batch")
    if state_pool.shape[1:] != (num_heads, head_dim, head_dim):
        raise ValueError("KDA replay requires K-major [pages,H,K,V] state")
    if state_out.shape[1:] != state_pool.shape[1:]:
        raise ValueError("KDA replay destination layout must match the state pool")
    if state_pool.dtype != torch.float32 or state_out.dtype != torch.float32:
        raise ValueError("KDA replay recurrent state must be FP32")

    has_history = read_indices >= 0
    safe_reads = torch.where(
        has_history, read_indices, torch.zeros_like(read_indices)
    ).long()
    valid_write = write_indices >= 0
    safe_writes = torch.where(
        valid_write, write_indices, torch.zeros_like(write_indices)
    ).long()
    steps = accepted_length.to(torch.int64).clamp(0, draft_token_num)

    conv_pool = _channel_major_conv_state_view(conv_states, channels)
    conv_destination = _channel_major_conv_state_view(conv_out, channels)
    history = conv_pool.index_select(0, safe_reads)
    history = torch.where(
        has_history[:, None, None], history, torch.zeros_like(history)
    )
    state = state_pool.index_select(0, safe_reads).float()
    state = torch.where(
        has_history[:, None, None, None], state, torch.zeros_like(state)
    )
    projected = mixed_qkv.view(batch, draft_token_num, channels)
    gate = F.linear(f_a_out, f_b_weight).view(
        batch, draft_token_num, num_heads, head_dim
    )
    beta = beta_logits.view(batch, draft_token_num, num_heads, head_dim)
    signal = torch.cat((history, projected.transpose(1, 2)), dim=-1)
    convolved = F.silu(
        F.conv1d(signal, conv_weights.unsqueeze(1), groups=channels)
    ).transpose(1, 2)
    conv_windows = signal.unfold(-1, 3, 1)[:, :, 1:].permute(0, 2, 1, 3)

    native_replay = (
        _load_flash_recurrent_kda_replay() if mixed_qkv.device.type == "npu" else None
    )
    if native_replay is not None:
        _, k, v = convolved.view(batch, draft_token_num, 3, num_heads, head_dim).unbind(
            2
        )
        native_replay(
            k,
            v,
            gate,
            beta,
            A_log,
            dt_bias.view(num_heads, head_dim),
            state_pool,
            state_out,
            read_indices,
            write_indices,
            accepted_length,
            float(lower_bound),
        )
        chosen = (steps - 1).clamp_min(0)
        final_history = conv_windows[torch.arange(batch, device=chosen.device), chosen]
        publish = valid_write & (steps > 0)
        old_conv = conv_destination.index_select(0, safe_writes)
        conv_destination.index_copy_(
            0,
            safe_writes,
            torch.where(publish[:, None, None], final_history, old_conv),
        )
        return

    a = A_log.float().exp().view(1, num_heads, 1)
    dt = dt_bias.float().view(1, num_heads, head_dim)

    for token in range(draft_token_num):
        next_history = conv_windows[:, token]
        q, k, v = convolved[:, token].view(batch, 3, num_heads, head_dim).unbind(1)
        q_fp32 = F.normalize(q.float(), p=2, dim=-1)
        k_fp32 = F.normalize(k.float(), p=2, dim=-1)
        beta_scale = torch.sqrt(torch.sigmoid(beta[:, token].float()) + 1.0e-10)
        k_fp32 = k_fp32 * beta_scale
        v_fp32 = v.float() * beta_scale
        decay = torch.exp(
            float(lower_bound) * torch.sigmoid(a * (gate[:, token].float() + dt))
        )
        decayed = state * decay.unsqueeze(-1)
        prediction = torch.matmul(k_fp32.unsqueeze(-2), decayed).squeeze(-2)
        next_state = decayed + k_fp32.unsqueeze(-1) * (v_fp32 - prediction).unsqueeze(
            -2
        )
        take = token < steps
        history = torch.where(take[:, None, None], next_history, history)
        state = torch.where(take[:, None, None, None], next_state, state)

    publish = valid_write & (steps > 0)
    old_conv = conv_destination.index_select(0, safe_writes)
    conv_destination.index_copy_(
        0,
        safe_writes,
        torch.where(publish[:, None, None], history, old_conv),
    )
    old_state = state_out.index_select(0, safe_writes)
    state_out.index_copy_(
        0,
        safe_writes,
        torch.where(publish[:, None, None, None], state, old_state),
    )


def _supports_flash_recurrent_kda(
    q,
    k,
    v,
    g_raw,
    beta_logits,
    A_log,
    dt_bias,
    state_pool,
    read_indices,
    write_indices,
    cu_seqlens,
    lower_bound,
) -> bool:
    """Metadata-only admission after Decode validation; never pack inputs/state."""
    batch, heads, dim = q.shape[1:]
    activations = (q, k, v, g_raw, beta_logits)
    return (
        1 <= batch <= 256
        and heads in (4, 8, 16, 32, 64)
        and dim == 128
        and lower_bound == -5.0
        and all(
            t.shape == q.shape
            and t.dtype == torch.bfloat16
            and t.stride(-1) == 1
            and t.stride(2) >= 128
            and (batch == 1 or t.stride(1) >= (heads - 1) * t.stride(2) + 128)
            for t in activations
        )
        and A_log.dtype == dt_bias.dtype == torch.float32
        and all(
            t.is_contiguous()
            for t in (
                A_log,
                dt_bias,
                read_indices,
                write_indices,
                cu_seqlens,
            )
        )
        and state_pool.stride(3) == 1
        and state_pool.stride(2) == 128
        and state_pool.stride(1) >= 128 * 128
        and state_pool.stride(0) >= (heads - 1) * state_pool.stride(1) + 128 * 128
        and not any(t.requires_grad for t in (*activations, A_log, dt_bias, state_pool))
    )


def ref_kda_causal_conv1d(
    *,
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
    bias: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
    decode: bool,
    require_public: bool = False,
) -> torch.Tensor:
    """Ref composition: graph-safe Decode or per-request depthwise-conv Prefill."""
    del cu_seqlens, require_public
    if decode:
        return _decode(
            projected,
            weight,
            conv_state,
            read_indices,
            write_indices,
            bias,
            activation,
        )
    assert cu_seqlens_cpu is not None
    return _prefill(
        projected,
        weight,
        conv_state,
        read_indices,
        write_indices,
        cu_seqlens_cpu,
        bias,
        has_initial_state,
        activation,
    )


def public_kda_causal_conv1d(
    *,
    projected: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
    bias: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
    decode: bool,
    require_public: bool = False,
) -> torch.Tensor:
    """Use the packaged op on width-major state, including strided arena slots."""
    channels, width = weight.shape
    if conv_state.ndim != 3 or conv_state.shape[1:] != (width - 1, channels):
        raise ValueError(
            "public causal-conv requires state [slots, width - 1, channels]"
        )
    if (
        conv_state.stride(2) != 1
        or conv_state.stride(1) != channels
        or conv_state.stride(0) < (width - 1) * channels
    ):
        raise ValueError("public causal-conv requires state strides [S, channels, 1]")
    op_name = "npu_causal_conv1d"
    flash_ops = _load_flash_causal_conv_ops()
    op = getattr(flash_ops, op_name, None) if flash_ops is not None else None
    if op is None:
        if require_public:
            raise RuntimeError(
                f"flash_ops schema custom::{op_name} with strided-state support is unavailable; "
                "install the matching updated wheel and custom OPP"
            )
        return ref_kda_causal_conv1d(
            projected=projected,
            weight=weight,
            conv_state=conv_state,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            bias=bias,
            has_initial_state=has_initial_state,
            activation=activation,
            decode=decode,
        )
    if decode:
        return op(
            projected,
            weight.transpose(0, 1).contiguous(),
            conv_state,
            bias=bias,
            cache_indices=read_indices,
            write_indices=write_indices,
            activation_mode=0 if activation is None else 1,
            pad_slot_id=-1,
            run_mode=1,
        )

    # Keep the original arena view: the op reads its slot stride and writes
    # directly to independent destination slots, without a compact pool.
    return op(
        projected,
        weight.transpose(0, 1).contiguous(),
        conv_state,
        bias=bias,
        query_start_loc=cu_seqlens,
        cache_indices=read_indices,
        write_indices=write_indices,
        initial_state_mode=has_initial_state,
        activation_mode=0 if activation is None else 1,
        pad_slot_id=-1,
        run_mode=0,
    )


def _prefill_chunk_plan(
    cu_seqlens_cpu: torch.Tensor,
    token_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    boundaries = tuple(int(value) for value in cu_seqlens_cpu.tolist())
    if (
        not boundaries
        or boundaries[0] != 0
        or boundaries[-1] != token_count
        or any(left > right for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("KDA Prefill boundaries must cover the packed input")
    keep = tuple(
        index
        for index, (left, right) in enumerate(zip(boundaries, boundaries[1:]))
        if right > left
    )
    compact = [0]
    chunks: list[int] = []
    for sequence, request in enumerate(keep):
        length = boundaries[request + 1] - boundaries[request]
        compact.append(compact[-1] + length)
        for chunk in range(math.ceil(length / _PREFILL_CHUNK_SIZE)):
            chunks.extend((sequence, chunk))
    return tuple(compact), keep, tuple(chunks)


def _public_prefill_unsupported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    lower_bound: float | None,
) -> str | None:
    heads, key_dim, value_dim = q.shape[2], q.shape[3], v.shape[3]
    if q.dtype not in (torch.bfloat16, torch.float16) or any(
        tensor.dtype != q.dtype for tensor in (k, v, g_raw, beta_logits)
    ):
        return "public KDA Prefill requires matching BF16 or FP16 activations"
    if any(
        tensor.device != q.device
        for tensor in (
            k,
            v,
            g_raw,
            beta_logits,
            A_log,
            dt_bias,
            initial_state,
        )
    ):
        return "public KDA Prefill tensors must share one device"
    if key_dim % 16 or value_dim % 16 or value_dim > 256:
        return "public KDA Prefill requires aligned K/V dimensions and V <= 256"
    if value_dim != key_dim or beta_logits.shape != q.shape:
        return "public KDA Prefill requires Lite featurewise beta with K == V"
    if initial_state.shape[1:] != (heads, key_dim, value_dim) or (
        initial_state.dtype != torch.float32
    ):
        return "public KDA Prefill requires K-major FP32 recurrent state"
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        return "public KDA Prefill requires FP32 A_log matching local heads"
    if dt_bias.numel() != heads * key_dim or dt_bias.dtype != torch.float32:
        return "public KDA Prefill requires FP32 dt_bias matching local heads"
    if (
        lower_bound is None
        or not math.isfinite(lower_bound)
        or not -5.0 <= lower_bound < 0.0
    ):
        return "public KDA Prefill requires a safe-gate lower bound in [-5, 0)"
    return None


def public_kda_paged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    lower_bound: float | None,
    require_public: bool = False,
) -> KdaPrefillResult:
    """Run Lite featurewise-beta Prefill through the public split KDA ops."""
    from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult

    required = ("kda_gate_cumsum", "chunk_kda_fwd")
    missing = [name for name in required if not is_available(name)]
    unsupported = _public_prefill_unsupported(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        initial_state,
        lower_bound,
    )
    prefer_reference = q.shape[1] < _PREFILL_MIN_PUBLIC_TOKENS and not require_public
    if missing or unsupported or prefer_reference:
        if require_public:
            if unsupported:
                raise RuntimeError(unsupported)
            status = load_public_kda_ops()
            reason = status.reason or f"missing public ops: {missing}"
            raise RuntimeError(reason)
        from tokenspeed_kernel.ops.attention.kda_reference import (
            torch_kda_paged_prefill,
        )

        return torch_kda_paged_prefill(
            q,
            k,
            v,
            g_raw,
            beta_logits,
            A_log,
            dt_bias,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            lower_bound=lower_bound,
        )

    compact_cu, keep, chunk_indices = _prefill_chunk_plan(cu_seqlens_cpu, q.shape[1])
    if not keep:
        return KdaPrefillResult(torch.zeros_like(v), initial_state.clone())

    key_dim = q.shape[-1]
    if (
        q.device.type == "npu"
        and q.shape[2:] == (4, 128)
        and all(t.stride(-1) == 1 for t in (q, k, v, beta_logits))
    ):
        from tokenspeed_kernel_npu.ops.kda_prepare import prepare_kda_inputs

        q_prepared, k_prepared, v_prepared, beta = prepare_kda_inputs(
            q, k, v, beta_logits
        )
    else:
        q_prepared = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        beta_scale = torch.sqrt(torch.sigmoid(beta_logits.float()) + 1e-10)
        k_prepared = (F.normalize(k.float(), p=2, dim=-1) * beta_scale).to(k.dtype)
        v_prepared = (v.float() * beta_scale).to(v.dtype)
        beta = torch.ones(q.shape[:-1], dtype=q.dtype, device=q.device)

    gate_cumsum = torch.ops.tokenspeed_npu_public_kda.kda_gate_cumsum(
        g_raw.contiguous(),
        _PREFILL_CHUNK_SIZE,
        a_log=A_log.contiguous(),
        dt_bias=dt_bias.contiguous(),
        cu_seqlens=compact_cu,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        layout="BSND",
    )
    compacted = len(keep) != initial_state.shape[0]
    if compacted:
        keep_tensor = torch.tensor(keep, dtype=torch.int64, device=q.device)
        compact_initial = initial_state.index_select(0, keep_tensor).contiguous()
    else:
        compact_initial = initial_state.contiguous()
    output, compact_final = torch.ops.tokenspeed_npu_public_kda.chunk_kda_fwd(
        q_prepared,
        k_prepared,
        v_prepared,
        gate_cumsum,
        beta,
        key_dim**-0.5,
        _PREFILL_CHUNK_SIZE,
        "BSND",
        initial_state=compact_initial,
        cu_seqlens=compact_cu,
        chunk_indices=chunk_indices,
    )
    if not compacted:
        return KdaPrefillResult(output, compact_final)
    final_state = initial_state.clone()
    final_state.index_copy_(0, keep_tensor, compact_final)
    return KdaPrefillResult(output, final_state)


__all__ = [
    "public_kda_causal_conv1d",
    "public_kda_paged_prefill",
    "ref_kda_causal_conv1d",
    "torch_kda_paged_decode",
    "torch_kda_fused_paged_verify",
    "torch_kda_replay_commit",
    "kda_v5_t8_experimental_available",
    "kda_v5_t8_experimental_verify",
]
