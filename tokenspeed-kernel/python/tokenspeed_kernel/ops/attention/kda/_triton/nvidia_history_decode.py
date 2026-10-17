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

"""Single-token FP32 KDA producers with cache-owned accepted history."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kda import KdaReplayRecords
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import _recover_kda_history


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def _fused_history_decode_kernel(
    qkv_raw,  # [T, 3*P] pre-conv packed projections (token-strided)
    conv_w,  # [3*P, 4] fused conv bank
    conv_pool,  # [pages, 3*P, 3] conv state
    f_a,  # [T, D_fa] low-rank gate input
    w_fb,  # [P, D_fa] f_b weight
    beta,
    A_log,
    dt_bias,
    o,
    h_pool,
    read_indices,
    write_indices,
    cu_seqlens,
    lower_bound,
    history_u,
    history_key,
    history_decay,
    history_lengths,
    append_offsets,
    raw_payload,
    candidate_u,
    candidate_kd,
    stride_candidate_u,
    stride_candidate_kd,
    record_metadata,
    record_stamps,
    stride_hu_page,
    stride_hu_head,
    stride_hu_token,
    stride_hu_dim,
    stride_hk_page,
    stride_hk_head,
    stride_hk_token,
    stride_hk_dim,
    stride_hd_page,
    stride_hd_head,
    stride_hd_token,
    stride_hd_dim,
    stride_length,
    stride_offsets,
    stride_payload,
    CAPACITY: tl.constexpr,
    stride_raw_tok: tl.constexpr,
    stride_fa_tok: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    P: tl.constexpr,  # proj_local = HV * K
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_page: tl.constexpr,
    stride_conv_page: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T
    if T == 0:
        return

    slot = tl.load(record_metadata + i_n * 3).to(tl.int64)
    if slot < 0:
        return
    generation = tl.load(record_metadata + i_n * 3 + 1)
    position = tl.load(record_metadata + i_n * 3 + 2)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    b_read = tl.load(read_indices + i_n).to(tl.int64)
    b_write = tl.load(write_indices + i_n).to(tl.int64)
    read_ok = b_read >= 0

    # --- fused conv (4-tap depthwise + silu) over this program's features ---
    qf = i_h * K + o_k  # q features in section 0
    kf = P + i_h * K + o_k  # k features in section 1
    vf = 2 * P + i_hv * V + o_v  # v features in section 2

    x_q = tl.load(qkv_raw + bos * stride_raw_tok + qf, mask=mask_k, other=0.0).to(
        tl.float32
    )
    x_k = tl.load(qkv_raw + bos * stride_raw_tok + kf, mask=mask_k, other=0.0).to(
        tl.float32
    )
    x_v = tl.load(qkv_raw + bos * stride_raw_tok + vf, mask=mask_v, other=0.0).to(
        tl.float32
    )

    acc_q = x_q * tl.load(conv_w + qf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    acc_k = x_k * tl.load(conv_w + kf * 4 + 3, mask=mask_k, other=0.0).to(tl.float32)
    acc_v = x_v * tl.load(conv_w + vf * 4 + 3, mask=mask_v, other=0.0).to(tl.float32)
    s_q0 = tl.zeros([BK], dtype=tl.float32)
    s_k0 = tl.zeros([BK], dtype=tl.float32)
    s_v0 = tl.zeros([BV], dtype=tl.float32)
    s_q1 = tl.zeros([BK], dtype=tl.float32)
    s_k1 = tl.zeros([BK], dtype=tl.float32)
    s_v1 = tl.zeros([BV], dtype=tl.float32)
    s_q2 = tl.zeros([BK], dtype=tl.float32)
    s_k2 = tl.zeros([BK], dtype=tl.float32)
    s_v2 = tl.zeros([BV], dtype=tl.float32)
    if read_ok:
        cp = conv_pool + b_read * stride_conv_page
        s_q0 = tl.load(cp + qf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
        s_q1 = tl.load(cp + qf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
        s_q2 = tl.load(cp + qf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
        s_k0 = tl.load(cp + kf * 3 + 0, mask=mask_k, other=0.0).to(tl.float32)
        s_k1 = tl.load(cp + kf * 3 + 1, mask=mask_k, other=0.0).to(tl.float32)
        s_k2 = tl.load(cp + kf * 3 + 2, mask=mask_k, other=0.0).to(tl.float32)
        s_v0 = tl.load(cp + vf * 3 + 0, mask=mask_v, other=0.0).to(tl.float32)
        s_v1 = tl.load(cp + vf * 3 + 1, mask=mask_v, other=0.0).to(tl.float32)
        s_v2 = tl.load(cp + vf * 3 + 2, mask=mask_v, other=0.0).to(tl.float32)
    acc_q += (
        s_q0 * tl.load(conv_w + qf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
        + s_q1 * tl.load(conv_w + qf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
        + s_q2 * tl.load(conv_w + qf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    )
    acc_k += (
        s_k0 * tl.load(conv_w + kf * 4 + 0, mask=mask_k, other=0.0).to(tl.float32)
        + s_k1 * tl.load(conv_w + kf * 4 + 1, mask=mask_k, other=0.0).to(tl.float32)
        + s_k2 * tl.load(conv_w + kf * 4 + 2, mask=mask_k, other=0.0).to(tl.float32)
    )
    acc_v += (
        s_v0 * tl.load(conv_w + vf * 4 + 0, mask=mask_v, other=0.0).to(tl.float32)
        + s_v1 * tl.load(conv_w + vf * 4 + 1, mask=mask_v, other=0.0).to(tl.float32)
        + s_v2 * tl.load(conv_w + vf * 4 + 2, mask=mask_v, other=0.0).to(tl.float32)
    )
    b_q = acc_q * tl.sigmoid(acc_q)  # silu
    b_k = acc_k * tl.sigmoid(acc_k)
    b_v = acc_v * tl.sigmoid(acc_v)

    # The accepted convolution endpoint is committed after verification.
    if i_v == 0:
        tl.store(raw_payload + i_n * stride_payload + qf, x_q, mask=mask_k)
        tl.store(raw_payload + i_n * stride_payload + kf, x_k, mask=mask_k)
    tl.store(raw_payload + i_n * stride_payload + vf, x_v, mask=mask_v)

    # --- fused f_b: g_raw[c] = w_fb[c, :] . f_a for this head's K features ---
    o_fa = tl.arange(0, D_FA)
    fa = tl.load(f_a + bos * stride_fa_tok + o_fa).to(tl.float32)
    gc = i_hv * K + o_k  # gate feature = same head slice of P
    wfb = tl.load(
        w_fb + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    b_g = tl.sum(wfb * fa[None, :], axis=1)

    # --- recurrence (T=1 decode) ---
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    b_A = tl.load(A_log + i_hv).to(tl.float32)
    if HAS_DT_BIAS:
        b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        b_g = b_g + b_bias
    if USE_LOWER_BOUND:
        b_gk = lower_bound * tl.sigmoid(tl.exp(b_A) * b_g)
    else:
        b_gk = -tl.exp(b_A) * tl.where(
            b_g < 20.0, tl.math.log(1 + tl.math.exp(b_g)), b_g
        )

    p_h0 = h_pool + b_read * stride_state_page + i_hv * K * V
    p_h0 += o_v[:, None] * K + o_k[None, :]
    length = tl.load(history_lengths + b_read * stride_length, mask=b_read > 0, other=0)
    tl.device_assert((length >= 0) & (length <= CAPACITY), "invalid history length")
    # The same load handles an empty history; its recovery loop then runs zero times.
    b_h = _recover_kda_history(
        p_h0,
        mask_h & read_ok,
        history_u + b_read * stride_hu_page + i_hv * stride_hu_head,
        history_key + b_read * stride_hk_page + i_hv * stride_hk_head,
        history_decay + b_read * stride_hd_page + i_hv * stride_hd_head,
        o_k,
        o_v,
        mask_k,
        mask_v,
        length,
        stride_hu_token,
        stride_hu_dim,
        stride_hk_token,
        stride_hk_dim,
        stride_hd_token,
        stride_hd_dim,
        triton.next_power_of_2(CAPACITY),
        False,
        2,
    )
    flush = ((length + 1 > CAPACITY) | (b_read != b_write)) & (b_write > 0)
    offset = tl.where(flush, 0, length)
    if flush:
        tl.store(p_h0 + (b_write - b_read) * stride_state_page, b_h, mask=mask_h)
    if i_hv == 0 and i_v == 0:
        tl.store(append_offsets + i_n * stride_offsets, offset)

    b_h *= tl.exp(b_gk)[None, :]
    b_v = b_v - tl.sum(b_h * b_k[None, :], axis=1)
    b_beta = tl.load(beta + bos * stride_beta_tok + i_hv).to(tl.float32)
    b_beta = tl.sigmoid(b_beta)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], axis=1)
    tl.store(
        o + (bos * HV + i_hv) * V + o_v,
        b_o.to(o.dtype.element_ty),
        mask=mask_v,
    )
    # A separate candidate workspace keeps old history immutable during verify.
    tl.store(candidate_u + slot * stride_candidate_u + i_hv * V + o_v, b_v, mask=mask_v)
    if i_v == 0:
        tl.store(
            candidate_kd + slot * stride_candidate_kd + i_hv * (2 * K) + o_k,
            b_k,
            mask=mask_k,
        )
        tl.store(
            candidate_kd + slot * stride_candidate_kd + i_hv * (2 * K) + K + o_k,
            tl.exp(b_gk),
            mask=mask_k,
        )

    if i_hv == 0 and i_v == 0:
        tl.store(record_stamps + slot * 2, position)
        tl.store(record_stamps + slot * 2 + 1, generation)


def fused_history_decode(
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
    records: KdaReplayRecords,
    replay_mixed_qkv: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Verify one token and save FP32 factors for the shared accepted commit.

    Inputs retain the fused decode's FP32 convolution and gate intermediates.
    records supplies normalized-key/decay destinations, stamps, query boundaries
    and the cache-owned accepted history. replay_mixed_qkv receives request-major
    raw projections for the subsequent convolution commit. Return packed BF16
    attention output shaped [1, tokens, heads, value_dim].
    """
    history = records.history
    if history is None or records.key_gate_format != "normalized_key_decay":
        raise ValueError("fused history decode requires normalized FP32 records")
    batch, heads, width, dim = records.correction.shape
    if width != 1 or dim != 128 or records.key_gate.dtype != torch.float32:
        raise ValueError("fused history decode requires one 128-channel FP32 record")
    if history.use_tensor_cores:
        raise ValueError("fused history decode requires ordered scalar recovery")
    out = torch.empty(
        (1, mixed_qkv.shape[0], heads, dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    # Match the original fused decode's full-head reduction layout.
    tile = 128
    _fused_history_decode_kernel[(batch * heads * triton.cdiv(dim, tile),)](
        qkv_raw=mixed_qkv,
        conv_w=conv_weights,
        conv_pool=conv_states,
        f_a=f_a_out,
        w_fb=f_b_weight,
        beta=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h_pool=state_pool,
        read_indices=read_indices,
        write_indices=history.write_indices,
        cu_seqlens=records.query_start_loc,
        lower_bound=lower_bound,
        history_u=history.correction,
        history_key=history.key,
        history_decay=history.decay,
        history_lengths=history.lengths,
        append_offsets=history.append_offsets,
        raw_payload=replay_mixed_qkv,
        record_metadata=records.metadata,
        record_stamps=records.stamps,
        candidate_u=records.correction,
        candidate_kd=records.key_gate,
        stride_candidate_u=records.correction.stride(0),
        stride_candidate_kd=records.key_gate.stride(0),
        stride_hu_page=history.correction.stride(0),
        stride_hu_head=history.correction.stride(1),
        stride_hu_token=history.correction.stride(2),
        stride_hu_dim=history.correction.stride(3),
        stride_hk_page=history.key.stride(0),
        stride_hk_head=history.key.stride(1),
        stride_hk_token=history.key.stride(2),
        stride_hk_dim=history.key.stride(3),
        stride_hd_page=history.decay.stride(0),
        stride_hd_head=history.decay.stride(1),
        stride_hd_token=history.decay.stride(2),
        stride_hd_dim=history.decay.stride(3),
        stride_length=history.lengths.stride(0),
        stride_offsets=history.append_offsets.stride(0),
        stride_payload=replay_mixed_qkv.stride(0),
        CAPACITY=history.correction.shape[2],
        stride_raw_tok=mixed_qkv.stride(0),
        stride_fa_tok=f_a_out.stride(0),
        stride_beta_tok=beta_logits.stride(0),
        scale=dim**-0.5,
        N=batch,
        T=1,
        H=heads,
        HV=heads,
        K=dim,
        V=dim,
        P=heads * dim,
        D_FA=f_a_out.shape[-1],
        BK=triton.next_power_of_2(dim),
        BV=tile,
        stride_state_page=state_pool.stride(0),
        stride_conv_page=conv_states.stride(0),
        num_warps=4,
        num_stages=2,
    )
    return out
