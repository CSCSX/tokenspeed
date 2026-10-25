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

"""Pure-torch Ascend fallback for the ring sconv entry point.

The Triton ring sconv kernels crash Triton-Ascend 3.2.1's vector cores
nondeterministically (device-side "DDR address out of range" faults that
persist even with fully clamped tile addresses), so on Ascend
``inkling_ring_sconv`` runs this reference-grade implementation instead.
Correctness only: it is a per-request python loop, not a fused kernel.
"""

from __future__ import annotations

import torch


def _checkpoint_window(
    ckpt_a: torch.Tensor,
    ckpt_b: torch.Tensor | None,
    page: int,
) -> torch.Tensor:
    """The ``[W - 1, D]`` window stored in one checkpoint page."""
    rows_a = ckpt_a[page].float()
    if ckpt_b is None:
        return rows_a
    return torch.cat([rows_a, ckpt_b[page].float()], dim=-1)


def torch_npu_inkling_ring_sconv(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_idx: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    ckpt_a: torch.Tensor,
    ckpt_b: torch.Tensor | None,
    *,
    num_extends: int,
    publish_live: bool,
    activation: str | None = None,
    use_residual: bool = True,
) -> torch.Tensor:
    """Torch implementation of the ring sconv contract; see the Triton
    kernels in ``ops/conv/_triton/sconv.py`` for the semantics."""
    T, D = x.shape
    W = weight.shape[1]
    R = conv_cache.shape[1]
    device = x.device
    extend = num_extends > 0
    # value[pos] for pos in [pre - (W-1), through): the window each request
    # convolves over, with history from the checkpoint (extend) or the ring.
    y = torch.empty_like(x)
    if T == 0:
        return y

    cu = cu_seqlens.tolist()
    ci_list = cache_indices.tolist()
    has_state = has_initial_state.tolist()
    through_list = seq_lens.tolist()
    table = block_table.tolist()
    w_float = weight.float()

    for i in range(len(ci_list)):
        s, e = cu[i], cu[i + 1]
        L = e - s
        through = through_list[i]
        pre = through - L
        ci = ci_list[i]
        use_cache = ci != -1 and has_state[i]
        page0, page1 = table[i][0], table[i][1]

        # History rows for positions pre-(W-1) .. pre-1.
        if extend:
            ckpt_ok = use_cache and pre > 0 and page0 > 0
            if ckpt_ok:
                prefix = _checkpoint_window(ckpt_a, ckpt_b, page0)
            else:
                prefix = torch.zeros(W - 1, D, device=device, dtype=torch.float32)
        else:
            # Decode/verify taps read the ring rows of the history positions.
            rows = []
            for w in range(W - 1):
                pos = pre - (W - 1) + w
                if use_cache and pos >= 0:
                    rows.append(conv_cache[ci, pos % R].float())
                else:
                    rows.append(torch.zeros(D, device=device, dtype=torch.float32))
            prefix = torch.stack(rows)

        window = torch.cat([prefix, x[s:e].float()])  # [L + W - 1, D]
        conv = sum(
            window[w : w + L] * w_float[:, w] for w in range(W)
        )
        if activation in ("silu", "swish"):
            conv = conv * torch.sigmoid(conv)
        if use_residual:
            conv = conv + x[s:e].float()
        y[s:e] = conv.to(x.dtype)

        if ci == -1:
            continue

        def value_at(pos: int) -> torch.Tensor:
            """Window value at absolute position pos (zeros pre-history when cold)."""
            return window[pos - pre + (W - 1)]

        # Ring persistence.
        if extend:
            # The last min(L + W - 1, R) positions: chunk rows always, the
            # checkpoint window only when it was actually read (ckpt_ok).
            first = max(through - R, pre)
            for pos in range(first, through):
                conv_cache[ci, pos % R] = x[s + pos - pre]
            if ckpt_ok:
                for pos in range(max(through - R, pre - (W - 1), 0), pre):
                    conv_cache[ci, pos % R] = value_at(pos).to(conv_cache.dtype)
        else:
            for pos in range(pre, through):
                conv_cache[ci, pos % R] = x[s + pos - pre]

        # Chunk-end window publish to the live slot.
        if publish_live and page1 > 0:
            for w in range(W - 1):
                pos = through - (W - 1) + w
                if pos >= pre:
                    row = x[s + pos - pre]
                elif extend:
                    row = (
                        value_at(pos).to(x.dtype)
                        if ckpt_ok
                        else torch.zeros(D, device=device, dtype=x.dtype)
                    )
                else:
                    row = (
                        conv_cache[ci, pos % R]
                        if use_cache and pos >= 0
                        else torch.zeros(D, device=device, dtype=x.dtype)
                    )
                a_width = ckpt_a.shape[-1]
                ckpt_a[page1, w] = row[:a_width].to(ckpt_a.dtype)
                if ckpt_b is not None:
                    ckpt_b[page1, w] = row[a_width:].to(ckpt_b.dtype)

    return y
