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

"""Compile actual AMD KDA wrapper launches without an AMD device."""

import importlib
import os
from pathlib import Path

import pytest
import torch
from amd.ops.attention.kda_replay_baseline import OriginalKda
from tokenspeed_kernel_amd._triton import gluon, triton


def _offline_launch(monkeypatch, kernel, arch):
    target = triton.backends.compiler.GPUTarget(
        "hip", arch, 64 if arch == "gfx950" else 32
    )
    backend = triton.compiler.make_backend(target)
    binder = triton.runtime.jit.create_function_from_signature(
        kernel.signature, kernel.params, backend
    )

    def compile_launch(*args, grid, warmup=False, **kwargs):
        # Use the real binder, specialization rules and compiler. Only the GPU
        # launch is replaced; unknown arguments and invalid JIT code still fail.
        bound, specialization, options = binder(*args, **kwargs)
        opts, signature, constants, attrs = kernel._pack_args(
            backend, options, bound, specialization, options
        )
        compiled = triton.compile(
            gluon.GluonASTSource(kernel, signature, constants, attrs),
            target=target,
            options=opts.__dict__,
        )
        assert compiled.asm["hsaco"]
        return compiled

    monkeypatch.setattr(kernel, "run", compile_launch)


def _inputs(window, device):
    batch, heads, dim, pages = 3, 12, 128, 8
    arena = torch.empty(pages, 884736, dtype=torch.uint8, device=device)
    state_bytes = heads * dim * dim * 4
    conv_bytes = 3 * heads * dim * 3 * 2
    return {
        "qkv_raw": torch.empty(
            batch * window, 3 * heads * dim, dtype=torch.bfloat16, device=device
        ),
        "conv_w": torch.empty(3 * heads * dim, 4, dtype=torch.bfloat16, device=device),
        "conv_pool": arena[:, state_bytes : state_bytes + conv_bytes]
        .view(torch.bfloat16)
        .view(pages, 3 * heads * dim, 3),
        "f_a": torch.empty(batch * window, dim, dtype=torch.bfloat16, device=device),
        "w_fb": torch.empty(heads * dim, dim, dtype=torch.bfloat16, device=device),
        "beta": torch.empty(batch * window, heads, dtype=torch.bfloat16, device=device),
        "A_log": torch.empty(heads, device=device),
        "dt_bias": torch.empty(heads * dim, device=device),
        "h_pool": arena[:, :state_bytes]
        .view(torch.float32)
        .view(pages, heads, dim, dim),
        "read_indices": torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        "write_indices": torch.arange(
            batch + 1, 2 * batch + 1, dtype=torch.int32, device=device
        ),
    }


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("window", [1, 4, 8])
def test_kda_records_wrapper_launches_compile(monkeypatch, arch, window):
    module = importlib.import_module(
        f"tokenspeed_kernel_amd.ops.{arch}.attention.kda.decode"
    )
    for name in (
        f"gluon_kda_fused_paged_verify_nostore_vmajor_{arch}",
        f"gluon_kda_fused_replay_{arch}",
    ):
        _offline_launch(monkeypatch, getattr(module, name), arch)
    x = _inputs(window, "cpu")
    batch, heads, dim, layers = 3, 12, 128, 2
    # A capacity wider than the live window also exercises masked ring loads.
    capacity = window + 1
    correction = torch.empty(batch, capacity, heads, dim)
    key_gate = torch.empty(batch, capacity, heads, 2 * dim, dtype=torch.bfloat16)
    metadata = torch.tensor(
        [[row, 7, 2**33 + 1] for row in range(batch)], dtype=torch.int64
    )
    stamps = torch.empty(layers, batch, capacity, 2, dtype=torch.int64)
    output = getattr(module, f"gluon_kda_fused_verify_{arch}")(
        mixed_qkv=x["qkv_raw"],
        conv_weights=x["conv_w"],
        conv_pool=x["conv_pool"],
        raw_g=torch.empty(batch * window, heads * dim, dtype=torch.bfloat16),
        beta_logits=x["beta"],
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        state_pool=x["h_pool"],
        read_indices=x["read_indices"],
        correction=correction.transpose(1, 2),
        key_gate=key_gate.transpose(1, 2),
        record_metadata=metadata,
        record_stamps=stamps[0],
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32) * window,
        num_heads=heads,
        head_dim=dim,
        draft_token_num=window,
        lower_bound=-5.0,
    )
    assert output.shape == (1, batch * window, heads, dim)
    assert output.dtype == x["qkv_raw"].dtype
    getattr(module, f"launch_gluon_kda_fused_replay_{arch}")(
        torch.empty(layers, 10, dtype=torch.uint64),
        torch.zeros(layers, dtype=torch.int32),
        x["read_indices"][None],
        x["write_indices"][None],
        torch.full((batch,), window, dtype=torch.int32),
        draft_token_num=window,
        num_heads=heads,
        head_dim=dim,
        qkv_stride=x["qkv_raw"].stride(0),
        conv_stride=x["conv_pool"].stride(0),
        correction_stride=correction.stride(1),
        key_gate_stride=key_gate.stride(1),
        state_stride=x["h_pool"].stride(0),
        conv_width=4,
        activation_dtype=x["qkv_raw"].dtype,
        conv_state_dtype=x["conv_pool"].dtype,
        record_metadata=metadata,
        record_stamps=stamps,
        lower_bound=-5.0,
    )


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("window", [1, 4])
def test_original_kda_comparison_launches_compile(monkeypatch, arch, window):
    source_dir = os.environ.get("TOKENSPEED_KDA_BASELINE_DIR")
    if source_dir is None:
        pytest.skip("set TOKENSPEED_KDA_BASELINE_DIR to immutable original AMD sources")
    source = Path(source_dir) / f"original_{arch}_decode.py"
    # Substitute the missing GPU device check and launch. Tensor storage,
    # shape/stride validation, adapter calls and compilation remain real.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: True))
    x = _inputs(window, "cpu")
    original = OriginalKda(
        source,
        arch,
        [x],
        torch.zeros(1, dtype=torch.int32),
        x["read_indices"][None],
        x["write_indices"][None],
        torch.full((3,), window, dtype=torch.int32),
        window,
    )
    for name in (
        f"gluon_kda_fused_paged_verify_nostore_vmajor_{arch}",
        f"gluon_kda_fused_replay_{arch}",
        f"gluon_kda_fused_paged_decode_vmajor_{arch}",
    ):
        _offline_launch(monkeypatch, getattr(original.module, name), arch)
    original.replay_lifecycle()
    if window == 1:
        original.decode_lifecycle()
    assert original.outputs[0].shape == (1, 3 * window, 12, 128)
