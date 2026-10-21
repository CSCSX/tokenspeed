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
from amd.ops.attention.kda_replay_baseline import (
    OriginalKda,
    RecordsKda,
    _bf16_neighbors,
    _check_gate_rounding,
    _original_gate_projection_gfx950,
    _original_module,
    original_gate_projection,
    prepare_records_verify,
)
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


def _inputs(window, device, *, batch):
    heads, dim, pages = 12, 128, 2 * batch + 2
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


@pytest.mark.parametrize(
    ("arch", "window", "batch", "capacity", "lower_bound"),
    [
        (arch, window, 3, window + 1, -5.0)
        for arch in ("gfx950", "gfx1250")
        for window in (1, 4, 8)
    ]
    # Cover a wide ring, unsplit T1, and the GEMM producer above 64 rows.
    + [
        ("gfx950", 1, 1, 129, -5.0),
        ("gfx950", 1, 17, 2, -5.0),
        ("gfx950", 4, 17, 5, -5.0),
    ]
    # Softplus keeps the existing recovery path for every tested window.
    + [("gfx950", window, 3, window + 1, None) for window in (1, 4, 8)]
    + [("gfx950", 1, 4, 2, None)],
)
def test_kda_records_wrapper_launches_compile(
    monkeypatch, arch, window, batch, capacity, lower_bound
):
    module = importlib.import_module(
        f"tokenspeed_kernel_amd.ops.{arch}.attention.kda.decode"
    )
    for name in (
        f"gluon_kda_fused_paged_verify_nostore_vmajor_{arch}",
        f"gluon_kda_fused_replay_{arch}",
    ):
        _offline_launch(monkeypatch, getattr(module, name), arch)
    x = _inputs(window, "cpu", batch=batch)
    heads, dim, layers = 12, 128, 2
    # A capacity wider than the live window also exercises masked ring loads.
    correction = torch.empty(batch, capacity, heads, dim)
    key_gate = torch.empty(batch, capacity, heads, 2 * dim, dtype=torch.bfloat16)
    metadata = torch.tensor(
        [[row, 7, 2**33 + 1] for row in range(batch)], dtype=torch.int64
    )
    stamps = torch.empty(layers, batch, capacity, 2, dtype=torch.int64)
    width = x["qkv_raw"].shape[1]
    retained = torch.empty_strided(
        x["qkv_raw"].shape, (width + 7, 1), dtype=x["qkv_raw"].dtype
    )
    for inline in ((False, True) if arch == "gfx950" else (False,)):
        projection = {}
        if arch == "gfx950":
            projection = {
                "f_a_out": (
                    torch.empty_strided(
                        x["f_a"].shape, (dim + 7, 1), dtype=x["f_a"].dtype
                    )
                    if inline
                    else None
                ),
                "f_b_weight": x["w_fb"] if inline else None,
            }
        for original_qkv in (None, retained):
            output = getattr(module, f"gluon_kda_fused_verify_{arch}")(
                mixed_qkv=x["qkv_raw"],
                original_qkv=original_qkv,
                conv_weights=x["conv_w"],
                conv_pool=x["conv_pool"],
                raw_g=(
                    None
                    if inline
                    else torch.empty(batch * window, heads * dim, dtype=torch.bfloat16)
                ),
                **projection,
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
                lower_bound=lower_bound,
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
        lower_bound=lower_bound,
    )


@pytest.mark.parametrize("both", [False, True])
def test_gate_launcher_requires_one_producer(both):
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_verify_gfx950,
    )

    x = _inputs(1, "cpu", batch=3)
    batch, heads, dim = 3, 12, 128
    message = "excludes projection inputs" if both else "requires either a raw gate"
    with pytest.raises(ValueError, match=message):
        gluon_kda_fused_verify_gfx950(
            mixed_qkv=x["qkv_raw"],
            original_qkv=None,
            conv_weights=x["conv_w"],
            conv_pool=x["conv_pool"],
            raw_g=(
                torch.empty(batch, heads * dim, dtype=torch.bfloat16) if both else None
            ),
            f_a_out=x["f_a"] if both else None,
            f_b_weight=x["w_fb"] if both else None,
            beta_logits=x["beta"],
            A_log=x["A_log"],
            dt_bias=x["dt_bias"],
            state_pool=x["h_pool"],
            read_indices=x["read_indices"],
            correction=torch.empty(batch, heads, 1, dim),
            key_gate=torch.empty(batch, heads, 1, 2 * dim, dtype=torch.bfloat16),
            record_metadata=torch.zeros(batch, 3, dtype=torch.int64),
            record_stamps=torch.zeros(batch, 1, 2, dtype=torch.int64),
            query_start_loc=torch.arange(batch + 1, dtype=torch.int32),
            num_heads=heads,
            head_dim=dim,
            draft_token_num=1,
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
    x = _inputs(window, "cpu", batch=3)
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


def test_original_gfx950_gate_projection_launch_compiles(monkeypatch):
    _offline_launch(monkeypatch, _original_gate_projection_gfx950, "gfx950")
    x = _inputs(1, "cpu", batch=3)
    projected = original_gate_projection(x["f_a"], x["w_fb"])
    assert projected.shape == (3, 12 * 128)
    assert projected.dtype == torch.bfloat16


def test_bf16_rounding_intervals_include_both_signs_and_binade_edges():
    value = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0, -2.0], dtype=torch.bfloat16)
    lower, upper = _bf16_neighbors(value)
    low = (lower.double() + value.double()) * 0.5
    high = (upper.double() + value.double()) * 0.5
    expected_low = torch.tensor(
        [-(2**-134), -(2**-134), 1 - 2**-9, -1 - 2**-8, 2 - 2**-8, -2 - 2**-7],
        dtype=torch.float64,
    )
    expected_high = torch.tensor(
        [2**-134, 2**-134, 1 + 2**-8, -1 + 2**-9, 2 + 2**-7, -2 + 2**-8],
        dtype=torch.float64,
    )
    torch.testing.assert_close(low, expected_low, atol=0, rtol=0)
    torch.testing.assert_close(high, expected_high, atol=0, rtol=0)


def test_gate_reference_allows_midpoint_rounding_but_rejects_wrong_values():
    projected = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
    exact = torch.tensor([1 + 2**-8, -1 - 2**-8], dtype=torch.float64)
    error = torch.full_like(exact, 2**-24)
    _check_gate_rounding(projected, exact, error)
    with pytest.raises(AssertionError, match="rounding interval"):
        _check_gate_rounding(projected + 0.25, exact, error)


def test_gate_reference_accepts_valid_fp32_cancellation_orders():
    # The FP32 accumulation error can span many BF16 ULPs near zero.
    terms = torch.tensor([1.0, 2**-26, -1.0], dtype=torch.float32)
    left = (terms[0] + terms[1]) + terms[2]
    cancelled = (terms[0] + terms[2]) + terms[1]
    alternatives = torch.stack((left, cancelled)).to(torch.bfloat16)
    exact = torch.full((2,), 2**-26, dtype=torch.float64)
    gamma = 128 * 2**-24 / (1 - 128 * 2**-24)
    error = torch.full_like(exact, gamma * (2 + 2**-26))
    _check_gate_rounding(alternatives, exact, error)


@pytest.mark.parametrize(
    ("batch", "window"), [(1, 1), (4, 1), (8, 1), (17, 1), (17, 4)]
)
def test_kda_reference_lifecycles_launches_compile(monkeypatch, batch, window):
    directory = os.environ.get("TOKENSPEED_KDA_BASELINE_DIR")
    if directory is None:
        pytest.skip("requires the accepted-parent and trial-reference snapshots")
    parent = _original_module(
        Path(directory) / "accepted-parent-kda-gfx950.py", "gfx950"
    )
    trial = _original_module(
        Path(directory) / "trial-reference-kda-gfx950.py", "gfx950"
    )
    current = importlib.import_module(
        "tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode"
    )
    heads, dim = 12, 128
    layers = [_inputs(window, "cpu", batch=batch) for _ in range(2)]
    stamps = torch.empty(len(layers), batch, window, 2, dtype=torch.int64)
    retained = [torch.empty_like(x["qkv_raw"]) for x in layers]
    for layer, x in enumerate(layers):
        x["correction"] = torch.empty(batch, window, heads, dim)
        x["key_gate"] = torch.empty(batch, window, heads, 2 * dim, dtype=torch.bfloat16)
        x["metadata"] = torch.tensor(
            [[row, 7, 2**33 + 1] for row in range(batch)], dtype=torch.int64
        )
        x["stamps"] = stamps[layer]
    descriptors = torch.tensor(
        [
            [
                qkv.data_ptr(),
                x["conv_w"].data_ptr(),
                x["conv_pool"].data_ptr(),
                x["correction"].data_ptr(),
                x["w_fb"].data_ptr(),
                x["key_gate"].data_ptr(),
                x["A_log"].data_ptr(),
                x["dt_bias"].data_ptr(),
                x["h_pool"].data_ptr(),
                0,
            ]
            for x, qkv in zip(layers, retained, strict=True)
        ],
        dtype=torch.uint64,
    )
    raw_gate = torch.empty(batch * window, heads * dim, dtype=torch.bfloat16)
    for module in (current, parent, trial):
        for kernel in (
            module.gluon_kda_fused_paged_verify_nostore_vmajor_gfx950,
            module.gluon_kda_fused_replay_gfx950,
        ):
            _offline_launch(monkeypatch, kernel, "gfx950")
        output = prepare_records_verify(
            module.gluon_kda_fused_verify_gfx950,
            layers[0],
            window,
            raw_gate=raw_gate,
            original_qkv=retained[0],
            lower_bound=-5.0,
        )()
        assert output.shape == (1, batch * window, heads, dim)
        assert output.dtype == layers[0]["qkv_raw"].dtype
        runner = RecordsKda(
            module,
            layers,
            retained,
            descriptors,
            torch.arange(len(layers), dtype=torch.int32),
            torch.stack([x["read_indices"] for x in layers]),
            torch.stack([x["write_indices"] for x in layers]),
            torch.arange(batch, dtype=torch.int32) % (window + 1),
            stamps,
            window,
            lower_bound=-5.0,
        )
        runner.lifecycle()
        for x in layers:
            assert x["verify_output"].shape == (1, batch * window, heads, dim)
            assert x["verify_output"].dtype == x["qkv_raw"].dtype
