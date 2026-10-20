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

"""Adapters for the original AMD KDA implementation used by the CI matrix."""

from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path

import torch
from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def _original_gate_projection_gfx950(
    f_a,
    f_b,
    output,
    H: gl.constexpr,
    D: gl.constexpr,
    F_A: gl.constexpr,
    F_A_ROW_STRIDE: gl.constexpr,
):
    # Match the original decode reduction before its BF16 rounding boundary.
    head = gl.program_id(0)
    row = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [4, 1], [1, 0])
    channels = gl.arange(0, D, layout=gl.SliceLayout(1, layout))
    columns = gl.arange(0, F_A, layout=gl.SliceLayout(0, layout))
    weight = gl.load(f_b + (head * D + channels)[:, None] * F_A + columns[None, :])
    activation = gl.load(f_a + row * F_A_ROW_STRIDE + columns)
    gate = gl.sum(weight.to(gl.float32) * activation.to(gl.float32)[None, :], 1)
    gl.store(
        output + row * H * D + head * D + channels, gate.to(output.dtype.element_ty)
    )


def original_gate_projection(f_a: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
    """Reproduce only the original gfx950 gate producer, outside timed calls."""
    output = torch.empty(
        (f_a.shape[0], f_b.shape[0]), dtype=f_a.dtype, device=f_a.device
    )
    _original_gate_projection_gfx950[(f_b.shape[0] // 128, f_a.shape[0])](
        f_a,
        f_b,
        output,
        H=f_b.shape[0] // 128,
        D=128,
        F_A=128,
        F_A_ROW_STRIDE=f_a.stride(0),
        num_warps=4,
        num_stages=2,
    )
    return output


def _bf16_neighbors(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bits = value.view(torch.int16).to(torch.int32)
    lower = torch.where(value > 0, bits - 1, bits + 1)
    upper = torch.where(value > 0, bits + 1, bits - 1)
    # Both signed zeros have the same adjacent nonzero BF16 values.
    lower = torch.where(value == 0, -32767, lower)
    upper = torch.where(value == 0, 1, upper)
    return (
        lower.to(torch.int16).view(torch.bfloat16),
        upper.to(torch.int16).view(torch.bfloat16),
    )


def _check_gate_rounding(projected, exact, error, gemm):
    if not bool((projected.abs() < torch.finfo(torch.bfloat16).max).all()):
        raise AssertionError("Gate reference must have finite BF16 neighbors")
    lower, upper = _bf16_neighbors(projected)
    low = (lower.double() + projected.double()) * 0.5
    high = (upper.double() + projected.double()) * 0.5
    if not bool(((low - error <= exact) & (exact <= high + error)).all()):
        raise AssertionError("Gate reference is outside its BF16 rounding interval")
    different = projected != gemm
    adjacent = (lower == gemm) | (upper == gemm)
    midpoint = (projected.double() + gemm.double()) * 0.5
    straddles = (exact - midpoint).abs() <= error
    if not bool((~different | (adjacent & straddles)).all()):
        raise AssertionError("Gate producers differ away from a BF16 rounding boundary")


def check_original_gate_projection(projected, f_a, f_b):
    """Check projection and boundary differences against independent FP64 math."""
    activation, weight = f_a.double(), f_b.double()
    exact = torch.mm(activation, weight.t())
    # BF16 products are exact in FP32; bound the sum's accumulated rounding.
    unit_roundoff = 2**-24
    gamma = 128 * unit_roundoff / (1 - 128 * unit_roundoff)
    error = gamma * torch.mm(activation.abs(), weight.abs().t())
    _check_gate_rounding(projected, exact, error, torch.mm(f_a, f_b.t()))


@cache
def _original_module(source: Path, arch: str):
    spec = importlib.util.spec_from_file_location(f"kda_original_{arch}", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load original AMD KDA source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OriginalKda:
    """Run immutable baseline source against the candidate's input and page views."""

    def __init__(
        self,
        source: Path,
        arch: str,
        layers,
        groups,
        reads,
        writes,
        accepted,
        window: int,
    ):
        self.module = _original_module(source.resolve(), arch)
        self.verify_kernel = getattr(self.module, f"gluon_kda_fused_verify_{arch}")
        self.commit_kernel = getattr(
            self.module, f"launch_gluon_kda_fused_replay_{arch}"
        )
        self.decode_kernel = getattr(self.module, f"gluon_kda_fused_decode_{arch}")
        self.arch = arch
        self.layers = layers
        self.groups = groups
        self.reads = reads
        self.writes = writes
        self.accepted = accepted
        self.window = window
        self.heads = layers[0]["beta"].shape[-1]
        self.dim = layers[0]["h_pool"].shape[-1]
        self.raw_gates = [torch.mm(x["f_a"], x["w_fb"].t()) for x in layers]
        self.payloads = [
            (
                torch.empty_like(x["qkv_raw"]),
                torch.empty_like(g),
                torch.empty_like(x["beta"]),
            )
            for x, g in zip(layers, self.raw_gates, strict=True)
        ]
        self.outputs: list[torch.Tensor | None] = [None] * len(layers)
        self.boundaries = torch.arange(
            accepted.numel() + 1, device=accepted.device, dtype=torch.int32
        )
        self.descriptors = torch.tensor(
            [
                [
                    q.data_ptr(),
                    x["conv_w"].data_ptr(),
                    x["conv_pool"].data_ptr(),
                    x["f_a"].data_ptr(),
                    x["w_fb"].data_ptr(),
                    b.data_ptr(),
                    x["A_log"].data_ptr(),
                    x["dt_bias"].data_ptr(),
                    x["h_pool"].data_ptr(),
                    g.data_ptr(),
                ]
                for x, (q, g, b) in zip(layers, self.payloads, strict=True)
            ],
            device=accepted.device,
            dtype=torch.uint64,
        )

    def verify(self, layer: int):
        x = self.layers[layer]
        qkv, gate, beta = self.payloads[layer]
        self.outputs[layer] = self.verify_kernel(
            x["qkv_raw"],
            x["conv_w"],
            x["conv_pool"],
            x["conv_pool"],
            self.raw_gates[layer],
            x["beta"],
            x["A_log"],
            x["dt_bias"],
            state_pool=x["h_pool"],
            state_scratch=None,
            read_indices=x["read_indices"],
            write_indices=x["write_indices"],
            num_heads=self.heads,
            head_dim=self.dim,
            draft_token_num=self.window,
            lower_bound=-5.0,
            replay_mixed_qkv=qkv,
            replay_gate=gate,
            replay_beta=beta,
        )

    def commit(self):
        x = self.layers[0]
        qkv, gate, beta = self.payloads[0]
        self.commit_kernel(
            self.descriptors,
            self.groups,
            self.reads,
            self.writes,
            self.accepted,
            draft_token_num=self.window,
            num_heads=self.heads,
            head_dim=self.dim,
            f_a_dim=x["f_a"].shape[-1],
            qkv_stride=qkv.stride(0),
            conv_stride=x["conv_pool"].stride(0),
            f_a_stride=x["f_a"].stride(0),
            beta_stride=beta.stride(0),
            state_stride=x["h_pool"].stride(0),
            gate_stride=gate.stride(0),
            conv_width=4,
            lower_bound=-5.0,
        )

    def replay_lifecycle(self):
        for layer, x in enumerate(self.layers):
            torch.mm(x["f_a"], x["w_fb"].t(), out=self.raw_gates[layer])
            self.verify(layer)
        self.commit()

    def decode_lifecycle(self):
        if self.window != 1:
            raise ValueError("Original standard decode requires a one-token window")
        for layer, x in enumerate(self.layers):
            common = dict(
                state_pool=x["h_pool"],
                read_indices=x["read_indices"],
                write_indices=x["write_indices"],
                num_heads=self.heads,
                head_dim=self.dim,
                cu_seqlens=self.boundaries,
                lower_bound=-5.0,
            )
            if self.arch == "gfx950":
                self.outputs[layer] = self.decode_kernel(
                    x["qkv_raw"],
                    x["conv_w"],
                    x["conv_pool"],
                    x["f_a"],
                    x["w_fb"],
                    x["beta"],
                    x["A_log"],
                    x["dt_bias"],
                    **common,
                )
            else:
                torch.mm(x["f_a"], x["w_fb"].t(), out=self.raw_gates[layer])
                self.outputs[layer] = self.decode_kernel(
                    x["qkv_raw"],
                    x["conv_w"],
                    x["conv_pool"],
                    self.raw_gates[layer],
                    x["beta"],
                    x["A_log"],
                    x["dt_bias"],
                    **common,
                )
