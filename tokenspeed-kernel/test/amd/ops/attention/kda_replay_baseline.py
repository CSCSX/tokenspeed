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
