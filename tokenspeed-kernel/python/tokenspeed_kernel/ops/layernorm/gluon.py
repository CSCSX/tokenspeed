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

"""gfx950 normalization producers with the shared operators' rounding order."""

import torch
from tokenspeed_kernel.ops.layernorm.qk_rmsnorm_ssmax import (
    QK_RMSNORM_SSMAX_SIGNATURES,
    qk_rmsnorm_ssmax_traits,
    validate_qk_norm_arguments,
)
from tokenspeed_kernel.ops.layernorm.triton import UnscaledFP8QKVGate, rmsnorm
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel

NAME = "gluon_qk_rmsnorm_ssmax_gfx950"
TRAITS = {
    "tokens_min": frozenset({1}),
    "head_dim": frozenset({128}),
    "eps_ok": frozenset({True}),
    "heads": frozenset({(6, 1)}),
    "is_cuda": frozenset({True}),
    "ndim": frozenset({2}),
    "positions_contiguous": frozenset({True}),
    "rows_disjoint": frozenset({True}),
    "unit_stride": frozenset({True}),
    "weights_contiguous": frozenset({True}),
}


def gluon_rmsnorm_bf16_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
    out: torch.Tensor | None,
    round_residual_sum_bf16: bool,
    enable_pdl: bool,
) -> bool:
    """Return whether the gfx950 kernel serves these BF16 [M,5120] rows.

    Weight is dense BF16 [5120]. A residual needs BF16-sum rounding;
    caller-provided norm output and PDL retain the existing implementation.
    """

    def rows_supported(value):
        return (
            value.is_cuda
            and value.dtype == torch.bfloat16
            and value.ndim == 2
            and value.shape[1] == 5120
            and value.stride(1) == 1
            and value.stride(0) >= 5120
            and value.stride(0) % 8 == 0
            and value.data_ptr() % 16 == 0
        )

    return (
        out is None
        and not enable_pdl
        and rows_supported(x)
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.shape == (5120,)
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
        and (
            residual is None
            or (
                round_residual_sum_bf16
                and rows_supported(residual)
                and residual.shape == x.shape
                and residual.device == x.device
            )
        )
        and current_platform().is_cdna4
    )


def gluon_rmsnorm_bf16_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: torch.Tensor | None,
    output: torch.Tensor,
    residual_output: torch.Tensor | None,
    x_scale: float,
    residual_scale: float,
    token_fp8_out: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    """Write the qualified BF16 norm, rounded residual and optional FP8 outputs.

    Arguments follow ``rmsnorm``; ``output`` and optional ``residual_output``
    are its caller-owned contiguous destinations. FP8 buffers hold E4M3 codes
    and FP32 per-token scales. This entry does not support PDL. Return None.
    """
    if not current_platform().is_cdna4:
        raise ValueError("gluon_rmsnorm_bf16_gfx950 requires AMD gfx950")
    from tokenspeed_kernel_amd.ops.gfx950.layernorm.rmsnorm import (
        launch_gluon_rmsnorm_bf16_gfx950,
    )

    launch_gluon_rmsnorm_bf16_gfx950(
        x,
        weight,
        eps,
        residual=residual,
        output=output,
        residual_output=residual_output,
        x_scale=x_scale,
        residual_scale=residual_scale,
        token_fp8_out=token_fp8_out,
    )


@register_kernel(
    "layernorm",
    "qk_rmsnorm_ssmax",
    name=NAME,
    solution="gluon",
    capability=CapabilityRequirement(
        vendors=frozenset({"amd"}),
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
    ),
    signatures=QK_RMSNORM_SSMAX_SIGNATURES,
    traits=TRAITS,
    priority=Priority.SPECIALIZED,
)
def gluon_qk_rmsnorm_ssmax_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    *,
    dequant: UnscaledFP8QKVGate | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse only Q normalization and SSMax; keep the portable K norm unchanged.

    Accept BF16 [M,768]/[M,128] Q/K, dense BF16 [128] weights and dense
    int32/int64 [M] positions on gfx950. Disjoint, unit-column-stride views
    are supported. Optional unscaled-FP8 dequantization is not admitted and
    ranked selection retains the portable operation for that feature.
    """
    validate_qk_norm_arguments(q, k, q_weight, k_weight, positions)
    if dequant is not None:
        raise ValueError(f"{NAME} does not support unscaled FP8 dequantization")
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or q_weight.dtype != torch.bfloat16
        or k_weight.dtype != torch.bfloat16
    ):
        raise ValueError(f"{NAME} requires BF16 queries, keys and weights")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{NAME} requires int32/int64 positions")
    traits = qk_rmsnorm_ssmax_traits(q, k, q_weight, k_weight, eps, positions)
    if traits["tokens"] < 1 or any(
        traits[name] not in values
        for name, values in TRAITS.items()
        if name != "tokens_min"
    ):
        raise ValueError(
            f"{NAME} requires positive rows, Q6/KV1 D128 and supported contiguous-column metadata"
        )
    platform = current_platform()
    if not platform.is_amd or platform.arch_version != ArchVersion(9, 5):
        raise ValueError(f"{NAME} requires AMD gfx950")
    from tokenspeed_kernel_amd.ops.gfx950.layernorm.qk_rmsnorm_ssmax import (
        launch_gluon_qk_rmsnorm_ssmax_gfx950,
    )

    q_out = launch_gluon_qk_rmsnorm_ssmax_gfx950(q, q_weight, positions, eps)
    k_out = rmsnorm(
        k.reshape(-1, 1, 128), k_weight, eps, residual=None, out=None, enable_pdl=None
    )
    return q_out, k_out.flatten(1)
