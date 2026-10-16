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

"""Registration shims for AMD Gluon activation kernels."""

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.activation.gating import (
        launch_gluon_attention_output_gate_gfx950,
    )

    @register_kernel(
        "activation",
        "attention_output_gate",
        name="gluon_attention_output_gate_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            "output", "dense", {torch.bfloat16, torch.float16, torch.float32}
        ),
        traits={
            "layout_supported": frozenset({True}),
            "zero_floor": frozenset({True}),
            "unit_temperature": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    def gluon_attention_output_gate_gfx950(
        output: torch.Tensor,
        gate: torch.Tensor,
        gate_bias: torch.Tensor,
        floor: float,
        temperature: float,
    ) -> torch.Tensor:
        """Apply the unit-temperature, zero-floor gate without mutating inputs."""
        if floor != 0.0 or temperature != 1.0:
            raise ValueError(
                "Gluon attention gating requires floor=0 and temperature=1"
            )
        return launch_gluon_attention_output_gate_gfx950(output, gate, gate_bias)
