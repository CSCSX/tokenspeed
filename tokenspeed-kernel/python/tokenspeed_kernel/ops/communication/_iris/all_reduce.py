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

"""Iris all-reduce protocols and the synchronization used by its fusions."""

import math

import torch
from tokenspeed_kernel._triton import gl, gluon, hip_utils, tl, triton
from tokenspeed_kernel.ops.communication._iris import IrisGroup, allocation_bytes
from tokenspeed_kernel.ops.communication.iris import (
    _PACKED_WORD_BYTES,
    _PRODUCER_DIRECT_GL_DTYPES,
    _STAGED_TWO_SHOT_MIN_BYTES,
    _TWO_SHOT_WORLD_SIZES,
    _use_two_stage_plain,
    _use_two_stage_producer_direct,
    producer_direct_all_reduce_can_run,
)
from tokenspeed_kernel.platform import current_platform

_PULL_MAX_PROGRAMS = 84
_PULL_WORDS_PER_LANE = 2
_TWO_SHOT_NUM_SUBGROUPS = 8


def _pull_scratch_numel(max_numel: int, world_size: int) -> int:
    return (
        triton.cdiv(max_numel, world_size) if world_size in _TWO_SHOT_WORLD_SIZES else 0
    )


class ProducerBuffer:
    """Own consecutive symmetric outputs written by upstream producers."""

    @staticmethod
    def heap_bytes(max_numel: int, dtype: torch.dtype) -> int:
        return allocation_bytes((max_numel,), dtype)

    def __init__(self, peers: IrisGroup, max_numel: int, dtype: torch.dtype):
        self.peers = peers
        self.data = peers.ctx.zeros((max_numel,), dtype=dtype)

    @staticmethod
    def views(
        buffer: torch.Tensor, shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[torch.Tensor, ...]:
        views = []
        offset = 0
        for shape in shapes:
            numel = math.prod(shape)
            views.append(buffer.narrow(0, offset, numel).view(shape))
            offset += numel
        return tuple(views)

    def acquire(self, shapes: tuple[tuple[int, ...], ...]) -> tuple[torch.Tensor, ...]:
        if not shapes or any(math.prod(shape) <= 0 for shape in shapes):
            raise ValueError("Iris requires non-empty symmetric output shapes")
        if sum(math.prod(shape) for shape in shapes) > self.data.numel():
            raise ValueError("Iris symmetric outputs exceed the input buffer")
        return self.views(self.data, shapes)

    def owns(self, tensors: tuple[torch.Tensor, ...]) -> bool:
        if not tensors or any(
            tensor.dtype != self.data.dtype
            or tensor.device != self.data.device
            or not tensor.is_contiguous()
            or tensor.numel() <= 0
            for tensor in tensors
        ):
            return False
        offset = 0
        for tensor in tensors:
            if (
                tensor.data_ptr()
                != self.data.data_ptr() + offset * self.data.element_size()
            ):
                return False
            offset += tensor.numel()
        elements_per_word = _PACKED_WORD_BYTES // self.data.element_size()
        return offset % elements_per_word == 0 and offset <= self.data.numel()


class StagedPullOneShot:
    """Own rotating slots and epochs for one staging geometry."""

    BLOCK_SIZE = 2048
    NUM_SUBGROUPS = 4
    INPUT_SLOTS = 2

    @staticmethod
    def heap_bytes(
        max_numel: int, world_size: int, dtype: torch.dtype, block_size: int
    ) -> int:
        return allocation_bytes(
            (StagedPullOneShot.INPUT_SLOTS, max_numel), dtype
        ) + allocation_bytes(
            (triton.cdiv(max_numel, block_size), world_size), torch.int32
        )

    def __init__(
        self,
        peers: IrisGroup,
        max_numel: int,
        dtype: torch.dtype,
        *,
        block_size: int,
        num_subgroups: int,
    ):
        self.peers = peers
        self.block_size = block_size
        self.num_subgroups = num_subgroups
        self.arch_major = current_platform().arch_version.major
        self.slots = peers.ctx.zeros((self.INPUT_SLOTS, max_numel), dtype=dtype)
        # Kernels mark peer slot bases as 16-byte aligned.
        assert self.slots.data_ptr() % 16 == 0
        self.ready_flags = peers.ctx.zeros(
            (triton.cdiv(max_numel, self.block_size), peers.world_size),
            dtype=torch.int32,
        )

    def run(self, tensor: torch.Tensor, out: torch.Tensor) -> None:
        iris_pull_oneshot_staged[(triton.cdiv(tensor.numel(), self.block_size),)](
            tensor.view(-1),
            self.slots.view(-1),
            out.view(-1),
            self.ready_flags,
            *self.peers.addresses,
            tensor.numel(),
            RANK=self.peers.rank,
            WORLD_SIZE=self.peers.world_size,
            BLOCK_SIZE=self.block_size,
            SLOT_STRIDE=self.slots.shape[1],
            NUM_SLOTS=self.slots.shape[0],
            NUM_WARPS=self.num_subgroups,
            ARCH_MAJOR=self.arch_major,
            num_warps=self.num_subgroups,
        )


def _launch_pull_twoshot(
    peers: IrisGroup,
    inputs: torch.Tensor,
    scratch: torch.Tensor,
    ready_flags: torch.Tensor,
    out: torch.Tensor,
    *,
    exit_barrier: bool,
) -> None:
    elements_per_word = _PACKED_WORD_BYTES // inputs.element_size()
    partition_words = out.numel() // peers.world_size // elements_per_word
    block_words = (
        _TWO_SHOT_NUM_SUBGROUPS
        * triton.runtime.driver.active.get_current_target().warp_size
        * _PULL_WORDS_PER_LANE
        // peers.world_size
    )
    num_tiles = triton.cdiv(partition_words, block_words)
    num_programs = min(num_tiles, _PULL_MAX_PROGRAMS)
    iris_pull_twoshot[(num_programs,)](
        inputs,
        scratch,
        out,
        ready_flags,
        *peers.addresses,
        RANK=peers.rank,
        WORLD_SIZE=peers.world_size,
        PARTITION_WORDS=partition_words,
        BLOCK_WORDS=block_words,
        NUM_PROGRAMS=num_programs,
        NUM_TILES=num_tiles,
        NUM_WARPS=_TWO_SHOT_NUM_SUBGROUPS,
        WORDS_PER_LANE=_PULL_WORDS_PER_LANE,
        ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[inputs.dtype],
        ELEMENTS_PER_WORD=elements_per_word,
        EXIT_BARRIER=exit_barrier,
        num_warps=_TWO_SHOT_NUM_SUBGROUPS,
    )


class StagedPullTwoShot:
    """Own stable staging, partition scratch, and completion epochs."""

    @staticmethod
    def supported(world_size: int, dtype: torch.dtype) -> bool:
        return (
            current_platform().is_cdna4
            and world_size in _TWO_SHOT_WORLD_SIZES
            and dtype in _PRODUCER_DIRECT_GL_DTYPES
        )

    @staticmethod
    def heap_bytes(max_numel: int, world_size: int, dtype: torch.dtype) -> int:
        return (
            allocation_bytes((max_numel,), dtype)
            + allocation_bytes((_pull_scratch_numel(max_numel, world_size),), dtype)
            + allocation_bytes((_PULL_MAX_PROGRAMS, world_size), torch.int32)
        )

    def __init__(self, peers: IrisGroup, max_numel: int, dtype: torch.dtype):
        self.peers = peers
        self.input = peers.ctx.zeros((max_numel,), dtype=dtype)
        self.scratch = peers.ctx.zeros(
            (_pull_scratch_numel(max_numel, peers.world_size),), dtype=dtype
        )
        self.ready_flags = peers.ctx.zeros(
            (_PULL_MAX_PROGRAMS, peers.world_size), dtype=torch.int32
        )

    def can_run(self, numel: int) -> bool:
        world_size = self.peers.world_size
        return (
            0 < numel <= self.input.numel()
            and numel * self.input.element_size()
            >= _STAGED_TWO_SHOT_MIN_BYTES[world_size]
            and _use_two_stage_plain(world_size, numel, self.input.dtype)
        )

    def run(self, tensor: torch.Tensor, out: torch.Tensor) -> None:
        self.input[: tensor.numel()].copy_(tensor.view(-1))
        # Packed stores require an aligned destination; ranks keep the same
        # collective when a caller supplies an element-aligned tensor view.
        aligned_output = (
            torch.empty_like(out) if out.data_ptr() % _PACKED_WORD_BYTES else out
        )
        # Graph replay reuses this staging address; peers must finish reading it.
        _launch_pull_twoshot(
            self.peers,
            self.input,
            self.scratch,
            self.ready_flags,
            aligned_output,
            exit_barrier=True,
        )
        if aligned_output is not out:
            out.copy_(aligned_output)


class ProducerPull:
    """Borrow producer input; own scratch and epochs shared by one-/two-shot pull."""

    BLOCK_SIZE = 512
    NUM_SUBGROUPS = 1

    @staticmethod
    def heap_bytes(max_numel: int, world_size: int, dtype: torch.dtype) -> int:
        return allocation_bytes(
            (_pull_scratch_numel(max_numel, world_size),), dtype
        ) + allocation_bytes((_PULL_MAX_PROGRAMS, world_size), torch.int32)

    def __init__(self, producer: ProducerBuffer):
        self.producer = producer
        peers = producer.peers
        scratch_numel = _pull_scratch_numel(producer.data.numel(), peers.world_size)
        self.scratch = (
            peers.ctx.zeros((scratch_numel,), dtype=producer.data.dtype)
            if scratch_numel
            else None
        )
        self.ready_flags = peers.ctx.zeros(
            (_PULL_MAX_PROGRAMS, peers.world_size), dtype=torch.int32
        )

    def can_run(self, shapes: tuple[tuple[int, ...], ...]) -> bool:
        data = self.producer.data
        return current_platform().is_cdna4 and producer_direct_all_reduce_can_run(
            self.producer.peers.world_size,
            sum(math.prod(shape) for shape in shapes),
            data.dtype,
            data.numel() * data.element_size(),
        )

    def run(self, out: torch.Tensor) -> None:
        peers = self.producer.peers
        inputs = self.producer.data
        if _use_two_stage_producer_direct(peers.world_size, out.numel(), inputs.dtype):
            assert self.scratch is not None
            _launch_pull_twoshot(
                peers, inputs, self.scratch, self.ready_flags, out, exit_barrier=False
            )
            return
        num_tiles = triton.cdiv(out.numel(), self.BLOCK_SIZE)
        num_programs = min(num_tiles, _PULL_MAX_PROGRAMS)
        iris_pull_oneshot[(num_programs,)](
            inputs,
            out,
            self.ready_flags,
            *peers.addresses,
            RANK=peers.rank,
            WORLD_SIZE=peers.world_size,
            TOTAL_NUMEL=out.numel(),
            BLOCK_SIZE=self.BLOCK_SIZE,
            NUM_PROGRAMS=num_programs,
            NUM_TILES=num_tiles,
            NUM_WARPS=self.NUM_SUBGROUPS,
            WORDS_PER_LANE=_PULL_WORDS_PER_LANE,
            PUBLISH_READY=False,
            ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[inputs.dtype],
            ELEMENTS_PER_WORD=_PACKED_WORD_BYTES // inputs.element_size(),
            num_warps=self.NUM_SUBGROUPS,
        )


class ProducerLamport:
    """Own a TP8 BF16 Lamport mailbox for the shapes selected during preparation."""

    WORLD_SIZE = 8
    BLOCK_ELEMENTS = 512
    NUM_STAGES = 3
    NUM_SUBGROUPS = 1
    TRANSACTION_BYTES = 16

    @staticmethod
    def supported(world_size: int, dtype: torch.dtype) -> bool:
        return (
            current_platform().is_cdna4
            and world_size == ProducerLamport.WORLD_SIZE
            and dtype == torch.bfloat16
        )

    @staticmethod
    def heap_bytes(max_numel: int) -> int:
        return allocation_bytes(
            (ProducerLamport.NUM_STAGES, ProducerLamport.WORLD_SIZE, max_numel),
            torch.bfloat16,
        )

    def __init__(
        self,
        producer: ProducerBuffer,
        *,
        max_numel: int,
    ):
        self.producer = producer
        peers = producer.peers
        assert self.supported(peers.world_size, producer.data.dtype)
        assert 0 < max_numel <= producer.data.numel()
        assert max_numel % self.BLOCK_ELEMENTS == 0
        self.max_numel = max_numel
        self.mailbox = peers.ctx.zeros(
            (self.NUM_STAGES, peers.world_size, self.max_numel), dtype=torch.bfloat16
        )
        self.epochs = torch.zeros(
            (self.max_numel // self.BLOCK_ELEMENTS,),
            dtype=torch.int32,
            device=peers.device,
        )
        # Reserve negative zero as the mailbox sentinel.
        self.mailbox.view(torch.int32).fill_(-2147483648)
        offset = self.mailbox.data_ptr() - peers.addresses[peers.rank]
        self.peer_mailboxes = tuple(base + offset for base in peers.addresses)
        assert all(
            address % self.TRANSACTION_BYTES == 0 for address in self.peer_mailboxes
        )

    def run(self, out: torch.Tensor) -> None:
        iris_lamport_allreduce[(out.numel() // self.BLOCK_ELEMENTS,)](
            self.producer.data,
            self.mailbox,
            out,
            self.epochs,
            *self.peer_mailboxes,
            RANK=self.producer.peers.rank,
            WORLD_SIZE=self.producer.peers.world_size,
            TOTAL_ELEMENTS=out.numel(),
            MAX_ELEMENTS=self.max_numel,
            NUM_STAGES=self.NUM_STAGES,
            num_warps=self.NUM_SUBGROUPS,
        )


def staged_tuning(
    max_numel: int, world_size: int, dtype: torch.dtype
) -> tuple[int, int, int] | None:
    """Return the measured TP4 geometry when this state can serve its payload."""
    # This exact CDNA4 TP4 payload is faster with the one-shot geometry.
    if (
        current_platform().is_cdna4
        and world_size == 4
        and dtype == torch.bfloat16
        and max_numel >= 16 * 4096
    ):
        return 16 * 4096, 512, 1
    return None


def all_reduce_heap_bytes(
    staged_max_numel: int,
    producer_max_numel: int,
    world_size: int,
    dtype: torch.dtype,
) -> int:
    size = 0
    if producer_max_numel:
        size += ProducerBuffer.heap_bytes(producer_max_numel, dtype)
        size += ProducerPull.heap_bytes(producer_max_numel, world_size, dtype)
    if staged_max_numel:
        size += StagedPullOneShot.heap_bytes(
            staged_max_numel, world_size, dtype, StagedPullOneShot.BLOCK_SIZE
        )
        tuning = staged_tuning(staged_max_numel, world_size, dtype)
        if tuning is not None:
            numel, block_size, _ = tuning
            size += StagedPullOneShot.heap_bytes(numel, world_size, dtype, block_size)
        if StagedPullTwoShot.supported(world_size, dtype):
            size += StagedPullTwoShot.heap_bytes(staged_max_numel, world_size, dtype)
    return size


@gluon.jit
def _iris_drain_subgroup_vmem():
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r,~{memory}",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _iris_staged_drain_vmem(ARCH_MAJOR: gl.constexpr):
    if ARCH_MAJOR >= 12:
        gl.inline_asm_elementwise(
            "s_wait_loadcnt 0; s_wait_storecnt 0",
            "=r,~{memory}",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        _iris_drain_subgroup_vmem()


@gluon.jit
def _iris_staged_pull_reduce(
    input_ptr,
    slots_ptr,
    ready_flags,
    flag_row,
    offsets,
    mask,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    SLOT_STRIDE: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    ARCH_MAJOR: gl.constexpr,
):
    """Stage local values, pull every peer's copy, and return the FP32 sum.

    Staging precedes the entry wait, so a rank called straight back would
    overwrite a slot a slower peer is still reading. With two or more rotating
    slots, the entry wait is sufficient: before a rank returns to epoch E's
    slot it must pass the wait at E+1, and a peer publishes E+1 only after it
    has finished reading E. A single slot instead needs a completion
    rendezvous before returning, trading heap for one more peer wait.

    Each ``flag_row`` is an independent epoch domain; every rank launches the
    same rows for a collective, so a shrinking grid leaves unused rows' epochs
    consistent. Epochs are int32 and wrap: the slot comes from the low bits
    and the wait compares modulo 2**32.
    """
    gl.static_assert(NUM_SLOTS & (NUM_SLOTS - 1) == 0)
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    epoch_ptr = ready_flags + flag_row * WORLD_SIZE + RANK
    # Only this rank writes its own counter, and the previous launch finished.
    epoch = gl.load(epoch_ptr) + 1
    slot = (epoch & (NUM_SLOTS - 1)) * SLOT_STRIDE

    local = gl.load(input_ptr + offsets, mask=mask, other=0.0)
    gl.store(slots_ptr + slot + offsets, local, mask=mask, cache_modifier=".wt")
    # Join every subgroup's staging stores before peers can observe the epoch.
    _iris_staged_drain_vmem(ARCH_MAJOR)
    gl.barrier()
    gl.atomic_xchg(epoch_ptr, epoch, sem="release", scope="sys")
    # Pushing epochs lets each rank poll local memory for all peers at once.
    _iris_sync_rank_epoch(
        ready_flags,
        flag_row,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        PUBLISH=True,
    )

    # Every rank adds in rank order so all ranks hold bitwise-equal results.
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            term = local
        else:
            peer_slots = _iris_translate(
                slots_ptr,
                local_heap,
                _iris_heap_base(
                    peer,
                    heap_base_0,
                    heap_base_1,
                    heap_base_2,
                    heap_base_3,
                    heap_base_4,
                    heap_base_5,
                    heap_base_6,
                    heap_base_7,
                ),
            )
            term = gl.load(
                peer_slots + slot + offsets,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
            )
        if peer == 0:
            acc = term.to(gl.float32)
        else:
            acc += term.to(gl.float32)

    if NUM_SLOTS == 1:
        # Peers must finish reading this slot before the next call restages it.
        _iris_staged_drain_vmem(ARCH_MAJOR)
        gl.barrier()
        gl.atomic_xchg(epoch_ptr, epoch + 1, sem="release", scope="sys")
        _iris_sync_rank_epoch(
            ready_flags,
            flag_row,
            epoch + 1,
            local_heap,
            heap_base_0,
            heap_base_1,
            heap_base_2,
            heap_base_3,
            heap_base_4,
            heap_base_5,
            heap_base_6,
            heap_base_7,
            RANK,
            WORLD_SIZE,
            NUM_WARPS,
            PUBLISH=True,
        )
    return acc


@gluon.jit
def iris_pull_oneshot_staged(
    input_ptr,
    slots_ptr,
    output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    NUMEL,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    SLOT_STRIDE: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    ARCH_MAJOR: gl.constexpr,
):
    """Reduce an ordinary tensor through rotating symmetric staging slots."""
    SUBGROUP_SIZE: gl.constexpr = hip_utils.num_threads() // hip_utils.num_warps()
    gl.static_assert(BLOCK_SIZE % (SUBGROUP_SIZE * NUM_WARPS) == 0)
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_SIZE // (SUBGROUP_SIZE * NUM_WARPS)], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    block_id = gl.program_id(0)
    offsets = block_id * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < NUMEL
    acc = _iris_staged_pull_reduce(
        input_ptr,
        slots_ptr,
        ready_flags,
        block_id,
        offsets,
        mask,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        SLOT_STRIDE,
        NUM_SLOTS,
        NUM_WARPS,
        ARCH_MAJOR,
    )
    gl.store(output_ptr + offsets, acc.to(output_ptr.dtype.element_ty), mask=mask)


@gluon.jit
def iris_pull_oneshot_rmsnorm(
    input_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    residual_out_ptr,
    slots_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    NUM_ROWS,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN_SIZE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    SLOT_STRIDE: gl.constexpr,
    NUM_SLOTS: gl.constexpr,
    EPS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    ARCH_MAJOR: gl.constexpr,
):
    """Reduce rows, add their residuals, and RMS-normalize them in one launch.

    Each row is its own epoch domain, so a persistent grid may stride rows.
    """
    SUBGROUP_SIZE: gl.constexpr = hip_utils.num_threads() // hip_utils.num_warps()
    gl.static_assert(BLOCK_SIZE % (SUBGROUP_SIZE * NUM_WARPS) == 0)
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_SIZE // (SUBGROUP_SIZE * NUM_WARPS)], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    columns = gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = columns < HIDDEN_SIZE
    weight = gl.load(weight_ptr + columns, mask=mask, other=0.0).to(gl.float32)
    for row in range(gl.program_id(0), NUM_ROWS, gl.num_programs(0)):
        offsets = row * HIDDEN_SIZE + columns
        acc = _iris_staged_pull_reduce(
            input_ptr,
            slots_ptr,
            ready_flags,
            row,
            offsets,
            mask,
            heap_base_0,
            heap_base_1,
            heap_base_2,
            heap_base_3,
            heap_base_4,
            heap_base_5,
            heap_base_6,
            heap_base_7,
            RANK,
            WORLD_SIZE,
            SLOT_STRIDE,
            NUM_SLOTS,
            NUM_WARPS,
            ARCH_MAJOR,
        )
        residual = gl.load(residual_ptr + offsets, mask=mask, other=0.0)
        residual_out = acc + residual.to(gl.float32)
        gl.store(
            residual_out_ptr + offsets,
            residual_out.to(residual_out_ptr.dtype.element_ty),
            mask=mask,
        )
        variance = gl.sum(residual_out * residual_out, axis=0) / HIDDEN_SIZE
        norm = residual_out * gl.rsqrt(variance + EPS) * weight
        gl.store(
            norm_out_ptr + offsets,
            norm.to(norm_out_ptr.dtype.element_ty),
            mask=mask,
        )


@gluon.jit
def _iris_sanitize_lamport_bf16(values):
    bits = values.to(gl.uint16, bitcast=True)
    return gl.where(bits == 0x8000, 0, bits).to(gl.bfloat16, bitcast=True)


@gluon.jit
def _iris_wait_lamport_peers(
    region,
    generation,
    offsets,
    valid,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    values = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        values += (gl.full([64, 8], 0, gl.bfloat16, LAYOUT),)
    active = valid
    while gl.any(active, 0):
        loaded = ()
        # A lane reloads every peer until all seven packs are ready. Issue the
        # independent reads before checking them; retired lanes keep their data.
        # .cv controls hardware caches, not compiler volatility. The compiler
        # regression test checks that these reads remain in the polling cycle.
        for delta in gl.static_range(1, WORLD_SIZE):
            peer = (RANK + delta) % WORLD_SIZE
            pointer = (
                region + generation * WORLD_SIZE * MAX_ELEMENTS + peer * MAX_ELEMENTS
            )
            loaded += (
                gl.amd.cdna4.buffer_load(
                    pointer,
                    offsets,
                    mask=active[:, None],
                    other=values[delta - 1],
                    cache=".cv",
                ),
            )
        active = gl.full([64], False, gl.int1, gl.SliceLayout(1, LAYOUT))
        for delta in gl.static_range(0, WORLD_SIZE - 1):
            active |= valid & (
                gl.max(
                    (loaded[delta].to(gl.uint16, bitcast=True) == 0x8000).to(gl.int32),
                    1,
                )
                != 0
            )
        values = loaded
    return values


@gluon.jit
def iris_lamport_allreduce(
    input_sym_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    region_0,
    region_1,
    region_2,
    region_3,
    region_4,
    region_5,
    region_6,
    region_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TOTAL_ELEMENTS: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """Push BF16 tiles and poll all peer packs with one subgroup per tile."""
    gl.static_assert(WORLD_SIZE == 8)
    gl.static_assert(TOTAL_ELEMENTS % 512 == 0)
    gl.static_assert(TOTAL_ELEMENTS <= MAX_ELEMENTS)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * 8 + element[None, :]
    mask = offsets < TOTAL_ELEMENTS
    generation = (
        gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    )
    stride: gl.constexpr = WORLD_SIZE * MAX_ELEMENTS
    local = gl.amd.cdna4.buffer_load(input_sym_ptr, offsets, mask=mask, other=0.0)
    local = _iris_sanitize_lamport_bf16(local)
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = _iris_heap_base(
            peer,
            region_0,
            region_1,
            region_2,
            region_3,
            region_4,
            region_5,
            region_6,
            region_7,
        )
        # Preserve 16-byte publication stores after casting integer addresses.
        destination = gl.multiple_of(destination.to(gl.pointer_type(gl.bfloat16)), 16)
        destination += generation * stride + RANK * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(local, destination, offsets, mask=mask, cache=".wt")

    peers = _iris_wait_lamport_peers(
        region_sym_ptr,
        generation,
        offsets,
        pack * 8 < TOTAL_ELEMENTS,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        layout,
    )
    # All ranks use the same FP32 addition order, then round once to BF16.
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            term = local
        else:
            term = peers[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        if peer == 0:
            accumulator = term.to(gl.float32)
        else:
            accumulator += term.to(gl.float32)
    gl.amd.cdna4.buffer_store(
        accumulator.to(gl.bfloat16), output_ptr, offsets, mask=mask
    )

    # Clear only this tile's consumed generation. Skipped tiles keep their
    # own epochs, so mixed row counts need no global counter or tail clearing.
    sentinel = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = region_sym_ptr + generation * stride + peer * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(
            sentinel, destination, offsets, mask=mask, cache=".wt"
        )
    gl.barrier()
    gl.store(epochs + gl.program_id(0), ((generation + 1) % NUM_STAGES).to(gl.int32))


@gluon.jit
def _iris_heap_base(
    rank: gl.constexpr,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
):
    if rank == 0:
        return heap_base_0
    if rank == 1:
        return heap_base_1
    if rank == 2:
        return heap_base_2
    if rank == 3:
        return heap_base_3
    if rank == 4:
        return heap_base_4
    if rank == 5:
        return heap_base_5
    if rank == 6:
        return heap_base_6
    return heap_base_7


@gluon.jit
def _iris_translate(pointer, local_heap, peer_heap):
    """Map a 16-byte-aligned symmetric pointer from this heap to a peer's."""
    offset = tl.cast(pointer, gl.uint64) - local_heap
    return gl.multiple_of(tl.cast(peer_heap + offset, pointer.dtype), 16)


@gluon.jit
def _iris_sync_rank_token(
    flags,
    row,
    token,
    local_heap,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    SUBGROUP_SIZE: gl.constexpr = hip_utils.num_threads() // hip_utils.num_warps()
    layout: gl.constexpr = gl.BlockedLayout([1], [SUBGROUP_SIZE], [NUM_WARPS], [0])
    peers = gl.arange(0, WORLD_SIZE, layout=layout)
    peer_mask = peers != RANK
    peer_heaps = gl.where(peers == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peers == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peers == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peers == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peers == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peers == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peers == 6, heap_base_6, peer_heaps)
    flags_heap_offset = tl.cast(flags, gl.uint64) - local_heap
    peer_flags = tl.cast(
        peer_heaps + flags_heap_offset,
        gl.pointer_type(gl.int32),
    )
    gl.atomic_xchg(
        peer_flags + row * WORLD_SIZE + RANK,
        token,
        mask=peer_mask,
        sem="release",
        scope="sys",
    )
    local_flags = flags + row * WORLD_SIZE + peers
    seen = gl.load(
        local_flags,
        mask=peer_mask,
        other=token,
        cache_modifier=".cv",
        volatile=True,
    )
    while gl.max(gl.where(peer_mask & (seen != token), 1, 0), axis=0) != 0:
        seen = gl.load(
            local_flags,
            mask=peer_mask,
            other=token,
            cache_modifier=".cv",
            volatile=True,
        )
    gl.atomic_add(
        local_flags,
        0,
        mask=peer_mask,
        sem="acquire",
        scope="sys",
    )
    _iris_drain_subgroup_vmem()
    # The acquire is subgroup-local. Keep every subgroup at the protocol
    # boundary until all of them have observed the peer publications; the
    # caller consumes the peer inbox immediately after this helper returns.
    gl.barrier()


@triton.constexpr_function
def _peer_lanes(world_size):
    return triton.next_power_of_2(world_size)


@gluon.jit
def _iris_sync_rank_epoch(
    ready_flags,
    block_id,
    epoch,
    local_heap,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PUBLISH: gl.constexpr,
):
    """Synchronize by polling peer epochs or publishing them locally."""
    SUBGROUP_SIZE: gl.constexpr = hip_utils.num_threads() // hip_utils.num_warps()
    ready_layout: gl.constexpr = gl.BlockedLayout(
        [1], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    peer_ids = gl.arange(0, _peer_lanes(WORLD_SIZE), layout=ready_layout)
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)
    flags_heap_offset = tl.cast(ready_flags, gl.uint64) - local_heap
    peer_mask = (peer_ids != RANK) & (peer_ids < WORLD_SIZE)
    if PUBLISH:
        remote_flags = tl.cast(
            peer_heaps + flags_heap_offset,
            gl.pointer_type(gl.int32),
        )
        remote_flags += block_id * WORLD_SIZE + RANK
        gl.store(
            remote_flags,
            epoch,
            mask=peer_mask,
            cache_modifier=".wt",
        )
        wait_flags = ready_flags + block_id * WORLD_SIZE + peer_ids
    else:
        wait_flags = tl.cast(
            peer_heaps + flags_heap_offset,
            gl.pointer_type(gl.int32),
        )
        wait_flags += block_id * WORLD_SIZE + peer_ids
    seen = gl.load(
        wait_flags,
        mask=peer_mask,
        other=epoch,
        cache_modifier=".cv",
        volatile=True,
    )
    # Compare modulo 32 bits, including when a peer has passed the wrap before
    # this rank. A zero or negative epoch still requires observing every peer.
    while gl.min((seen - epoch).to(gl.int32), axis=0) < 0:
        seen = gl.load(
            wait_flags,
            mask=peer_mask,
            other=epoch,
            cache_modifier=".cv",
            volatile=True,
        )


@gluon.jit
def _unpack_16bitx4(packed, dtype: gl.constexpr):
    value_0 = (packed & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    value_1 = (
        ((packed >> 16) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    )
    value_2 = (
        ((packed >> 32) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    )
    value_3 = (
        ((packed >> 48) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    )
    return value_0, value_1, value_2, value_3


@gluon.jit
def _pack_16bitx4(value_0, value_1, value_2, value_3, dtype: gl.constexpr):
    bits_0 = value_0.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_1 = value_1.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_2 = value_2.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_3 = value_3.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    return bits_0 | (bits_1 << 16) | (bits_2 << 32) | (bits_3 << 48)


@gluon.jit
def _unpack_word(packed, dtype: gl.constexpr, elements_per_word: gl.constexpr):
    # Explicit branches keep Gluon from type-checking the inactive bitcast.
    if elements_per_word == 4:
        return _unpack_16bitx4(packed, dtype)
    else:
        value_0 = (packed & 0xFFFFFFFF).to(gl.uint32).to(dtype, bitcast=True)
        value_1 = ((packed >> 32) & 0xFFFFFFFF).to(gl.uint32).to(dtype, bitcast=True)
        return value_0, value_1, value_0, value_1


@gluon.jit
def _pack_word(
    value_0,
    value_1,
    value_2,
    value_3,
    dtype: gl.constexpr,
    elements_per_word: gl.constexpr,
):
    if elements_per_word == 4:
        return _pack_16bitx4(value_0, value_1, value_2, value_3, dtype)
    else:
        bits_0 = value_0.to(dtype).to(gl.uint32, bitcast=True).to(gl.uint64)
        bits_1 = value_1.to(dtype).to(gl.uint32, bitcast=True).to(gl.uint64)
        return bits_0 | (bits_1 << 32)


@gluon.jit
def iris_pull_oneshot(
    input_sym_ptr,
    output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    # The reduced size and the tile/program counts derived from it follow the
    # batch; runtime so every batch shape shares one binary.
    TOTAL_NUMEL,
    BLOCK_SIZE: gl.constexpr,
    NUM_PROGRAMS,
    NUM_TILES,
    NUM_WARPS: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    PUBLISH_READY: gl.constexpr,
    ELEMENT_DTYPE: gl.constexpr,
    ELEMENTS_PER_WORD: gl.constexpr,
):
    """Reduce producer outputs placed consecutively in symmetric memory."""
    block_id = gl.program_id(0)
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        PUBLISH=PUBLISH_READY,
    )

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    SUBGROUP_SIZE: gl.constexpr = hip_utils.num_threads() // hip_utils.num_warps()
    layout: gl.constexpr = gl.BlockedLayout(
        [WORDS_PER_LANE], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    lane = gl.arange(0, BLOCK_SIZE // ELEMENTS_PER_WORD, layout=layout)
    total_packed = TOTAL_NUMEL // ELEMENTS_PER_WORD
    tile_id = block_id
    while tile_id < NUM_TILES:
        packed_offset = tile_id * (BLOCK_SIZE // ELEMENTS_PER_WORD) + lane
        mask = packed_offset < total_packed
        local_packed = gl.amd.cdna4.buffer_load(
            tl.cast(input_sym_ptr, gl.pointer_type(gl.uint64)),
            packed_offset.to(gl.int32),
            mask=mask,
            other=0,
        )
        acc_0, acc_1, acc_2, acc_3 = _unpack_word(
            local_packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
        )
        for peer in gl.static_range(0, WORLD_SIZE):
            if peer != RANK:
                peer_heap = _iris_heap_base(
                    peer,
                    heap_base_0,
                    heap_base_1,
                    heap_base_2,
                    heap_base_3,
                    heap_base_4,
                    heap_base_5,
                    heap_base_6,
                    heap_base_7,
                )
                peer_input = tl.cast(
                    peer_heap + input_heap_offset, gl.pointer_type(gl.uint64)
                )
                peer_packed = gl.amd.cdna4.buffer_load(
                    peer_input,
                    packed_offset.to(gl.int32),
                    mask=mask,
                    other=0,
                    cache=".cg",
                )
                peer_0, peer_1, peer_2, peer_3 = _unpack_word(
                    peer_packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
                )
                acc_0 += peer_0
                acc_1 += peer_1
                acc_2 += peer_2
                acc_3 += peer_3

        packed_output = _pack_word(
            acc_0,
            acc_1,
            acc_2,
            acc_3,
            ELEMENT_DTYPE,
            ELEMENTS_PER_WORD,
        )
        gl.amd.cdna4.buffer_store(
            packed_output,
            tl.cast(output_ptr, gl.pointer_type(gl.uint64)),
            packed_offset.to(gl.int32),
            mask=mask,
        )
        tile_id += NUM_PROGRAMS

    # Do not return while a peer program can still be reading this rank's
    # input. The next producer reuses the same symmetric buffer.
    completion_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        completion_epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        PUBLISH=PUBLISH_READY,
    )


@gluon.jit
def iris_pull_twoshot(
    input_sym_ptr,
    scratch_sym_ptr,
    output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    # The partition size and the tile/program counts derived from it follow
    # the batch; runtime so every batch shape shares one binary.
    PARTITION_WORDS,
    BLOCK_WORDS: gl.constexpr,
    NUM_PROGRAMS,
    NUM_TILES,
    NUM_WARPS: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    ELEMENT_DTYPE: gl.constexpr,
    ELEMENTS_PER_WORD: gl.constexpr,
    EXIT_BARRIER: gl.constexpr,
):
    """Reduce-scatter producer outputs, then all-gather the rank partitions."""
    block_id = gl.program_id(0)
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        PUBLISH=True,
    )

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    scratch_heap_offset = tl.cast(scratch_sym_ptr, gl.uint64) - local_heap
    SUBGROUP_SIZE: gl.constexpr = hip_utils.num_threads() // hip_utils.num_warps()
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, WORDS_PER_LANE],
        [1, SUBGROUP_SIZE],
        [WORLD_SIZE, NUM_WARPS // WORLD_SIZE],
        [1, 0],
    )
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        [WORLD_SIZE, 1], [1, SUBGROUP_SIZE], [1, NUM_WARPS], [0, 1]
    )
    peer_layout: gl.constexpr = gl.SliceLayout(1, load_layout)
    word_layout: gl.constexpr = gl.SliceLayout(0, load_layout)
    reduce_word_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    peer_ids = gl.arange(0, WORLD_SIZE, layout=peer_layout)
    words = gl.arange(0, BLOCK_WORDS, layout=word_layout)
    reduce_words = gl.arange(0, BLOCK_WORDS, layout=reduce_word_layout)
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)
    peer_inputs = tl.cast(
        gl.expand_dims(peer_heaps, 1) + input_heap_offset,
        gl.pointer_type(gl.uint64),
    )
    peer_scratch = tl.cast(
        gl.expand_dims(peer_heaps, 1) + scratch_heap_offset,
        gl.pointer_type(gl.uint64),
    )
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[32, 4]],
        [WORLD_SIZE, BLOCK_WORDS],
        [1, 0],
    )
    peer_values = gl.allocate_shared_memory(
        gl.uint64,
        [WORLD_SIZE, BLOCK_WORDS],
        shared_layout,
    )
    rank_start = RANK * PARTITION_WORDS

    # Reduce only this rank's partition of the full input into symmetric scratch.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        input_offset = rank_start + partition_offset
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_inputs + gl.expand_dims(input_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=".cg",
        )
        peer_values.store(values)

        packed = peer_values.load(reduce_layout)
        value_0, value_1, value_2, value_3 = _unpack_word(
            packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
        )
        reduced = _pack_word(
            gl.sum(value_0, axis=0),
            gl.sum(value_1, axis=0),
            gl.sum(value_2, axis=0),
            gl.sum(value_3, axis=0),
            ELEMENT_DTYPE,
            ELEMENTS_PER_WORD,
        )
        gl.amd.cdna4.buffer_store(
            reduced,
            tl.cast(scratch_sym_ptr, gl.pointer_type(gl.uint64)),
            (tile_id * BLOCK_WORDS + reduce_words).to(gl.int32),
            mask=tile_id * BLOCK_WORDS + reduce_words < PARTITION_WORDS,
            cache=".wt",
        )
        tile_id += NUM_PROGRAMS

    partitions_ready = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        partitions_ready,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        PUBLISH=True,
    )

    # Gather one reduced partition from every rank into the local output.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_scratch + gl.expand_dims(partition_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=".cg",
        )
        output_offset = gl.expand_dims(peer_ids * PARTITION_WORDS, 1) + gl.expand_dims(
            partition_offset, 0
        )
        gl.store(
            tl.cast(output_ptr, gl.pointer_type(gl.uint64)) + output_offset,
            values,
            mask=gl.expand_dims(mask, 0),
        )
        tile_id += NUM_PROGRAMS

    if EXIT_BARRIER:
        # Callers that stage into the symmetric input before launching cannot
        # rotate buffers safely: under graph capture the staging copy records a
        # fixed address and replays it, so a rank one invocation ahead would
        # overwrite an input its slower peers are still reducing. Holding the
        # kernel until every peer has finished reading makes a single staging
        # buffer correct by construction, at the price of one more rendezvous.
        # Producer-direct callers own their input and pass False.
        reads_done = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
        _iris_sync_rank_epoch(
            ready_flags,
            block_id,
            reads_done,
            local_heap,
            heap_base_0,
            heap_base_1,
            heap_base_2,
            heap_base_3,
            heap_base_4,
            heap_base_5,
            heap_base_6,
            heap_base_7,
            RANK,
            WORLD_SIZE,
            NUM_WARPS,
            PUBLISH=True,
        )
