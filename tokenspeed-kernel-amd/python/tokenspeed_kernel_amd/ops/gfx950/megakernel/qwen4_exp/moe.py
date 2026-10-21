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

"""Sparse MoE stages (TokenSpeed ``Qwen3_5MoeSparseMoeBlock``).

``out = sum_k w_k E_k(x) + sigmoid(g . x) S(x)`` over the top 10 of 512
routed experts (softmax weights renormalized over the 10) and the shared
expert ``S``. ``E(x) = W_down (silu(W_gate x) * W_up x)``, with the
intermediate rounded to BF16 and quantized to MXFP8 like TokenSpeed's MXFP8
expert pipeline. The block output joins the residual streams with the MLP
mixer's inject logits: ``h_b + bf16(out) * 2 sigmoid(inject_b)``.

Every task serves every row of the launch:

* router (M3), 65 blocks of 8 router rows (BF16); the last block is the
  shared expert gate. Past 2 rows, two tasks per block split the rows
  (:func:`router_rows`). M3 publishes the 512 logits as BF16.
* expert up (M4), 200 routed tasks: rank ``r`` (0-9) x 32 intermediate
  columns, then the shared expert's tasks (:func:`shared_blocks` blocks of
  32 columns each). A task polls its input rows first (M2 published them
  before M3 ran). Then it routes (:func:`routing`, the CTA's waves together).
  Then it streams units of weights, two in flight: a routed task's unit is
  a row's expert, a shared task's unit is a block, read once for every row.
  A routed task keeps each row's result in its own MFMA rows of one
  accumulator, so one epilogue publishes ``silu(gate) * up`` of every row
  as MXFP8 blocks.
* expert down (M5), 160 tasks of 16 hidden columns. A task routes again and
  polls every row's 11 intermediates. Its chains are the (row, rank) down
  projections, 16 per rank-3 MFMA whose batch dimension is the chain. The
  epilogue combines each row's chains in rank order (the same order at any
  row count) and writes the layer's output streams.

The stages read routed experts in place from TokenSpeed's N16 MoE banks. The
MXFP8 pipeline pads the intermediate to ``INTER_PAD``, and its scales use the
``"n32"`` cells of ``mx._scale_bytes``. The shared expert's weights are the
dense projections' preshuffled tiles, with scale copies in the same cells.
Its gate / up blocks alternate like the banks' (``weights.alternate_gate_up``).
Thus every M4 rank reads its weights the same way.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._scheduling import wave_uniform_i32
from tokenspeed_kernel_amd._triton import aggregate, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import Stage, placement
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp.dims import HIDDEN

EXPERTS = gl.constexpr(512)
TOP_K = gl.constexpr(10)
RANKS = gl.constexpr(11)  # the routed top 10, then the shared expert
INTER = gl.constexpr(640)
INTER_PAD = gl.constexpr(768)
ROUTER_TASKS = gl.constexpr(65)
UP_BLOCKS = gl.constexpr(220)  # RANKS * INTER / 32: 32-column blocks
DOWN_TASKS = gl.constexpr(160)  # HIDDEN / 16
W13_CELLS = gl.constexpr(10)  # 256-wide K cells of HIDDEN
W2_CELLS = gl.constexpr(3)  # of INTER_PAD

REGIONS = (
    ("lg", 256),  # router logits, BF16 pairs
    ("sg", 1),  # shared expert gate logit (f32)
    ("dn8", 1760),  # MXFP8 words of each rank's intermediate, [RANKS, 160]
    ("dne", 220),  # its exponents, [RANKS, 20]
)


def shared_blocks(rows: int) -> int:
    """32-column blocks of the shared expert per M4 task. Past one row, a
    task reads its weights once for every row, so it takes more blocks."""
    return {1: 1, 2: 2}.get(rows, 4)


def up_tasks(rows: int) -> int:
    """M4's tasks: 20 per routed rank, then the shared expert's."""
    return int(TOP_K) * 20 + 20 // shared_blocks(rows)


def router_rows(rows: int) -> int:
    """Rows of one router task. Past two rows, two: the tasks of a block of
    experts split the rows, so a task's per-row work stays short."""
    return 2 if rows > 2 else rows


def stages(
    bases: dict[str, int], rows: int, X: str, W: str, OUT: tuple[str, ...]
) -> tuple[Stage, ...]:
    """M3-M5 for ``rows`` rows, placed by ``bases``. ``X`` is the MLP
    mixer's region prefix and
    ``W`` is the region of the residual streams. M5 also publishes the output
    streams in the regions ``OUT`` (none or one)."""
    m3, m4, m5 = placement(bases, ("m3", "m4", "m5"))
    return (
        Stage(
            "m3",
            int(ROUTER_TASKS) * -(-rows // router_rows(rows)),
            m3,
            reads=(X + "xbf",),
            writes=("lg", "sg"),
        ),
        Stage(
            "m4",
            up_tasks(rows),
            m4,
            reads=("lg", X + "x8", X + "xe"),
            writes=("dn8", "dne"),
        ),
        Stage(
            "m5",
            int(DOWN_TASKS),
            m5,
            reads=("lg", "dn8", "dne", W, X + "inj", "sg"),
            writes=OUT,
        ),
    )


@aggregate
class Weights:
    """The MoE block's weights (``weights.MoeWeights``)."""

    router_w: gl.tensor  # router, BF16 [EXPERTS, HIDDEN]
    gate_w: gl.tensor  # shared expert gate, BF16 [HIDDEN]
    # Routed experts' N16 gate / up bank [EXPERTS, 2 INTER_PAD, HIDDEN]
    # (16-row gate and up blocks alternate) and its "n32" scales.
    w13: gl.tensor
    s13: gl.tensor
    # The shared expert's gate / up [2 INTER, HIDDEN] preshuffled tiles and
    # "n32" scales, in the banks' block order (weights.alternate_gate_up).
    shared_gu: gl.tensor
    shared_gu_s: gl.tensor
    # Routed experts' N16 down bank [EXPERTS, HIDDEN, INTER_PAD] and its
    # "n32" scales.
    w2: gl.tensor
    s2: gl.tensor
    # The shared expert's down [HIDDEN, INTER] preshuffled tiles and "n32"
    # scale copy (W2_CELLS cells).
    shared_dn: gl.tensor
    shared_dn_s: gl.tensor


@gluon.constexpr_function
def _rank_layout(num_warps):
    """``[16]`` per-rank values, rank ``r`` on lane ``r`` of every wave."""
    return gl.BlockedLayout([1], [64], [num_warps], [0])


@gluon.jit
def _pick(values, k):
    """Rank ``k`` (a runtime scalar) of a :func:`_route_row` vector."""
    ranks = gl.arange(0, 16, values.type.layout)
    return gl.sum(gl.where(ranks == k, values, 0), axis=0)


@gluon.jit
def router_stage(mb, task, weights, X: gl.constexpr, STAGE: gl.constexpr):
    """M3 task ``task``: for rows ``RT h .. RT h + RT - 1`` (``RT`` =
    :func:`router_rows`, ``h = task // 65``), the logits of experts
    ``8 b .. 8 b + 7`` (BF16, ``b = task % 65``), or for the last block the
    shared expert gate logit (FP32).

    Args:
        weights: The block's :class:`Weights`.
        X: Region prefix of the MLP mixer (its BF16 row ``xbf``).
    """
    mb.stamp(STAGE, trace.START)
    R: gl.constexpr = schedule.rows(mb.plan)
    RT: gl.constexpr = _task_rows(mb.plan)
    RTP: gl.constexpr = triton.next_power_of_2(RT)
    block = task % ROUTER_TASKS
    row0 = task // ROUTER_TASKS * RT
    is_gate = block == ROUTER_TASKS - 1
    rl: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [gl.num_warps(), 1], [1, 0])
    r = gl.arange(0, 8, gl.SliceLayout(1, rl))
    k = gl.arange(0, 4096, gl.SliceLayout(0, rl))
    rows = gl.where(
        is_gate,
        weights.gate_w + 0 * (r[:, None] * HIDDEN),
        weights.router_w + (block * 8 + r)[:, None] * HIDDEN,
    )
    w = gl.load(rows + k[None, :], mask=(k < HIDDEN)[None, :], other=0.0)
    gl.amd.hint.sched_barrier()
    # The task's BF16 rows (past the last row: the last row again): poll
    # compactly, then share them through LDS.
    pl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, gl.num_warps()], [1, 0])
    j = gl.arange(0, RTP, gl.SliceLayout(1, pl))[:, None]
    i = gl.arange(0, 2048, gl.SliceLayout(0, pl))[None, :]
    index = mb.row_offset(gl.minimum(row0 + j, R - 1)) + gl.minimum(i, HIDDEN // 2 - 1)
    (words,) = mb.poll_pairs((mb.pairs(X + "xbf", index),))
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    x_words = gl.allocate_shared_memory(gl.int32, [RTP * 2048], flat)
    x_words.store(gl.reshape(gl.where(i < HIDDEN // 2, words, 0), [RTP * 2048]))
    gl.barrier()
    mb.stamp(STAGE, trace.READY)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    for t in gl.static_range(RT):
        row = row0 + t
        if row < R:
            x = x_words.slice(t * 2048, 2048).reinterpret(gl.bfloat16, [4096], flat)
            x = x.load(gl.SliceLayout(0, rl))
            logit = gl.sum(w.to(gl.float32) * x.to(gl.float32)[None, :], axis=1)
            if is_gate:
                one = gl.arange(0, 1, lanes)
                mb.row(row).put(
                    "sg",
                    one,
                    gl.full([1], 0.0, gl.float32, lanes) + gl.max(logit, axis=0),
                )
            else:
                mx.put_bf16(mb.row(row), "lg", block * 4, logit)
    mb.stamp(STAGE, trace.COMPUTED)
    mb.stamp(STAGE, trace.END)


@gluon.constexpr_function
def _task_rows(plan):
    """:func:`router_rows` of a launch's plan."""
    return 2 if plan.plan.rows > 2 else plan.plan.rows


@gluon.jit
def _up_tiles(
    task, unit: gl.constexpr, routes, weights, R: gl.constexpr, C: gl.constexpr
):
    """Issue the weight loads of unit ``unit`` of M4 task ``task``
    (``routes`` from :func:`routing`): blocks ``4 q .. 4 q + 3`` (gate, up,
    gate, up rows of columns ``32 q ..``) of an expert. A routed rank's
    unit is a row: the row's expert, block ``q = task % 20``. A shared expert
    task's unit is its block ``q = C (task - 200) + unit``.

    The shared expert's blocks alternate like the banks'. Thus every unit is
    one load sequence with a scalar base. A branch per rank would join two
    sequences' registers, and the copies there wait for the loads.
    """
    rank = task // 20
    routed = rank < TOP_K
    # Wave-uniform, so the loads take a scalar base (else every load becomes
    # a waterfall loop over the lanes' bases).
    row = _route_row(routes, _last_row(unit, R))[0]
    expert = wave_uniform_i32(_pick(row, rank)).to(gl.int64)
    q = task % 20
    if C > 1:
        q = gl.where(routed, q, (task - TOP_K * 20) * C + unit)
    if routed:
        tiles, scales = mx.load_weight_tiles(
            weights.w13 + expert * (2 * INTER_PAD * HIDDEN // 2),
            weights.s13 + expert * (INTER_PAD // 16 * W13_CELLS * 256),
            nb0=q * 4,
            K=HIDDEN,
            kit0=0,
            KITS=20,
            SPLIT=2,
            NW=4,
            GROUP=4,
            GROUP_STRIDE=0,
            SCALES="n32",
        )
    else:
        tiles, scales = mx.load_weight_tiles(
            weights.shared_gu,
            weights.shared_gu_s,
            nb0=q * 4,
            K=HIDDEN,
            kit0=0,
            KITS=20,
            SPLIT=2,
            NW=4,
            GROUP=4,
            GROUP_STRIDE=0,
            SCALES="n32",
        )
    return tiles, scales


@gluon.jit
def _up_publish(mb, task, out):
    """SwiGLU of M4 task ``task``'s GEMV result ``out [RP, 64]`` (a row per
    row), published as one MXFP8 block of ``dn8`` / ``dne`` per row."""
    RP: gl.constexpr = out.shape[0]
    # Result column n is block n // 16: gate, up, gate, up. Round to BF16
    # like TokenSpeed's epilogues.
    out = gl.reshape(out.to(gl.bfloat16).to(gl.float32), [RP, 2, 2, 16])
    lay: gl.constexpr = out.type.layout
    i1 = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, lay))))
    gate = gl.sum(gl.where((i1 == 0)[None, None, :, None], out, 0.0), axis=2)
    up = gl.sum(gl.where((i1 == 1)[None, None, :, None], out, 0.0), axis=2)
    h = (mx.silu(gate) * up).to(gl.bfloat16).to(gl.float32)
    mx.put_mxfp8_rows(mb, "dn8", "dne", task, gl.reshape(h, [RP, 32]))


@gluon.constexpr_function
def _batch_layouts(num_warps):
    """Rank-3 MFMA over a group of 16 M5 chains: 2 chains per wave."""
    mma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[num_warps, 1, 1],
    )
    return mma, gl.DotOperandLayout(0, mma, 16), gl.DotOperandLayout(1, mma, 16)


@gluon.constexpr_function
def _batch_scale_layout(num_warps):
    """E8M0 scales ``[16 chains, 16 rows, 4 K groups]`` of :func:`_batch_layouts`.

    Waves own chains ``0 .. num_warps - 1`` and a register repeat adds
    ``num_warps`` (as the MFMA layout places its batch dimension).
    """
    warp_bits = num_warps.bit_length() - 1
    return gl.DistributedLinearLayout(
        reg_bases=[[1 << i, 0, 0] for i in range(warp_bits, 4)],
        lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]],
        warp_bases=[[1 << i, 0, 0] for i in range(warp_bits)],
        block_bases=[],
        shape=[16, 16, 4],
    )


@gluon.constexpr_function
def _chain_groups(rows):
    """Groups of 16 M5 chains. Chain ``c < 11 rows`` is rank ``c % 11`` of
    row ``c // 11`` (rank 10: the shared expert)."""
    return (rows * RANKS + 15) // 16


@gluon.constexpr_function
def _chain_slots(rows):
    """LDS rows per k-iteration of M5's intermediates (chain ``c``'s at row
    ``c``), a power of 2."""
    return triton.next_power_of_2(rows * RANKS)


@gluon.jit
def _chain_rows(c, R: gl.constexpr):
    """``(valid, row, rank)`` of chain ``c``."""
    valid = c < R * RANKS
    return valid, gl.where(valid, c // RANKS, 0), gl.where(valid, c % RANKS, 0)


@gluon.jit
def _chain_values(table, c, R: gl.constexpr):
    """``table[c // 11, c % 11]`` (a :func:`routing` tensor) of each chain
    ``c [16]``, in the layout of ``c``."""
    rl: gl.constexpr = table.type.layout
    cj = gl.convert_layout(c, gl.SliceLayout(0, rl))
    _, row, rank = _chain_rows(cj, R)
    rr = gl.arange(0, table.shape[0], gl.SliceLayout(1, rl))[:, None]
    picked = gl.gather(table, gl.zeros_like(table.to(gl.int32)) + rank[None, :], 1)
    # A thread holds every row of its rank (rows in registers).
    value = gl.sum(gl.where(rr == row[None, :], picked, 0), axis=0)
    return gl.convert_layout(value, c.type.layout)


@gluon.jit
def _down_tiles(task, group: gl.constexpr, experts, weights, R: gl.constexpr):
    """Issue the weight loads of chains ``16 group ..`` of M5 task ``task``
    (``experts`` from :func:`routing`). Chains past the last row load
    nothing."""
    bl: gl.constexpr = _batch_layouts(gl.num_warps())[2]
    rb = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    kb = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    nb = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    cb = group * 16 + rb
    eb = _chain_values(experts, cb, R)
    used_b, _, rank_b = _chain_rows(cb, R)
    routed = used_b & (rank_b < TOP_K)
    tile_base = gl.where(
        routed,
        (eb.to(gl.int64) * HIDDEN + task * 16) * (INTER_PAD // 2),
        task * 16 * (INTER // 2),
    )
    tile = (
        tile_base[:, None, None]
        + ((kb // 16) * 256 + kb % 16)[None, :, None]
        + (nb * 16)[None, None, :]
    )
    base_ptr = gl.where(
        routed[:, None, None], weights.w2 + tile, weights.shared_dn + tile
    )
    sl: gl.constexpr = _batch_scale_layout(gl.num_warps())
    rs = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(2, sl)))
    ns = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, sl)))
    gs = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, sl)))
    cs = group * 16 + rs
    es = _chain_values(experts, cs, R)
    used_s, _, rank_s = _chain_rows(cs, R)
    cell = (
        (task // 2) * (W2_CELLS * 256)
        + gs[None, None, :] * 64
        + ns[None, :, None] * 4
        + task % 2
    )
    s_base = gl.where(
        (used_s & (rank_s < TOP_K))[:, None, None],
        weights.s2
        + (es.to(gl.int64) * (HIDDEN // 32 * W2_CELLS * 256))[:, None, None]
        + cell,
        weights.shared_dn_s + cell,
    )
    used = used_b[:, None, None]
    tiles = ()
    scales = ()
    for i in gl.static_range(INTER // 128):
        tiles += (gl.load(base_ptr + i * 1024, mask=used, other=0),)
        scales += (
            gl.load(
                s_base + (i // 2) * 256 + (i % 2) * 2,
                mask=used_s[:, None, None],
                other=127,
            ),
        )
    return tiles, scales


@gluon.constexpr_function
def _stream_layout(num_warps):
    """``[rows, 4, 16]`` (row, stream, column) with a BF16 pair per lane."""
    return gl.BlockedLayout([1, 1, 2], [1, 8, 8], [1, num_warps, 1], [2, 1, 0])


@gluon.jit
def _down_inputs(mb, task, M: gl.constexpr, W: gl.constexpr):
    """Poll M5 task ``task``'s inputs for every row: each rank's
    intermediate (into LDS), the residual columns of region ``W``, and the
    MLP mixer's (prefix ``M``) inject gates and the shared expert gate.

    The polls take two round trips, which bounds a thread's polled pairs
    (and their registers).

    Returns:
        ``(x_words, xs_smem, residual, mix, shared_gate)``: LDS ``[8 S, 32]``
        int32 and ``[8 S, 4]`` uint8 (k-iteration ``i``'s intermediate of
        chain ``c`` at LDS row ``i S + c``, ``S = _chain_slots(rows)``), then
        FP32 ``[RP, 4, 16]``
        (:func:`_stream_layout`), ``[RP, 4]`` and ``[RP]``.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    S: gl.constexpr = _chain_slots(R)
    NW: gl.constexpr = gl.num_warps()
    # Residual columns (a BF16 pair per lane), and the inject half-logits
    # ([row, stream, 0..1]) with the shared gate ([row, stream, 2..3]).
    sl: gl.constexpr = _stream_layout(NW)
    r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, sl)))[:, None, None]
    b = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(2, sl)))[None, :, None]
    c = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, sl)))[None, None, :]
    column = b * HIDDEN + task * 16 + c
    gate_l: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [1, 4, 16], [1, 1, NW], [2, 1, 0]
    )
    gr = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, gate_l)))[:, None, None]
    gb = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(2, gate_l)))[None, :, None]
    gh = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, gate_l)))[None, None, :]
    gates = gl.where(
        gh < 2,
        mb.pairs(M + "inj", mb.row_offset(gr) + gh * 4 + gb),
        mb.pairs("sg", mb.row_offset(gr) + gb * 0),
    )
    # The words of k-iteration i ([S, 32]) and every k-iteration's exponents
    # ([8, S, 4], k-iterations 5 to 7 padding). Padding polls the row's
    # pair 0, so it costs one cache line per instruction.
    wl: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [NW, 1], [1, 0])
    ok, row, rank = _chain_rows(gl.arange(0, S, gl.SliceLayout(1, wl)), R)
    word = mb.row_offset(row)[:, None] + gl.where(
        ok[:, None],
        (rank * 160)[:, None] + gl.arange(0, 32, gl.SliceLayout(0, wl))[None, :],
        0,
    )
    el: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 16, 4], [NW, 1, 1], [2, 1, 0])
    ei = gl.arange(0, 8, gl.SliceLayout(1, gl.SliceLayout(2, el)))[:, None, None]
    ec = gl.arange(0, S, gl.SliceLayout(0, gl.SliceLayout(2, el)))[None, :, None]
    ee = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, el)))[None, None, :]
    e_ok, e_row, e_rank = _chain_rows(ec, R)
    e_ok = e_ok & (ei < INTER // 128)
    exponent = mb.row_offset(e_row) + gl.where(e_ok, e_rank * 20 + ei * 4 + ee, 0)
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    x_words = gl.allocate_shared_memory(gl.int32, [8 * S, 32], flat)
    xs_smem = gl.allocate_shared_memory(gl.uint8, [8 * S, 4], flat)
    # Throttle: wave 0 waits for M4's exponents (each M4 task writes its
    # block's exponent after its words) before every wave polls the inputs.
    mb.row(R - 1).pre_poll_spread("dne", UP_BLOCKS)
    got_0, got_1, got_e, got_h = mb.poll_pairs(
        (
            mb.pairs("dn8", word),
            mb.pairs("dn8", word + gl.where(ok[:, None], 32, 0)),
            mb.pairs("dne", exponent),
            mb.pairs(W, mb.row_offset(r) + column // 2),
        )
    )
    x_words.slice(0, S, 0).store(gl.where(ok[:, None], got_0, 0))
    x_words.slice(S, S, 0).store(gl.where(ok[:, None], got_1, 0))
    xs_smem.store(gl.reshape(gl.where(e_ok, got_e, 127).to(gl.uint8), [8 * S, 4]))
    got_2, got_3, got_4, got_g = mb.poll_pairs(
        (
            mb.pairs("dn8", word + gl.where(ok[:, None], 64, 0)),
            mb.pairs("dn8", word + gl.where(ok[:, None], 96, 0)),
            mb.pairs("dn8", word + gl.where(ok[:, None], 128, 0)),
            gates,
        )
    )
    x_words.slice(2 * S, S, 0).store(gl.where(ok[:, None], got_2, 0))
    x_words.slice(3 * S, S, 0).store(gl.where(ok[:, None], got_3, 0))
    x_words.slice(4 * S, S, 0).store(gl.where(ok[:, None], got_4, 0))
    residual = mx.bf16_half(got_h, column)
    halves = gl.where(gh < 2, got_g.to(gl.float32, bitcast=True), 0.0)
    logit = gl.sum(halves, axis=2).to(gl.bfloat16).to(gl.float32)
    mix = gl.convert_layout(2.0 * mx.sigmoid(logit), gl.SliceLayout(2, sl))
    gate = gl.max(gl.where(gh == 2, got_g.to(gl.float32, bitcast=True), -1e30), axis=2)
    shared_gate = gl.convert_layout(
        mx.sigmoid(gl.max(gate, axis=1)), gl.SliceLayout(1, gl.SliceLayout(1, sl))
    )
    gl.barrier()
    return x_words, xs_smem, residual, mix, shared_gate


@gluon.jit
def _down_chains(x_view, xs_smem, loaded, R: gl.constexpr):
    """Every group's rank-batched MFMA chain (A rows from
    :func:`_down_inputs`' LDS, broadcast to every MFMA row; ``loaded`` from
    :func:`_down_tiles`)."""
    mma: gl.constexpr = _batch_layouts(gl.num_warps())[0]
    al: gl.constexpr = _batch_layouts(gl.num_warps())[1]
    sl: gl.constexpr = _batch_scale_layout(gl.num_warps())
    S: gl.constexpr = _chain_slots(R)
    a_full = gl.zeros([16, 16, 128], gl.uint8, al)
    as_full = gl.zeros([16, 16, 4], gl.uint8, sl)
    accs = ()
    for g in gl.static_range(len(loaded)):
        tiles, scales = loaded[g]
        acc = gl.zeros([16, 16, 16], gl.float32, mma)
        for i in gl.static_range(INTER // 128):
            x = x_view.slice(i * S + g * 16, 16, 0).load(gl.SliceLayout(1, al))
            a, _ = gl.broadcast(gl.expand_dims(x, 1), a_full)
            xs = xs_smem.slice(i * S + g * 16, 16, 0).load(gl.SliceLayout(1, sl))
            a_s, _ = gl.broadcast(gl.expand_dims(xs, 1), as_full)
            acc = gl.amd.cdna4.mfma_scaled(
                a.to(gl.float8e4nv, bitcast=True),
                a_s,
                "e4m3",
                tiles[i],
                scales[i],
                "e2m1",
                acc,
            )
        accs += (acc,)
    return accs


@gluon.jit
def _down_epilogue(mb, task, accs, routes, inputs, out, OUT: gl.constexpr):
    """Every row of M5 task ``task`` from the chains' accumulators ``accs``
    (:func:`_down_chains`). The epilogue:

    1. Combines each row's ranks with its weights (:func:`routing`) in rank
       order. The order does not depend on the row count, so a row's result
       is the same at any row count.
    2. Adds the result to the residual.
    3. Stores the layer's output streams ``out`` (BF16 ``[rows, 4 * HIDDEN]``).

    ``OUT`` names a region that also receives the output streams as BF16
    pairs (for stages after the layer in the same launch), or is ``""``.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    NW: gl.constexpr = gl.num_warps()
    _, _, residual, mix, shared_gate = inputs
    # Each chain's result (every MFMA row agrees) through LDS.
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    G: gl.constexpr = triton.next_power_of_2(len(accs)) * 16
    chains = gl.allocate_shared_memory(gl.float32, [G, 16, 16], flat)
    for g in gl.static_range(len(accs)):
        chains.slice(g * 16, 16, 0).store(accs[g].to(gl.bfloat16).to(gl.float32))
    gl.barrier()
    sl: gl.constexpr = _stream_layout(NW)
    ol: gl.constexpr = gl.SliceLayout(1, sl)
    # MFMA row 0 of every row's 16 chains from its first, [rows x 16, 1, 16]
    # with a column's 16 chains in one thread.
    wl: gl.constexpr = gl.BlockedLayout(
        [16, 1, 1], [RP, 1, 64 // RP], [1, 1, NW], [2, 1, 0]
    )
    x = gl.arange(0, RP * 16, gl.SliceLayout(1, gl.SliceLayout(2, wl)))[:, None, None]
    n = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, wl)))[None, None, :]
    chain = gl.minimum((x // 16) * RANKS + x % 16, G - 1)
    window = gl.reshape(chains.gather(chain + 0 * n, 0), [RP, 16, 16])
    k = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, window.type.layout)))
    k = k[None, :, None]
    w = gl.convert_layout(routes[1], gl.SliceLayout(2, window.type.layout))
    routed = gl.sum(gl.where(k < TOP_K, window * w[:, :, None], 0.0), axis=1)
    shared = gl.sum(gl.where(k == TOP_K, window, 0.0), axis=1)
    routed = gl.convert_layout(routed, ol)
    shared = gl.convert_layout(shared, ol)
    routed = routed.to(gl.bfloat16).to(gl.float32)
    block = (routed + shared_gate[:, None] * shared).to(gl.bfloat16).to(gl.float32)
    streams = (residual + block[:, None, :] * mix[:, :, None]).to(gl.bfloat16)
    r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, sl)))[:, None, None]
    bb = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(2, sl)))[None, :, None]
    c = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(1, sl)))[None, None, :]
    gl.store(out + r * (4 * HIDDEN) + bb * HIDDEN + task * 16 + c, streams, mask=r < R)
    if OUT != "":
        streams = streams.to(gl.float32)
        for i in gl.static_range(4):
            stream = gl.sum(gl.where(bb == i, streams, 0.0), axis=1)
            mx.put_bf16_rows(mb, OUT, (i * HIDDEN + task * 16) // 2, stream)


@gluon.jit
def routing(mb):
    """Every row's top 10 of the router logits (ties: the lower
    expert) and their renormalized softmax weights.

    The key ``(monotone(bf16) << 9) | (511 - expert)`` orders by logit then
    by lower expert. Rounds of a max pick experts in rank order. The CTA's
    waves share the work. Wave ``w`` takes row ``w % RP`` and one slice of
    the experts and keeps its slice's top 10. The slices' picks meet in LDS.
    Then wave ``w`` takes row ``w % RP``'s top 10 of those picks and their
    softmax weights. The top 10 of the slices' top 10s is the top 10.

    Returns:
        ``(experts, weights)``: ``[rows, 16]`` by rank. Ranks past the top 10
        hold expert 0 and weight 0. :func:`_route_row` reads one row.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    NW: gl.constexpr = gl.num_warps()
    SLICES: gl.constexpr = NW // RP
    SLICE: gl.constexpr = EXPERTS // SLICES
    # One wave per [row, slice] pair: wave w = slice * RP + row.
    wl: gl.constexpr = gl.BlockedLayout([1, SLICE // 64], [1, 64], [NW, 1], [1, 0])
    w = gl.arange(0, NW, gl.SliceLayout(1, wl))[:, None]
    e = (w // RP) * SLICE + gl.arange(0, SLICE, gl.SliceLayout(0, wl))[None, :]
    word = mb.poll("lg", mb.row_offset(w % RP) + e // 2)
    bits = (word >> (16 * (e % 2))) & 0xFFFF
    # BF16 bits to an order-preserving unsigned key.
    monotone = gl.where(bits >= 0x8000, bits ^ 0xFFFF, bits | 0x8000)
    key = (monotone << 9) | (EXPERTS - 1 - e)
    picks = _top_keys(key)
    # Each row's slice picks side by side, [RP, SLICES * 16], in LDS.
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    shared = gl.allocate_shared_memory(gl.int32, [RP, SLICES * 16], flat)
    shared.store(
        gl.reshape(
            gl.permute(gl.reshape(picks, [SLICES, RP, 16]), [1, 0, 2]),
            [RP, SLICES * 16],
        )
    )
    gl.barrier()
    # Wave w merges row w % RP: rows over waves (repeated past RP).
    ml: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW, 1], [1, 0])
    best = _top_keys(shared.load(ml))
    rank = gl.arange(0, 16, gl.SliceLayout(0, best.type.layout))[None, :]
    m = best >> 9
    logits = (gl.where(m >= 0x8000, m & 0x7FFF, m ^ 0xFFFF) << 16).to(
        gl.float32, bitcast=True
    )
    first = gl.max(gl.where(rank == 0, logits, float("-inf")), axis=1)
    weights = gl.where(rank < TOP_K, gl.exp(logits - first[:, None]), 0.0)
    normed = weights / gl.sum(weights, axis=1)[:, None]
    experts = gl.where(rank < TOP_K, EXPERTS - 1 - (best & (EXPERTS - 1)), 0)
    # Every wave needs every row.
    rl: gl.constexpr = gl.BlockedLayout([RP, 1], [1, 64], [NW, 1], [1, 0])
    experts = gl.convert_layout(experts, rl)
    normed = gl.convert_layout(normed, rl)
    return experts, normed


@gluon.jit
def _top_keys(key):
    """The 16 largest of each row of ``key [rows, N]`` (N >= 16, distinct
    keys) in descending order, ``[rows, 16]`` in ``key``'s layout. Callers
    use only the first ``TOP_K`` (the rest are padding)."""
    rank = gl.arange(0, 16, gl.SliceLayout(0, key.type.layout))[None, :]
    picks = gl.zeros([key.shape[0], 16], gl.int32, key.type.layout)
    for k in gl.static_range(TOP_K):
        best = gl.max(key, axis=1)
        key = gl.where(key == best[:, None], 0, key)
        picks = gl.where(rank == k, best[:, None], picks)
    return picks


@gluon.jit
def _route_row(routes, r: gl.constexpr):
    """Row ``r`` of :func:`routing` as ``[16]`` per-rank vectors
    (:func:`_rank_layout`)."""
    experts, weights = routes
    rows = gl.arange(0, experts.shape[0], gl.SliceLayout(1, experts.type.layout))[
        :, None
    ]
    ranks: gl.constexpr = _rank_layout(gl.num_warps())
    return (
        gl.convert_layout(gl.sum(gl.where(rows == r, experts, 0), axis=0), ranks),
        gl.convert_layout(gl.sum(gl.where(rows == r, weights, 0.0), axis=0), ranks),
    )


@gluon.jit
def up_stage(mb, task, weights, X: gl.constexpr):
    """M4 task ``task``: rank ``task // 20``, intermediate columns
    ``32 (task % 20) ..`` of every row, published as MXFP8 blocks
    of ``dn8`` / ``dne``.

    Args:
        weights: The block's :class:`Weights`.
        X: Region prefix of the MLP mixer (its MXFP8 rows ``x8`` / ``xe``).
    """
    mb.stamp("m4", trace.START)
    x = _up_inputs(mb, X)
    routes = routing(mb)
    _up_finish(mb, task, routes, x, weights, 0, False)


@gluon.jit
def down_stage(
    mb, task, weights, out, M: gl.constexpr, W: gl.constexpr, OUT: gl.constexpr
):
    """M5 task ``task``: hidden columns ``16 task .. 16 task + 15`` of every
    row's block output, combined into the layer's output streams.

    Args:
        weights: The block's :class:`Weights`.
        out: The layer's output streams, BF16 ``[rows, 4 * HIDDEN]`` (global).
        M: Region prefix of the MLP mixer (its ``inj`` logits).
        W: Region of the residual streams (the output projection's ``w``).
        OUT: A region that also receives the output streams as BF16 pairs
            (for stages after the layer in the same launch), or ``""``.
    """
    mb.stamp("m5", trace.START)
    routes = routing(mb)
    head = _down_issue(task, routes, weights, schedule.rows(mb.plan))
    _down_finish(mb, task, routes, head, weights, out, M, W, OUT)


@gluon.jit
def up_down_stage(
    mb,
    up_task,
    down_task,
    weights,
    out,
    X: gl.constexpr,
    M: gl.constexpr,
    W: gl.constexpr,
    OUT: gl.constexpr,
):
    """M4 task ``up_task`` then M5 task ``down_task`` on one CTA (arguments
    as :func:`up_stage` and :func:`down_stage`).

    Both tasks' expert weights depend only on the routing. Thus the CTA issues
    the weight loads of the first M5 rows before M4's last row computes. These
    loads stream while M4 finishes, not after.
    """
    mb.stamp("m4", trace.START)
    x = _up_inputs(mb, X)
    routes = routing(mb)
    head = _up_finish(mb, up_task, routes, x, weights, down_task, True)
    mb.stamp("m5", trace.START)
    _down_finish(mb, down_task, routes, head, weights, out, M, W, OUT)


@gluon.jit
def _up_inputs(mb, X: gl.constexpr):
    """Poll M4's input, the mixed rows (MLP mixer prefix ``X``), into LDS.

    M4 polls them before the routing and its weight loads. M2 published
    them before M3 ran, so the poll returns at once. Each row's GEMV then
    waits only for its own weights (a poll waits for every earlier load).
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    x_smem, xs_smem, _ = mx.load_mxfp8_rows(mb, X + "x8", X + "xe", 20, R)
    return x_smem, xs_smem


@gluon.jit
def _up_finish(
    mb,
    task,
    routes,
    x,
    weights,
    down_task,
    DOWN: gl.constexpr,
):
    """M4 task ``task`` for every row (``routes`` from :func:`routing`, ``x``
    from :func:`_up_inputs`). With ``DOWN``, it also issues the loads of the
    first chain group of M5 task ``down_task`` before the last GEMV (see
    :func:`_down_issue`) and returns them. Else it returns ``()``.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    C: gl.constexpr = _shared_blocks(mb.plan)
    U: gl.constexpr = max(R, C)
    x_smem, xs_smem = x
    mb.stamp("m4", trace.READY)
    # Two units' weights are in flight. Units 0 and 1 load at once. Unit
    # u + 2 loads as soon as unit u computes (its registers are free then).
    # Each unit's GEMV runs every row (an MFMA has 16 rows anyway). A routed
    # task keeps row u of unit u and publishes once. A shared expert task
    # publishes every unit's block for every row.
    shared = task >= TOP_K * 20
    gl.static_assert(
        C * triton.next_power_of_2(R) <= 16, "a block per 16 // RP MFMA rows"
    )
    loaded = ()
    for u in gl.static_range(_up_ahead(U)):
        loaded += (_up_tiles(task, u, routes, weights, R, C),)
    head = ()
    if (U == 1) & DOWN:
        head = _down_issue(down_task, routes, weights, R)
    RP: gl.constexpr = xs_smem.shape[0]
    acc = mx.gemv_zero(2, 4)
    m = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, acc.type.layout)))
    units = ()
    for u in gl.static_range(U):
        tiles, scales = loaded[u]
        own = mx.gemv_chain(x_smem, xs_smem, tiles, scales)
        # Keep each unit's loads after the chain that frees their registers.
        gl.amd.hint.sched_barrier()
        if u + 2 < U:
            loaded += (_up_tiles(task, u + 2, routes, weights, R, C),)
        elif (u + 2 == U) & DOWN:
            # M5's first chains stream while M4's last unit computes.
            head = _down_issue(down_task, routes, weights, R)
        # MFMA row m keeps row m % RP of unit u: a routed task's row u, or
        # every row of a shared task's block m // RP.
        # A routed task keeps row u of unit u. A shared task keeps every
        # unit (its blocks), and publishes them after its last load.
        units += (own,)
        acc = gl.where((m % RP == u)[None, :, None], own, acc)
    if C > 1:
        if shared:
            for k in gl.static_range(C):
                _up_publish(
                    mb,
                    TOP_K * 20 + (task - TOP_K * 20) * C + k,
                    mx.gemv_result(units[k], RP),
                )
        else:
            _up_publish(mb, task, mx.gemv_result(acc, RP))
    else:
        _up_publish(mb, task, mx.gemv_result(acc, RP))
    mb.stamp("m4", trace.COMPUTED)
    mb.stamp("m4", trace.END)
    return head


@gluon.constexpr_function
def _last_row(unit, rows):
    return min(unit, rows - 1)


@gluon.constexpr_function
def _shared_blocks(plan):
    """:func:`shared_blocks` of a launch's plan."""
    return {1: 1, 2: 2}.get(plan.plan.rows, 4)


@gluon.constexpr_function
def _up_ahead(rows):
    return min(rows, 2)


@gluon.constexpr_function
def _before_poll(groups):
    """M5 chain groups whose loads issue before the input poll."""
    return max(1, groups - 1)


@gluon.jit
def _down_issue(task, routes, weights, R: gl.constexpr):
    """Issue M5 task ``task``'s weight loads of its first chain group."""
    return (_down_tiles(task, 0, routes[0], weights, R),)


@gluon.jit
def _down_finish(
    mb,
    task,
    routes,
    head,
    weights,
    out,
    M: gl.constexpr,
    W: gl.constexpr,
    OUT: gl.constexpr,
):
    """M5 task ``task`` for every row, with the first chain groups' loads
    (``head``) already issued. The task issues every other group's loads,
    then polls every row's inputs. A poll waits for the loads, which land
    while the slowest M4 task finishes. Then the groups' chains run, and one
    epilogue serves every row.
    """
    R: gl.constexpr = schedule.rows(mb.plan)
    G: gl.constexpr = _chain_groups(R)
    loaded = head
    for g in gl.static_range(len(head), _before_poll(G)):
        loaded += (_down_tiles(task, g, routes[0], weights, R),)
    inputs = _down_inputs(mb, task, M, W)
    # The last group issues after the poll, whose registers it reuses. Its
    # loads stream while the first groups' chains run.
    for g in gl.static_range(_before_poll(G), G):
        loaded += (_down_tiles(task, g, routes[0], weights, R),)
    mb.stamp("m5", trace.READY)
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    x_view = inputs[0].reinterpret(gl.uint8, [8 * _chain_slots(R), 128], flat)
    accs = _down_chains(x_view, inputs[1], loaded, R)
    mb.stamp("m5", trace.COMPUTED)
    _down_epilogue(mb, task, accs, routes, inputs, out, OUT)
    mb.stamp("m5", trace.END)
