# Iris communication

Collectives share one process-wide Iris context and symmetric heap. Each
prepared operation retains its buffers; execution never searches for a suitable
state. Allocate in the same order on participating ranks before graph capture;
prepared capacities and addresses stay fixed.

The first preparation sizes the shared heap. Later states allocate from that
same fixed heap; Iris's allocator enforces capacity.

- **State:** the persistent buffers and references needed by an operation.
- **Epoch:** a GPU counter that distinguishes successive uses of synchronization
  flags, including graph replays. Epochs are 32-bit and wrap during long
  serving runs: compare them modulo 2**32 (`(seen - epoch) < 0` as int32), and
  derive slot indices from the unsigned value with a power-of-two slot count.
- **Scratch:** reusable buffers holding intermediate computation results.
- **Borrowed tensor:** a view, without a copy, into storage owned by the state.
  Another operation can overwrite its contents even while the view is alive.

Each implementation owns the synchronization and scratch it needs. Sharing is
explicit: K3 fusions reuse producer storage and pull scratch, with producers
finished and consumers ordered on one stream. Sharded activations are borrowed;
ordinary producer-direct reductions return results in independent storage.

`IrisBackend` retains generic operations by group and dtype. Models prepare
additional fusion state before cache sizing and bind it to their callers.
Generic operations can also initialize during warmup. Calls select among
prepared variants using the existing shape and topology policies. Different
staging geometries have separate synchronization buffers because their workgroup
IDs cover different element ranges. Triton RSAG remains independent.

Model preparation, dispatch, and fusions live in `kimi_k3/` and
`muse_spark/`. General collectives live in `all_reduce.py`,
`reduce_scatter.py`, `all_gather.py`, and `broadcast.py`. The all-gather module
also owns the width-parameterized RMSNorm/MXFP4 fusion. When Lamport mailboxes are
prepared, that fusion pushes each shard into them and polls for the data itself,
sharing the BF16 reductions' per-tile stage counters; otherwise it pulls producer
storage between two peer rendezvous. Exchanges with
model-specific head layouts, route packets, or normalization semantics stay in
the model package, along with dispatch and capacities.

Lamport reductions use data as the ready signal: peers push BF16 packs into
mailboxes initialized to an empty marker (negative zero in odd elements, so
inputs and reduced values replace negative zero with zero), the receiver polls
until no marker remains, then restores the markers and advances a stage
counter over three stages. The one-shot pushes the whole payload to every
peer; its mailbox and one stage counter per 512-element tile are shared with
the gather-norm-MXFP4 fusion and Muse's routed reduction, which keep that
element-to-tile mapping. For larger TP8 payloads the two-shot reduce-scatters
through one mailbox and all-gathers through another with its own per-program
counters, sending each peer 2/8 of the payload. Its reduced values also replace
negative zero with zero. These all-reduces, including Kimi K3's one-shot,
rotate their push order by program so concurrent stores use all links; the
gather and routed fusions keep a fixed order.

Muse Spark prepares its collectives for TP4/EP4, TP8/EP8, or expert TP on gfx950.
Pure expert TP and EP share the owned-row and QKVG token thresholds; full-row
reductions retain their payload-based protocol policies. Owned rows use
ceil(global rows / world size), with
padded rows masked; QKVG exchange derives each rank's head widths from the group
size. Expert TP reduce-scatters complete groups of eight raw routes to token
owners before normalization and weighted combination. Its separate prepared
route state owns symmetric input staging, local output scratch, and completion
flags; it does not enlarge the attention producer or gather buffers. All peer
reads finish before input reuse, and the local norm consumes borrowed output
before the next route reduction. Pool/unpool, the residual boundary, and the
next attention input remain on owned rows; QKVG exchange restores head sharding.

The route state lends its canonical symmetric input through `acquire_input`.
Eligible TP/EP1 A4W4 experts write directly into it; `run` skips staging only
for the exact contiguous destination with no route map. Other sources keep
copy/gather staging and must not overlap its destination. These ordinary
producer stores retain the reduction's release publication and exit barrier;
being in symmetric memory does not establish a write-through store contract.
Producer, reduction, and local normalization use the same stream.

Full-row expert TP reduces raw route partials before normalization. Oversized
ordinary payloads use consecutive capacity-bounded Iris reductions, preserving
the prepared addresses and the existing fallback for excluded policies. Its model-private
`reduce_borrowed` reads ordinary canonical rows and returns a borrowed gather
or push output; the per-route norm consumes it on the same stream before any
projection or collective reuses the storage. Generic all-reduce outputs keep
their independent-storage contract.

Configure the allocator and optional heap capacity on every rank before the
first preparation:

```python
from tokenspeed_kernel.ops.communication.iris import configure_iris

configure_iris(heap_size=None, allocator_type="torch")
```

`heap_size=None` sizes the heap automatically. `allocator_type=None` uses Iris's
default (`torch` in the pinned version). Names pass through to Iris;
`IRIS_ALLOCATOR` takes precedence. A live context cannot be reconfigured.
