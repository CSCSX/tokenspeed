# Iris communication

Collectives share one process-wide Iris context and symmetric heap. Each
prepared operation retains its buffers; execution never searches for a suitable
state. Allocate in the same order on participating ranks before graph capture;
prepared capacities and addresses stay fixed.
Group preparation also records the device subgroup size for launch geometry;
launchers reuse that value.

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

`IrisBackend` retains generic operations by group and dtype. K3 additionally
prepares `K3IrisState` before cache sizing and binds it to attention, MoE,
and NextN callers. Other models initialize during warmup. Calls select among
prepared variants using the existing shape and topology policies. Different
staging geometries have separate synchronization buffers because their workgroup
IDs cover different element ranges. Triton RSAG remains independent.

K3 preparation, dispatch, and fused implementations live in `kimi_k3/`.
General all-reduce implementations remain in `all_reduce.py`, separate from
K3's shapes and fusion policies.

Ordinary tensors use a Gluon one-shot that stages, publishes each workgroup's
epoch, and pulls peers in rank order. All payloads share its staging geometry.
On gfx950, BF16 staged two-shot starts at
1 MiB for TP4 and 448 KiB for TP8; smaller payloads avoid its extra rendezvous.
Producer-direct thresholds are independent. Fused all-reduce/residual/RMSNorm
uses the same two-slot staging mechanism per row, avoiding a second peer wait
before reuse. Its persistent grid strides rows with independent row epochs;
group-relative peer addresses also support noncontiguous subgroups.
The row layout aims for one packed word per lane, up to 16 subgroups per
workgroup, independently of model and row count.

Configure the allocator and optional heap capacity on every rank before the
first preparation:

```python
from tokenspeed_kernel.ops.communication.iris import configure_iris

configure_iris(heap_size=None, allocator_type="torch")
```

`heap_size=None` sizes the heap automatically. `allocator_type=None` uses Iris's
default (`torch` in the pinned version). Names pass through to Iris;
`IRIS_ALLOCATOR` takes precedence. A live context cannot be reconfigured.
