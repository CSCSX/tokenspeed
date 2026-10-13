# Iris communication

Collectives share one process-wide Iris context and symmetric heap. Each
prepared operation retains its buffers; execution never searches for a suitable
state. Allocate in the same order on participating ranks before graph capture;
prepared capacities and addresses stay fixed.

Serving budgets the shared heap from all target and draft buffer requirements,
including allocation alignment. There is no minimum heap size or fixed headroom.
`--iris-heap-size BYTES` overrides the automatic total; an undersized override
fails before allocation. Standalone callers sharing several operations must pass
their combined capacity to `configure_iris` before preparing the first state.

- **State:** the persistent buffers and references needed by an operation.
- **Epoch:** a GPU counter that distinguishes successive uses of synchronization
  flags, including graph replays.
- **Scratch:** reusable buffers holding intermediate computation results.
- **Borrowed tensor:** a view, without a copy, into storage owned by the state.
  Another operation can overwrite its contents even while the view is alive.

Each implementation owns the synchronization and scratch it needs. Sharing is
explicit: K3 fusions reuse producer storage and pull scratch, with producers
finished and consumers ordered on one stream. Sharded activations are borrowed;
ordinary producer-direct reductions return results in independent storage.

Serving prepares generic reductions and fused RMSNorm before model warmup,
cache sizing and graph capture. `IrisBackend` retains generic operations by
group and dtype. K3 additionally binds `K3IrisState` to attention, MoE, and NextN
callers. Calls select among
prepared variants using the existing shape and topology policies. Different
staging geometries have separate synchronization buffers because their workgroup
IDs cover different element ranges. Triton RSAG remains independent.

K3 preparation, dispatch, and fused implementations live in `kimi_k3/`.
General all-reduce implementations remain in `all_reduce.py`, separate from
K3's shapes and fusion policies.

Configure the allocator and optional heap capacity on every rank before the
first preparation:

```python
from tokenspeed_kernel.ops.communication.iris import configure_iris

configure_iris(heap_size=None, allocator_type="torch")
```

`heap_size=None` sizes a standalone heap for its first operation. `allocator_type=None` uses Iris's
default (`torch` in the pinned version). Names pass through to Iris;
`IRIS_ALLOCATOR` takes precedence. A live context cannot be reconfigured.
