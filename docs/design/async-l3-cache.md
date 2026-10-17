# Asynchronous global L3 cache

L3 extends the existing Device → Host cache hierarchy with shared Mooncake
objects. The scheduler prepares Host data before ordinary Device admission.
All requests still use the same matcher, admission planner, H2D operations and
forward path. Selecting `--kvstore-storage-backend mooncake` selects this
asynchronous implementation; there is no second synchronous scheduling mode.

## Ownership

| Owner | Responsibilities |
| --- | --- |
| C++ scheduler and `StorageTransferManager` | Request preparation, prefix boundaries, pressure admission, soft deadlines, operation IDs, credits, Host references, publication, cancellation and quarantine |
| `CacheCoordinator` | Group matching, allocation and pinning; ordinary probes see published Device/Host data only |
| Python `L3Executor` | Host job snapshots, namespace and backend lifetime, bounded query/read/write queues, completion/pressure facts, and private backend health monitoring |
| `L3CacheHooks` | Replica agreement on time, permissions and completion facts; feedback to the existing event-loop head |
| `HostCacheStorage` | Shared pinned Host allocation and CPU-only block geometry; holds no device buffers or transfer layout |
| `DeviceHandle` / `HostCacheExecutor` | Ordered D2H/H2D and device completion fences |

Python never selects a reusable prefix or allocates scheduler blocks. Workers
never call the scheduler, GPU APIs or collectives. Host references remain in
C++ until every cache-owning rank proves retirement. L2 and L3 independently
retain one Host allocation from `cache/host_storage.py`. L3 owns its backend and accepts only Host storage
and immutable namespace facts; it has no reference to L2 or DeviceHandle.
The executor imports no device or numerical libraries; the runtime supplies
an owner translation when sharding is used. It binds the backend and Host
resource directly and resolves object keys and local byte ranges before
enqueueing immutable descriptors. Workers consume only those snapshots.
Runtime QUERY/GET/PUT use its bounded queues and explicit
completion/retirement protocol. Idle administrative namespace
purge remains a synchronous management call.

Device construction returns a complete DeviceHandle and a startup-only
`L3CacheInputs` record defined beside the L3 factory: shared Host storage, owner translation and immutable
layout/partition/producer facts, with no Device buffers or L2 executor.
The startup caller constructs L3 separately through
`create_l3_executor` in `cache/l3/factory.py`. Its lower-level
`create_storage_backend` selects only the byte-storage client.
`DeviceSpecs` contains no L3 configuration; scheduler configuration receives
the executor's storage limits explicitly. Without a storage backend the factory returns
before checkpoint I/O, client registration, workers or collectives. Checkpoint
agreement uses the caller's explicit Gloo world group, never a default device
process group.
If construction fails after PD starts its background threads, the worker closes
its ready pipe. Readiness reports failure immediately without joining or killing
the worker; the existing launcher owns subprocess teardown and its budget.
This does not set a deadline on healthy model loading or checkpoint agreement.
The loop captures `plan.cache` once, submits its Host-storage subset to L3 and
passes its Device-copy subset explicitly to DeviceHandle from the same plan,
including frozen-round cleanup. L3 hooks poll the L3 executor directly
and return agreed facts to the existing head feedback point. RequestHandler
coordinates namespace/version changes through that same L3 capability.
The factory collects static checkpoint, layout and producer facts once into an
immutable deployment identity. Producer options are structured fields; target
compute dtype is already in the packed-cache signature. The weight-version
callback only combines that identity with explicit target/draft generations.
It retains no mutable launch configuration. This consolidation changes namespace
hashes, so earlier objects remain cold; the compatibility epoch stays unchanged.
The factory validates page-sharded payloads and constructs the backend,
asynchronous executor with the shared Host resource. It retains no L2 executor or device
capability. The device builder only extracts resolved facts; it neither selects
an L3 backend nor starts storage work. L2 copies remain inside device construction
because they bind Device buffers, streams and slot-state exporters.

No additional HiCacheEngine is needed: C++ owns policy, the existing loop owns
round sequencing, and the shared Host resource owns its allocation.

Device-copy and L3 teardown are attempted independently. A failed Device
shutdown must still retire L3 workers before its backend can be closed;
unproven native retirement retains the registered Host allocation.

`StorageConfig` is the read-only scheduling contract: credits, prefetch limits,
deadlines and pressure watermarks. `StorageRuntimeConfig` contains that contract
plus Python worker counts and backend-health settings; it does not duplicate
the scheduling fields. The executor and replica hooks read the same parsed
limits that are passed to the scheduler, which takes its normal value snapshot.
`MooncakeStoreConfig` stays inside the backend and contains only connection
and storage-segment settings. L3 names the cache tier; Storage names its
backend-neutral operations and scheduling contract. Neither selects a second
execution mode. The existing JSON `async` object supplies runtime tuning values.

A request waiting for QUERY/GET is in the main FSM's `Prefetching` state,
not in `Submitted` with a second readiness flag. It owns no Device pages or
request slot and never enters forward candidates, prefill token reservation
or capacity-retraction demand. A one-shot preparation bit prevents retries
after a miss, pressure bypass or soft timeout; it does not gate admission.

`Prefetching` carries a typed continuation, either `Bootstrapping` or
`Submitted`. The PD bootstrap ACK updates that continuation without ending
the L3 wait. L3 completion or a soft deadline resumes the continuation,
so both dependencies still overlap. A completed read's Ready hold moves with
the pre-admission state, expires independently and remains charged to its
storage ticket until ordinary admission acquires the Host sources. Native
I/O and its memory references may outlive cancellation or a soft deadline;
that lifetime stays in `StorageTransferManager`, independently of the FSM.
`plan.cache` contains both Device copies and four Host-only command types:
`StorageQueryOp`, `StorageGetOp`, `StoragePutOp` and `StorageCancelOp`.
Query carries keys, Get carries Host destinations, Put carries Host sources,
and Cancel names the existing operation without creating another ticket.
Their read-only `kind` is fixed by type. They share one L3 executor and
completion protocol; storage completion never substitutes for an H2D fence.

The control plane reads and validates the plan once. Its immutable
`CacheOperations` snapshot preserves order within the Device and storage
subsets. The cache hooks count only Device tickets, DeviceHandle receives
that same Device tuple explicitly, and L3Executor receives the storage tuple.
A missing executor or an unsupported operation fails explicitly. Frozen and
DP-idle rounds still dispatch cancellation and Device cleanup. This grouping
adds no ordering between the two executors: tickets express the dependencies.

## Read transaction

1. The scheduler probes local Device and Host caches. A local hit covering
   the maximum reusable boundary, or an extension below the configured
   minimum, bypasses L3. Otherwise it considers a bounded extension of that
   prefix, respecting replay/logprob limits and group geometry. It expands
   full prefix boundaries and matching tails from the local `TailTable`, plus
   the exact reusable request bound. Query credits first cover the longest
   boundary whose complete missing dependencies fit: Full history, the SWA
   window and each State checkpoint at that boundary. Shorter fallback
   boundaries contribute their distinct missing keys only when the whole
   additional set fits. Historical State checkpoints cannot crowd out the
   prompt-end checkpoint needed for the longest restore.
2. the storage subset of `plan.cache` carries a query with a monotonically increasing operation
   ID and immutable keys. Querying reserves no Device pages or batch slots.
3. After all owners report existence, C++ runs the existing group matcher
   with these temporary hints. It pins all warm dependencies before allocating
   Host destinations. Under shortage it shortens the boundary and re-runs the
   matcher, including sliding-window and snapshot-state requirements.
4. A GET job receives fixed backend keys, Host offsets and lengths. No worker
   reads a mutable namespace or a later scheduler plan.
5. `StorageDone` carries per-key success, confirmed object-failure facts,
   and an operation-wide retirement fact. Only all-owner, full-length, retired successes publish Host entries.
   Partial success may populate a shorter legal prefix; ordinary matching
   determines what is actually reusable.
6. A short Ready hold protects the result until ordinary admission transfers
   ownership to the existing H2D ticket. Ready expiry releases the hold. H2D
   and the forward use the same execution path as every Host hit.

Tail candidates carry their exact token count and hash through reservation.
A retired, attached GET records a local tail hint only after publishing Host
data under it. Hints alone never establish a hit: the ordinary group matcher
still requires Full history, the SWA lookback and every State checkpoint at
the same boundary. A partial history block follows the usual tail copy-on-write
path. Under Host shortage the matcher retries the next shorter candidate,
including an aligned boundary immediately below a tail.

Tail discovery is local and bounded. A cold instance can probe the reusable
request bound, but cannot discover an arbitrary older nonaligned checkpoint
in a longer prompt without a local hint. There is no shared tail directory
or scan of every token position. These misses use ordinary computation.

Query planning uses the same matcher masks and key traversal as reservation,
with temporary existence assumptions and no cache-block allocation. It may
skip history below the aligned local Device/Host coverage, but considers only
boundaries beyond the complete local hit. Reservation always matches again
from the Device floor and pins the real Host dependencies before allocation;
eviction between QUERY and GET can shorten the restore, never establish a hit.

`max_host_blocks` counts logical references across all groups and outstanding
storage operations, including recurrent-state checkpoints. It is neither a
byte budget nor a quota per worker. Page sharding changes the transferred bytes,
not this logical count. A read with no legal prefix within the remaining budget
uses ordinary local admission; deployments must budget for the prefix lengths
and concurrency they intend to restore together.

Retraction restores exact images through `HostCacheExecutor`; it never starts
a fresh L3 preparation. State groups use the current fixed two-slot checkpoint
contract. Runtime compatibility epoch 2 isolates this implementation from
earlier producer and storage contracts. A different runtime epoch starts a
separate namespace even with the same weights; objects are not converted.
The literal `tsl3-` prefix identifies the cache without a second version number;
the namespace hash includes `runtime_compat` as its single compatibility epoch.
Scheduling, queue and query-budget changes do not advance that epoch.

The request has one soft deadline shared by query and GET. A soft timeout
ends its wait and permits local admission; it does not retract the request,
raise admission headroom, mark the backend unhealthy or release native I/O
buffers. Detached completions cannot publish data. Another request's local
publication may also satisfy the wait before the remote operation completes.

Client cancellation enters at the existing loop head. A fused request still
in Submitted or Prefetching can abort immediately when C++ confirms it has no
forward resources. PD requests detach L3 but retain their existing transport
completion fence and output lifecycle. This removes the extra L3 delay; it
does not make PD cancellation immediate or bound it to one forward chunk.

## Write transaction

A D2H completion publishes Host L2 and releases its Device source immediately
after replica agreement. It never waits for Mooncake. C++ queues bounded,
nonowning backup candidates. At PUT dispatch it rechecks Host residency and
acquires source references; evicted candidates are skipped. Queued candidates
therefore cannot pin Host memory indefinitely.

PUT is create-only. A retired failure leaves the local Host entry usable.
An in-flight PUT retains its source references independently of the request,
and source ownership ends only after all owners report retirement. Local Host
publication does not imply that a shared object exists. Different compatible
instances use the same namespace and per-TP-shard key format; DP replica IDs
are deliberately absent from object keys. The Mooncake master owns metadata,
not the cached bytes. Independently mounted data segments can outlive inference
workers: setting `global_segment_size=0` explicitly makes an inference worker
a client without mounting storage of its own.

## Controller queues, monitor and bounds

Host allocation uses the ordinary per-node rank share and actual rounded
pinned sizes for L2, retraction snapshots and slot-state arenas. Ratio-based
L2 sizing is capped to a common count across ranks; an explicit size that
does not fit is rejected. L2 and L3 share this allocation; attaching L3 does
not allocate another arena. This is not an atomic reservation across containers
or independent P/D worlds. Deployment sizing must account for their combined
allocations, Mooncake segments/staging and independent storage owners.
Runtime operation/Host credits bound L3 ownership, not process RSS.

Query, GET and PUT have separate bounded queues and workers. The total
outstanding-operation limit includes queued jobs, native work and undelivered
results. C++ separately limits key count, Host references, lookahead tokens,
negative metadata and backup candidates. Each held CacheBlock is charged
conservatively as one potentially pinned LCM parent, including warm
requirements and quarantine. This can underuse packed Host capacity but cannot
oversubscribe it.

Backups may occupy at most half of the operation credits and half of the Host
credits, rounded down. Queries and reads may use the entire shared budget.
These are admission limits in the C++ cache owner, not separate allocators or
worker-side decisions. They keep slow PUTs from exhausting the resources a
new prefetch needs; they do not preempt native I/O or guarantee a large read
fits. A one-credit budget admits reads but no backups. Backup candidates stay
bounded and nonowning until admission, so write pressure cannot pin queued
data indefinitely.

Temporary credit exhaustion leaves candidates queued until credits return;
it does not discard a still-resident prefix. Dispatch rechecks Host residency.
A permanently zero write budget rejects candidates immediately. The bounded
queue drops its oldest candidate on overflow; pressure closure and frozen-round
cleanup still discard queued work. Scheduler counters distinguish pressure,
overflow, unavailable Host data, disabled writes and cleanup, exported through
`tokenspeed:l3_dropped_backup_keys` with a fixed `reason` label. These count
candidate occurrences, not distinct objects. Coalescing a candidate already
covered by a pending PUT is not a loss. The existing pressure-only counter
retains its meaning. None of these observations change retirement or read reserves.

The executor's completion poll also returns a small pressure snapshot under the
same lock: the number and payload bytes of queued QUERY/GET/PUT jobs, and the
age of the oldest queued or running job. Query payload bytes are zero. Retired
work is excluded; detached native work remains included until it retires.
Sampling scans at most the configured operation limit and never calls storage
or the GPU. Metrics snapshots remain a separate, lower-frequency observation. L3 exports
use the shared metric adapter and the process-local metrics service for the
selected outputs. Local I/O remains labeled by TP rank; exporters do not make
storage admission or retirement decisions.

Hooks MAX-reduce these durations and counts across cache owners through the
existing fixed CPU header. C++ closes a shared read/write admission gate when
any high watermark is reached and reopens it only when all observations reach
their low watermarks. Local wall-clock timestamps never cross this boundary.
The gate is independent of pause/freeze, backend health and quarantine. It does
not pause forward execution, completion handling or soft deadlines.

Closed admission bypasses a new request's L3 preparation. A request still in
QUERY detaches and uses ordinary local admission without waiting for its soft
deadline; no new GET may acquire destinations. GETs already submitted keep
their original lifetime and deadline, and Ready results remain usable. Host
credit exhaustion also rejects a new QUERY before backend work. Closing the
gate drops nonowning backup candidates and rejects new ones before acquiring
Host references. Already submitted PUTs continue; L2 publication is unaffected.
Recovery does not replay dropped candidates. Native retirement is never inferred
from pressure recovery, and an empty runtime queue cannot clear quarantine.

The L2 executor binds immutable transfer geometry and mapped-buffer address
tables for every write lane and load workspace during construction. Their
initial uploads must not occur on the first serving H2D/D2H submission.
Per-operation block tables still use pinned, asynchronous uploads and their
existing lifetime fences.

Queued cancellation is acknowledged without waiting for an occupied worker.
A cancelled queue item becomes inert. Started I/O has a separate lifetime:
request cancellation is not native retirement. Operation IDs never reuse a
request slot or an earlier operation's ID.

The executor's private health monitor rejects work during cooldown and permits
one recovery probe. It has no separate thread or public control interface.
Health epochs prevent a stale successful call from undoing a newer outage.
Existence misses and safely retired GET misses do not indicate an outage;
exceptions, PUT failures and unknown native failures do. A soft timeout is
only a scheduling decision. The first version uses fixed deadlines and
credits and explicit pressure watermarks; it does not estimate per-request
transfer benefit or auto-tune limits. Pressure does not report a backend fault,
and a successful backend call does not override C++ pressure admission.

PD bootstrap and L3 preparation may run together while the request owns no
Device pages. Bootstrap completion does not discard a pending or Ready storage
transaction; ordinary admission still requires both dependencies. The D role
continues to obtain its missing prefix from the prefill peer.

## Completion agreement and ordering

Every attention-TP owner sends its newly completed records once over the CPU
group. Hooks retain partial-owner records until the other owners finish.
Each round first exchanges a fixed CPU header containing the clock, permissions
presence of completion/error facts, and the three pressure observations. Empty
and pressure-only rounds need no object serialization;
nonempty rounds exchange the bounded facts after every owner sees that header.
The agreed tick is still delivered every round, including idle and frozen rounds.
Results are agreed per operation, so a slow earlier operation cannot block a
later complete one. Per-key validity and retirement are AND-reduced;
confirmed missing/invalid object facts are OR-reduced. A single missing shard
invalidates the shared object, but a peer that did not read contributes no
object-failure evidence. The
window is bounded by operation and key credits; it is not truncated to a
prefix of pending operations. A local polling error joins the collective
before every owner raises the protocol error.
Local completion records are validated and captured inside that same error
boundary, before header exchange. Success, object-failure and retirement
fields must be actual booleans; per-key masks have equal lengths and success
cannot also report an object failure. Query/PUT cannot report read failures.
A truthy value is not proof of native retirement. Malformed backend
results follow the backend's explicit exception-retirement contract.

Page-cyclic groups use the existing Host owner translation. Each worker copies
only its locally owned rows; other rows contribute success and retirement but
no object failure or transferred bytes. Their keys name the payload replica
(`attention_tp_rank // shard_count`), so a new Host allocation may restore the
same object on another cyclic owner. Replicated state groups retain rank-local
keys. Startup rejects head-partitioned fields in a page-sharded group: changing
page owner must not change what bytes an object means. The producer namespace
also includes KVP, QCP and linear-attention TP geometry.

The TP leader supplies the common monotonic clock; permissions are intersected.
The head advance applies that tick and cancellation before completion records,
so an already expired/detached read cannot publish a late result. The scheduler
emits storage commands beside cache and forward operations. Storage dispatch
bypasses the GPU FIFO; ordinary D2H → zeroing → H2D → forward ordering is
unchanged. Frozen rounds still service the same planner with forward/storage
permissions disabled, allowing completions, expiration and cancellation.

## Retirement and maintenance

For the Mooncake adapter, exact byte counts prove GET success. Nonnegative
short reads are retired failures. `OBJECT_NOT_FOUND` (`-704`) is a metadata
miss before destination transfer submission. PUT `OBJECT_ALREADY_EXISTS`
(`-705`) is a create-only rejection before source transfer and is idempotent
success. Other negative transfer errors, exceptions after native submission,
and malformed transfer results do not prove retirement.

`failures_retire` declares the backend's guarantee when a call raises or returns
a malformed result: whether it has stopped accessing the supplied Host buffers.
It is not an operation's success flag. Valid results carry their own `retired`
fact, including safely retired misses. The in-process store guarantees retirement
on failure; Mooncake does not infer it from an exceptional SDK return. QUERY
accesses no supplied Host payload, so its failure does not retain payload buffers.

Only confirmed missing or invalid GET objects enter C++'s bounded negative
metadata. Health/queue/shutdown rejection, queued cancellation and backend
exceptions carry no object evidence. They still complete the request's attempt
and preserve the retirement contract, but the same keys can be queried and read
by a new request after recovery. Genuine negative metadata retains its existing
bounded FIFO lifetime; expiry/reprobe after external object repair is separate
work. Python reports facts and does not choose negative-cache policy.

An unretired result quarantines every reference of that job. It is never
published, freed or repaired by an existence recheck. The first version
stops new L3 work for the replica after quarantine and requires process
restart to reclaim that storage safely. Ordinary computation and valid L2
hits can continue within the remaining capacity. A deployment SDK must honor
the adapter's return-code contract; real transport fault tests remain part of
hardware validation.

A full cache clear, namespace rotation or weight update must reject outstanding
storage operations, including Ready holds and quarantine; it must not
synchronously drain QUERY/GET/PUT on the scheduling thread. After replica-agreed
preflight, remote namespace deletion retains the existing synchronous
administrative call on the control thread. This purge can delay request and
control processing until the backend returns; preparation soft timeouts do not
bound or cancel it. A deletion failure prevents local cache clearing.
Asynchronous administrative purge is outside the first-version scope.
L1-only clear may proceed while Host-only storage I/O remains active.
Shutdown attempts device-copy cleanup and L3 retirement independently from the
control plane. L3 joins its workers with a deadline and then closes its backend;
if native retirement remains unknown it refuses backend close/Host deregistration
and retains the backend and Host allocation even if the runtime owner is collected.

## Initial scope and validation

### Observability

With Prometheus metrics enabled, L3 exports rank-local query/GET/PUT counters,
queue/backend cumulative durations, queued/running/completed job counts and
their byte ranges. Completed jobs remain charged until their facts are polled.
Separate scheduler gauges retain visibility of Ready holds and quarantined
references after runtime completion. Waiting soft timeouts and Ready expiration
are different counters; client detachment increments neither.
Pressure bypasses and dropped backup candidate keys have separate counters;
the admission gate is a mirrored scheduler gauge. These do not count unique
remote objects or saved prefill tokens.

Health state/epoch and unproven native retirement are separate gauges. Reading
health never consumes a recovery probe. Snapshots carry no request IDs or object
keys. Labels identify model, engine role, DP rank and TP rank: I/O is shard-local,
while scheduler resource/deadline values are mirrored and must not be summed
across TP owners. Successful adapter byte counters include idempotent PUT hits
and detached GET results; they measure neither network traffic nor published
cache hits. Metrics do not participate in scheduler decisions or agreement.
The `backend_calls` event counts adapter invocations; a create-only PUT can
finish after its existence check without calling the native PUT API.

### Supported scope

The first version supports PP=1 and the existing Host-KVStore model/cache-group
capabilities. DCP remains excluded by Host KVStore. Fused and PD-prefill roles
can prepare L3 hits. Ordinary PD-decode remote landing keeps its Device-only
prefix contract; local recovery uses the existing matcher. There is no new PD
handshake, direct L3-to-Device DMA, shared read deduplication or layer-wise GET
publication in this version.

Scheduler tests exercise mixed block granularities, sliding windows, sparse
state, partial results, deadline expiry, cancellation, ID reuse and ownership.
Executor tests cover bounded queues, health epochs, queued cancellation,
shared-store byte round trips and quarantine. Two-process Gloo tests cover
out-of-order owner agreement and collective error handling. GPU H2D/D2H and
real Mooncake/RDMA validation are distinct from these CPU protocol tests.

The Mooncake CI workflow runs CPU KVP owner relocation tests and the scheduler
tail roundtrip through both memory and native storage. Its native job starts a
local master and independent DRAM owner with `MOONCAKE_REQUIRE_MASTER=1`, then
checks exact Host bytes after the writer closes. Native RDMA and device-byte
checks still require the separate GPU validation environment.

`test/runtime/test_l3_flat_kv_e2e.py` checks packed history/state groups through
D2H, asynchronous PUT/GET and H2D, including untouched GPU bytes. Its native
case requires `MOONCAKE_GPU_TEST_CONFIG`, a JSON backend configuration with
`global_segment_size=0`, and an already running master plus an independent
storage owner. It exercises both `direct` and `kernel` I/O settings. Each writer
process exits before a new reader allocates Host/GPU buffers; the reader restores
the objects into different block slots and checks exact bytes, then deletes
them and verifies safely retired misses. Without that environment variable,
the native case skips and only the memory-backend case runs. This transfer test
complements serving tests that verify scheduler admission and model output.

Validation counters must distinguish successful adapter PUT results from new
native writes: create-only PUT can succeed because an object already exists.
Native PUT success can also mean that the SDK skipped an existing object;
successful submitted payload is not proof of new writes or network traffic.
Use SDK transport counters when reporting wire bytes. Model
reuse evidence should include fresh inference processes, successful GETs on
every owning rank, and answers requiring information in the restored prefix.

The primary latency measure for L3 prefix reuse is end-to-end time to first
token (TTFT), compared with recomputation when the tested prefix is absent
from both Device and Host caches. Report the prefix actually used by admission,
remaining prefill work and fallback frequency alongside TTFT. Separate remote
DRAM hits, SSD hits with warm or cold file caches, and misses. Concurrent decode
time per output token (TPOT) is an interference check; reducing idle L3 hook
overhead does not establish a prefill benefit. A storage call overlapping GPU
work also does not establish how much latency was hidden on the requesting
request's critical path. PD measurements must include transfer and decode-side
admission through the first delivered token.

Benchmark observers belong in isolated test deployments. Idle-only local
resets distinguish Host hits from remote restores without purging the shared
namespace. Cleanup removes only keys written by that test and waits for
store-side offload to settle. These calls stay outside timed cohorts. Formal
comparisons use equal Device capacity, warm execution paths, the default L3
wait budget, and cache-off/L2-only/L3 runs; partial hits, misses and fallback
remain in the results. Profiling and artificial I/O delays are separate runs.
Check the engine's reported single-request limit, including both PD roles,
against rendered input and generation tokens. Nominal pool token capacity
does not imply that a hybrid request of that length fits every cache group;
context-budget truncation invalidates an answer-comparison workload.
SSE chunk intervals are not token ITL, and small-cohort empirical p99 values
do not establish a production tail guarantee.

See [server configuration](../configuration/server.md#host-l2-and-mooncake-store-l3)
for limits and backend settings, [cache concepts](cache-concepts.md) for
geometry and key compatibility, and [event loop](event-loop.md) for the two
centralized feedback points.

## Disabled storage and scope

Without a storage backend, startup creates no L3 executor or SDK client, hooks
return without I/O or owner agreement, and the scheduler skips storage
preparation and reference-handoff scans. Ordinary Device/Host matching,
admission and execution remain the same path. The runtime requires the
scheduler built from matching source even when storage is disabled.

This change retains the shared Host-transfer startup bindings and ordinary
Host-copy lifetime fences. L3 preparation does not change Decode destination
publication or receiver lifetime: restored Host data enters the existing H2D,
forward and PD transfer path. Model metadata, sampling communication, unrelated
MoE kernels and toolchain tuning are not dependencies of the storage protocol.
Storage does not add model-specific forward or attention implementations.
