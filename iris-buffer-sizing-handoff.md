# Iris heap sizing and eager preparation: handoff

## Scope and decisions

Continue from the latest `origin/main`. The requested delivery is a two-commit stack:

1. Automatically size the shared Iris heap from all planned communication buffers, with an optional explicit byte-size override. Remove the arbitrary 256 MiB minimum and heuristic headroom.
2. Allocate those buffers before memory profiling, KV-cache sizing, model warmup, and graph capture across supported models, extending the existing Kimi K3 preparation.

Keep dispatch eligibility and numerical behavior unchanged. Use existing numerical and integration coverage, adding only a meaningful regression test for preparation/capture. Keep source comments and documentation short. Explain the memory-accounting and graph-capture advantages in the second commit message.

The eventual implementation must be reviewed with `$tokenspeed-reviewer`, then pushed to `efric/...` branches. Do not open remote PRs. Use the `gh-stack` skill for the stack. Run the exact `pre-commit run --all-files` before each commit and sign off commits.

**Current stopping point:** the user requested this Markdown handoff after confirming that the intended GLM weights are on another machine, then requested publication of the prototype branches as WIP. Implementation, final review, and benchmarking are not complete.

## WIP branches

The inspected and tested main revision was `7b08e77e9` (`refactor(amd): make Iris a first class comms backend (#2210)`). Main had advanced to `676acaa5b` when preparing the WIP push. The published prototype preserves its tested base; fetch main and rebase or port the final stack before continuing. Do not base it on the earlier context-ownership branch.

| Local reference | State |
| --- | --- |
| `iris-sizing-main-base` | Local stack trunk at `7b08e77e9`, tracking `origin/main`. This alias avoided changing `main`, which was checked out in another worktree. |
| `efric/iris-heap-sizing` | Signed-off prototype commit `4d42b1c8d`, `fix(iris): size the heap from all communication buffers`. |
| `efric/iris-eager-buffers` | Second WIP commit on top of `efric/iris-heap-sizing`, including this handoff. |

Both `efric/...` branches are published to `origin` as WIP; no PRs were opened. `git show 4d42b1c8d` and `git diff origin/efric/iris-heap-sizing..origin/efric/iris-eager-buffers` contain the two prototype layers. They are implementation references, **not review-approved patches**.

The second layer changes `docs/design/unified_path.md`, runtime `distributed/iris.py` and `execution/device.py`, `test/runtime/distributed/test_comm_ops.py`, and the kernel communication entry point, `iris.py`, and `_iris/README.md`.

## Research: ownership and users

Paths are relative to the repository root. Line references refer to the preserved prototype unless explicitly identified as inspected main.

Iris has one fixed-capacity, process-wide context in `tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/_iris/__init__.py`. A process group does not own a separate heap. Consequently, sizing for the first operation or taking the maximum individual state's requirement is insufficient: separately allocated states must be summed, while shared target/draft and K3 states must be counted once.

| Allocation family | Owners and use sites | Planning requirement |
| --- | --- | --- |
| Ordinary and producer-direct all-reduce | Runtime `distributed/comm_backend/iris.py`: `IrisBackend`, `_get_or_create`, `install`, `prepare_all_reduce_buffers`; kernel `ops/communication/iris.py`: `IrisAllReduce` | Deduplicate by ordered group and dtype; include staged data, producer data, and synchronization allocations using the existing heap-size helpers. |
| Fused all-reduce + residual + RMSNorm | Runtime `layers/layernorm.py:158` and `:389`; kernel `ops/communication/__init__.py:127`; `IRIS_AR_RMSNORM_STATES` in kernel `communication/iris.py` | Deduplicate by process-group identity, configured maximum token count, hidden width, and dtype. This is separate storage from ordinary all-reduce. |
| K3 all-reduce, AttnRes, and sharded MoE tail | Runtime `models/kimi_k3.py`, `kimi_k3_nextn.py`, `kimi_k3_comm.py`; kernel `communication/_iris/kimi_k3/` | Retain K3 capacity rules and state sharing. Include K3 extras without counting its reused ordinary all-reduce state twice. |

The repository audit followed both direct Iris references and indirect communication callers:

- `CommManager`, compiled model `CommOps`, `RowParallelLinear`, and `VocabParallelEmbedding` expose the ordinary route to many models.
- Direct reductions occur in Qwen models, DFLASH, DeepSeek V3 head-TP, DeepSeek V4.1 output/Engram paths, Nemotron latent FC2, DCP attention, and query-sharded MTP paths. Examples: `models/dflash.py:277`, `models/deepseek_v3.py:1025`, `models/deepseek_v41.py:958`, `models/nemotron_h.py:714`, and `layers/attention/dcp/comm.py:108`.
- Fused norms are reached through `CommManager` and model-specific norm calls, including Qwen and EAGLE3. Generic decoder users, GLM5, GLM5.3, LongCat, MiniMax, and Nemotron must therefore be considered even without a literal `iris` import in the model file.
- Relevant mapping groups include attention TP, head TP, KVP, dense TP, MoE TP, MoE TP-EP, linear-attention TP, vision TP, and LM-head TP. Confirm QCP group coverage against current mapping rules. Named communication backends currently route through NCCL rather than this generic Iris path.
- `AutoBackend.can_prepare_all_reduce_buffers` in `distributed/comm_backend/auto.py:423` already checks the AMD/BF16 route, group size, deterministic and batch-invariant modes, node boundaries, and an existing TRT-LLM lane. Reuse the routing contract; do not make more calls eligible just because buffers exist.

Generic ordinary all-reduce currently has a 512 KiB payload limit; the generic producer-direct budget is 1 MiB. These are dispatch/capacity policies, not the process heap size. Kernel-level producer support also includes FP16 and FP32; the generic runtime route examined here is BF16. Keep that distinction explicit.

### GLM 5.3

`models/glm53_flash.py` has two relevant paths. The mHC decoder path calls `CommManager.post_attn_comm` at line 1399 and `post_mlp_comm` at line 1427, reaching ordinary reductions when the configured communication route requires them. Its other path uses `input_reduce_norm`, `post_attn_reduce_norm`, and `post_mlp_fused` at lines 1435, 1456, and 1470. `glm53_flash_nextn.py` reuses the decoder machinery.

Verified repository path, with dispatch conditional on topology, tensor shape/dtype, and backend policy:

```text
Glm53FlashDecoderLayer.forward (python/tokenspeed/runtime/models/glm53_flash.py:1399)
    -> CommManager.post_attn_comm (python/tokenspeed/runtime/distributed/comm_manager.py:513)
        -> distributed all_reduce / AutoBackend.all_reduce
            -> IrisBackend.all_reduce, when eligible
                -> IrisBackend._get_or_create
                    -> create_iris_state
```

The existence of this lazy path is verified. A GLM startup failure caused by it, or a throughput gain from fixing it, has **not** been demonstrated. The prototype conservatively reserves some fused-norm shapes even when an mHC path may not use them; refine this if it materially wastes memory.

### K3 precedent and startup order

[PR #1457](https://github.com/lightseekorg/tokenspeed/pull/1457) introduced K3 communication preparation after weights and before cache sizing/capture. Its advantage is that KV-cache planning sees the communication reservation, and first use does not need to allocate capture-sensitive storage. Apply that rationale to the general path without changing kernel dispatch thresholds.

K3 already prepares through `ModelRunner.prepare_communication_runtime`. Preserve its target/NextN sharing, staged/producer capacities, AttnRes rules, and sharded MoE-tail eligibility. The existing K3 preparation caps its planning rows at 8192 and has specialized TP8/shape rules; derive the budget from those rules instead of duplicating simplified approximations. The user also identified NVIDIA K3 eager preparation as precedent; broader NVIDIA model coverage was not audited in this work.

Inspected main order (`python/tokenspeed/runtime/execution/device.py`, `7b08e77e9`):

```text
build_device_side
    -> create_model_runner: load target and draft (device.py:1251)
    -> prepare_multimodal_runtime (device.py:1254)
    -> compute max_forward_tokens (device.py:1257)
    -> prepare_communication_runtime, target then draft (device.py:1266)
    -> build_components / create_attn_components: cache planning (device.py:1279)
    -> graph-memory probing and graph preparation
```

Multimodal preparation can execute model warmup. The desired order is weights → plan all target/draft communication → reserve heap and allocate communication → multimodal warmup → cache planning → graph capture. Retain `ModelRunner.prepare_communication_runtime`'s existing DP-linear preparation and expert-load-row reservation (`execution/model_runner.py:270`).

## Design: shared heap budget

Use a planning pass before the first Iris allocation, followed by a preparation pass. Sum the aligned allocations of every unique planned state, then add context-owned storage once:

```text
required heap bytes = ordinary/producer state allocations
                    + K3-only allocations
                    + unique fused-norm allocations
                    + context barrier allocation
                    + allocator-specific reserved prefix
```

The 256 MiB minimum is a local heuristic, not a required Iris capacity. The inspected main also adds 16 MiB headroom; fused RMSNorm separately uses `max(256 MiB, 4 * input_bytes + 16 MiB)`. Replace these guesses with the allocation inventory. Real GPU tests below succeeded with heaps below 256 MiB.

Two non-obvious upstream allocations matter:

- The inspected Iris Torch allocator rounds each allocation to 1024 bytes. Round each allocation separately, not just the final sum. VMem uses HIP allocation granularity; query it rather than assuming Torch alignment.
- `ctx.device_barrier()` lazily allocates one WORLD-sized `int32` counter. Budget it and initialize it before capture. VMem additionally reserves an initial exported mapping prefix; the prototype budgets `max(2 MiB, aligned barrier bytes)` for that prefix. Omitting this prefix caused a real VMem test failure, fixed in the prototype. Recheck these details against the Iris version installed on the next machine.

Expose `--iris-heap-size` in bytes, default `None` meaning automatic. Validate that an explicit override covers the planned requirement. Preserve the fixed-capacity/lifetime contract: no heap growth or state replacement after captured pointers exist. Keep the optional allocator configuration and upstream `IRIS_ALLOCATOR` precedence intact.

Iris initializes over WORLD. The prototype takes a WORLD maximum of local required bytes and initializes the context on all ranks if any rank needs it. This handles differing capacity totals but does **not yet prove** that heterogeneous pipeline stages execute compatible VMem allocation collectives; that is a review item below.

For standalone kernel callers without a whole-model plan, document that multiple distinct states require configuring their combined budget before the first allocation. Removing headroom must not silently imply that an already-created heap can grow.

## Implementation split

### Commit 1: automatic sizing and override

The prototype commit `4d42b1c8d` implements these pieces:

- New `python/tokenspeed/runtime/distributed/iris.py`: `IrisBufferPlan` collects ordinary groups, fused-norm descriptors, and K3 capacities across target and draft; `heap_bytes()` totals them; `configure()` resolves the WORLD capacity and override; `plan_iris_buffers()` runs the planning pass.
- `IrisBackend.default_capacities` exposes the existing staged/producer budgets rather than creating another set of constants.
- K3 extracts `k3_iris_buffer_requirements` from preparation, exposes model/wrapper requirements for target and NextN, and uses a shared kernel `k3_communication_heap_bytes` calculation. Planning and allocation must agree exactly.
- Kernel `_iris/__init__.py` removes the floor/headroom, accounts for allocator alignment, and initializes the context barrier. Kernel `communication/iris.py` provides `iris_heap_bytes` for shared overhead and corrects per-state estimates.
- `ServerArgs.iris_heap_size` and `--iris-heap-size` supply the optional override. `build_device_side` plans after loading both models and determining the maximum forward rows.
- Existing mixed-state numerical tests configure the combined heap, including context overhead. Concise heap-contract documentation explains fixed capacity and standalone use.

This layer sets the complete heap capacity; generic state allocation remains lazy until the second layer. K3 retains its existing eager hook. Verify the first layer independently as well as the stack.

### Commit 2: eager allocation across models

The second WIP commit adds:

- `IrisBufferPlan.prepare()` (`runtime/distributed/iris.py:151`) initializes the planned context, prepares ordinary/producer buffers through the existing backend, and prepares cached fused-norm states.
- Public kernel helpers `prepare_iris_heap` (`communication/iris.py:66`) and `prepare_iris_ar_rmsnorm` (`:521`). The fused forward wrapper calls the same cache helper so preparation and execution share state identity.
- `build_device_side` executes plan preparation before existing target/draft model hooks, then multimodal warmup. K3 reuses the ordinary state and allocates its additional storage through its existing hook.
- A four-GPU integration test in `test/runtime/distributed/test_comm_ops.py` captures the first ordinary all-reduce and fused RMSNorm forward without a warmup forward. It checks numerical output, repeated replay/eager execution, stable storage, and target/draft deduplication with a heap below 256 MiB.
- Short updates to `docs/design/unified_path.md` and the Iris README describe the common startup contract.

Suggested second commit message rationale: reserve communication memory before profiling so cache capacity reflects the buffers serving will retain; prepare stable storage before graph capture and remove first-forward allocation work across models. Do not claim a measured performance improvement until benchmarks exist.

## Review items still open

The `tokenspeed-reviewer` skill and its review references were consulted, but a final review of the complete stack was **not completed**. Resolve these before calling the implementation ready:

1. **Consumer coverage and over-reservation.** The planner enumerates mapping groups and model hidden widths instead of collecting requirements from every actual consumer. Check direct fused-norm callers, flags that may bypass `enable_allreduce_fusion`, model/draft width differences, alternate dtypes, vision groups, and QCP. Ensure every possible serving allocation is planned without reserving large buffers for unreachable paths.
2. **Rank-uniform initialization and allocation.** Check pipeline stages with different layer/norm inventories and a draft only on some stages, noncontiguous subgroups, and multiple local groups. A WORLD maximum alone does not establish matching VMem peer-map collective order or symmetric offsets. Test or redesign this before claiming heterogeneous PP support.
3. **Allocator override preservation.** `IrisBufferPlan.configure()` currently calls `configure_iris(..., allocator_type=None)`. It may overwrite a programmatically selected allocator, even though the environment override still works. Preserve explicit caller configuration and make budget calculation use the allocator actually selected.
4. **K3 sharing and capacity merge.** Confirm that planned maxima and later target/NextN preparation agree, that generic states are reused, and that no model hook tries to resize a prepared state. Replace any accidental change from an explicit `None` check to `value or fallback` for `routed_expert_hidden_size` with the original semantics.
5. **Standalone callers and allocator accounting.** Audit callers that formerly relied on the 256 MiB floor to accommodate later states. Confirm all flags/counters and VMem prefix behavior for the pinned Iris dependency. Keep third-party Iris imports inside the kernel package boundary.
6. **Startup ordering.** Check that moving communication preparation ahead of multimodal warmup is valid for supported multimodal runners and retains DP-linear/load-counter preparation. Confirm NVIDIA behavior is preserved; no NVIDIA GPU validation was performed.

Review against `docs/design/unified_path.md`, the cache design contracts, current repository instructions, and the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Prefer addressing the ownership and initialization questions over adding guard-only or mocked-behavior tests.

## Validation already performed

These results apply to the preserved prototype, not arbitrary later main revisions:

| Check | Result |
| --- | --- |
| Runtime suites: `test_auto_backend.py`, `test_kimi_k3_comm_arming.py`, `test_device_handle.py`, `test_cache_setup.py`, `test_cudagraph_probe_arena.py` | 242 passed, including a rerun after the eager layer. |
| Existing Iris numerical GPU cases | Four passing cases covered Torch all-reduce at two/four ranks and both four-rank RMSNorm launch variants. |
| VMem two-rank mixed-state case | Initially failed due to the missing exported prefix; passed after the accounting fix. |
| Existing runtime all-reduce at two/four ranks | 2 passed. |
| New eager first-forward/capture integration test | 1 passed on four GPUs; 44 deselected. |
| `pre-commit run --all-files`, base layer | Passed before commit `4d42b1c8d`. |
| `pre-commit run --all-files`, eager layer | Passed before the WIP commit. Earlier formatting changes are included. |

An additional attempt including `test_k3_moe_tail_tier.py` hit a fixture/model missing `iris_states`. It appeared unrelated to the changed path, but was **not separately confirmed on baseline**; investigate rather than recording it as a proven pre-existing failure. Eight-rank K3 cases, heterogeneous PP, NVIDIA, and end-to-end GLM serving remain unvalidated.

A local `.venv` was created and used. It reuses dependencies from another local environment through a `.pth`, so it is not a portable environment recipe. Use the current repository dependency setup on the destination machine. The local scheduler was pinned to `0.1.31` after `0.1.34` produced a `State`/`block_granularity` API mismatch during the baseline attempt.

For this checkout, tests used source roots on `PYTHONPATH`:

```bash
source .venv/bin/activate
export HIP_VISIBLE_DEVICES="${GPU_IDS:?select idle physical GPUs first}"
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-amd/python"
python -m pytest -q test/runtime/distributed/test_comm_ops.py -k eager_iris_buffers
```

Run runtime and kernel test files in separate pytest invocations: combining their separate `test` package roots caused a collection import collision locally. Useful existing kernel coverage is in `tokenspeed-kernel/test/amd/ops/test_iris_communication.py` and `test_iris_moe_tail.py`.

Local, ephemeral logs: `/tmp/iris-eager-gpu-tests.log`, `/tmp/iris-eager-runtime-tests.log`, `/tmp/iris-heap-gpu-tests.log`, `/tmp/iris-vmem-tests.log`, `/tmp/iris-comm-tests.log`, `/tmp/iris-heap-hooks.log`, and `/tmp/iris-eager-hooks.log`. No task server or pytest processes remained when this handoff was written.

## Deferred GLM benchmark

The intended weights are on another machine. **No GLM throughput or latency comparison was completed.** A substitute local checkpoint failed during weight processing on unmodified inspected main, before communication preparation or cache sizing; that is not evidence of an Iris regression. Do not use the substitute as the requested benchmark.

Run baseline latest main and the completed stack on the same four idle physical AMD MI355X GPUs (`gfx950`), admitted using `amd-smi`. The user explicitly authorized any four inactive physical GPUs. Recheck availability on the destination machine; do not reuse the previous physical IDs blindly.

Use the intended `GLM-5.3-Flash` checkpoint. The originally supplied snapshot was `03eb5366286afd40d2221b1d9c63a6dd1ba4832e` under `/data/models/hf/hub/models--zai-org--GLM-5.3-Flash`; verify its location and identity there. Use a serving environment that actually imports the checkout being measured.

Preserve the supplied serving configuration:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${GPU_IDS:?provide four admitted physical GPUs}"
: "${MODEL:?provide the verified GLM-5.3-Flash snapshot directory}"
: "${TOKENSPEED_BIN:?provide the tokenspeed executable for the checkout under test}"
export HIP_VISIBLE_DEVICES="${GPU_IDS}"
unset TOKENSPEED_LOG_SPEC_ACCEPT_LENGTHS

exec "${TOKENSPEED_BIN}" serve \
  --model "${MODEL}" \
  --served-model-name glm-5.3-flash \
  --trust-remote-code \
  --attn-tp-size 4 \
  --ep-size 1 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --max-total-tokens 1048576 \
  --max-prefill-tokens 8192 \
  --chunked-prefill-size 8192 \
  --max-cudagraph-capture-size 16 \
  --cudagraph-capture-sizes 1 2 4 8 16 \
  --sampling-backend triton \
  --disable-kvstore \
  --kvstore-ratio 0 \
  --disable-prefix-caching \
  --speculative-algorithm MTP \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disable-health-check \
  --host 127.0.0.1 \
  --port 18000 \
  --control-port 18001 \
  --policy round_robin \
  --engine-startup-timeout 7200 \
  --gateway-startup-timeout 600
```

Workload: **50,000 input tokens / 500 output tokens**, at **C1, C2, C4, C8, and C16**. Keep prompts, tokenizer, request counts, warmup, sampling, and hardware consistent; record the client command and actual token counts. Record heap bytes, cache capacity/pages, graph memory, startup outcome, throughput, TTFT, and inter-token latency for both revisions. No benchmark client command was validated in this session.

There are no intended KV algorithm changes, but both earlier reservation and removal of the heap floor can change available KV capacity. Therefore, treat the requested benchmark as necessary validation of the finished stack, and report measured cache differences even if latency is unchanged.

## Pickup sequence

1. Read the destination repository's `AGENTS.md`/`AGENTS.local.md`, activate its local environment, and fetch current main. Start an isolated checkout from `origin/main`, preserving this prototype if it is available locally.
2. Reconcile the audited call sites and allocation helpers with current main. Resolve consumer coverage, allocator preservation, and rank/VMem ordering before relying on an exact heap budget.
3. Build the automatic-sizing base commit and then the eager-preparation commit. Port useful prototype code selectively; do not assume its pending review items are solved.
4. Run focused existing numerical/integration coverage and the first-forward capture regression. Run the GLM baseline/stack benchmark on the machine with the intended weights.
5. Complete `$tokenspeed-reviewer` review, fix findings, run `pre-commit run --all-files`, and sign off both commits. Push the reviewed stack to `efric/...` branches based on latest main. Do not create PRs.
