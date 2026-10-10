# RecoverSSM PR1 progress

Status: implementation in progress. The T4/two-accepted-token attention matrix
improves, but B32 with one accepted token still regresses by 3.5%. The full
performance gate has not passed.

## Branch and commits

- Branch: `kda-recoverssm-pr1`.
- Unchanged main baseline: `f4ac1affe11ad404720bcd150970487f75fbf59a`.
- Design commit: `7a47827e6ce2112aa32f252fbc2e80514bfd411e`.
- Implementation results below refer to uncommitted work after that commit.
- Claude approved the revised design, including the two-precision producer
  contract. This is not code approval or a performance exception.
- Keep all design versions on this branch until validation and review finish.

## Environment

Attention-only checks use one NVIDIA GB300 from a persistent one-node Slurm
allocation. No model weights are needed for these checks.

| Component | Version |
| --- | --- |
| Driver | 580.167.08 |
| PyTorch | 2.14.0+cu130 |
| CUDA | 13.0 |
| tokenspeed-triton | 3.8.10.post20260920 |

Baseline and candidate use the same allocation, container and dependencies.
Main is a separate unchanged worktree. Local launchers, allocation details
and raw results are in ignored `outputs/pr1/`.

## Implementation milestones

| Milestone | Status |
| --- | --- |
| New branch, design retained, rebase to main | Done |
| Claude design review | Approved |
| Licensed vLLM kernel copy and current-round runtime adapter | Implemented; uncommitted |
| Strict accepted-state compatibility | Initial tests pass |
| Combined convolution and gate producers | Initial tests pass |
| KDA performance non-regression | T4/A2 improves; B32/A1 still regresses |
| Real-model correctness, AIME and E2E performance | Not run |

The adapter keeps the existing accepted-state planner and convolution commit.
FP32 K/U/D replaces the NVIDIA per-round replay payload. Workspace accounting
includes these larger records. No scheduler or persistent-cache geometry
change is required for PR1.

## Accuracy

The first port saved corrections from BF16 verify inputs. It failed an existing
state regression test. Main replay uses FP32 convolution and gate results.
Casting rounded verify inputs to FP32 does not restore those results.

The current producer keeps two state chains. One computes verify outputs.
The independent FP32 chain creates recovery records. Convolution retains
each path's tap order. The original state limit remains `atol=1e-5, rtol=1e-3`.

Run the suites separately because their conftest roots differ:

```bash
python -m pytest -q test/runtime/test_kimi_k3_kda_eager_commit.py
python -m pytest -q -s tokenspeed-kernel/test/nvidia/ops/attention/test_kda_vllm_recoverssm.py
```

After producer fusion:

- Existing runtime suite: 19 passed.
- New kernel suite: 11 passed.
- Existing replay-kernel suite: 33 passed, 1 platform-specific test skipped.
- 128 sequential rounds at B4/T1, B4/T4 and B16/T4:
  maximum state error `2.3841858e-7`; state RMS about `1.0e-9`.
- Maximum BF16 verify-output error: `6.1035156e-5`.
- Combined verify convolution matches the original producer exactly.
- Perturbing rounded verify inputs leaves recovery records unchanged.

Width-one speculative verification is not a standard-decode serving test.
These rounds commit an exact checkpoint each time. They do not validate
PR2 retained history or ring wraparound.

## Diagnostic performance

Scope: 69 KDA layers, 12 local heads, head dimension 128; B4/T4, two accepted
tokens. The graph includes conv/gates, verify, recurrent commit and conv
commit. It excludes projections, MLA, MoE, sampling, communication and
metadata refresh.

These are medians of 50 CUDA-event samples after warmup, from one process per
arm. They do not satisfy the independent-start acceptance protocol.
Commit is inside the measurement graph, but remains outside the production
forward graph until acceptance is available.

| Implementation | Total KDA time | Versus main |
| --- | ---: | ---: |
| Unchanged main | 850.4 us | Reference |
| Separate precision producers | 1304.1 us | +53.3% |
| Wider verify tile experiment | 1354.2 us | +59.2%; discarded |
| Combined producers | 1109.5 us | +30.5% |
| Combined producers with overlap | 1072.6 us | +26.1% |
| Rolled recurrence loop with overlap | 1005.0 us | +18.2% |
| Fixed-width specialization experiment | 1133.1 us | +33.2%; discarded |

Raw JSON files are in `outputs/pr1/`: `main-b4-t4-a2.json`,
`candidate-b4-t4-a2.json`, `candidate-b4-t4-a2-bv32.json`,
and `candidate-b4-t4-a2-fused-producers.json`. Later measurements are
`candidate-b4-t4-a2-overlapped.json`,
`candidate-b4-t4-a2-rolled-loop.json`, and
`candidate-b4-t4-a2-fixed-windows.json`.

The short Nsight reports `main-b4-t4-a2-kernel-detail.nsys-rep` and
`candidate-b4-t4-a2-kernel-detail.nsys-rep` each contain five attention graph
replays. They compare main with the combined-producer revision before overlap
and loop tuning. They are diagnostic traces, not profiles of the final code.

In those traces, average verify kernel time is 7.27 us per layer for main and
9.36 us for the two-chain kernel. Recurrent commit falls from 76.1 us to
52.3 us across all 69 layers; the candidate also removes main's 12.5 us
batched gate-precompute launch.

A local unrolled-kernel tile sweep found 137 registers per thread and zero
spill slots for BV4/one warp. Larger tiles did not improve the full attention
result. Occupancy and final rolled-kernel resource reporting remain pending.
These register results do not describe the later rolled-loop binary.

## Producer and launch optimization

The retained implementation gives verify and record production separate CTAs
within one launch. Each CTA keeps one state tile. Convolution prepares FP32
normalized Q/K after the original BF16 rounding. Gate preparation produces
separate verify and replay decay values. Replay retains its FP32 inputs.
This removes repeated normalization and gate transforms from value tiles.

The four-token recurrence is unrolled. Small batches use a 16-value tile
with four warps. Larger batches use a four-value tile with one warp. Wider
tiles spilled registers or reduced throughput. A separate verify/record tile
experiment increased register use and was discarded.

Gate row count and commit batch count are runtime values. Verify uses a
power-of-two batch bucket. A new test checks that request counts within the
same launch configuration do not cause another Triton compilation.

### Matched attention results

The replacement node is also a GB300. Both arms use the same GPU and software
listed above. These results use T4, two accepted tokens and 69 KDA layers.
They remain diagnostic single-process measurements, not an E2E result or
the independent-start acceptance test.

| Batch | Main | Earlier candidate | Retained candidate | Retained versus main |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 852.4 us | 928.3 us | 862.9 us | +1.2% |
| 8 | 1337.8 us | 1460.0 us | 1229.3 us | -8.1% |
| 16 | 1567.2 us | 2184.7 us | 1761.8 us | +12.4% |
| 32 | 2416.1 us | 3631.6 us | 2843.1 us | +17.7% |

The earlier candidate had prepared decay but not prepared verify Q/K. It used
the same four-warp tile at every batch size. The retained candidate improves
all four rows, but does not pass the no-regression gate.

The earlier milestone's table values use `outputs/pr1/main-final-b*-t4-a2.json` and
`candidate-final-b*-t4-a2.json`. The earlier candidate uses
`candidate-newnode-b*-t4-a2.json`. A preceding matched run is retained in
`main-newnode-b*-t4-a2.json` and `candidate-prepared-qk-b*-t4-a2.json`.
Both matched runs show the same performance pattern.
The local launcher is `outputs/pr1/submit.sh`. The attention harness is
`outputs/pr1/bench_runtime.py`; it accepts `--source`, `--output`, `--batch`,
`--width` and `--accepted`. Select the unchanged main worktree for the baseline.

### Accuracy and profile evidence

- Retained kernel suite: 12 passed, including the compilation-reuse check.
- Existing runtime suite: 19 passed.
- Existing replay-kernel suite: 33 passed, 1 platform-specific test skipped.
- Full `pre-commit run --all-files`: passed after formatting.
- Across 128 rounds, maximum state error is `2.9802322e-7` and maximum
  BF16 output error is `3.0517578e-5`. State RMS remains about `1.1e-9`.
- State and output tolerance limits are unchanged.

The new reports are `outputs/pr1/main-b32-t4-a2-prepared-qk.nsys-rep`
and `candidate-b32-t4-a2-prepared-qk.nsys-rep`. Each records five attention
graph replays. They include the prepared-input implementation, unlike the
earlier profiles above. Later source changes add validation and comments;
the retained kernel arithmetic and launch geometry match these reports.

| B32 component | Main average | Candidate average |
| --- | ---: | ---: |
| Verify, per layer; candidate also produces records | 19.05 us | 24.65 us |
| Recurrent commit, all 69 layers | 568.72 us | 536.91 us |
| Convolution producer, per layer | 3.64 us | 5.93 us |
| Gate producer, per layer | 2.78 us | 5.98 us |

Main also launches per-layer payload capture and an all-layer replay gate
precompute. The candidate incorporates this work into its producers. Producer
streams overlap, so these component times must not be summed as wall time.
The largest remaining difference is verify plus FP32 record production.
The next optimization target is that work, not looser numerical tolerances.

### Reproduction snapshot

No implementation commit was made for these measurements. The code diff from
the design commit is saved as `outputs/pr1/retained-optimization-code.patch`.
Its SHA-256 is:

```text
13b003efdb83aaddb17d824838bb3fa53d309944085214bd0e974d3cf6e6f056
```

Apply that patch to the design commit listed above to reproduce the code.
It excludes design documents. Test logs are in
`outputs/pr1/final-optimization-tests.log`; the repeated matched benchmark
log is `outputs/pr1/final-matched-performance.log`. The paired timelines are
also packaged in `outputs/pr1/kda-main-vs-recoverssm-b32-t4-a2-cuda-graph.zip`.

## Regression analysis and reduction-layout optimization

This milestone follows the prepared-Q/K results above. Earlier timelines and
source snapshots do not describe this revision.

### Cause

PR1 must preserve two producer precisions. Main verifies BF16 inputs, but
recomputes accepted tokens from FP32 inputs during recovery. The initial port
produced FP32 records for every candidate before acceptance. It still committed
a full state each round. At T4 with two accepted tokens, record production
therefore processed four tokens where main recovery processed two.

The B32 profile above shows the cost: verify plus records took 24.65 us per
layer, compared with 19.05 us for main verify. Commit saved only 31.81 us across
all 69 layers. Reading and writing those FP32 states moves about 3.47 GB.
The candidate commit's 536.91 us corresponds to about 6.47 TB/s, excluding
smaller record and metadata traffic. This is a memory-heavy operation, so
removing arithmetic there did not offset the extra per-layer recurrence.

The implicit verify layout also put 32 lanes on each key reduction. Address
calculations used runtime strides even when the model layout fixed them.
Masked state selects retained a full tile after invalid suffix tokens, although
no output or record could observe it.

### Retained changes

- Use an explicit Triton Gluon verify layout. Four value rows share a warp;
  eight lanes reduce each key row. This reduces warp shuffle work and avoids
  layout-conversion shared memory in the selected one-warp kernel.
- Keep token updates ordered and FP32. The within-token reduction tree changes.
  This is not a bitwise-equivalence claim. Precision and test limits stay fixed.
- Specialize fixed model-layout strides. Keep the request-dependent group
  pitch and commit batch count as runtime values.
- Remove unobservable suffix-state selects. Mask every output and record store.
  A combined ragged-input test covers empty rows, null pages, unchanged input
  checkpoints and untouched record suffixes.
- Use an eight-value tile through B4 and a 16-value tile above B4 for the
  separate-record producer. Both use one warp. Retain the existing grid order.

The B32 cold-state test cycles through 69 different layer states. Verify plus
records falls from about 21.90 us to 17.75 us per layer after the layout change.
The selected kernel uses 161 registers per thread, no spill slots and no shared
memory. These are compiler resource counts, not measured achieved occupancy.
The previous layout used 91 registers and 32 bytes of shared memory at its
selected tile. A higher register count did not mean a slower kernel here.

Fixed recovery strides reduce the selected commit kernel from 98 to 78
registers per thread. The commit-only harness, including planning and conv
commit, falls from about 598 us to 574 us at B32.

### Discarded experiments

Separate verify and record launches increased launch overhead. Keeping both
state chains in one CTA increased register pressure. A one-warp convolution
producer and a head-major verify grid also made the full attention path slower.
These changes are not retained. Tile sweeps that reuse one state pool can
overstate cache reuse; retained decisions use separate layer states and the
full attention harness.

### Validation

The retained code passes 14 new kernel tests, 19 existing runtime tests and
33 existing replay-kernel tests. One platform-specific test is skipped.
Across 128 rounds, maximum accepted-state error is `2.9802322e-7`.
State RMS is about `1.08e-9`; maximum BF16 output error is `6.1035156e-5`.
The state limit remains `atol=1e-5, rtol=1e-3`.

These checks do not establish real-model token or acceptance equivalence.
True standard-decode serving tests, AIME and E2E performance remain pending.

### Repeated attention comparison

Same GB300, container and software as above; 69 KDA layers, 12 local heads,
dimension 128, T4 and two accepted tokens. Each row is the median of three
independent process medians. Each process measures 50 CUDA-event samples.
Baseline/candidate order reverses in the second trial. Main remains unchanged.

| Batch | Main | Earlier candidate | Current candidate | Current versus main |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 852.5 us | 862.9 us | 823.8 us | -3.4% |
| 8 | 1337.8 us | 1229.3 us | 1060.3 us | -20.7% |
| 16 | 1567.2 us | 1761.8 us | 1547.7 us | -1.2% |
| 32 | 2415.1 us | 2843.1 us | 2381.3 us | -1.4% |

No measured row regresses in these three starts. The B16/B32 margin is small.
B32 current medians range from 2379.9 to 2381.3 us; main ranges from 2414.2 to
2415.1 us. Compared with the earlier candidate, B32 is about 16.2% faster.

The earlier-candidate column repeats the prior milestone, not another arm in
this three-start run. All columns use the same attention harness and settings.
The current run includes conv/gate producers, verify, recurrent commit,
planning and conv commit. It excludes model projections, MLA, MoE, sampling,
communication, prefill and metadata refresh. Commit is included in the test
graph but remains outside the production forward graph until acceptance.

Raw files are `outputs/pr1/regression-{main,candidate}-r{1,2,3}-b*-t4-a2.json`.
The log is `outputs/pr1/regression-matched.log`. This is a local diagnostic
result, not the full ten-start real-model performance gate.

### Acceptance-length sensitivity and remaining regression

The extra checks keep T4 and vary accepted length. These are single-process
diagnostics with the same 50-sample harness.

| Batch | Accepted tokens | Main | Current candidate | Versus main |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 834.0 us | 817.7 us | -2.0% |
| 32 | 1 | 2285.0 us | 2364.9 us | +3.5% |
| 4 | 4 | 889.3 us | 842.2 us | -5.3% |
| 32 | 4 | 2692.4 us | 2481.6 us | -7.8% |

B32 with one accepted token still regresses. The full no-regression gate has
not passed. The candidate creates all four FP32 records regardless of acceptance;
main reduces recovery work when fewer tokens are accepted.

The additional lane-count, vector-layout and rolled-loop experiments did not
improve the selected eight-lane layout. Fully unrolled recurrence is retained.
Do not remove the low-acceptance case from future performance checks.

Raw files are `outputs/pr1/regression-acceptance-{main,candidate}-b*-t4-a*.json`.
The focused A1 timeline shows main recurrent commit at 439.09 us and the
candidate at 497.90 us. Verify is 19.04 us for main and 18.27 us for the
candidate. The 58.81 us commit difference explains most of the approximately
80 us attention regression.

Main commit drops from 567.70 us at A2 to 439.09 us at A1. Candidate commit only
drops from 513.61 us to 497.90 us. Its all-candidate record cost remains fixed,
and each commit still reads and writes a full state. This locates the remaining
cost; the exact memory/scheduling mechanism is not yet resolved.

Commit tile sweeps, processing all value tiles in one CTA, and main's streaming
state-cache hints did not improve the A1 result. They are not retained. The
next target is low-acceptance commit efficiency, alongside the fixed cost of
FP32 record production. Keep precision and phase ordering unchanged.

### Current timeline and source snapshot

The B32/T4/A2 reports each contain five attention graph replays:

- `outputs/pr1/regression-main-GB300-B32-T4-A2-cuda-graph.nsys-rep`
- `outputs/pr1/regression-candidate-GB300-B32-T4-A2-cuda-graph.nsys-rep`

The matching A1 reports use the same names with `A1` in place of `A2`.
All four reports are in
`outputs/pr1/kda-recoverssm-main-vs-optimized-GB300-B32.zip`.
Archive integrity was checked. The A1 profiles confirm the remaining regression.

| Component | Main average | Current candidate average |
| --- | ---: | ---: |
| Verify per layer; candidate also creates records | 19.05 us | 18.29 us |
| Recurrent commit, all layers | 567.70 us | 513.61 us |
| Convolution producer, per layer | 3.62 us | 5.94 us |
| Gate producer, per layer | 2.77 us | 5.98 us |

Main also needs payload capture and batched replay gate preparation. The
candidate includes those tasks in its producers. Producer streams overlap;
these component times must not be summed as elapsed attention time.

The final restored code passes the same 14 + 19 + 33 tests, with one skip.
See `outputs/pr1/regression-restored-tests.log`. Full repository hooks pass;
the log is `outputs/pr1/regression-hooks-final.log`.

The uncommitted code snapshot is `outputs/pr1/regression-optimized-code.patch`.
Apply it to design commit `7a47827e6ce2112aa32f252fbc2e80514bfd411e`.
It includes runtime/kernel code, kernel documentation and tests, but excludes
design documents. SHA-256:

```text
67422187e7b55b372082d2ef90a5d93957f7e2abe5453794e897615d74b44a54
```

The local harness remains `outputs/pr1/bench_runtime.py`; the persistent-job
launcher remains `outputs/pr1/submit.sh`. The accepted-length and A2 profile
commands are in `outputs/pr1/validate-regression-tail.sh`. The A1 profile
commands are in `outputs/pr1/profile-regression-a1.sh`.

## Remaining work

1. Remove the remaining B32/T4/one-accepted-token regression. Extend the
   attention matrix and measure achieved occupancy.
2. Complete composable-record, boundary, padding, alias and backend coverage.
3. Run the B4/8/16/32 matrix, including true standard decode.
4. Validate real Kimi-K3 NVFP4 TP8: deterministic tokens and acceptance,
   agentic workflow, CUDA graphs, eager smoke test and AIME 2026.
5. Run the independent-start performance protocol. Record the tested commit.
6. Run all hooks and obtain code review.
7. Remove temporary design documents only when the work is complete.


## Code-quality review follow-up

Claude Fable 5.1 reviewed the implementation. A second review corrected two
claims: undefined padded output is allowed by the unified-path contract, and
the incompatible per-layer fallback hits an existing assertion rather than
demonstrating silent corruption. Neither claim justifies a numerical change.

The review fixes remain uncommitted after the design commit listed above:

- Pass the required replay-record argument from the benchmark generator.
- Remove dead gate and stride-array inputs from the standalone recovery API
  and the copied recovery kernel. Update both test callers.
- Document the producer-only fields retained by the shared runtime adapter.
- Guard the split-record path against the legacy per-layer fallback.
- Name record fields explicitly and document backend-specific payload meanings.
- Register the RecoverSSM operations at module initialization.
- Check producer layouts without device reads or synchronization.
- Add production batched-commit tests for mixed groups, all accepted lengths,
  in-place writes, padding, pointer refresh and graph capture.
- Add padded runtime graph coverage and a CPU benchmark-caller regression test.

FP32 update order, the two producer precisions and numerical limits are
unchanged. No memset or kernel launch is added. Old kernels used as offline
references remain during validation; final removal of unused legacy code is
not complete.

Claude's follow-up found a per-head layout error in the new test oracle.
The oracle now splits each head's record into K and decay; the production
layout did not change. The existing 128-round differential test already
checks prepared Q/K and decay against main.

Python syntax and static launch-signature checks pass. The full repository
hook suite ran and applied formatting changes. GPU tests and matched timing
await a persistent one-node allocation under the existing binding. The
login-node test environment lacks dependencies available in the GPU
container. This is not a numerical test pass. Existing performance and E2E
gates remain open; this review does not approve the whole PR.

The source snapshot and review logs are in `outputs/pr1/review-fixes-code.patch`
and `outputs/pr1/claude-review-fixes*.json`. The current launcher is
`outputs/pr1/submit.sh`. Run the suites separately in its existing container:

```bash
bash outputs/pr1/submit.sh python -m pytest -q tokenspeed-kernel/test/test_benchmark_kda_generator.py
bash outputs/pr1/submit.sh python -m pytest -q -s tokenspeed-kernel/test/nvidia/ops/attention/test_kda_vllm_recoverssm.py
bash outputs/pr1/submit.sh python -m pytest -q test/runtime/test_kimi_k3_kda_eager_commit.py
bash outputs/pr1/submit.sh python -m pytest -q tokenspeed-kernel/test/ops/attention/test_kda_replay_commit.py
```
