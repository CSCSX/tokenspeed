# Kernel benchmarks

This directory contains suites for measuring TokenSpeed operations and their
selected kernel registrations. The benchmark harness separates operation-specific
input and correctness logic from shared device timing and result reporting.

Suites are organized as `<vendor>/<arch>.json`, one per target platform.
Benchmark cases are grouped by model and operation family in files referenced by
the hardware suite. They use representative model inputs and exercise normal
operation dispatch, covering both kernel selection and execution.

Each operation family and mode owns one generator under
`tokenspeed_kernel/benchmark/generators/`. The harness loads built-in
generators, and `set_benchmark_generator` registers additional ones.
Every generator reuses the same harness, timer, and validators.

## AMD KDA verify and commit matrix

`test/amd/ops/attention/test_gluon_kda_replay_matrix.py` adds a focused BF16
matrix for gfx950 and gfx1250. It reuses the existing replay input/record
helpers and the graph timer. Concurrency is the number of requests in one
kernel batch. The pytest matrix covers concurrency 1, 2, 4, 8, 16,
32, 64 and 128, with T=1 and T=4 (MTP3), all integer accepted lengths from
zero through T, and mixed lengths within a batch. Both distinct-page and
in-place commit run eagerly and through graph replay.

Correctness checks compare verify outputs, FP32 U, K/raw-g records and
accepted state with an independent torch recurrence over the existing
BF16 convolution producer's outputs. They also check record stamps, exact
convolution history, verify's read-only checkpoint, and untouched pool pages.
The unit tests use three independent layers and cache groups; the timing
command checks every requested layer before measuring.

Run the matrix from the repository root in a prepared AMD kernel environment.
The three source revisions must be available in the local Git checkout:

```bash
export PYTHONPATH="$PWD/tokenspeed-kernel/test:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-amd/python${PYTHONPATH:+:$PYTHONPATH}"
baseline_revision=3461560d467a76401ce3cb3d72742fccb05dd591
accepted_parent_revision=d84269c5a222c2dcfb46c333c84b4967bd892a7a
trial_reference_revision=668d50a198d005a9b39bf4837968437d902f880b
source_path=tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/kda/decode.py
git show "$baseline_revision:$source_path" > /tmp/original-kda.py
git show "$accepted_parent_revision:$source_path" > /tmp/accepted-parent-kda.py
git show "$trial_reference_revision:$source_path" > /tmp/trial-reference-kda.py
records_path=tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/_kda_records.py
git show "$accepted_parent_revision:$records_path" > /tmp/accepted-parent-kda-records.py
git show "$trial_reference_revision:$records_path" > /tmp/trial-reference-kda-records.py
python tokenspeed-kernel/test/amd/ops/attention/test_gluon_kda_replay_matrix.py compare \
  --concurrency 1 2 4 8 16 32 64 128 \
  --window-lengths 1 4 \
  --layers 69 \
  --page-relation distinct \
  --cache cold \
  --measurement-blocks 30 \
  --paired-rounds 3 \
  --baseline-source /tmp/original-kda.py \
  --baseline-revision "$baseline_revision" \
  --accepted-parent-source /tmp/accepted-parent-kda.py \
  --accepted-parent-records-source /tmp/accepted-parent-kda-records.py \
  --accepted-parent-revision "$accepted_parent_revision" \
  --trial-reference-source /tmp/trial-reference-kda.py \
  --trial-reference-records-source /tmp/trial-reference-kda-records.py \
  --trial-reference-revision "$trial_reference_revision" \
  --output /tmp/kda-replay-matrix.json
```

For gfx950, use the same snapshots to measure two successive records lifecycles:

```bash
python tokenspeed-kernel/test/amd/ops/attention/test_gluon_kda_replay_matrix.py reuse \
  --concurrency 1 2 4 8 16 32 64 128 \
  --window-lengths 1 4 \
  --layers 69 \
  --page-relation distinct \
  --cache cold \
  --measurement-blocks 30 \
  --paired-rounds 3 \
  --trial-reference-source /tmp/trial-reference-kda.py \
  --trial-reference-records-source /tmp/trial-reference-kda-records.py \
  --trial-reference-revision "$trial_reference_revision" \
  --output /tmp/kda-replay-reuse.json
```

The second verify reads the first commit's state. No reset or cache clear runs
between steps. The report measures the whole sequence against the trial reference;
it does not predict model latency. Zero acceptance and explicit page swaps cover
the kernel API, not serving scheduler or allocator policy. CI publishes both
original comparison matrices before it runs the reuse matrices.

The `compare` command checks each case eagerly and with graph replay. Full-lifecycle
timing includes all requested layers. The candidate runs gate production,
records verify and QKV retention, then one batched commit. The immutable
original uses fused decode at T1, and gate GEMM, capture verify and raw-gate
replay at T4. On gfx1250, T1 also includes gate GEMM. These directly measured
totals determine performance acceptance.

GFX950 also pairs the full lifecycle with an accepted parent and a separate
trial reference. Each source has its own module, revision, SHA256 and labeled
regression count. The fixed trial revision isolates the current optimization;
it does not replace the immutable original or accepted-parent baseline.

The separate one-layer verify timing includes gate production and retained replay
payloads on both sides. The original runs its gate GEMM and capture verify in one
timed invocation; the candidate uses its automatic gate producer and records
verify. Comparison schema version 3 reports this complete verify time and its
speedup ratio. Earlier versions excluded the original gate GEMM. At T1 this is a
capture-verify comparison; the original standard fused decode remains the
full-lifecycle baseline. Commit timing includes convolution-history updates and
excludes verify. Neither phase is subtracted from the full-lifecycle timing.

On gfx950, an auxiliary measurement compares one-layer verify with an accepted
parent source. Both sides use the same supplied raw gate and retain QKV; gate
projection and commit are excluded. Both pass independent eager and graph
correctness checks before paired timing. Results name the parent revision and
source hash separately from the immutable original.

A second auxiliary measurement compares the complete current and accepted-parent
records lifecycles on the same buffers. Each side runs its own automatic gate
producer, verify with QKV retention, and batched commit. All acceptance patterns
have equal work, including zero and mixed lengths. Both sides pass independent
eager checks and reproduce those results through one complete captured graph.
The timer alternates sides across rounds and resets pools outside the timed
interval. The report includes each round's parent/current latency ratio, along
with aggregate medians and MADs. Neither parent comparison runs on gfx1250.

After accepting a kernel change, advance the accepted-parent revision in the
CI task and this example for the next comparison. Read the calibration whenever
the parent changes. Identical-source runs show the measured spread; a later
candidate within that spread needs another measurement before acceptance.
The immutable original remains fixed, and its complete-operation timings remain
the final performance target.

Use `--page-relation in-place` to measure aliasing; the timer restores the
checkpoint outside the timed interval. `--cache hot` omits the cache clear after
that reset; it does not preserve state across measured invocations.
JSON includes all samples, median/p90, accepted counts, layer count, device,
revision and dirty-worktree status. Keep these settings identical when comparing
revisions. A failed correctness check stops the command without publishing a
timing for that case.

The MI355 unit-test job discovers the pytest matrix automatically.
The manual CI task `kda-replay-matrix-amd-gfx950` runs the 69-layer timing command
and publishes its JSON artifact. On `amd_replay_ssm_opt`, the `AMD Replay SSM`
workflow invokes only this task on pushes or manual dispatch. Other jobs in the
branch's workflow definitions skip this branch. K8s Dispatch permits only this
exact task YAML, and the reusable matrix runner permits only a single entry for
this task. Other branches retain their existing conditions. Repository automation
that GitHub loads from the default branch follows that branch's configuration.
Simulator timings are not GPU performance measurements.

## AMD KDA serving comparison

The manual `kda-replay-e2e-amd-gfx950` task compares pinned main and records
revisions on one eight-GPU AMD runner. Each revision uses its own source tree
and a scheduler built from that revision. It serves Kimi-K3 with BF16 activations,
TP8/EP1 and the same pinned model snapshot. The report records the actual GPU
properties and package versions.

Select modes explicitly with `--modes`. T1 and MTP3 cover concurrency 32, 64
and 128. Native MTP3 requires a checkpoint that declares and contains MTP
weights; the pinned public Kimi-K3 snapshot has none, so the CI task selects T1.
MTP3 uses greedy sampling with synthetic accepted length 3.7. Each request has
4096 input tokens and 371 output tokens; prefix caching is disabled. Both modes use main/current/current/main
server starts, with two warmup and three measured waves per concurrency per
start. Requests have identical token IDs and launch from one client barrier;
HTTP arrival and scheduler batch identity are not guaranteed.

Reported request E2E includes prefill. The report also records TTFT, output-token
latency, throughput and observed acceptance counters from the engine metrics
port, separately from the HTTP gateway. This is a comparison of
complete branches, including their scheduler differences, rather than an isolated
KDA attribution or natural-acceptance accuracy test. Startup failures and
incomplete waves remain failures; they do not produce speedup claims.

## Benchmark requests

Each request identifies an operation family and mode, supplies parameters for
that operation's generator, and may select a solution or exact registration.
The generator interprets parameters such as shapes and data types and returns
the callable, arguments, and correctness work needed by the harness. For
example, a dense BF16 batched GEMM request against one exact registration:

```python
from tokenspeed_kernel.benchmark import (
    BenchmarkRequest,
    GraphBenchmarkConfig,
    GraphTimer,
    KernelBenchmarkHarness,
)
from tokenspeed_kernel.platform import current_platform

harness = KernelBenchmarkHarness(
    GraphTimer(
        GraphBenchmarkConfig(
            eager_warmup_iterations=5,
            replay_warmup_iterations=3,
        )
    ),
    platform_provider=current_platform,
)
result = harness.run(
    BenchmarkRequest(
        family="gemm",
        mode="bmm",
        parameters={
            "batch": 12,
            "M": 1,
            "N": 512,
            "K": 128,
            "dtype": "bfloat16",
            "validation": {
                "runs": 5,
                "atol": 0.015,
                "rtol": 0.015,
            },
        },
        solution=None,
        registration="gluon_bmm_a16w16_gfx950",
        cold_cache=True,
        seed=42,
    ),
    measurement_blocks=30,
)
```

Exact-registration benchmarks invoke the named registration through its normal
public behavior. Any internal fallback remains owned by the operation, and the
benchmark harness does not change it.

## Timing

The shared timer measures warmed graph replay of the selected registration.
Input creation, selection, compilation, eager warmup, graph capture, replay
warmup, correctness checks, and result serialization are outside the reported
device time.

By default, each captured invocation clears the device caches immediately
before the operation runs. Device events surround only the operation, so the
reported time excludes cache clearing. A case can set `cold_cache` to
`false` for hot-cache experiments.

Each measurement block reports the device time for one captured operation
invocation. Cases inherit the suite's `measurement_blocks` value unless they set
an explicit override. Eager and graph-replay warmup counts remain suite-wide.
Warmup settings and cache mode must match across revisions before measurements
can be compared. Measurement-block counts may differ.

The result contains the raw device-time samples, resolved registration, timing
mode, and structured failure information. Suite-level comparison uses the
sample median.

## Correctness

The operation generator owns correctness checking, which is opt-in. The
generator selects a registered reference solution, constructs candidate and
reference calls over the same fresh inputs, and declares a validator for
each output that needs checking. Outputs that do not require validation use
no validator.

The generator declares how many fresh input sets to run, and each validator
receives every run's candidate/reference pair for its output along with
validator-specific options. The built-in `close` validator accepts absolute
and relative tolerances; `set_output_validator` registers additional
validators. Generators typically compare against the operation's
registered `torch` solution, the PyTorch references under
`tokenspeed_kernel/numerics/torch/`.

Correctness runs before timing. A failure prevents the case from producing a
successful measurement. Correctness remains within the revision-local process,
and tensor values are never serialized for cross-revision comparison.

## Suite contract

A suite declares:

- the required hardware vendor and architecture;
- shared eager and graph-replay warmup counts, plus a default measurement-block
  count;
- stable case IDs, comparison epochs, and definitions;
- per-case relative and absolute regression limits; and
- `min_regression_absolute_us`, the smallest absolute limit any case may use.

The minimum absolute limit reflects timing resolution rather than performance
intent: separate processes can measure the same small kernel a fraction of a
microsecond apart, so a tighter limit would flag process-to-process noise.

A case may set `measurement_blocks` when its noise profile requires more samples
than the suite default. The suite runner records the effective value with that
case's result; the suite's single benchmark harness receives that value for
each case.

The optional `case_files` list composes model-and-operation case files into the
hardware suite. Case IDs remain unique across the composed suite.
Case files may define `common_parameters`; the loader merges them into every
case, with parameters written on an individual case taking precedence.

A top-level parameter may be a non-empty list. The loader expands all such lists
as a Cartesian product and appends an index (`_0`, `_1`, and higher) to the case
ID. Parameter names are sorted alphabetically to determine dimension order;
list order determines each dimension's value order. A one-element product
retains the original case ID.
Use separate case entries when shape classes need
different policies or do not form a Cartesian product. This expansion applies only
to direct parameter values; lists inside nested parameter objects remain literal.
Wrap a literal top-level list in an outer expansion list, such as `[[1, 2, 3]]`.
Changing list contents or order can remap numbered IDs, so treat the grid as part of
the case identity.

`comparison_epoch` is an opaque equality token used only to decide whether
baseline and candidate measurements are comparable. It has no ordering and
creates no backward-compatibility requirement. When an operation's
performance-relevant semantics change while retaining the same operation and
case identity, advance it. For example, adding a consumer fusion without
renaming the operation starts a new comparison epoch. For ordinary kernel
implementation changes or benchmark harness and CI changes, do not advance it.
Those kernel implementation changes are what the benchmark is intended to
compare, while the comparison tooling represents and checks
timing-infrastructure compatibility separately.

Compatible cases must have the same ID, comparison epoch, definition, warmup
settings, cache mode, and recorded hardware. Their measurement-block counts may
differ. The comparison reports the resolved registration, which may change
because dispatch is part of the behavior under test. Added and changed cases
appear in the report without comparison. A baseline case missing from the
candidate also appears in the report.

Regression policy comes from the merge-base suite, so a candidate cannot weaken
its own gate by changing a threshold. A benchmark is a regression only when its
median slowdown exceeds both the configured relative and absolute limits.

The variation between benchmark processes is several times larger than the
sample spread within one process, and it occasionally shifts a single
process by more than a budget. A single base/candidate pair therefore cannot
settle a verdict. The comparison runs every case once per revision, then
re-measures only the cases outside their budgets in further rounds. Each
round starts fresh baseline and candidate processes, and the revision that runs
first alternates between rounds. The round with the median relative change
supplies the reported times and the verdict. If one round finds a regression
and another an improvement, the measurement cannot tell the direction, and the
result is inconclusive. A wide spread alone does not void the verdict: noise
grows with the size of the change, so a large regression that every round
reproduces still fails.

## Running a suite

From the repository root and a prepared TokenSpeed kernel environment, run the
revision-local worker:

```bash
python3 -m tokenspeed_kernel.benchmark.ci \
  --suite tokenspeed-kernel/benchmarks/amd/gfx950.json \
  --revision "$(git rev-parse HEAD)" \
  --output /tmp/tokenspeed-kernel-result.json
```

On AMD, a direct worker run profiles the measured graph replays with Proton and
adds a per-case diagnostic summary to the result JSON. It reuses launch metadata
from the final eager warm-up to calculate per-kernel TFLOP/s. To disable it,
use `--profiler none`. Automated CI explicitly selects no profiler so
its regression measurements stay as close to production execution as possible.
Repeated launches with the same runtime kernel name appear as one combined
kernel entry in the summary because Proton groups them in graph-replay profiles.
`--case-filter REGEX` selects expanded case IDs with `re.search`; repeat the
option to match any expression. For example, `--case-filter '_0$'` selects the
first expanded combination for each parameterized case.

Run a complete base/candidate comparison from the repository root:

```bash
python3 test/ci_system/kernel_benchmark_ci.py \
  --base-ref <target-commit> \
  --candidate-ref <candidate-commit> \
  --output-dir /tmp/tokenspeed-kernel-benchmark \
  --measurement-rounds 3
```

`--measurement-rounds` is the total number of rounds for cases that the first
round finds outside their budgets. It must be odd so that a median round
exists; `1` reports first-round verdicts without re-measuring.

The default comparison creates separate worktrees and virtual environments for
the two revisions and installs each revision's ROCm kernel requirements. To
reuse an already prepared environment during local development, use
`--environment-mode current`.

The output directory contains revision-local JSON results and logs, setup logs,
the structured comparison, and a Markdown summary. When the merge base does not
contain the suite, the comparison degrades to a candidate-only bootstrap
because no compatible baseline exists.

See the [CI documentation](../../test/ci/README.md#registration-level-kernel-benchmarks)
for GitHub Actions triggers, artifacts, pull request comments, and runner setup.
