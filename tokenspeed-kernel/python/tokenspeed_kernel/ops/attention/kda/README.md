# KDA verification producers

Multi-token record-producing verify uses one operation for the four-tap
convolution and the second gate projection. Separate thread blocks perform
these independent producers in one launch. Verification consumes both outputs
on the same stream.

Convolution reads the committed three-token window without modifying it.
It accumulates in FP32, applies SiLU, and stores BF16 QKV. The operation also
copies the original projection rows into the accepted-commit payload.
An exact input/payload alias skips the copy, because concurrent identical
stores would still race with convolution reads.

Gate projection uses BF16 matrix operands and FP32 accumulation, then rounds
once to BF16. Its reduction order can differ from a standalone GEMM.
Numerical tests check an FP64 oracle with an FP32 accumulation-error bound
and the final BF16 rounding interval. Exactly representable sums must match
bit for bit. Separate exact tests cover ties-to-even rounding, small terms,
and every reduction position across tile boundaries. The error-bound tests
use finite inputs away from overflow and subnormal ranges; they do not
establish behavior for unrestricted IEEE inputs or universal cuBLAS identity.
Recurrence, candidate record precision, and accepted-prefix
ownership use the existing RecoverSSM contracts.

## Accepted history

Each cache checkpoint retains FP32 correction, normalized key and decay for
every accepted token. Appending a new token leaves older factors unchanged.
Scalar reconstruction applies each decay multiplication before the fused
correction/key update. Keep key normalization as a one-dimensional reduction
with one warp; changing its layout can change FP32 rounding.

The optional TF32x3 specialization forms decay-weighted keys during recovery.
It preserves the same accepted interval but can round differently because it
groups products. Serving currently uses scalar recovery. Per-token decay uses
more checkpoint storage than an accumulated decay; the cache plan includes
the complete field shape when it selects the number of physical pages.

## Single-token FP32 records

Standard decode retains the fused convolution, SiLU and gate projection in
FP32 registers. It writes FP32 correction, normalized key and decay records;
`key_gate_format` distinguishes these from the multi-token raw key/gate
records. Accepted commit copies normalized factors without another transform.
Both producers use the same cache-owned history and acceptance lifecycle.

The tiled recurrence preserves the fused decode reduction groups and rounds
products before their sums. Candidate factors stay in separate scratch while
all value tiles read the previous accepted history. Only the first value tile
writes shared key/decay records. Convolution publication remains after verify,
so an in-place update cannot race another tile reading the old window.

Zero acceptance can still publish a consolidated base or copy a checkpoint to
its private destination. Current record stamps distinguish that case from an
inactive row. Both accepted factors and convolution commit skip nonpositive
destination pages, preserving the reserved null page.
