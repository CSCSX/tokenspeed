# KDA verification producers

The record-producing verify path uses one operation for the four-tap
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
bit for bit. Recurrence, candidate record precision, and accepted-prefix
ownership use the existing RecoverSSM contracts.
