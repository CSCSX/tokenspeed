# Fused Ascend accepted-prefix ReplaySSM

The runtime retains its existing all-layer descriptor, metadata and arena-anchor
contract. Set `TOKENSPEED_KDA_REPLAY_FUSED_LIBRARY` to the absolute path of
`libreplayssm_fused.so` before importing TokenSpeed to select the standalone
`flash_replay::fused` extension. Build it from Flash's
`cann-flash-ops/experimental/replayssm_fused` using the same PyTorch, torch_npu
and CANN runtime as the model. The ordinary Flash wheel does not include it.

The NPU adapter loads it during kernel registry initialization, validates the
22-argument schema and writable arena anchor, and maps the existing keyword
arguments to its positional ABI. Configured loading or execution errors are
fatal, never a silent legacy fallback. An unset variable preserves compatibility
with older deployments. Change the selection only by restarting the process
and recapturing graphs, not by changing the environment after import.

The extension fuses accepted-token gate calculation, FP32 recurrence, and raw
Q/K/V history commit in one kernel. It preserves the old Replay FP32 arithmetic
boundaries; V5 verify's BF16 fences must not be applied to replay. Existing
descriptor storage, payloads, page strides, acceptance lengths and state-anchor
lifetimes remain backend-owned. The caller must provide unique writable pages
and no cross-request read/write races. A zero accepted length or invalid write
page is a no-op; an invalid read supplies zero initial state.

Supported geometry is BF16 packed payload, FP32 key-major recurrent state,
T1–8, D128, FA128, convolution width4 and H4/8/16/32/64. AIV blocks loop over
logical layer/request/head work, avoiding the legacy logical-grid limit.
The legacy gate workspace remains allocated for ABI compatibility but is not
an output. Verify, prefill, one-token execution, cache allocation and acceptance
policy are unchanged by this adapter.

Validation must include old-Replay/CPU state and exact convolution-history
comparison, changing metadata in native graphs, and next-verify outputs. A
whole-model performance trace must show one fused replay per step and no old
gate/replay/conv-commit kernels. Performance evidence does not waive numerical
precision debt or constitute a model-quality result.
