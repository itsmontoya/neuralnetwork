# NN-29: `MTLComputeCommandEncoder` Coalescing per `Scope`

**Date:** 2026-11-27  
**Repository:** `github.com/itsmontoya/neuralnetwork`  
**Base:** `main` @ `c6e3ae1fa305f065f2fc958ee040700fa84e00a3` (`v0.16.0`)  
**Public API changes:** None

## Summary

Previously, every compute dispatch in `internal/device/metal_backend.m` opened a new `MTLComputeCommandEncoder`, encoded one dispatch, and immediately called `[encoder endEncoding]`. For multi-op resident graphs (e.g., 16-dispatch `CausalAttention`, 19-op ONNX Transformer encoder blocks, or ~240-op Llama decode steps), opening and closing a separate compute encoder per dispatch added substantial Metal driver and GPU scheduling boundary overhead.

NN-29 keeps a single active `MTLComputeCommandEncoder` open on `NNMetalScopeRecord` across consecutive compute dispatches within a `Scope`, ending and releasing it only when:

1. Transitioning to an `MTLBlitCommandEncoder` (`nn_metal_scope_encode_copy`, `nn_metal_scope_encode_gather_rows`, `nn_metal_scope_encode_kv_append`).
2. Committing the scope (`nn_metal_scope_commit`).
3. Releasing the scope (`nn_metal_scope_release`), ensuring discarded, canceled, or mid-scope-failed scopes always end and release any open compute encoder without leaks or Metal API validation errors.

## Lifecycle & Safety Design

- **State (`NNMetalScopeRecord`):** Added `id<MTLComputeCommandEncoder> computeEncoder;` (zero-initialized by `calloc` in `nn_metal_scope_new`).
- **Helpers (`internal/device/metal_backend.m`):**
  - `nn_metal_scope_compute_encoder(record, context)`: returns `record->computeEncoder` if already open; otherwise creates `[record->commandBuffer computeCommandEncoder]`, retains it under MRC, stores it in `record->computeEncoder`, and returns it.
  - `nn_metal_scope_end_compute_encoder(record)`: if `record->computeEncoder != nil`, calls `[record->computeEncoder endEncoding]`, `[record->computeEncoder release]`, and sets `record->computeEncoder = nil`.
- **Pre-validation before encoder acquisition:** Threadgroup-limit checks in `nn_metal_scope_encode_tensor_conv2d` and `nn_metal_scope_encode_tensor_transpose` were hoisted before encoder acquisition (matching the matmul paths) so validation errors never touch encoder state.
- **Hazard tracking:** All device `MTLBuffer`s use `MTLResourceStorageModeShared` with default hazard tracking (`MTLHazardTrackingModeDefault`), and `MTLComputeCommandEncoder` defaults to serial dispatch (`MTLDispatchTypeSerial`), preserving exact read-after-write and write-after-read ordering across consecutive dispatches within the coalesced encoder.
- **Buffer retention:** `scopeRecord->retainedResources` continues to retain every buffer referenced by any dispatch until `nn_metal_scope_release` after command buffer completion.

## Verification

- Verified with **Metal API Validation Enabled** (`METAL_DEVICE_WRAPPER_TYPE=1`) across `internal/device` and `gpu` with zero validation warnings or encoder-lifecycle assertions.
- Added dedicated Metal tests in `internal/device/encoder_coalescing_metal_test.go` and `gpu/encoder_coalescing_metal_test.go`:
  - Interleaved compute → blit → compute → blit → compute chains in a single `Scope` verified bit-exact (`math.Float32bits`).
  - Discarding an uncommitted `Scope` with an active open compute encoder (`ResourceSnapshot` confirms zero leaks and zero submitted commands).
  - Mid-scope encoding failures and `context.Canceled` before `Submit` with an active open compute encoder.
  - 8-worker `-race -tags=metal` concurrent stress across interleaved compute + blit + discard scopes.
- All 14 entries of the Required Test Matrix pass.

## Apple M3 Paired Benchmarks

**Hardware:** Apple M3 (`darwin/arm64`, 8-core CPU / 10-core GPU)  
**Methodology:** Paired before (`v0.16.0`) vs after (`NN-29`) runs via `CGO_ENABLED=1 GOWORK=off go test -tags=metal -run='^$' -bench=... -benchmem -count=5 ./gpu`. Table reports median latency across 5 samples; raw logs are stored in `.benchmarks/gpu-nn-29/run1/{before,after}.txt`.

### `Benchmark_CausalAttentionMetal` (16 consecutive compute dispatches / scope)

| Benchmark | Before (ms/op) | After (ms/op) | Delta | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `M1_H38_x16/float32` | 0.985 | 0.345 | -64.9% | **2.85×** |
| `M1_H38_x16/float16` | 0.980 | 0.344 | -64.9% | **2.85×** |
| `M1_H128_x16/float32` | 1.074 | 0.363 | -66.2% | **2.95×** |
| `M1_H128_x16/float16` | 1.065 | 0.361 | -66.1% | **2.95×** |
| `M1_H512_x16/float32` | 1.436 | 0.747 | -48.0% | **1.92×** |
| `M1_H512_x16/float16` | 1.509 | 0.747 | -50.5% | **2.02×** |
| `M1_H1024_x16/float32` | 2.275 | 1.121 | -50.7% | **2.03×** |
| `M1_H1024_x16/float16` | 2.475 | 1.159 | -53.2% | **2.14×** |
| `M29_H29_x16/float32` | 1.302 | 0.529 | -59.4% | **2.46×** |
| `M29_H29_x16/float16` | 1.309 | 0.528 | -59.6% | **2.48×** |

### `Benchmark_FFNMixedMetal` (4 consecutive compute dispatches / scope)

| Benchmark | Before (ms/op) | After (ms/op) | Delta | Speedup | Encode Before → After (µs/op) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `M1/float32` | 2.976 | 2.918 | -1.9% | 1.02× | 18.5 → 4.8 |
| `M1/float16` | 1.474 | 1.424 | -3.4% | 1.04× | 7.5 → 4.8 |
| `M1/bfloat16` | 1.502 | 1.457 | -3.0% | 1.03× | 7.9 → 4.8 |
| `M1/q8_0` | 0.987 | 0.868 | -12.0% | **1.14×** | 5.7 → 4.7 |
| `M1/q4_0` | 0.691 | 0.576 | -16.6% | **1.20×** | 5.4 → 4.8 |
| `M1/q4_1` | 1.001 | 0.611 | -38.9% | **1.64×** | 6.0 → 4.7 |
| `M29/float32` | 15.620 | 14.805 | -5.2% | 1.06× | 12.7 → 4.9 |
| `M29/float16` | 2.828 | 2.761 | -2.4% | 1.02× | 15.5 → 4.5 |
| `M29/bfloat16` | 2.718 | 2.757 | +1.4% | 0.99× | 5.5 → 4.4 |
| `M29/q8_0` | 2.192 | 2.205 | +0.6% | 0.99× | 5.5 → 4.4 |
| `M29/q4_0` | 2.548 | 2.156 | -15.4% | **1.18×** | 9.8 → 4.4 |
| `M29/q4_1` | 2.394 | 2.157 | -9.9% | **1.11×** | 9.2 → 4.4 |

### `Benchmark_TransformerBlockMixedMetal` (full Llama-style layer / scope)

| Benchmark | Before (ms/op) | After (ms/op) | Delta | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `M1_H511/FP32_KV32` | 3.855 | 3.779 | -2.0% | 1.02× |
| `M1_H511/FP16_KV32` | 1.867 | 1.783 | -4.5% | 1.05× |
| `M1_H511/FP16_KV16` | 1.862 | 1.765 | -5.2% | **1.06×** |
| `M1_H511/BF16_KV32` | 1.914 | 1.785 | -6.7% | **1.07×** |
| `M1_H511/BF16_KV16` | 1.859 | 1.755 | -5.6% | **1.06×** |
| `M29_H0/FP32_KV32` | 19.103 | 17.733 | -7.2% | **1.08×** |
| `M29_H0/FP16_KV32` | 3.789 | 3.731 | -1.5% | 1.02× |
| `M29_H0/FP16_KV16` | 3.727 | 3.723 | -0.1% | 1.00× |
| `M29_H0/BF16_KV32` | 3.755 | 3.733 | -0.6% | 1.01× |
| `M29_H0/BF16_KV16` | 3.733 | 3.718 | -0.4% | 1.00× |

### `Benchmark_ONNXTransformerEncoderBlockMetal` & `Benchmark_ResNetBottleneckMetal`

| Benchmark | Before (ms/op) | After (ms/op) | Delta | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `ONNXTransformerEncoderBlock/BERTBase_B2_S64_E768/float32` | 2.409 | 1.644 | -31.8% | **1.47×** |
| `ONNXTransformerEncoderBlock/BERTBase_B2_S64_E768/float16` | 1.639 | 1.621 | -1.1% | 1.01× |
| `ONNXTransformerEncoderBlock/BERTBase_B2_S64_E768/bfloat16` | 1.606 | 1.610 | +0.2% | 1.00× |
| `ResNetBottleneck/Metal/float32` | 6.847 | 6.807 | -0.6% | 1.01× |
| `ResNetBottleneck/Metal/float16` | 6.856 | 6.843 | -0.2% | 1.00× |
| `ResNetBottleneck/Metal/bfloat16` | 6.874 | 6.838 | -0.5% | 1.01× |

### Key Takeaways

- Coalescing eliminates ~40 µs per avoided compute-encoder boundary on Apple M3, delivering **1.92×–2.95×** speedups on dispatch-heavy scopes (`CausalAttentionMetal` 16× dispatches), **1.14×–1.64×** on fast quantized `M=1` FFN decode (`Q8_0`/`Q4_0`/`Q4_1`), **1.05×–1.07×** on full single-layer `M=1` Transformer blocks, and **1.47×** on FP32 BERT-base encoder blocks (with much tighter run-to-run latency stability).
- Host-side `encode-ns/op` in `FFNMixedMetal` dropped to a consistent **4.3–4.9 µs/op** across all dtypes and batch sizes.
- Heavy compute-bound scopes (`ResNetBottleneck`, `M=29` mixed-precision tiled GEMM) remain unchanged or slightly faster with zero regressions.

## Handoff Notes for NN-30

- Any new compute dispatch added in `internal/device/metal_backend.m` (such as NN-30's rank-2 `Float32` GEMV/GEMM kernels) must acquire its encoder via `nn_metal_scope_compute_encoder(scopeRecord, context)` and must **not** call `[encoder endEncoding]`.
- Any new blit dispatch (such as NN-31's `outer == 1` `Concat` fast path) must call `nn_metal_scope_end_compute_encoder(scopeRecord)` before calling `[scopeRecord->commandBuffer blitCommandEncoder]`.
