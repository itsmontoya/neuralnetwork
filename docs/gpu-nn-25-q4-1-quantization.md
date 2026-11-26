# NN-25: Asymmetric `Q4_1` Block Quantization & `GATE-I` Release Readiness Checklist

**Date:** 2026-11-26  
**Base:** `gpu-nn-24` (`d2eef0f05b76e0c6ca871b49233ce2ed9ad693b7`, PR #165; stacked on `v0.14.0`)  
**Hardware:** Apple M3 (`darwin/arm64`, Metal, `CGO_ENABLED=1`)

## Summary

Implemented GGUF-compatible asymmetric 4-bit block quantization **`gpu.Q4_1`** (`20` bytes per `32` weights = `0.625 B/param`, **6.40×** compression vs FP32) end-to-end across `gpu` and `internal/device`:

- **Public `gpu` API (`gpu/dtype.go`, `gpu/quantize.go`, `gpu/tensor.go`, `gpu/scope.go`, `gpu/backend.go`):**
  - `gpu.Q4_1`: `String() == "q4_1"`, `Valid() == true`, `IsQuantized() == true`, `Bytes() == 0`, `BlockSize() == 32`, `BlockBytes() == 20`, `QuantBlockBytesQ4_1 = 20`.
  - Deterministic host helpers `QuantizeQ4_1` and `DequantizeQ4_1` handling `NaN`, constant blocks (`max == min` → `d = 0, m = Float32ToFloat16(min), qs = 0`), non-finite/overflow `d`/`m`, and round-to-nearest-even clamped to `0..15`.
  - Wired through `Tensor.UploadFloat32As`, `Tensor.DownloadFloat32Into`, `Tensor.UploadBytes`, `Tensor.DownloadBytesInto`, `Scope.Copy`, `Scope.MatMulRightTranspose` (`K % 32 == 0`), and `Scope.GatherRows` (`Q4_1 → Float32` dequantizing gather and `Q4_1 → Q4_1` matching raw-byte blit), rejecting non-transposed `Scope.MatMul`, `Tensor.View`, `KVCache`, and all unsupported ops with `ErrInvalid`.
- **Metal kernels (`internal/device/matmul_quant_metal_source.h`, `attention_metal_source.h`, `metal_backend.m`, `scope.go`):**
  - **`M == 1` SIMD GEMV** (`nn_matmul_q4_1_row_right_transpose`): 128 threads (`4` SIMD groups, `2` rows/SIMD, `4` lanes/block = `8` blocks/step), coalesced `float4` activation loads, naturally 4-byte-aligned `half` `d`/`m` and `uint` payload loads, FP32 `fma(x, d * q + m, acc)`, `simd_sum`.
  - **`M > 1` `simdgroup_float8x8` tiled GEMM** (`nn_matmul_q4_1_tiled`, plus scalar fallback `nn_matmul_q4_1` for `M < 8 || N < 8`): `(BM=32, BN=64, BK=32)` matching `QuantBlockSize = 32` so each 20-byte block is dequantized once into `threadgroup float` shared memory and reused across `BM = 32` activation rows.
  - **Dequantizing `GatherRows`** (`nn_tensor_gather_rows_q4_1_to_f32`): 2D compute gather reusing the 512-row `setBytes` chunking loop, plus matching `Q4_1 → Q4_1` blit (`rowBytes = (columns / 32) * 20`).
  - All kernels compiled lazily via `pthread_once` with `[options setFastMathEnabled:NO]` and FP32 accumulation.

## Exact `Q4_1` Block Layout (GGUF-Compatible)

Every block encodes `QuantBlockSize = 32` contiguous elements along the inner dimension `K` (`K % 32 == 0`) of a rank-2 `[N, K]` weight matrix in `QuantBlockBytesQ4_1 = 20` bytes (naturally 4-byte aligned):

| Byte offset | Type | Description |
| --- | --- | --- |
| `0..1` | little-endian `uint16` | IEEE-754 binary16 scale `d` (`Float16`) |
| `2..3` | little-endian `uint16` | IEEE-754 binary16 minimum `m` (`Float16`) |
| `4..19` | `16 × uint8` | Packed 4-bit unsigned nibbles `qs[0..15]` (GGUF split-half order, `0..15`) |

- `value[j] = Float16ToFloat32(d) * float32(qs[j] & 0x0F) + Float16ToFloat32(m)` (`j = 0..15`)
- `value[j + 16] = Float16ToFloat32(d) * float32(qs[j] >> 4) + Float16ToFloat32(m)` (`j = 0..15`)

## Numerical Verification

### 1. Synthetic FFN SQNR Comparison (`Q4_0` vs `Q4_1` vs `Q8_0`)

Measured by `Test_MetalFFNQuantized` (`hidden=2048, intermediate=8192`, 3 projections, vs unquantized FP64 FFN reference):

| DType | Block Bytes / 32 | Compression vs FP32 | FFN Weight Bytes | M=1 SQNR | M=29 SQNR | M=1 Max Abs Err | M=29 Max Abs Err |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q8_0` | 34 B | 3.76× | 53,477,376 B | `43.64 dB` | `43.68 dB` | `0.00231` | `0.00284` |
| **`Q4_1`** | **20 B** | **6.40×** | **31,457,280 B** | **`19.44 dB`** | **`19.49 dB`** | **`0.03466`** | **`0.04651`** |
| `Q4_0` | 18 B | 7.11× | 28,311,552 B | `19.05 dB` | `18.90 dB` | `0.04332` | `0.05349` |

`Q4_1` achieves **+0.39 to +0.59 dB higher SQNR** and **13–20% lower max absolute error** than `Q4_0` even on zero-centered synthetic FFN weights (and larger gains on asymmetric real LLM blocks).

### 2. Independent FP64 Dequantized-Weight Accuracy (`Q4_1`)

Measured by `Test_MetalTensorMatMulQuant/q4_1` against an independent FP64 dot-product reference over `DequantizeQ4_1` weights:

| Shape | Kernel Path | p50 abs err | p95 abs err | p99 abs err | max abs err |
| --- | --- | ---: | ---: | ---: | ---: |
| `M=1, K=256, N=33` | `M=1` SIMD GEMV | `3.69e-07` | `8.63e-07` | `9.41e-07` | `9.42e-07` |
| `M=3, K=96, N=17` | Scalar fallback | `1.11e-07` | `3.79e-07` | `5.66e-07` | `5.93e-07` |
| `M=32, K=256, N=64` | `simdgroup_float8x8` tiled | `7.81e-07` | `3.05e-06` | `4.78e-06` | `8.04e-06` |
| `M=1, K=2048, N=8192` | `M=1` SIMD GEMV | `1.05e-06` | `3.12e-06` | `4.20e-06` | `7.09e-06` |
| `M=1, K=8192, N=2048` | `M=1` SIMD GEMV | `3.39e-06` | `9.58e-06` | `1.29e-05` | `1.84e-05` |
| `M=1, K=2048, N=128256` | `M=1` SIMD GEMV | `1.04e-06` | `3.12e-06` | `4.14e-06` | `8.04e-06` |
| `M=29, K=2048, N=8192` | `simdgroup_float8x8` tiled | `5.99e-06` | `2.58e-05` | `4.10e-05` | `1.16e-04` |
| `M=29, K=8192, N=2048` | `simdgroup_float8x8` tiled | `2.39e-05` | `1.03e-04` | `1.66e-04` | `4.51e-04` |

`Test_MetalGatherRowsQuantizedBitExact/q4_1` and `Test_MetalGatherRowsQuantizedSpecialScales/q4_1` verify **bit-exact** `float32` agreement (`math.Float32bits`) between GPU `GatherRows` (`Q4_1 → Float32`) and host `DequantizeQ4_1`, and byte-exact `Q4_1 → Q4_1` blit gathers, including `M=600` multi-chunk gathers and special `d`/`m` combinations (`d=0` with non-zero `m`, negative `m`, subnormal, max FP16, `NaN`, `±Inf`).

## Apple M3 Benchmark Results

Raw logs stored under `.benchmarks/gpu-nn-25/run-1/bench.txt` (gitignored, `-benchtime=500ms`).

### `M == 1` Decode Projections (`Benchmark_MatMulRightTransposeMixedMetal`)

| Projection | `BFloat16` | `Q8_0` | `Q4_0` | **`Q4_1`** | `Q4_1` Weight Bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `FFN_GateUp` `[1,2048] × [8192,2048]^T` | 0.791 ms | 0.540 ms | 0.577 ms | **0.617 ms** | 10.0 MiB |
| `FFN_Down` `[1,8192] × [2048,8192]^T` | 0.653 ms | 0.495 ms | 0.527 ms | **0.575 ms** | 10.0 MiB |
| `Attn_QO` `[1,2048] × [2048,2048]^T` | 0.509 ms | 0.465 ms | 0.480 ms | **0.440 ms** | 2.5 MiB |
| `Attn_KV` `[1,2048] × [512,2048]^T` | 0.333 ms | 0.401 ms | 0.349 ms | **0.353 ms** | 0.625 MiB |
| `Classifier` `[1,2048] × [128256,2048]^T` | 6.293 ms | 3.503 ms | 1.952 ms | **2.114 ms** (`77.9 GB/s`) | 156.6 MiB |

### Multi-Row `M > 1` Tiled GEMM (`Benchmark_MatMulQuantBatchMetal`, `K=2048, N=8192`)

| Batch (`M`) | `Q8_0` | `Q4_0` | **`Q4_1`** | `Q4_1` Effective GFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 2 (fallback) | 0.490 ms | 0.806 ms | **0.749 ms** | 89.6 GFLOP/s |
| 32 (tiled) | 0.895 ms | 1.322 ms | **0.868 ms** | 1,236 GFLOP/s |
| 128 (tiled) | 2.608 ms | 2.346 ms | **2.307 ms** | 1,862 GFLOP/s |
| 256 (tiled) | 4.216 ms | 4.186 ms | **4.177 ms** | 2,056 GFLOP/s |

### Full Resident FFN (`Benchmark_FFNMixedMetal`, `hidden=2048, intermediate=8192`)

| Batch (`M`) | `BFloat16` | `Q8_0` | `Q4_0` | **`Q4_1`** | `Q4_1` Weight Bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `M=1` | 1.456 ms | 1.423 ms | 0.638 ms | **1.270 ms** | 30.0 MiB |
| `M=29` | 2.722 ms | 2.185 ms | 2.668 ms | **2.208 ms** | 30.0 MiB |

### Tied `[128256, 2048]` Embedding & Classifier (`Benchmark_TiedEmbeddingClassifierMetal`)

| Operation | `BFloat16` (501.0 MiB) | `Q8_0` (266.2 MiB) | `Q4_0` (140.9 MiB) | **`Q4_1` (156.6 MiB)** |
| --- | ---: | ---: | ---: | ---: |
| `GatherRows_M1` | 0.133 ms | 0.159 ms | 0.121 ms | **0.143 ms** |
| `GatherRows_M29` | 0.159 ms | 0.158 ms | 0.157 ms | **0.193 ms** |
| `Classifier_M1` | 6.298 ms | 3.588 ms | 1.920 ms | **2.114 ms** |
| `TiedGatherAndClassifier_M1` (1 `Scope`) | 6.385 ms | 3.532 ms | 1.965 ms | **2.162 ms** (`2.95×` vs BF16) |

---

## `GATE-I` Release Readiness Checklist (`v0.15.0` — Covering `NN-24` & `NN-25`)

- [x] **NN-24 — In-Place `KVCache` Append** (`gpu-nn-24`, PR #165, `d2eef0f05b76e0c6ca871b49233ce2ed9ad693b7`):
  - Exact **−50% KV device memory** and buffer count (1 `MTLBuffer` per `KVCache` verified via `ResourceSnapshot`).
  - `O(1)` 16-layer `AppendKV` latency across `H = 38..4096`: **0.165–0.184 ms** (FP32, up to **40.71×** faster at `H = 4096`) and **0.216–0.231 ms** (FP16, up to **16.66×** faster at `H = 4096`).
  - Zero public API changes; committed prefix `[0..length)` preserved across failed/discarded/canceled appends; `KVCache.Reset()` enforces `ErrBusy` against pending appends and active readers.
  - Long-context `H = 2048, 4096` FP64 parity verified (`max abs err <= 3.34e-9`).
- [x] **NN-25 — Asymmetric `Q4_1` Block Quantization** (`gpu-nn-25`):
  - GGUF-compatible `Q4_1` (`20 B / 32`, `6.40×` compression, naturally 4-byte aligned) with deterministic `QuantizeQ4_1` / `DequantizeQ4_1`, `UploadFloat32As` / `DownloadFloat32Into` / `UploadBytes` / `DownloadBytesInto`, `Copy`, `MatMulRightTranspose` (`M=1` GEMV, fallback, `M >= 8 && N >= 8` `simdgroup_float8x8` tiled GEMM), and `GatherRows` (`Q4_1 → Float32` bit-exact and `Q4_1 → Q4_1` blit).
  - Higher SQNR than `Q4_0` on synthetic FFN (`19.44–19.50 dB` vs `18.90–19.05 dB`) and near-`Q4_0` Apple M3 throughput (`2.162 ms` tied `[128256, 2048]` gather+classifier, `2,056 GFLOP/s` at `M=256`).
  - Quantization ADR updated (`docs/adr-gpu-quantization.md`).
- [x] **Backward Compatibility & Portability:**
  - All existing `Float32`, `Float16`, `BFloat16`, `Q8_0`, `Q4_0`, `KVCache`, ANN Metal (`Sequential`), CPU default, `purego`, `CGO_ENABLED=0`, and Linux `amd64` builds preserved and passing.
  - All new Metal pipelines compile lazily via `pthread_once` with `[options setFastMathEnabled:NO]` and FP32 accumulation.
  - Zero native buffer or scope leaks verified via `internal/device.ResourceSnapshot`; 8-worker `-race -tags=metal` concurrency stress tests pass.
- [x] **14-Entry Required Test Matrix:** All 14 entries verified green on `gpu-nn-25` (including `-race -tags=metal`, `CGO_ENABLED=0`, `purego`, and `GOOS=linux GOARCH=amd64`).
- [ ] **Owner Action (GATE-I):** Review and merge PR #165 (`NN-24`) and NN-25 PR, tag immutable release `v0.15.0`, and record commit SHA and `go.sum` module checksums in `TODO.md`.
