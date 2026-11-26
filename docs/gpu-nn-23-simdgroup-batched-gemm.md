# NN-23: `simdgroup_float8x8` Tiled GEMM for `BatchedMatMul`, `BatchedMatMulRightTranspose`, & `MatMulLeftTranspose`

**Date:** 2026-11-26  
**Hardware:** Apple M3 (arm64, 8-core CPU, 10-core GPU, unified memory)  
**Baseline:** `gpu-nn-22` (`33f250f6fb5f50eaf74489680e8d8d93a3c3f8d3`)  
**Raw benchmark logs:** `.benchmarks/gpu-nn-23/run1/` (`before.txt`, `after.txt`, `sweep_after.txt`, `fp64_tests.txt`)

---

## 1. Summary

`NN-23` accelerates `Scope.BatchedMatMul`, `Scope.BatchedMatMulRightTranspose`, and `Scope.MatMulLeftTranspose` across `Float32`, `Float16`, and `BFloat16` right operands with Apple Silicon `simdgroup_float8x8` tiled GEMM kernels compiled lazily via `pthread_once` with `[options setFastMathEnabled:NO]` and FP32 accumulation.

- **Public API:** Zero changes.
- **Preserved untouched:** Existing 1-thread-per-output kernels (`nn_batched_matmul_{f32,f16,bf16}`, `nn_matmul_left_transpose_{f32,f16,bf16}`) remain as the clean fallback for tiny shapes; rank-2 `MatMul`/`MatMulRightTranspose` and `M == 1` GEMV paths are untouched.
- **End-to-end BERT-base Transformer encoder block (`B=2, S=64, H=12, D=64, E=768, FFN=3072`, 1 `gpu.Scope`):** **`14.28–14.43 ms` → `1.60–1.65 ms`** (**8.68×–8.94× speedup** across `Float32`, `Float16`, and `BFloat16` weights).
- **Isolated projections:** **4.69×–4.87×** on `SharedProj_2x64x768_x_768x768`, **2.53×–2.74×** on `MatMulLeftTranspose` (`K=768, M=32, N=768`), and up to **1.07 TFLOP/s** FP32-accumulated on `LeftT_K768_M128_N3072` and `SharedProj_2x64x768_x_768x3072`.

---

## 2. Tile Geometry & Kernel Architecture

Implemented in `internal/device/batched_matmul_metal_source.h` via two shared templates instantiated into nine Metal compute kernels:

- **Left-transposed GEMM (`nn_matmul_left_transpose_tiled_impl<RightT, RightVec4T>`):**
  - `nn_matmul_left_transpose_f32_tiled`, `nn_matmul_left_transpose_f16_tiled`, `nn_matmul_left_transpose_bf16_tiled`
- **3D+ broadcast-batched GEMM (`nn_batched_matmul_tiled_impl<RightT, RightVec4T, RightTranspose>`):**
  - `nn_batched_matmul_f32_tiled_nn`, `nn_batched_matmul_f32_tiled_nt`
  - `nn_batched_matmul_f16_tiled_nn`, `nn_batched_matmul_f16_tiled_nt`
  - `nn_batched_matmul_bf16_tiled_nn`, `nn_batched_matmul_bf16_tiled_nt`

### Tile Dimensions & Threadgroup Layout

| Parameter | Value | Rationale |
| --- | --- | --- |
| Threadgroup tile `(BM, BN, BK)` | `(32, 64, 32)` | Reuses the proven `NN-21`/`NN-22` layout: zero wasted M-rows at `M=32, 64, 128, 256`; balanced shared-memory footprint and register pressure. |
| Threads per threadgroup | `128` (`4` SIMD groups, `2×2` grid) | Each SIMD group owns a `16×32` sub-tile (`2×4 = 8` `simdgroup_float8x8` accumulators). |
| 3D batch grid (`BatchedMatMul`) | `(ceil(N/64), ceil(M/32), totalBatches)` | Computes `leftBase` and `rightBase` once per threadgroup via `nn_batched_offsets(params, tgid.z, leftBase, rightBase)`, supporting 1..4D batch broadcasting. |
| `As` shared memory (`BatchedMatMul`) | `32 × 36` `float` (`4,608 B`) | Coalesced `float4` loads along `K` when `K % 4 == 0`; `simdgroup_load(..., transpose=false)`. |
| `As` shared memory (`MatMulLeftTranspose`) | `32 × 36` `float` (`4,608 B`, `BK × BM`) | Coalesced `float4` loads from `left[K, M]` along `M` when `M % 4 == 0`; hardware-transposed `simdgroup_load(..., transpose=true)` feeds `aFrag[2]` directly. |
| `Bs` shared memory (`NN` / `LeftT`) | `32 × 68` `float` (`8,704 B`) | Coalesced `float4`/`half4`/`bfloat4` loads along `N` when `N % 4 == 0`; `simdgroup_load(..., transpose=false)`. |
| `Bs` shared memory (`RT`) | `64 × 36` `float` (`9,216 B`) | Coalesced `float4`/`half4`/`bfloat4` loads along `K` when `K % 4 == 0`; hardware-transposed `simdgroup_load(..., transpose=true)`. |
| Total threadgroup memory | `13,312 B` (`NN`/`LeftT`) / `13,824 B` (`RT`) | Fits 2 resident threadgroups per 32 KB threadgroup memory. |
| Output store | Interior: direct `simdgroup_store` to `outMat`; Edge: stage through reused `Bs` (`32 × 64` `float`) | Zero extra shared-memory allocation for partial boundary tiles (`M % 32 != 0` or `N % 64 != 0`). |

### Dispatch Thresholds (`internal/device/metal_backend.m`)

```c
// MatMulLeftTranspose (scalar fallback)
int useTiled = (m >= 4 && n >= 8 && k >= 8);

// BatchedMatMul / BatchedMatMulRightTranspose (matches NN-21)
int useTiled = (params.n >= 8 && params.k >= 8 &&
    (params.rightTransposed != 0 ? params.m >= 8 : params.m >= 4));
```

- Standard `BatchedMatMul` (`rightTransposed == 0`) and `MatMulLeftTranspose` have unvectorized scalar fallbacks, so tiled execution wins starting at `M >= 4, N >= 8, K >= 8`.
- `BatchedMatMulRightTranspose` (`rightTransposed != 0`) has a vectorized `float4`/`half4`/`bfloat4` fallback, so `M >= 8, N >= 8, K >= 8` (matching `NN-21`) avoids `BM = 32` underutilization on tiny `M < 8`.

### Special-Value Safety (`NaN`, `±Inf`, `0 * Inf`)

Out-of-bounds `k >= K` lanes are explicitly zero-padded in **both** `As` and `Bs` during every K-tile load, guaranteeing `0.0f * 0.0f = 0.0f` and preventing padded zeros from ever multiplying an in-bounds `±Inf` or `NaN`. Out-of-bounds `row >= M` and `col >= N` outputs are masked during edge-tile writeback.

---

## 3. Before vs After Apple M3 Benchmarks

### A. Paired Before vs After (`Benchmark_MatMulLeftTransposeMetal`, `Benchmark_BatchedMatMulMetal`, `Benchmark_ONNXTransformerEncoderBlockMetal`)

| Benchmark | DType | Before (`ms/op`) | After (`ms/op`) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `MatMulLeftTranspose/K768_M32_N768` | `float32` | 0.736 | 0.276 | **2.67×** |
| `MatMulLeftTranspose/K768_M32_N768` | `float16` | 0.668 | 0.264 | **2.53×** |
| `MatMulLeftTranspose/K768_M32_N768` | `bfloat16` | 0.720 | 0.263 | **2.74×** |
| `BatchedMatMul/AttnScores_2x12x64x64_RT` | `float32` | 0.237 | 0.198 | **1.19×** |
| `BatchedMatMul/AttnContext_2x12x64x64` | `float32` | 0.288 | 0.195 | **1.48×** |
| `BatchedMatMul/SharedProj_2x64x768_x_768x768` | `float32` | 1.393 | 0.297 | **4.69×** |
| `BatchedMatMul/SharedProj_2x64x768_x_768x768` | `float16` | 1.415 | 0.291 | **4.87×** |
| `BatchedMatMul/SharedProj_2x64x768_x_768x768` | `bfloat16` | 1.391 | 0.291 | **4.78×** |
| `ONNXTransformerEncoderBlock/BERTBase_B2_S64_E768` | `float32` | 14.28 | 1.646 | **8.68×** |
| `ONNXTransformerEncoderBlock/BERTBase_B2_S64_E768` | `float16` | 14.43 | 1.614 | **8.94×** |
| `ONNXTransformerEncoderBlock/BERTBase_B2_S64_E768` | `bfloat16` | 14.29 | 1.604 | **8.91×** |

*Note: Single-dispatch `AttnScores_2x12x64x64_RT` and `AttnContext_2x12x64x64` now execute near the ~0.18–0.19 ms single-scope Metal submit+wait floor; inside the 1-`Scope` BERT-base block where 22 ops share one command buffer, all 10 batched matmuls run back-to-back without submit overhead, driving the 8.68×–8.94× end-to-end speedup.*

### B. Large-Shape Throughput Sweep (`Benchmark_BatchedMatMulTiledSweepMetal`)

| Shape & Operation | DType | After (`ms/op`) | Achieved Throughput |
| --- | --- | ---: | ---: |
| `LeftT_K768_M128_N3072` | `float32` | 0.627 | 963.8 GFLOP/s |
| `LeftT_K768_M128_N3072` | `float16` | 0.564 | **1,070 GFLOP/s** |
| `LeftT_K768_M128_N3072` | `bfloat16` | 0.575 | **1,050 GFLOP/s** |
| `SharedProj_2x64x768_x_768x3072` | `float32` | 0.624 | 968.7 GFLOP/s |
| `SharedProj_2x64x768_x_768x3072` | `float16` | 0.565 | **1,069 GFLOP/s** |
| `SharedProj_2x64x768_x_768x3072` | `bfloat16` | 0.569 | **1,062 GFLOP/s** |
| `SharedProj_2x64x3072_x_768x3072_RT` | `float32` | 0.635 | 951.1 GFLOP/s |
| `SharedProj_2x64x3072_x_768x3072_RT` | `float16` | 0.688 | 878.6 GFLOP/s |
| `SharedProj_2x64x3072_x_768x3072_RT` | `bfloat16` | 0.761 | 793.9 GFLOP/s |
| `AttnScores_2x12x256x64_RT` | `float32` | 0.379 | 531.8 GFLOP/s |
| `AttnContext_2x12x256x256` | `float32` | 0.393 | 512.1 GFLOP/s |

---

## 4. Numerical Verification (Independent FP64 Reference)

Representative error distributions from `.benchmarks/gpu-nn-23/run1/fp64_tests.txt`:

| Test / Shape | DType | p50 abs err | p95 abs err | p99 abs err | Max abs err |
| --- | --- | ---: | ---: | ---: | ---: |
| `MatMulLeftTranspose 128×65×128` | `float32` | `3.67e-07` | `1.56e-06` | `2.50e-06` | `6.95e-06` |
| `MatMulLeftTranspose 128×65×128` | `float16` | `3.70e-07` | `1.60e-06` | `2.60e-06` | `6.16e-06` |
| `MatMulLeftTranspose 128×65×128` | `bfloat16` | `3.69e-07` | `1.63e-06` | `2.58e-06` | `5.71e-06` |
| `MatMulLeftTranspose 129×33×65` | `float32` | `3.72e-07` | `1.59e-06` | `2.69e-06` | `4.35e-06` |
| `BatchedMatMul 3d_x_2d_large_edge_65x128x128` | `float32` | `3.80e-07` | `1.60e-06` | `2.53e-06` | `5.89e-06` |
| `BatchedMatMul 3d_x_2d_large_edge_65x128x128` | `float16` | `3.72e-07` | `1.63e-06` | `2.60e-06` | `7.18e-06` |
| `BatchedMatMul 3d_x_2d_large_edge_65x128x128` | `bfloat16` | `3.76e-07` | `1.65e-06` | `2.59e-06` | `6.41e-06` |
| `BatchedMatMul 4d_attention_scores` | `float32` | `1.88e-07` | `8.04e-07` | `1.30e-06` | `3.67e-06` |
| `BatchedMatMul 4d_attention_scores_rt` | `float32` | `1.91e-07` | `8.34e-07` | `1.36e-06` | `2.86e-06` |
| `BatchedMatMul 6d_broadcast_tiled` | `float16` | `5.27e-08` | `2.20e-07` | `3.63e-07` | `1.30e-06` |
| `BatchedMatMul 6d_broadcast_tiled_rt` | `bfloat16` | `5.16e-08` | `2.13e-07` | `3.54e-07` | `8.39e-07` |
| `ONNX Transformer Encoder Block (1 Scope)` | `float32` | `4.68e-08` | `1.44e-07` | `2.03e-07` | `3.18e-07` |
| `ONNX Transformer Encoder Block (1 Scope)` | `float16` | `4.81e-08` | `1.48e-07` | `2.06e-07` | `3.18e-07` |
| `ONNX Transformer Encoder Block (1 Scope)` | `bfloat16` | `4.82e-08` | `1.49e-07` | `2.09e-07` | `3.11e-07` |

Additional verification in `gpu/transpose_batched_matmul_metal_test.go`:
- `Test_MetalMatMulLeftTransposeAgainstFP64`: 11 shapes (`1, 3, 4, 5, 7, 8, 15, 16, 17, 19, 23, 31, 32, 33, 64, 65, 128, 129`) × 3 dtypes (`Float32`, `Float16`, `BFloat16`).
- `Test_MetalMatMulLeftTransposeSpecialValues`: `NaN`, `+Inf`, `0*Inf`, opposing `±Inf`, and edge-element checks on `fallback_2x3x3`, `tiled_partial_9x8x9`, and `tiled_edge_35x33x65` × 3 dtypes.
- `Test_MetalBatchedMatMulAgainstFP64`: 24 shapes covering 2D/3D/4D/6D, broadcast batch dims, both `NN` and `RT`, fallback and tiled boundaries × 3 dtypes.
- `Test_MetalBatchedMatMulSpecialValues`: `NaN`, `+Inf`, `0*Inf`, opposing `±Inf`, and batch-1 edge checks on fallback, broadcast partial tiled, and multi-tile edge shapes (`NN` and `RT`) × 3 dtypes.
- `Test_MetalBatchedMatMulTiledConcurrentStress`: 8 concurrent workers × 16 iterations sharing immutable `Float32`/`Float16`/`BFloat16` weights on tiled edge shapes (`MatMulLeftTranspose`, `BatchedMatMul`, `BatchedMatMulRightTranspose`) under `-race -tags=metal` with zero resource leaks.

---

## 5. `GATE-H` Release Readiness Checklist (`NN-21`, `NN-22`, `NN-23`)

Phase E (`simdgroup_matrix` tiled `M > 1` GEMM) is ready for owner review, merge, and immutable release tagging (e.g., `v0.13.4` or `v0.14.0`):

- [x] **NN-21 (`gpu-nn-21`, PR #162):** `simdgroup_float8x8` tiled `M > 1` GEMM for `Float16` & `BFloat16` (`MatMul` & `MatMulRightTranspose`) — up to **1.60 TFLOP/s** FP32-accumulated on Apple M3; **3.3×–10.1×** on `MatMul`, **1.5×–2.9×** on `MatMulRightTranspose`, **2.0×–2.1×** on full resident Llama 3.2 1B FFN prefill (`M=27..29`); `M == 1` GEMV and `Float32 × Float32` untouched; report in `docs/gpu-nn-21-simdgroup-gemm-mixed.md`.
- [x] **NN-22 (`gpu-nn-22`, PR #163):** `simdgroup_float8x8` tiled `M > 1` GEMM for quantized `Q8_0` & `Q4_0` (`MatMulRightTranspose`, `BK = 32 == QuantBlockSize`) — up to **2.04 TFLOP/s** FP32-accumulated on Apple M3; **2.15×–5.00×** on batched quantized projections, **2.94×–3.48×** on full resident Llama 3.2 1B FFN prefill (`2.16–2.23 ms`, eliminating the NN-12 prefill inversion vs `FP16`/`BF16`); 2-byte alignment strictly preserved; `M == 1` GEMV untouched; report in `docs/gpu-nn-22-simdgroup-gemm-quant.md`.
- [x] **NN-23 (`gpu-nn-23`):** `simdgroup_float8x8` tiled GEMM for `BatchedMatMul`, `BatchedMatMulRightTranspose`, & `MatMulLeftTranspose` (`Float32`, `Float16`, `BFloat16`) — up to **1.07 TFLOP/s** FP32-accumulated; **4.69×–4.87×** on batched projections, **2.53×–2.74×** on `MatMulLeftTranspose`, and **8.68×–8.94×** on the full 1-`Scope` BERT-base Transformer encoder block (`14.28–14.43 ms` → `1.60–1.65 ms`); report in `docs/gpu-nn-23-simdgroup-batched-gemm.md`.
- [x] **Zero public API / backward-compatibility breaks:** All existing `gpu` v0.12.1–v0.13.3 APIs, validation contracts, error sentinels, and `M == 1` GEMV / `Float32 × Float32` rank-2 paths are preserved unchanged.
- [x] **Numerics & safety:** All Phase E Metal kernels compile lazily via `pthread_once` with `[options setFastMathEnabled:NO]` and FP32 accumulation; dual `k >= K` zero-padding preserves `NaN`, `±Inf`, and `0 * Inf` semantics; zero leaks and zero data races under 8-worker `-race -tags=metal` stress tests.
- [x] **Portability & full 14-entry test matrix:** Verified across default CPU, `-race`, `CGO_ENABLED=0`, `purego`, `metal`, `metal purego`, `GOOS=linux GOARCH=amd64`, and `CGO_ENABLED=1 -tags=metal` (with and without `-race`).
- [ ] **Owner action (not agent):** Review and merge PR #162 (`NN-21`), PR #163 (`NN-22`), and the `NN-23` PR; tag and publish immutable `neuralnetwork` release (`GATE-H`); record release tag, commit SHA, and module checksums in `TODO.md`.
