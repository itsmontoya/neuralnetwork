# NN-30: Rank-2 `Float32 × Float32` SIMD GEMV (`M == 1`) & `simdgroup_float8x8` Tiled GEMM (`M > 1`)

**Date:** 2026-11-27  
**Repository:** `github.com/itsmontoya/neuralnetwork`  
**Hardware:** Apple M3 (`darwin/arm64`, 8-core CPU / 10-core GPU, unified memory)  
**Base:** `gpu-nn-29` @ `6ff23b96af10511612b38f939726e7e93b4e780b` (PR #170)  
**Public API changes:** None  
**Raw benchmark logs:** `.benchmarks/gpu-nn-30/run1/{before*,after*}.txt` (gitignored)

---

## 1. Summary

`NN-30` accelerates rank-2 `Float32 × Float32` `Scope.MatMul` and `Scope.MatMulRightTranspose` by replacing the legacy 1-thread-per-output scalar `nn_matmul` dispatch on tensor scopes (`tensorPipeline == 1`) with:

- **`M == 1` `MatMulRightTranspose` SIMD GEMV (`nn_matmul_f32_row_right_transpose`)**: 4 SIMD groups (128 threads, 2 output rows/SIMD), coalesced `float4` loads along `K`, FP32 `fma` + `simd_sum` reduction, and an exact scalar tail for `K % 4 != 0`.
- **`M > 1` `simdgroup_float8x8` tiled GEMM (`nn_matmul_f32_tiled_nn` & `nn_matmul_f32_tiled_nt`)**: `(BM=32, BN=64, BK=32)`, 4 SIMD groups (128 threads), coalesced `float4` loads into bank-conflict-padded `threadgroup float` shared memory, and dual `k >= K` zero-padding.

All new kernels live in `internal/device/matmul_f32_metal_source.h`, compile lazily via `pthread_once` with `[options setFastMathEnabled:NO]`, acquire their encoder through `nn_metal_scope_compute_encoder` without calling `[encoder endEncoding]` (preserving `NN-29` encoder coalescing), keep `nn_matmul` as the tiny-shape fallback, and leave ANN `matrix` / `model.Sequential` (`nnMetalMatMulPipeline`, `tensorPipeline == 0`) completely untouched.

- **Peak FP32 GEMV bandwidth:** **86.4 GB/s** (`Classifier_M1_K2048_N128256`), with **1.20×–4.59×** speedups across Llama 3.2 1B single-row decode projections.
- **Peak FP32 GEMM throughput:** **1.66 TFLOP/s** (`MatMulRT_M256_K2048_N8192`), with **2.19×–8.28×** speedups across `M=4..256` `MatMul` and `MatMulRightTranspose`.
- **End-to-end resident FP32 speedups:** **3.65×–3.92×** on `Benchmark_FFNMixedMetal` (`M=27..29`), **1.17×** on `M=1` FFN, **3.88×** on `Benchmark_TransformerBlockMixedMetal/M29_H0/FP32_KV32`, and **1.23×** on `M1_H511/FP32_KV32`.

---

## 2. Kernel Architecture & Dispatch Thresholds

### Tile Geometry (`nn_matmul_f32_tiled_nn`, `nn_matmul_f32_tiled_nt`)

| Parameter | Value | Rationale |
| --- | --- | --- |
| Threadgroup tile `(BM, BN, BK)` | `(32, 64, 32)` | Matches `NN-21`/`NN-22`/`NN-23`; zero wasted M-rows at `M=32, 64, 128, 256` and balanced shared-memory footprint. |
| Threads per threadgroup | `128` (`4` SIMD groups, `2×2` grid) | Each SIMD group owns a `16×32` sub-tile (`2×4 = 8` `simdgroup_float8x8` accumulators). |
| `As` shared memory | `32 × 36` `float` (`4,608 B`) | Stride `36 = 32 + 4` maintains 16-byte `float4` alignment and avoids bank conflicts. |
| `Bs` shared memory (`nn` / `nt`) | `32 × 68` (`8,704 B`) / `64 × 36` (`9,216 B`) | Coalesced `float4` loads along `N` (`nn`) or `K` (`nt`, loaded with `simdgroup_load(..., transpose=true)`). |
| Output writeback | Interior: direct `simdgroup_store`; Edge: staged via reused `Bs` (`32 × 64`) | Zero extra threadgroup memory for partial boundary tiles. |

### Dispatch Thresholds (`internal/device/metal_backend.m`)

```c
int useRowRT = (dimensions.resultRows == 1 && dimensions.variant == 2);
int useTiled = (!useRowRT &&
    (dimensions.variant == 0 || dimensions.variant == 2) &&
    dimensions.resultCols >= 8 &&
    dimensions.leftCols >= 8 &&
    dimensions.resultRows >= 4);
```

- **Why `M >= 4` for both `NN` (`variant == 0`) and `NT` (`variant == 2`):** Unlike `Float16`/`BFloat16` (whose fallback had a vectorized `half4`/`bfloat4` loop), the legacy FP32 `nn_matmul` fallback is pure scalar for both orientations. On Apple M3, `nn_matmul_f32_tiled` wins decisively starting at `M = 4` for both `MatMul` (**3.07×**) and `MatMulRightTranspose` (**2.69×**), and at `M = 6` (`FFN_GateUp_M6_K2048_N8192`: **1.25×**; `MatMulRT_M6_K2048_N2048`: **2.19×**).
- **Fallback (`nn_matmul`):** Handles `M == 1` non-transposed (`variant == 0`), `M ∈ {2, 3}`, and tiny `N < 8` or `K < 8`.

### Special-Value Safety (`NaN`, `±Inf`, `0 * Inf`)

Out-of-bounds `k >= K` lanes are explicitly zero-padded in **both** `As` and `Bs` during every K-tile load, ensuring `0.0f * 0.0f = 0.0f` and preventing padded zeros from multiplying an in-bounds `±Inf` or `NaN`. All 9 special-value cases (`nan`, `±inf`, `0*inf`, opposing infinities, large finite, finite cancellation, small normal, overflow) pass across `M=1` GEMV (scalar and `float4`), `M=2` fallback, `8×9×9` partial tiles, and `33×35×65` multi-tile boundaries for both `MatMul` and `MatMulRightTranspose`.

---

## 3. Before vs After Apple M3 Benchmarks

Methodology: Paired runs before vs after kernel changes via `CGO_ENABLED=1 GOWORK=off go test -tags=metal -run='^$' -bench=... -benchmem -benchtime=400ms -count=3 ./gpu`. Tables report median latency across 3 samples.

### A. `M == 1` `MatMulRightTranspose` SIMD GEMV (`Benchmark_MatMulRightTransposeMixedMetal/.*/float32`)

| Shape | Before (`ms/op`) | After (`ms/op`) | Speedup | Before GB/s | After GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Attn_KV_M1_K2048_N512` | 1.123 | 0.244 | **4.59×** | 3.7 | 17.2 |
| `Attn_QO_M1_K2048_N2048` | 1.051 | 0.408 | **2.58×** | 16.0 | 41.1 |
| `FFN_Down_M1_K8192_N2048` | 2.051 | 1.054 | **1.95×** | 32.7 | 63.7 |
| `FFN_GateUp_M1_K2048_N8192` | 1.262 | 1.049 | **1.20×** | 53.2 | 64.0 |
| `Classifier_M1_K2048_N128256` | 12.576 | 12.166 | **1.03×** | 83.6 | **86.4** |
| `OddTail_M1_K257_N33` | 0.416 | 0.117 | **3.55×** | 0.1 | 0.3 |

### B. `M = 4..256` `simdgroup_float8x8` Tiled GEMM (`Benchmark_MatMulFloat32BatchMetal`)

| Shape & Orientation | Before (`ms/op`) | After (`ms/op`) | Speedup | Before GFLOP/s | After GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `MatMul_M4_K2048_N2048` | 1.466 | 0.477 | **3.07×** | 22.9 | 70.3 |
| `MatMul_M6_K2048_N2048` | 1.446 | 0.477 | **3.03×** | 34.8 | 105.5 |
| `MatMul_M8_K2048_N2048` | 1.555 | 0.478 | **3.25×** | 43.1 | 140.3 |
| `MatMul_M29_K2048_N2048` | 2.416 | 0.494 | **4.89×** | 100.7 | 492.1 |
| `MatMul_M32_K2048_N2048` | 2.327 | 0.492 | **4.73×** | 115.3 | 545.0 |
| `MatMul_M64_K2048_N2048` | 2.821 | 0.735 | **3.84×** | 190.3 | 730.5 |
| `MatMul_M128_K2048_N2048` | 4.769 | 0.961 | **4.96×** | 225.1 | **1,117** |
| `MatMul_M256_K2048_N2048` | 8.604 | 1.436 | **5.99×** | 249.5 | **1,495** |
| `MatMulRT_M4_K2048_N2048` | 1.275 | 0.475 | **2.69×** | 26.3 | 70.7 |
| `MatMulRT_M6_K2048_N2048` | 1.051 | 0.481 | **2.19×** | 47.9 | 104.6 |
| `MatMulRT_M8_K2048_N2048` | 1.396 | 0.481 | **2.90×** | 48.1 | 139.4 |
| `MatMulRT_M29_K2048_N2048` | 2.055 | 0.500 | **4.11×** | 118.3 | 486.1 |
| `MatMulRT_M32_K2048_N2048` | 2.088 | 0.504 | **4.14×** | 128.5 | 532.4 |
| `MatMulRT_M64_K2048_N2048` | 3.077 | 0.745 | **4.13×** | 174.5 | 720.7 |
| `MatMulRT_M128_K2048_N2048` | 5.758 | 0.973 | **5.92×** | 186.4 | **1,103** |
| `MatMulRT_M256_K2048_N2048` | 11.111 | 1.634 | **6.80×** | 193.3 | **1,314** |
| `MatMulRT_M32_K2048_N8192` | 5.764 | 1.173 | **4.91×** | 186.3 | 915.0 |
| `MatMulRT_M128_K2048_N8192` | 21.652 | 2.891 | **7.49×** | 198.3 | **1,485** |
| `MatMulRT_M256_K2048_N8192` | 42.962 | 5.189 | **8.28×** | 199.9 | **1,655** |

### C. Full Resident FFN & Transformer Block (`Float32`)

| Benchmark | Before (`ms/op`) | After (`ms/op`) | Speedup |
| --- | ---: | ---: | ---: |
| `Benchmark_FFNMixedMetal/M1/float32` | 3.031 | 2.601 | **1.17×** |
| `Benchmark_FFNMixedMetal/M27/float32` | 12.277 | 3.365 | **3.65×** |
| `Benchmark_FFNMixedMetal/M28/float32` | 12.331 | 3.318 | **3.72×** |
| `Benchmark_FFNMixedMetal/M29/float32` | 13.250 | 3.381 | **3.92×** |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/FP32_KV32` | 3.908 | 3.181 | **1.23×** |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/FP32_KV32` | 17.187 | 4.433 | **3.88×** |

---

## 4. Numerical Accuracy & Concurrency Verification

All shapes were verified against independent FP64 references in `gpu/matmul_metal_test.go` (`Test_MetalTensorMatMulFloat32Paths`, `Test_MetalTensorMatMulTiledSpecialValues`, `Test_MetalTensorMatMulFloat32ConcurrentStress`) with `METAL_DEVICE_WRAPPER_TYPE=1` and `-race -tags=metal` (zero validation warnings, zero leaks):

| Shape & Path | Outputs | p50 abs err | p95 abs err | p99 abs err | max abs err |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1×32×32` (`rt_true`, GEMV `float4`) | 32 | `8.03e-08` | `2.01e-07` | `2.04e-07` | `3.00e-07` |
| `1×33×33` (`rt_true`, GEMV scalar tail) | 33 | `7.20e-08` | `2.51e-07` | `2.71e-07` | `4.50e-07` |
| `3×33×33` (fallback `nn_matmul`) | 99 | `1.14e-07` | `4.47e-07` | `7.25e-07` | `8.09e-07` |
| `4×16×16` (tiled min `NN`/`NT`) | 64 | `4.86e-08` | `1.81e-07` | `2.29e-07` | `3.30e-07` |
| `29×768×512` (tiled `NN`/`NT`) | 14,848 | `2.19e-06` | `9.61e-06` | `1.58e-05` | `3.59e-05` |
| `33×33×65` (tiled boundary `NN`/`NT`) | 2,145 | `9.58e-08` | `3.95e-07` | `6.80e-07` | `1.42e-06` |
| `64×257×129` (tiled odd K/N) | 8,256 | `7.37e-07` | `3.22e-06` | `5.16e-06` | `1.19e-05` |
| `128×64×128` (tiled interior) | 16,384 | `1.87e-07` | `7.97e-07` | `1.27e-06` | `3.07e-06` |
| `32×2048×2048` (tiled Llama QO) | 65,536 | `5.87e-06` | `2.57e-05` | `4.12e-05` | `1.24e-04` |
| `4×8192×512` (tiled long K) | 2,048 | `2.29e-05` | `9.89e-05` | `1.65e-04` | `2.49e-04` |
| `4×2048×8192` (tiled Llama GateUp) | 32,768 | `5.95e-06` | `2.58e-05` | `4.20e-05` | `1.06e-04` |

---

## 5. Handoff Notes for NN-31

- Rank-2 `Float32` matmul pipelines are now compiled lazily via `nnMetalTensorMatMulOnce` into `nnMetalTensorMatMulPipelines[NNMetalMatMulF32Count]` and share the coalesced `MTLComputeCommandEncoder` from `NN-29`.
- For `NN-31` (`Scope.Concat`), remember:
  - The `outer == 1` blit fast path must call `nn_metal_scope_end_compute_encoder(scopeRecord)` before creating an `MTLBlitCommandEncoder`.
  - The `outer > 1` strided compute path (`nn_tensor_concat_u32` / `nn_tensor_concat_u16`) must acquire its encoder via `nn_metal_scope_compute_encoder(scopeRecord, context)` and must **not** call `[encoder endEncoding]`.
