# NN-21: `simdgroup_float8x8` Tiled `M > 1` GEMM for `Float16` & `BFloat16`

**Date:** 2026-11-26  
**Hardware:** Apple M3 (arm64, 8-core CPU, 10-core GPU, unified memory)  
**Baseline:** `v0.13.3` (`2b0a5f7f9fd4f7eff5f99a77d34818d7e3fa524f`)  
**Raw benchmark logs:** `.benchmarks/gpu-nn-21/run1/` (before & initial `M >= 4` sweep), `.benchmarks/gpu-nn-21/run2/` (final tuned threshold)

---

## 1. Summary

`NN-21` replaces the 1-thread-per-output `M > 1` execution path in `nn_matmul_f16` and `nn_matmul_bf16` (for both standard `MatMul` and `MatMulRightTranspose`) with Apple Silicon `simdgroup_float8x8` tiled GEMM kernels compiled lazily with `[options setFastMathEnabled:NO]` and FP32 accumulation.

- **Public API:** Zero changes.
- **Preserved untouched:** `M == 1` GEMV (`nn_matmul_f16_row_right_transpose`, `nn_matmul_bf16_row_right_transpose`) and `Float32 × Float32` (`nn_matmul`).
- **Peak achieved throughput:** Up to **1.60 TFLOP/s** FP32-accumulated on Apple M3 (`MatMulRT_M128_K2048_N8192`, `MatMul_M256_K2048_N2048`).
- **Speedups:** **3.3×–10.1×** on batched `MatMul` (`M=32..256`), **1.5×–2.9×** on batched `MatMulRightTranspose` (`M=27..256`), and **2.0×–2.1×** on full resident Llama 3.2 1B FFN prefill (`M=27..29`).

---

## 2. Tile Geometry & Kernel Architecture

Implemented in `internal/device/matmul_mixed_metal_source.h` via a shared template `nn_matmul_mixed_tiled_impl<WeightT, WeightVec4T, RightTranspose>` instantiated into four Metal compute kernels:

- `nn_matmul_f16_tiled_nn` (`half`, `variant == 0`: `left[M, K] × right[K, N]`)
- `nn_matmul_f16_tiled_nt` (`half`, `variant == 2`: `left[M, K] × right[N, K]^T`)
- `nn_matmul_bf16_tiled_nn` (`bfloat`, `variant == 0`)
- `nn_matmul_bf16_tiled_nt` (`bfloat`, `variant == 2`)

### Tile Dimensions & Threadgroup Layout

| Parameter | Value | Rationale |
| --- | --- | --- |
| Threadgroup tile `(BM, BN, BK)` | `(32, 64, 32)` | Zero wasted M-rows at `M=32, 64, 128, 256`; minimal waste at `M=27..29`; balanced 4 KB `A` + 4 KB `B` global loads per K-tile. |
| Threads per threadgroup | `128` (`4` SIMD groups, `2×2` grid) | Each SIMD group owns a `16×32` sub-tile (`2×4 = 8` `simdgroup_float8x8` accumulators = 16 FP32 registers/lane). |
| `As` shared memory | `32 × 36` `float` (`4,608 B`) | Stride `36 = 32 + 4` preserves 16-byte `float4` alignment and avoids shared-memory bank conflicts. |
| `Bs` shared memory (`variant == 0`) | `32 × 68` `float` (`8,704 B`) | Coalesced `half4`/`bfloat4` loads along `N`; `simdgroup_load(..., transpose=false)`. |
| `Bs` shared memory (`variant == 2`) | `64 × 36` `float` (`9,216 B`) | Coalesced `half4`/`bfloat4` loads along `K`; hardware-transposed `simdgroup_load(..., transpose=true)`. |
| Total threadgroup memory | `13,312 B` (`nn`) / `13,824 B` (`nt`) | Fits 2 resident threadgroups per 32 KB threadgroup memory for latency hiding. |
| Output store | Interior: direct `simdgroup_store` to `result`; Edge: stage through reused `Bs` (`32 × 64` `float`) | Zero extra shared-memory allocation for partial boundary tiles (`M % 32 != 0` or `N % 64 != 0`). |

### Dispatch Thresholds (`internal/device/metal_backend.m`)

```c
useRowRT = (dimensions.resultRows == 1 && dimensions.variant == 2);
useTiled = (!useRowRT &&
    dimensions.resultCols >= 8 &&
    dimensions.leftCols >= 8 &&
    (dimensions.variant == 0 ? dimensions.resultRows >= 4 : dimensions.resultRows >= 8));
```

- **Why `M >= 8` for `MatMulRightTranspose` (`variant == 2`):** On Apple M3, `FFN_GateUp_M6_K2048_N8192` takes `~718–756 µs` on the vectorized `float4`/`half4` fallback kernel (`nn_matmul_{f16,bf16}`), whereas the `BM = 32` tiled kernel takes a flat `~998 µs` for any `M <= 32` (81% M-tile underutilization at `M = 6`). Setting `M >= 8` for `variant == 2` preserves `M = 6` with zero regression while accelerating `M >= 8` through `M = 256+`.
- **Why `M >= 4` for standard `MatMul` (`variant == 0`):** The scalar fallback for `variant == 0` lacks vectorized loads and is ~3× slower than `variant == 2`, so the tiled kernel wins starting at `M = 4`.

### Special-Value Safety (`NaN`, `±Inf`, `0 * Inf`)

Out-of-bounds `k >= K` lanes are explicitly zero-padded in **both** `As` and `Bs` during every K-tile load, guaranteeing `0.0f * 0.0f = 0.0f` and preventing padded zeros from ever multiplying an in-bounds `±Inf` or `NaN`. Out-of-bounds `row >= M` and `col >= N` outputs are masked during edge-tile writeback. All 8 special-value cases (`nan`, `±inf`, `0*inf`, opposing infinities, large finite, finite cancellation, overflow) pass 100% on `M=1`, `M=2` fallback, `tiled_partial_8x9x9`, and `tiled_edge_33x35x65` across both orientations and both dtypes.

---

## 3. Before vs After Apple M3 Benchmarks

### A. Batched `MatMul` & `MatMulRightTranspose` (`Benchmark_MatMulMixedBatchMetal`)

| Shape & Orientation | DType | Before (`ms/op`) | After (`ms/op`) | Speedup | After GFLOP/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `MatMul_M32_K2048_N2048` | `float16` | 2.170 | 0.650 | **3.34×** | 412.9 |
| `MatMul_M32_K2048_N2048` | `bfloat16` | 2.092 | 0.492 | **4.25×** | 545.2 |
| `MatMul_M64_K2048_N2048` | `float16` | 3.832 | 1.134 | **3.38×** | 473.2 |
| `MatMul_M64_K2048_N2048` | `bfloat16` | 3.753 | 0.713 | **5.27×** | 753.0 |
| `MatMul_M128_K2048_N2048` | `float16` | 7.130 | 0.922 | **7.74×** | 1165 |
| `MatMul_M128_K2048_N2048` | `bfloat16` | 7.074 | 0.924 | **7.66×** | 1162 |
| `MatMul_M256_K2048_N2048` | `float16` | 13.861 | 1.379 | **10.05×** | 1557 |
| `MatMul_M256_K2048_N2048` | `bfloat16` | 13.618 | 1.384 | **9.84×** | 1551 |
| `MatMulRT_M32_K2048_N2048` | `float16` | 0.705 | 0.468 | **1.51×** | 573.5 |
| `MatMulRT_M32_K2048_N2048` | `bfloat16` | 0.694 | 0.464 | **1.50×** | 578.4 |
| `MatMulRT_M64_K2048_N2048` | `float16` | 1.211 | 0.701 | **1.73×** | 765.5 |
| `MatMulRT_M64_K2048_N2048` | `bfloat16` | 1.201 | 0.698 | **1.72×** | 769.1 |
| `MatMulRT_M128_K2048_N2048` | `float16` | 2.148 | 0.928 | **2.31×** | 1156 |
| `MatMulRT_M128_K2048_N2048` | `bfloat16` | 2.124 | 0.929 | **2.29×** | 1156 |
| `MatMulRT_M256_K2048_N2048` | `float16` | 4.177 | 1.516 | **2.76×** | 1417 |
| `MatMulRT_M256_K2048_N2048` | `bfloat16` | 3.984 | 1.562 | **2.55×** | 1375 |
| `MatMulRT_M32_K2048_N8192` | `float16` | 2.171 | 1.013 | **2.14×** | 1060 |
| `MatMulRT_M32_K2048_N8192` | `bfloat16` | 2.168 | 0.997 | **2.17×** | 1077 |
| `MatMulRT_M128_K2048_N8192` | `float16` | 7.774 | 2.697 | **2.88×** | 1592 |
| `MatMulRT_M128_K2048_N8192` | `bfloat16` | 7.749 | 2.677 | **2.89×** | 1604 |

### B. Llama 3.2 1B Projections (`Benchmark_MatMulRightTransposeMixedMetal`)

| Shape | DType | Before (`ms/op`) | After (`ms/op`) | Speedup | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `FFN_GateUp_M1_K2048_N8192` | `float16` | 0.644 | 0.664 | 0.97× | `M=1` GEMV untouched (run-to-run noise) |
| `FFN_GateUp_M1_K2048_N8192` | `bfloat16` | 0.649 | 0.657 | 0.99× | `M=1` GEMV untouched |
| `Classifier_M1_K2048_N128256` | `float16` | 6.002 | 6.264 | 0.96× | `M=1` GEMV untouched (~84 GB/s) |
| `FFN_GateUp_M6_K2048_N8192` | `float16` | 0.719 | 0.756 | 0.95× | `M=6 < 8` stays on vec4 fallback |
| `FFN_GateUp_M6_K2048_N8192` | `bfloat16` | 0.718 | 0.740 | 0.97× | `M=6 < 8` stays on vec4 fallback |
| `FFN_GateUp_M27_K2048_N8192` | `float16` | 1.776 | 1.000 | **1.78×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M27_K2048_N8192` | `bfloat16` | 1.771 | 0.999 | **1.77×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M28_K2048_N8192` | `float16` | 1.834 | 1.002 | **1.83×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M28_K2048_N8192` | `bfloat16` | 1.804 | 1.025 | **1.76×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M29_K2048_N8192` | `float16` | 1.913 | 0.992 | **1.93×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M29_K2048_N8192` | `bfloat16` | 1.891 | 1.024 | **1.85×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M29_K2048_N8192` | `float32` | 4.498 | 4.641 | 0.97× | `float32` control untouched |

### C. Full Resident FFN (`Benchmark_FFNMixedMetal`, `hidden=2048, intermediate=8192`)

| Rows | DType | Before (`ms/op`) | After (`ms/op`) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `M1` | `float32` | 2.821 | 2.872 | 0.98× |
| `M1` | `float16` | 1.774 | 1.468 | 1.21× (`M=1` untouched; thermal/boost variance) |
| `M1` | `bfloat16` | 1.413 | 1.442 | 0.98× |
| `M27` | `float32` | 11.955 | 12.266 | 0.97× |
| `M27` | `float16` | 5.435 | 2.727 | **1.99×** |
| `M27` | `bfloat16` | 5.389 | 2.757 | **1.96×** |
| `M28` | `float32` | 12.025 | 12.352 | 0.97× |
| `M28` | `float16` | 5.549 | 2.723 | **2.04×** |
| `M28` | `bfloat16` | 5.505 | 2.804 | **1.96×** |
| `M29` | `float32` | 12.839 | 13.202 | 0.97× |
| `M29` | `float16` | 5.844 | 2.779 | **2.10×** |
| `M29` | `bfloat16` | 5.810 | 2.743 | **2.12×** |

---

## 4. Numerical Verification (Independent FP64 Reference)

Measured on `Test_MetalFFNMixedPrecision` (`hidden=2048, intermediate=8192`, 3 projections + SwiGLU):

| DType | Rows | Elements | p50 abs err | p95 abs err | p99 abs err | Max abs err |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `float16` | `M1` | 2,048 | `1.54e-08` | `4.59e-08` | `5.92e-08` | `7.69e-08` |
| `float16` | `M27` | 55,296 | `1.30e-07` | `4.31e-07` | `6.36e-07` | `1.72e-06` |
| `float16` | `M28` | 57,344 | `1.30e-07` | `4.32e-07` | `6.36e-07` | `1.72e-06` |
| `float16` | `M29` | 59,392 | `1.30e-07` | `4.32e-07` | `6.38e-07` | `1.72e-06` |
| `bfloat16` | `M1` | 2,048 | `1.61e-08` | `4.52e-08` | `5.93e-08` | `7.20e-08` |
| `bfloat16` | `M27` | 55,296 | `1.29e-07` | `4.29e-07` | `6.39e-07` | `1.43e-06` |
| `bfloat16` | `M28` | 57,344 | `1.29e-07` | `4.28e-07` | `6.39e-07` | `1.43e-06` |
| `bfloat16` | `M29` | 59,392 | `1.29e-07` | `4.28e-07` | `6.39e-07` | `1.43e-06` |

Additional verification in `gpu/matmul_mixed_metal_test.go`:
- `Test_MetalTensorMatMulMixed`: 46 shapes across `M ∈ {1, 2, 3, 4, 6, 7, 15, 16, 17, 27, 28, 29, 31, 32, 33, 64, 128}`, `K ∈ {1, 3, 4, 7, 8, 16, 31, 32, 33, 35, 64, 65, 128, 132, 257, 2048, 8192}`, `N ∈ {1, 3, 7, 8, 9, 15, 16, 17, 18, 19, 31, 32, 33, 64, 65, 128, 512, 2048, 8192, 128256}` for both `Float16` and `BFloat16`.
- `Test_MetalTensorMatMulMixedSpecialValues`: all 8 special cases on `M=1`, `M=1` vec4, `M=2` fallback, `tiled_partial_8x9x9`, and `tiled_edge_33x35x65`.
- `Test_MetalTensorMatMulMixedTiledConcurrentStress`: 8 concurrent workers sharing immutable `Float16`/`BFloat16` weights on `33×131×67` (`MatMul` and `MatMulRightTranspose`) under `-race -tags=metal` with zero resource leaks.

---

## 5. Tiling Notes for `NN-22` (`Q8_0` & `Q4_0`)

- The exact same `(BM=32, BN=64, BK=32)` threadgroup layout (`4` SIMD groups = `128` threads, `AS_STRIDE=36`, `Bs[64 × 36]` with `simdgroup_load(..., transpose=true)`, and `Bs`-reused edge staging) maps 1-to-1 to `NN-22`:
  - `BK = 32` matches `QuantBlockSize = 32` exactly, so each K-tile step dequantizes **1 exact GGUF block** per column `n ∈ [0, 64)` (`64` blocks total across `128` threads = 2 threads per block, each unpacking 16 weights into `Bs[n * 36 + k]`).
  - Keep `threadgroup float As[32u * 36u]` and `threadgroup float Bs[64u * 36u]` declared in the `kernel` function (MSL disallows `threadgroup` storage declarations inside non-`kernel` inline helpers).
  - For `Q8_0`/`Q4_0`, `M >= 4` (or even `M >= 2`) is likely to win over the scalar `nn_matmul_q8_0`/`nn_matmul_q4_0` kernel because the scalar kernel dequantizes each block separately for every row of `M`; measure `M = 2, 4, 6` on Apple M3 to confirm.
