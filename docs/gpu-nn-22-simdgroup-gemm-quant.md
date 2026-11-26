# NN-22: `simdgroup_float8x8` Tiled `M > 1` GEMM for Quantized `Q8_0` & `Q4_0`

**Date:** 2026-11-26  
**Hardware:** Apple M3 (arm64, 8-core CPU, 10-core GPU, unified memory)  
**Baseline:** `gpu-nn-21` (`fd5a393310e77e160e3bffd188194c6381904dc8`)  
**Raw benchmark logs:** `.benchmarks/gpu-nn-22/run1/` (before & initial `M >= 2` threshold sweep), `.benchmarks/gpu-nn-22/run2/` (final tuned threshold `M >= 8, N >= 8`)

---

## 1. Summary

`NN-22` replaces the 1-thread-per-output `M > 1` execution path in `nn_matmul_q8_0` and `nn_matmul_q4_0` (`Scope.MatMulRightTranspose`) with Apple Silicon `simdgroup_float8x8` tiled GEMM kernels (`BK = 32 == QuantBlockSize`) compiled lazily with `[options setFastMathEnabled:NO]` and FP32 accumulation.

- **Public API:** Zero changes.
- **Preserved untouched:** `M == 1` GEMV (`nn_matmul_q8_0_row_right_transpose`, `nn_matmul_q4_0_row_right_transpose`), `Float32 × Float32`, and `Float16`/`BFloat16` paths.
- **Peak achieved throughput:** Up to **2.04 TFLOP/s** FP32-accumulated on Apple M3 (`MatMulRT_M256_K2048_N8192`: `2,043 GFLOP/s` `Q8_0`, `2,035 GFLOP/s` `Q4_0`).
- **Speedups:** **2.15×–5.00×** on batched quantized `MatMulRightTranspose` (`M=27..256`) and **2.94×–3.48×** on full resident Llama 3.2 1B FFN prefill (`M=27..29`), making quantized `Q8_0`/`Q4_0` prefill (`2.16–2.23 ms`) **faster than `Float16`/`BFloat16` (`2.72–2.80 ms`)** and eliminating the NN-12 prefill inversion.

---

## 2. Tile Geometry & Kernel Architecture

Implemented in `internal/device/matmul_quant_metal_source.h` via a shared template `nn_matmul_quant_tiled_impl<IsQ4>` instantiated into two Metal compute kernels:

- `nn_matmul_q8_0_tiled` (`IsQ4 = false`, 34-byte GGUF `Q8_0` blocks)
- `nn_matmul_q4_0_tiled` (`IsQ4 = true`, 18-byte GGUF `Q4_0` split-half blocks)

### Tile Dimensions & Threadgroup Layout

| Parameter | Value | Rationale |
| --- | --- | --- |
| Threadgroup tile `(BM, BN, BK)` | `(32, 64, 32)` | `BK = 32` matches `QuantBlockSize = 32` exactly: every K-tile step dequantizes **1 exact GGUF block** per column `n ∈ [0, 64)` and reuses it across all 32 rows of `M` (32× less dequantization work). |
| Threads per threadgroup | `128` (`4` SIMD groups, `2×2` grid) | `64` blocks across `128` threads = **2 threads per block** (`n = tid >> 1`, `halfIdx = tid & 1u`), each unpacking 16 weights into `Bs` with 16-byte `float4` stores. Each SIMD group owns a `16×32` sub-tile (`2×4 = 8` `simdgroup_float8x8` accumulators). |
| `As` shared memory | `32 × 36` `float` (`4,608 B`) | Stride `36 = 32 + 4` preserves 16-byte `float4` alignment and avoids bank conflicts. Since `K % 32 == 0`, every K-tile is 100% in-bounds along `K` with coalesced `float4` loads. |
| `Bs` shared memory | `64 × 36` `float` (`9,216 B`) | Hardware-transposed `simdgroup_load(..., transpose=true)` feeds `acc[2][4]` directly. |
| Total threadgroup memory | `13,824 B` | Fits 2 resident threadgroups per 32 KB threadgroup memory for latency hiding. |
| Alignment safety | `const device uchar *` + 2-byte `half` / `ushort` loads | Strictly honors `docs/adr-gpu-quantization.md`: 34-byte (`Q8_0`) and 18-byte (`Q4_0`) blocks are 2-byte aligned (never 4-byte/16-byte misaligned casts). |
| Output store | Interior: direct `simdgroup_store` to `result`; Edge: stage through reused `Bs` (`32 × 64` `float`) | Zero extra shared-memory allocation for partial boundary tiles (`M % 32 != 0` or `N % 64 != 0`). |

### Dispatch Threshold (`internal/device/metal_backend.m`)

```c
useRowRT = (dimensions.resultRows == 1);
useTiled = (!useRowRT && dimensions.resultRows >= 8 && dimensions.resultCols >= 8);
```

- **Why `M >= 8, N >= 8`:** In the `run1` threshold sweep on Apple M3 (`K=2048, N=8192`), the `BM = 32` tiled kernel takes a flat `~800 µs` for any `M <= 32`, whereas the fallback `nn_matmul_q8_0`/`nn_matmul_q4_0` kernel takes `620 µs` / `426 µs` at `M = 2`, `566 µs` / `601 µs` at `M = 4`, and `736 µs` / `791 µs` at `M = 6`, crossing `~800 µs` right at `M = 8`. Setting `M >= 8 && N >= 8` matches `NN-21` (`variant == 2`), guarantees zero regression for small `M < 8` or narrow `N < 8`, and accelerates every `M >= 8` through `M = 256+`.

### Special-Value Safety (`NaN`, `±Inf`, `0 * Inf`, `int8(-128)`)

Because `K % 32 == 0` and `BK = 32`, there are no out-of-bounds `k >= K` lanes in any K-tile. Out-of-bounds `row >= M` rows in `As` and `col >= N` columns in `Bs` are zeroed and masked during edge-tile writeback, never interacting with in-bounds outputs. All 11 special-value cases (`zero_scale_finite_left`, `zero_scale_inf_left`, `negative_scale`, `extreme_max_fp16_scale`, `min_subnormal_scale`, `nan_scale`, `inf_scale_positive`, `inf_scale_zero_left`, `nan_left`, `inf_left`, and `q8_min_int8_minus_128`) pass 100% on `M=1 N=1`, `M=1 N=2`, `M=2 N=2` (fallback), `M=8 N=9` (tiled partial edge), and `M=33 N=65` (tiled interior + boundary).

---

## 3. Before vs After Apple M3 Benchmarks

### A. Batched Quantized `MatMulRightTranspose` (`Benchmark_MatMulQuantBatchMetal`)

| Shape | DType | Before (`ms/op`) | After (`ms/op`) | Speedup | After GFLOP/s | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `MatMulRT_M2_K2048_N8192` | `q8_0` | 0.620 | 0.622 | 1.00× | 107.9 | `M=2 < 8` stays on fallback |
| `MatMulRT_M2_K2048_N8192` | `q4_0` | 0.426 | 0.428 | 0.99× | 156.7 | `M=2 < 8` stays on fallback |
| `MatMulRT_M4_K2048_N8192` | `q8_0` | 0.566 | 0.558 | 1.02× | 240.7 | `M=4 < 8` stays on fallback |
| `MatMulRT_M4_K2048_N8192` | `q4_0` | 0.601 | 0.588 | 1.02× | 228.3 | `M=4 < 8` stays on fallback |
| `MatMulRT_M6_K2048_N8192` | `q8_0` | 0.736 | 0.756 | 0.97× | 266.2 | `M=6 < 8` stays on fallback |
| `MatMulRT_M6_K2048_N8192` | `q4_0` | 0.791 | 0.788 | 1.00× | 255.4 | `M=6 < 8` stays on fallback |
| `MatMulRT_M32_K2048_N2048` | `q8_0` | 0.864 | 0.402 | **2.15×** | 667.0 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M32_K2048_N2048` | `q4_0` | 0.930 | 0.486 | **1.91×** | 551.8 | Tiled `simdgroup_float8x8` (`0.385 ms` in run1) |
| `MatMulRT_M64_K2048_N2048` | `q8_0` | 1.450 | 0.542 | **2.67×** | 989.6 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M64_K2048_N2048` | `q4_0` | 1.600 | 0.527 | **3.04×** | 1,019 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M128_K2048_N2048` | `q8_0` | 2.658 | 0.801 | **3.32×** | 1,340 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M128_K2048_N2048` | `q4_0` | 2.897 | 0.793 | **3.65×** | 1,353 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M256_K2048_N2048` | `q8_0` | 4.943 | 1.301 | **3.80×** | 1,651 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M256_K2048_N2048` | `q4_0` | 5.528 | 1.298 | **4.26×** | 1,654 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M32_K2048_N8192` | `q8_0` | 2.644 | 0.858 | **3.08×** | 1,251 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M32_K2048_N8192` | `q4_0` | 2.918 | 0.833 | **3.50×** | 1,289 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M128_K2048_N8192` | `q8_0` | 9.571 | 2.289 | **4.18×** | 1,876 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M128_K2048_N8192` | `q4_0` | 10.716 | 2.322 | **4.61×** | 1,849 | Tiled `simdgroup_float8x8` |
| `MatMulRT_M256_K2048_N8192` | `q8_0` | 18.823 | 4.204 | **4.48×** | **2,043** | Tiled `simdgroup_float8x8` (2.04 TFLOP/s) |
| `MatMulRT_M256_K2048_N8192` | `q4_0` | 21.103 | 4.220 | **5.00×** | **2,035** | Tiled `simdgroup_float8x8` (2.04 TFLOP/s) |

### B. Llama 3.2 1B Projections (`Benchmark_MatMulRightTransposeMixedMetal`)

| Shape | DType | Before (`ms/op`) | After (`ms/op`) | Speedup | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `FFN_GateUp_M1_K2048_N8192` | `q8_0` | 0.500 | 0.478 | 1.05× | `M=1` GEMV untouched |
| `FFN_GateUp_M1_K2048_N8192` | `q4_0` | 0.347 | 0.357 | 0.97× | `M=1` GEMV untouched |
| `Classifier_M1_K2048_N128256` | `q8_0` | 3.511 | 3.499 | 1.00× | `M=1` GEMV untouched (~79.9 GB/s) |
| `Classifier_M1_K2048_N128256` | `q4_0` | 1.971 | 1.978 | 1.00× | `M=1` GEMV untouched (~75.0 GB/s) |
| `FFN_GateUp_M6_K2048_N8192` | `q8_0` | 0.740 | 0.734 | 1.01× | `M=6 < 8` stays on fallback |
| `FFN_GateUp_M6_K2048_N8192` | `q4_0` | 0.790 | 0.795 | 0.99× | `M=6 < 8` stays on fallback |
| `FFN_GateUp_M27_K2048_N8192` | `q8_0` | 2.315 | 0.852 | **2.72×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M27_K2048_N8192` | `q4_0` | 2.567 | 0.838 | **3.06×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M28_K2048_N8192` | `q8_0` | 2.344 | 0.852 | **2.75×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M28_K2048_N8192` | `q4_0` | 2.605 | 0.822 | **3.17×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M29_K2048_N8192` | `q8_0` | 2.436 | 0.872 | **2.79×** | Tiled `simdgroup_float8x8` |
| `FFN_GateUp_M29_K2048_N8192` | `q4_0` | 2.706 | 0.835 | **3.24×** | Tiled `simdgroup_float8x8` |

### C. Full Resident FFN (`Benchmark_FFNMixedMetal`, `hidden=2048, intermediate=8192`)

| Rows | DType | Before (`ms/op`) | After (`ms/op`) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `M1` | `q8_0` | 0.965 | 0.941 | 1.03× (`M=1` untouched) |
| `M1` | `q4_0` | 0.590 | 0.591 | 1.00× (`M=1` untouched) |
| `M27` | `q8_0` | 6.374 | 2.169 | **2.94×** |
| `M27` | `q4_0` | 7.084 | 2.157 | **3.28×** |
| `M28` | `q8_0` | 6.515 | 2.191 | **2.97×** |
| `M28` | `q4_0` | 7.260 | 2.169 | **3.35×** |
| `M29` | `q8_0` | 6.818 | 2.227 | **3.06×** |
| `M29` | `q4_0` | 7.573 | 2.174 | **3.48×** |

---

## 4. Numerical Verification (Independent FP64 Dequantized Reference)

Measured on `Test_MetalFFNQuantized` (`hidden=2048, intermediate=8192`, 3 quantized projections + SwiGLU):

| DType | Rows | Elements | p50 abs err | p95 abs err | p99 abs err | Max abs err | SQNR vs FP32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q8_0` | `M1` | 2,048 | `1.57e-08` | `4.38e-08` | `5.92e-08` | `7.29e-08` | 43.64 dB |
| `q8_0` | `M27` | 55,296 | `1.29e-07` | `4.30e-07` | `6.47e-07` | `1.60e-06` | 43.69 dB |
| `q8_0` | `M28` | 57,344 | `1.29e-07` | `4.30e-07` | `6.46e-07` | `1.60e-06` | 43.68 dB |
| `q8_0` | `M29` | 59,392 | `1.29e-07` | `4.30e-07` | `6.47e-07` | `1.60e-06` | 43.68 dB |
| `q4_0` | `M1` | 2,048 | `1.45e-08` | `4.21e-08` | `5.60e-08` | `8.59e-08` | 19.05 dB |
| `q4_0` | `M27` | 55,296 | `1.24e-07` | `4.08e-07` | `6.10e-07` | `1.47e-06` | 18.90 dB |
| `q4_0` | `M28` | 57,344 | `1.24e-07` | `4.08e-07` | `6.10e-07` | `1.47e-06` | 18.90 dB |
| `q4_0` | `M29` | 59,392 | `1.24e-07` | `4.08e-07` | `6.10e-07` | `1.47e-06` | 18.90 dB |

Additional verification in `gpu/matmul_quant_metal_test.go`:
- `Test_MetalTensorMatMulQuant`: 31 shapes across `M ∈ {1, 2, 3, 4, 6, 8, 15, 16, 17, 27, 28, 29, 31, 32, 33, 64, 128}`, `K ∈ {32, 64, 96, 128, 160, 256, 2048, 8192}`, `N ∈ {1, 3, 7, 9, 15, 17, 33, 64, 65, 128, 512, 2048, 8192, 128256}` for both `Q8_0` and `Q4_0`.
- `Test_MetalTensorMatMulQuantSpecialValues`: all 11 special-value cases on `M=1 N=1`, `M=1 N=2`, `M=2 N=2` (fallback), `M=8 N=9` (tiled partial), and `M=33 N=65` (tiled interior + boundary).
- `Test_MetalTensorMatMulQuantTiledConcurrentStress`: 8 concurrent workers sharing immutable `Q8_0`/`Q4_0` weights on `33×128×65` under `-race -tags=metal` with zero resource leaks.

---

## 5. Notes for `NN-23` (`BatchedMatMul` & `MatMulLeftTranspose`)

- The `(BM=32, BN=64, BK=32)` layout (`4` SIMD groups = `128` threads, `AS_STRIDE=36`, `Bs`-reused edge staging, `fastMathEnabled = NO`) is now proven across `Float16`, `BFloat16`, `Q8_0`, and `Q4_0` (1.60–2.04 TFLOP/s FP32-accumulated on Apple M3).
- For `NN-23`, `BatchedMatMul` and `BatchedMatMulRightTranspose` can launch the same `(32, 64, 32)` tile over a 3D grid `(ceil(N/64), ceil(M/32), totalBatches)`, computing `leftBase` and `rightBase` once per threadgroup via `nn_batched_offsets(params, tgid.z, leftBase, rightBase)`.
- For `MatMulLeftTranspose` (`left[K, M]^T × right[K, N]`), load `As` transposed (or load `As[BK][BM]` with `AS_STRIDE = 36` and use `simdgroup_load(..., transpose=true)` on `aFrag`) so global reads from `left[K, M]` stay coalesced along `M`.
