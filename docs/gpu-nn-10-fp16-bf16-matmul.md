# NN-10 Report: FP16 / BF16 Resident Weights with FP32-Accumulated Metal Projections

**Task:** NN-10 (`github.com/itsmontoya/neuralnetwork`)  
**Base commit:** `main` (`1dc533a1143de6afc880b62b2a4a8f8e74ca89cd`)  
**Hardware / Environment:** Apple M3 (`Mac15,13`, 10-core GPU, 24 GiB unified memory), macOS `26.5.2 (25F84)`, Darwin `25.5.0 arm64`, Go `1.27.3` (`GOWORK=off`, `CGO_ENABLED=1`)

## Summary

NN-10 adds mixed-precision Metal projection support to `gpu.Scope.MatMul` and `gpu.Scope.MatMulRightTranspose`:

| Left Activation | Right Weight | Destination | Accumulation | Path |
| --- | --- | --- | --- | --- |
| `Float32` `[M,K]` | `Float32` `[K,N]` or `[N,K]` | `Float32` `[M,N]` | FP32 (unchanged) | Existing `nn_matmul` (bit-for-bit untouched) |
| `Float32` `[M,K]` | `Float16` `[K,N]` or `[N,K]` | `Float32` `[M,N]` | FP32 (`fastMathEnabled = NO`) | `nn_matmul_f16_row_right_transpose` (`M=1`, RT) / `nn_matmul_f16` |
| `Float32` `[M,K]` | `BFloat16` `[K,N]` or `[N,K]` | `Float32` `[M,N]` | FP32 (`fastMathEnabled = NO`) | `nn_matmul_bf16_row_right_transpose` (`M=1`, RT) / `nn_matmul_bf16` |

Unsupported dtype combinations (e.g., `Float16`/`BFloat16` left or destination) are explicitly rejected with `gpu.ErrInvalid`.

## Kernel Design (`internal/device/matmul_mixed_metal_source.h`)

All four mixed-precision kernels compile lazily on first mixed-precision projection use via `pthread_once` with `[options setFastMathEnabled:NO]` (zero impact on ANN startup or FP32 `nn_matmul`):

1. **Single-row (`M=1`) right-transposed GEMV** (`nn_matmul_f16_row_right_transpose`, `nn_matmul_bf16_row_right_transpose`):
   - `128` threads per threadgroup (`4` SIMD groups of `32` lanes), `2` output rows per SIMD group (`8` rows per threadgroup), `ceil(N / 8)` threadgroups.
   - When `K % 4 == 0`: coalesced `float4` loads for `left` and `half4`/`bfloat4` loads for `right`, FP32 `fma` accumulation, lane reduction, and `simd_sum` across the 32 lanes.
   - When `K % 4 != 0`: exact scalar SIMD-coalesced stride-32 loop avoiding misaligned vector loads.
   - Uniform `row0 >= N` / `row1 < N` guards prevent out-of-bounds reads/writes on odd `N` with zero intra-SIMD divergence.
2. **Multi-row (`M>1`) right-transposed GEMM and standard `MatMul`** (`nn_matmul_f16`, `nn_matmul_bf16`):
   - Barrier-free `16×16` 2D grid (`float4` × `half4`/`bfloat4` vectorized inner loop when `variant == 2` and `K % 4 == 0`; exact scalar loop for `variant == 0` or odd `K`).
   - Avoids threadgroup barrier stalls on Apple M3 while benefiting from L1/L2 broadcast of `left` and 2× lower weight memory traffic (2.1×–2.2× faster than FP32 on `M=27/28/29` FFN, versus a 16×16 threadgroup-tiled prototype that regressed to 17.2–17.4 ms due to `256` barriers per tile).

## Numerical Verification vs Independent FP64 Reference

Each test uploads `Float16`/`BFloat16` weights via `UploadFloat32As`, independently dequantizes the exact uploaded bits (`Float16ToFloat32` / `BFloat16ToFloat32`) to `float64`, and compares every GPU output against the FP64 dot product (`Test_MetalTensorMatMulMixed`, `Test_MetalTensorMatMulMixedSpecialValues`, `Test_MetalFFNMixedPrecision`).

### Isolated Projections (FP64 Absolute Error)

| Shape | DType | Outputs | p50 | p95 | p99 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Attn_KV` `[1,2048]×[512,2048]^T` | `float16` | 512 | `1.03e-6` | `2.96e-6` | `4.17e-6` | `5.88e-6` |
| `Attn_KV` `[1,2048]×[512,2048]^T` | `bfloat16` | 512 | `9.37e-7` | `3.01e-6` | `3.92e-6` | `5.71e-6` |
| `Attn_QO` `[1,2048]×[2048,2048]^T` | `float16` | 2,048 | `1.04e-6` | `2.98e-6` | `4.24e-6` | `6.24e-6` |
| `Attn_QO` `[1,2048]×[2048,2048]^T` | `bfloat16` | 2,048 | `1.03e-6` | `3.00e-6` | `4.08e-6` | `6.33e-6` |
| `FFN_GateUp` `[1,2048]×[8192,2048]^T` | `float16` | 8,192 | `1.04e-6` | `3.13e-6` | `4.17e-6` | `6.38e-6` |
| `FFN_GateUp` `[1,2048]×[8192,2048]^T` | `bfloat16` | 8,192 | `1.06e-6` | `3.03e-6` | `4.02e-6` | `6.94e-6` |
| `FFN_Down` `[1,8192]×[2048,8192]^T` | `float16` | 2,048 | `3.13e-6` | `9.53e-6` | `1.28e-5` | `1.80e-5` |
| `FFN_Down` `[1,8192]×[2048,8192]^T` | `bfloat16` | 2,048 | `3.15e-6` | `9.01e-6` | `1.21e-5` | `1.56e-5` |
| `Classifier` `[1,2048]×[128256,2048]^T` | `float16` | 128,256 | `1.02e-6` | `3.05e-6` | `4.10e-6` | `8.80e-6` |
| `Classifier` `[1,2048]×[128256,2048]^T` | `bfloat16` | 128,256 | `1.03e-6` | `3.06e-6` | `4.11e-6` | `7.50e-6` |
| `Prefill` `[29,2048]×[8192,2048]^T` | `float16` | 237,568 | `3.73e-6` | `1.24e-5` | `1.76e-5` | `5.31e-5` |
| `Prefill` `[29,2048]×[8192,2048]^T` | `bfloat16` | 237,568 | `3.75e-6` | `1.24e-5` | `1.78e-5` | `4.49e-5` |
| `OddTail` `[1,257]×[33,257]^T` | `float16` | 33 | `1.85e-7` | `5.38e-7` | `6.13e-7` | `7.18e-7` |
| `OddTail` `[1,257]×[33,257]^T` | `bfloat16` | 33 | `3.47e-7` | `8.74e-7` | `1.13e-6` | `1.36e-6` |

### Complete Resident FFN (`Test_MetalFFNMixedPrecision`, FP64 Absolute Error)

| Rows | DType | Elements | p50 | p95 | p99 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `M=1` | `float16` / `bfloat16` | 2,048 | `1.54e-8` / `1.61e-8` | `4.59e-8` / `4.52e-8` | `5.92e-8` / `5.93e-8` | `7.69e-8` / `7.20e-8` |
| `M=27` | `float16` / `bfloat16` | 55,296 | `6.90e-8` / `6.83e-8` | `2.10e-7` / `2.11e-7` | `2.91e-7` / `2.86e-7` | `6.13e-7` / `6.14e-7` |
| `M=28` | `float16` / `bfloat16` | 57,344 | `6.90e-8` / `6.82e-8` | `2.10e-7` / `2.11e-7` | `2.91e-7` / `2.86e-7` | `6.13e-7` / `6.14e-7` |
| `M=29` | `float16` / `bfloat16` | 59,392 | `6.89e-8` / `6.83e-8` | `2.10e-7` / `2.11e-7` | `2.91e-7` / `2.85e-7` | `6.69e-7` / `6.14e-7` |

Special values (`NaN`, `+Inf`, `-Inf`, `0*Inf`, opposing infinities, large finite, finite cancellation, overflow) passed on scalar tail, vectorized `M=1` (`K=4, N=2`), and multi-row `M=2` paths for both `Float16` and `BFloat16`.

## Isolated Benchmarks on Apple M3

Raw logs stored in `.benchmarks/gpu-nn-10/2026-11-25-m3-v2/` (gitignored).

### Isolated Resident Projections (`Benchmark_MatMulRightTransposeMixedMetal`, `-benchtime=10x -count=5`, median)

| Shape | FP32 (ms) | FP16 (ms) | BF16 (ms) | FP16 / BF16 Speedup | FP16 / BF16 Eff. GB/s | Weight Bytes (FP32 → 16-bit) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `FFN_GateUp_M1_K2048_N8192` | 1.581 | **0.804** | **0.796** | **1.97× / 1.99×** | 41.41 / 41.76 | 67,108,864 → **33,554,432** (-50%) |
| `FFN_Down_M1_K8192_N2048` | 1.870 | **0.865** | **0.825** | **2.16× / 2.27×** | 38.49 / 40.53 | 67,108,864 → **33,554,432** (-50%) |
| `Attn_QO_M1_K2048_N2048` | 0.979 | **0.391** | **0.365** | **2.50× / 2.68×** | 21.29 / 22.80 | 16,777,216 → **8,388,608** (-50%) |
| `Attn_KV_M1_K2048_N512` | 0.402 | **0.207** | **0.206** | **1.94× / 1.95×** | 10.02 / 10.03 | 4,194,304 → **2,097,152** (-50%) |
| `Classifier_M1_K2048_N128256` | 12.453 | **6.207** | **6.249** | **2.01× / 1.99×** | **84.63 / 84.06** | 1,050,673,152 → **525,336,576** (-50%) |
| `FFN_GateUp_M6_K2048_N8192` | 1.707 | 1.736 | **1.271** | 0.98× / **1.34×** | 19.40 / 26.47 | 67,108,864 → **33,554,432** (-50%) |
| `FFN_GateUp_M27_K2048_N8192` | 4.349 | **2.272** | **2.061** | **1.91× / 2.11×** | 15.22 / 16.77 | 67,108,864 → **33,554,432** (-50%) |
| `FFN_GateUp_M28_K2048_N8192` | 4.345 | **2.298** | **2.151** | **1.89× / 2.02×** | 15.06 / 16.08 | 67,108,864 → **33,554,432** (-50%) |
| `FFN_GateUp_M29_K2048_N8192` | 4.584 | **2.247** | **2.246** | **2.04× / 2.04×** | 15.42 / 15.42 | 67,108,864 → **33,554,432** (-50%) |
| `OddTail_M1_K257_N33` | 0.235 | **0.161** | **0.160** | **1.46× / 1.47×** | 0.11 / 0.11 | 33,924 → **16,962** (-50%) |

Note: Small `M=1` shapes include ~150–200 µs of per-op `Scope` submit/wait overhead in isolated single-dispatch benchmarks; on the large `Classifier` shape where dispatch overhead is negligible, effective bandwidth reaches **84.6 GB/s** (85% of M3's 100 GB/s unified memory peak).

### Complete Resident FFN (`Benchmark_FFNMixedMetal`, `-benchtime=5x -count=3`, median resident interval)

| Rows | FP32 Resident (ms) | FP16 Resident (ms) | BF16 Resident (ms) | FP16 / BF16 Speedup | Weight Bytes (FP32 → 16-bit) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `M=1` | 3.194 | **1.715** | **1.753** | **1.86× / 1.82×** | 201,326,592 → **100,663,296** (-50%) |
| `M=27` | 12.236 | **5.689** | **5.460** | **2.15× / 2.24×** | 201,326,592 → **100,663,296** (-50%) |
| `M=28` | 12.151 | **5.783** | **5.755** | **2.10× / 2.11×** | 201,326,592 → **100,663,296** (-50%) |
| `M=29` | 13.113 | **5.893** | **5.936** | **2.23× / 2.21×** | 201,326,592 → **100,663,296** (-50%) |

## Handoff to NN-11

- **Ready for NN-11:** `Scope.MatMul` and `Scope.MatMulRightTranspose` support `Float32 × Float16 → Float32` and `Float32 × BFloat16 → Float32` with exact 50% weight byte reduction and ~1.9×–2.5× isolated `M=1` / ~2.1×–2.2× `M=27/28/29` FFN speedups.
- **Next bottlenecks (NN-11 scope):**
  1. Parallel SIMD `RMSNorm` (currently single-thread-per-row at `18.3 ms` across 33 dispatches in Llama 3.2 1B) with `Float32`/`Float16`/`BFloat16` norm weights.
  2. Tied `Float16`/`BFloat16` `GatherRows` into `Float32` (and matching 16-bit) destinations so the `[128256, 2048]` embedding/classifier table stays in a single `525 MB` allocation.
  3. Optional `Float16` `KVCache` storage with FP32 query/score/softmax/output accumulation in `CausalAttention` and `AppendKV`.
