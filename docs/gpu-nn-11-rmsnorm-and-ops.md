# NN-11 Report: Parallel SIMD `RMSNorm`, Tied FP16/BF16 `GatherRows`, and Optional FP16 `KVCache`

**Task:** NN-11 (`github.com/itsmontoya/neuralnetwork`)  
**Base commit:** `main` (`0c24093b595ad74de7a0fd34ea0e484da3f0b21f`)  
**Hardware / Environment:** Apple M3 (`Mac15,13`, 10-core GPU, 24 GiB unified memory), macOS `26.5.2 (25F84)`, Darwin `25.5.0 arm64`, Go `1.27.3` (`GOWORK=off`, `CGO_ENABLED=1`)

## Summary

NN-11 removes the rank-2 LL-04 Metal decode bottleneck (single-thread-per-row `RMSNorm` at `18.3 ms` for 33 dispatches) and completes the mixed-precision transformer primitive set so full decoder blocks and tied `[128256, 2048]` embedding/classifier tables execute end-to-end in a single Metal command scope with `Float32` activations, `Float16`/`BFloat16` weights, and optional `Float16` KV cache storage:

| Operation | Supported DTypes in NN-11 | Notes |
| --- | --- | --- |
| `Scope.RMSNorm` | `Float32` input/destination; `Float32`, `Float16`, or `BFloat16` weight | Threadgroup-cooperative SIMD reduction (`fastMathEnabled = NO`), FP32 max-scaling & variance accumulation, vectorized `float4`/`half4`/`bfloat4` when `cols % 4 == 0`, scalar fallback for arbitrary `cols >= 1`. |
| `Scope.GatherRows` | Matching (`Float32→Float32`, `Float16→Float16`, `BFloat16→BFloat16`) and mixed dequantizing (`Float16→Float32`, `BFloat16→Float32`) | Matching dtypes use zero-allocation Metal blit (`rowBytes = cols * dtype.Bytes()`, including odd 16-bit widths). Mixed gathers use a 2D compute kernel (`nn_tensor_gather_rows_f16_to_f32` / `nn_tensor_gather_rows_bf16_to_f32`) with chunked `setBytes` (512 rows/chunk, zero temporary buffers). Invalid combinations return `ErrInvalid`. |
| `NewKVCacheWithDType` / `KVCache.DType` | `Float32` (default via `NewKVCache`) and `Float16` | Halves KV slot memory and prefix copy bandwidth; unsupported dtypes (e.g., `BFloat16`, `0`) return `ErrInvalid`. |
| `Scope.AppendKV` | `Float32` keys/values into `Float32` or `Float16` `KVCache` | For `Float16` caches: blits committed `Float16` prefix and converts appended FP32 rows to `half` via `nn_tensor_kv_append_f16`, preserving exact double-buffered staged publication semantics. |
| `Scope.CausalAttention` | `Float32` queries/destination with `Float32` or `Float16` `KVCache` | `nn_tensor_causal_attention_f16` reads `half` K/V history and accumulates dot products, softmax, and weighted values in FP32 (`fastMathEnabled = NO`) across MHA, GQA, and MQA (`M=1` and `M>1`). |
| `Add`, `Mul`, `SiLU`, `SwiGLU`, `RotatePairs` | `Float32` only (unchanged) | Activations stay `Float32` throughout the mixed-precision path (`docs/adr-gpu-reduced-precision.md`), avoiding precision loss in residuals/RoPE and avoiding unused kernel matrix bloat. |

## Before vs. After: Isolated `RMSNorm` on Apple M3

Raw logs stored under `.benchmarks/gpu-nn-11/2026-11-25-m3/` (`rmsnorm-before.txt`, `rmsnorm-after.txt`; `-benchtime=10x -count=5`, median):

| Shape | Before FP32 (ms) | After FP32 (ms) | After FP16 wt (ms) | After BF16 wt (ms) | Speedup (FP32 / FP16 / BF16) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `M=1, C=2048` (1 dispatch) | 1.197 | **0.288** | **0.270** | **0.231** | **4.16× / 4.43× / 5.18×** |
| `M=1, C=2048` (**33 dispatches**) | **18.26** | **0.702** | **0.703** | **0.694** | **26.0× / 26.0× / 26.3×** |
| `M=6, C=2048` (33 dispatches) | 18.35 | **0.673** | **0.742** | **0.728** | **27.3× / 24.7× / 25.2×** |
| `M=27, C=2048` (33 dispatches) | 20.05 | **0.943** | **0.977** | **0.912** | **21.3× / 20.5× / 22.0×** |
| `M=28, C=2048` (33 dispatches) | 20.08 | **0.541** | **0.571** | **0.925** | **37.1× / 35.2× / 21.7×** |
| `M=29, C=2048` (33 dispatches) | 20.12 | **0.813** | **0.539** | **0.391** | **24.7× / 37.3× / 51.5×** |

Replacing the single-thread-per-row loop with 1 threadgroup per row (`32`–`256` threads cooperating via `simd_max`/`simd_sum` + threadgroup reduction and `float4` loads) removes **~17.6 ms per decode token** across Llama 3.2 1B's 33 `RMSNorm` dispatches.

## Synthetic Transformer Block Benchmark on Apple M3

Llama 3.2 1B layer dimensions (`hidden=2048`, `heads=32`, `kvHeads=8`, `headDim=64`, `ffn=8192`, `capacity=512`), 1 command scope per step (`Benchmark_TransformerBlockMixedMetal`, `-benchtime=5x -count=3`, median; raw log `transformer-block.txt`):

| Mode | `M=1, H=511` (ms) | `M=29, H=0` (ms) | Layer Weight Bytes | KV Cache Bytes (2 slots) |
| --- | ---: | ---: | ---: | ---: |
| `FP32` weights + `FP32` KV | 9.204 | 17.64 | 243,286,016 | 4,194,304 |
| `FP16` weights + `FP32` KV | **7.352** (1.25×) | **7.874** (2.24×) | **121,643,008** (-50%) | 4,194,304 |
| `FP16` weights + `FP16` KV | **7.480** (1.23×) | **10.80** (1.63×) | **121,643,008** (-50%) | **2,097,152** (-50%) |
| `BF16` weights + `FP32` KV | **7.460** (1.23×) | **7.720** (2.28×) | **121,643,008** (-50%) | 4,194,304 |
| `BF16` weights + `FP16` KV | **7.510** (1.23×) | **7.740** (2.28×) | **121,643,008** (-50%) | **2,097,152** (-50%) |

Note: At `M=1, H=511`, `CausalAttention` over 512 history rows accounts for ~4 ms of the single-layer time (and is constant across weight dtypes), while projection + `RMSNorm` time drops ~2×.

## Numerical Verification vs. Independent FP64 Reference

Verified in `Test_MetalRMSNormParallelAndMixedWeights`, `Test_MetalGatherRowsReducedPrecision`, `Test_MetalKVCacheFloat16AndAttention`, and `Test_MetalTransformerBlockMixedPrecision` (raw log `metal-tests.txt`):

### `RMSNorm` FP64 Absolute Error (`C=2048`)

| Shape / Weight DType | p50 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: |
| `M=1, C=2048` `float32` | `7.97e-9` | `5.72e-8` | `9.33e-8` | `1.33e-7` |
| `M=1, C=2048` `float16` | `7.86e-9` | `5.52e-8` | `9.21e-8` | `1.46e-7` |
| `M=1, C=2048` `bfloat16` | `8.08e-9` | `5.81e-8` | `9.37e-8` | `1.46e-7` |
| `M=29, C=2048` `float32` | `8.95e-9` | `6.78e-8` | `1.10e-7` | `2.05e-7` |
| `M=29, C=2048` `float16` | `9.09e-9` | `6.70e-8` | `1.09e-7` | `2.03e-7` |
| `M=29, C=2048` `bfloat16` | `9.00e-9` | `6.75e-8` | `1.09e-7` | `1.95e-7` |

Special values (all zeros, `±math.MaxFloat32` overflow guard, `NaN`, `+Inf`) pass across arbitrary widths (`1, 3, 6, 17, 32, 33, 64, 65, 128, 129, 256, 257, 2048, 8192`).

### End-to-End Synthetic Transformer Block + Tied Gather & Classifier (FP64 Absolute Error)

| Mode | Prefill (`M=6`) p50 / p95 / p99 / Max | Decode (`M=1`) p50 / p95 / p99 / Max |
| --- | --- | --- |
| `FP32` weights + `FP32` KV | `7.94e-8` / `3.06e-7` / `4.97e-7` / `6.71e-7` | `6.23e-8` / `1.53e-7` / `1.86e-7` / `1.94e-7` |
| `FP16` weights + `FP32` KV | `4.48e-8` / `1.31e-7` / `1.73e-7` / `2.53e-7` | `2.68e-8` / `7.46e-8` / `8.93e-8` / `1.00e-7` |
| `FP16` weights + `FP16` KV | `8.23e-8` / `1.94e-6` / `2.66e-6` / `3.24e-6` | `5.08e-7` / `1.02e-6` / `1.19e-6` / `1.31e-6` |
| `BF16` weights + `FP32` KV | `4.66e-8` / `1.52e-7` / `2.13e-7` / `3.52e-7` | `2.55e-8` / `7.03e-8` / `8.27e-8` / `8.46e-8` |
| `BF16` weights + `FP16` KV | `7.41e-8` / `8.51e-7` / `1.34e-6` / `2.03e-6` | `2.22e-7` / `5.58e-7` / `7.94e-7` / `8.38e-7` |

Tied `[128256, 2048]` `Float16` and `BFloat16` `GatherRows` + `MatMulRightTranspose` execute in 1 command with 3 buffers (`525,336,576` weight bytes = exact 50% of FP32) and exact bit-for-bit match against `Float16ToFloat32` / `BFloat16ToFloat32`.

## Handoff to NN-12

- **Ready for `llama` and `onnx`:** Full mixed-precision FP16/BF16 inference (`RMSNorm`, `GatherRows`, `MatMul`/`MatMulRightTranspose`, `RotatePairs`, `AppendKV`, `CausalAttention`, `Add`/`Mul`/`SiLU`/`SwiGLU`, and optional `Float16` `KVCache`) is complete, tested, and leak-free.
- **Next task (NN-12):** Add block-quantized weight tensors (`Q8_0` and `Q4_0`, 32-element blocks with FP16 scale) and fused dequantize-on-the-fly Metal `M=1` GEMV / `M>1` GEMM kernels with FP32 accumulation.
