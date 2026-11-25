# NN-12 Report: Block-Quantized Weight Tensors (`Q8_0` / `Q4_0`) & FP32-Accumulated Metal GEMV/GEMM

**Task:** NN-12 (`github.com/itsmontoya/neuralnetwork`)  
**Base commit:** `main` (`3c40a385497af418d81446a5c942e068e36515b1`)  
**Hardware / Environment:** Apple M3 (`Mac15,13`, 10-core GPU, 24 GiB unified memory), macOS `26.5.2 (25F84)`, Darwin `25.5.0 arm64`, Go `1.27.3` (`GOWORK=off`, `CGO_ENABLED=1`)

## Summary

NN-12 adds GGUF-compatible block-quantized weight storage (`Q8_0` and `Q4_0`, 32-element blocks along `K` with IEEE-754 binary16 scales) and fused dequantize-on-the-fly Metal `M=1` GEMV and `M>1` GEMM kernels with FP32 accumulation (`fastMathEnabled = NO`) through the uniform `gpu.Scope.MatMulRightTranspose` API:

| DType | Block Layout (`QuantBlockSize = 32`) | Bytes / Block | Compression vs FP32 | Public Helpers |
| --- | --- | ---: | ---: | --- |
| `Q8_0` | 2-byte LE `float16` scale + 32 signed `int8` values | `34` | **`3.76×`** (`1.0625` B/param) | `QuantizeQ8_0`, `DequantizeQ8_0` |
| `Q4_0` | 2-byte LE `float16` scale + 16 `uint8` nibble pairs (GGUF split-half: low nibble `0..15`, high nibble `16..31`, offset `-8`) | `18` | **`7.11×`** (`0.5625` B/param) | `QuantizeQ4_0`, `DequantizeQ4_0` |

Full design, alignment rules, and operation dtype contract are specified in [`docs/adr-gpu-quantization.md`](adr-gpu-quantization.md).

## Kernel Design (`internal/device/matmul_quant_metal_source.h`)

All four quantized kernels compile lazily on first quantized projection use via `pthread_once` with `[options setFastMathEnabled:NO]` (zero impact on ANN startup or existing `Float32`/`Float16`/`BFloat16` pipelines):

1. **Alignment safety:** Because block sizes `34` (`Q8_0`) and `18` (`Q4_0`) are multiples of `2` (not `4`), weight buffers are addressed as `const device uchar *` with 2-byte-aligned `half` scale loads and 2-byte-aligned `ushort` payload loads (zero misaligned 4-byte/16-byte struct or vector casts).
2. **`M=1` SIMD-cooperative GEMV** (`nn_matmul_q8_0_row_right_transpose`, `nn_matmul_q4_0_row_right_transpose`):
   - `128` threads/threadgroup (`4` SIMD groups of `32` lanes), `2` output rows per SIMD group (`8` rows/threadgroup), `ceil(N / 8)` threadgroups.
   - `Q8_0`: 8 lanes per 32-element block (4 blocks per SIMD step), coalesced `float4` loads on `left`, 2-byte-aligned `ushort` loads unpacking 4 `int8` weights per lane, `d * wq` FP32 `fma`, and final `simd_sum`.
   - `Q4_0`: 4 lanes per 32-element block (8 blocks per SIMD step), coalesced `float4` loads on `left` (low half + high half), 2-byte-aligned `ushort` loads unpacking 8 4-bit nibbles per lane, `d * lo`/`d * hi` FP32 `fma`, and final `simd_sum`.
   - Uniform `row0 >= N` / `row1 < N` guards prevent out-of-bounds reads/writes on odd `N` with zero intra-SIMD divergence.
3. **`M>1` right-transposed GEMM** (`nn_matmul_q8_0`, `nn_matmul_q4_0`):
   - Bounds-checked `16×16` 2D grid with `float4` activation loads, 2-byte-aligned `ushort` weight unpacking, and FP32 accumulation supporting `M=6, 27, 28, 29` and arbitrary odd `N >= 1` (`K % 32 == 0`).

## Numerical Verification vs. Independent FP64 Reference

Verified in `Test_MetalTensorMatMulQuant`, `Test_MetalTensorMatMulQuantSpecialValues`, `Test_MetalTensorMatMulQuantCopyAndConcurrency`, and `Test_MetalFFNQuantized` (raw log: `.benchmarks/gpu-nn-12/2026-11-25-m3-v2/quant-tests.txt`).

### Isolated Projections (FP64 Absolute Error vs. Exact Dequantized Weights)

| Shape | DType | Outputs | p50 | p95 | p99 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Attn_KV` `[1,2048]×[512,2048]^T` | `q8_0` | 512 | `1.05e-6` | `3.14e-6` | `4.05e-6` | `5.25e-6` |
| `Attn_KV` `[1,2048]×[512,2048]^T` | `q4_0` | 512 | `1.01e-6` | `2.85e-6` | `3.91e-6` | `4.47e-6` |
| `Attn_QO` `[1,2048]×[2048,2048]^T` | `q8_0` | 2,048 | `1.04e-6` | `3.03e-6` | `4.20e-6` | `5.52e-6` |
| `Attn_QO` `[1,2048]×[2048,2048]^T` | `q4_0` | 2,048 | `1.02e-6` | `3.20e-6` | `4.07e-6` | `6.35e-6` |
| `FFN_GateUp` `[1,2048]×[8192,2048]^T` | `q8_0` | 8,192 | `1.04e-6` | `3.14e-6` | `4.28e-6` | `6.27e-6` |
| `FFN_GateUp` `[1,2048]×[8192,2048]^T` | `q4_0` | 8,192 | `9.99e-7` | `2.95e-6` | `4.05e-6` | `6.35e-6` |
| `FFN_Down` `[1,8192]×[2048,8192]^T` | `q8_0` | 2,048 | `3.34e-6` | `9.36e-6` | `1.21e-5` | `1.78e-5` |
| `FFN_Down` `[1,8192]×[2048,8192]^T` | `q4_0` | 2,048 | `3.23e-6` | `9.02e-6` | `1.18e-5` | `1.65e-5` |
| `Classifier` `[1,2048]×[128256,2048]^T` | `q8_0` | 128,256 | `1.04e-6` | `3.10e-6` | `4.16e-6` | `7.67e-6` |
| `Classifier` `[1,2048]×[128256,2048]^T` | `q4_0` | 128,256 | `1.02e-6` | `3.04e-6` | `4.07e-6` | `7.96e-6` |
| `Prefill` `[29,2048]×[8192,2048]^T` | `q8_0` | 237,568 | `1.37e-6` | `5.01e-6` | `7.75e-6` | `3.03e-5` |
| `Prefill` `[29,2048]×[8192,2048]^T` | `q4_0` | 237,568 | `1.33e-6` | `4.92e-6` | `7.62e-6` | `2.04e-5` |
| `OddTail` `[1,256]×[33,256]^T` | `q8_0` | 33 | `2.56e-7` | `7.06e-7` | `1.19e-6` | `1.22e-6` |
| `OddTail` `[1,256]×[33,256]^T` | `q4_0` | 33 | `1.83e-7` | `5.57e-7` | `5.81e-7` | `6.58e-7` |

Special values (zero scale, `0 * Inf = NaN`, negative scale, max FP16 scale `65504`, min subnormal FP16 scale, `int8(-128)` raw `Q8_0`, NaN/`±Inf` scales and inputs) passed on `M=1` (`N=1` and `N=2`) and `M=2` paths for both `Q8_0` and `Q4_0`.

### Synthetic Resident FFN: Dequantized FP64 Error & End-to-End Quantization SQNR vs. Unquantized FP32

| Rows | DType | FP64 Error vs Dequantized (p50 / p95 / p99 / Max) | SQNR vs Unquantized FP32 | Max Abs Error vs Unquantized FP32 |
| --- | --- | --- | ---: | ---: |
| `M=1` | `q8_0` | `1.57e-8` / `4.38e-8` / `5.92e-8` / `7.29e-8` | **43.64 dB** | `2.31e-3` |
| `M=27` | `q8_0` | `2.54e-8` / `8.10e-8` / `1.16e-7` / `3.28e-7` | **43.69 dB** | `2.84e-3` |
| `M=28` | `q8_0` | `2.54e-8` / `8.11e-8` / `1.17e-7` / `3.28e-7` | **43.68 dB** | `2.84e-3` |
| `M=29` | `q8_0` | `2.54e-8` / `8.10e-8` / `1.17e-7` / `3.28e-7` | **43.68 dB** | `2.84e-3` |
| `M=1` | `q4_0` | `1.45e-8` / `4.21e-8` / `5.60e-8` / `8.59e-8` | **19.05 dB** | `4.33e-2` |
| `M=27` | `q4_0` | `2.47e-8` / `7.92e-8` / `1.16e-7` / `3.27e-7` | **18.90 dB** | `5.35e-2` |
| `M=28` | `q4_0` | `2.46e-8` / `7.91e-8` / `1.16e-7` / `3.27e-7` | **18.90 dB** | `5.35e-2` |
| `M=29` | `q4_0` | `2.46e-8` / `7.90e-8` / `1.16e-7` / `3.27e-7` | **18.90 dB** | `5.35e-2` |

## 5-DType Benchmarks on Apple M3

Raw logs stored in `.benchmarks/gpu-nn-12/2026-11-25-m3-v2/` (`matmul-5dtype.txt`, `ffn-5dtype.txt`).

### Isolated Resident Projections (`Benchmark_MatMulRightTransposeMixedMetal`, `-benchtime=10x -count=5`, median)

| Shape | FP32 (ms) | FP16 (ms) | BF16 (ms) | Q8_0 (ms) | Q4_0 (ms) | Q8_0 / Q4_0 Speedup vs FP32 | Q8_0 / Q4_0 Eff. GB/s | Weight Bytes (FP32 → Q8_0 → Q4_0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `FFN_GateUp_M1_K2048_N8192` | 1.370 | 0.845 | 0.811 | **0.732** | **0.673** | **1.87× / 2.04×** | 24.27 / 14.00 | 67.11 MB → **17.83 MB** (-3.76×) → **9.44 MB** (-7.11×) |
| `FFN_Down_M1_K8192_N2048` | 1.996 | 0.831 | 0.824 | **0.737** | **1.135** | **2.71× / 1.76×** | 23.94 / 8.30 | 67.11 MB → **17.83 MB** (-3.76×) → **9.44 MB** (-7.11×) |
| `Attn_QO_M1_K2048_N2048` | 0.699 | 0.385 | 0.370 | **0.320** | **0.301** | **2.18× / 2.32×** | 13.81 / 7.81 | 16.78 MB → **4.46 MB** (-3.76×) → **2.36 MB** (-7.11×) |
| `Attn_KV_M1_K2048_N512` | 0.496 | 0.278 | 0.208 | **0.191** | **0.200** | **2.60× / 2.48×** | 5.77 / 2.93 | 4.19 MB → **1.11 MB** (-3.76×) → **0.59 MB** (-7.11×) |
| `Classifier_M1_K2048_N128256` | 12.695 | 6.337 | 6.412 | **3.681** | **2.204** | **3.45× / 5.76×** | **75.82 / 67.04** | 1,050.67 MB → **279.09 MB** (-3.76×) → **147.75 MB** (-7.11×) |
| `FFN_GateUp_M6_K2048_N8192` | 1.705 | 1.616 | 1.561 | **1.529** | **1.682** | 1.12× / 1.01× | 11.77 / 5.74 | 67.11 MB → **17.83 MB** → **9.44 MB** |
| `FFN_GateUp_M27_K2048_N8192` | 4.346 | 2.143 | 2.490 | **2.589** | **2.759** | **1.68× / 1.58×** | 7.29 / 3.81 | 67.11 MB → **17.83 MB** → **9.44 MB** |
| `FFN_GateUp_M28_K2048_N8192` | 4.385 | 2.143 | 2.416 | **2.517** | **2.794** | **1.74× / 1.57×** | 7.52 / 3.78 | 67.11 MB → **17.83 MB** → **9.44 MB** |
| `FFN_GateUp_M29_K2048_N8192` | 4.744 | 2.122 | 2.617 | **2.706** | **2.847** | **1.75× / 1.67×** | 7.01 / 3.72 | 67.11 MB → **17.83 MB** → **9.44 MB** |

Notes:
- On `Classifier_M1_K2048_N128256` where per-scope submit/wait overhead (~150–200 µs) is negligible, `Q8_0` achieves **3.45×** speedup (**75.8 GB/s**) and `Q4_0` achieves **5.76×** speedup (**67.0 GB/s**, **2.88× faster than FP16**).
- For `M=27/28/29` prefill, `Q8_0` and `Q4_0` are **1.57×–1.75× faster than FP32** while cutting weight memory by 3.76×–7.11×; they are slightly slower than FP16/BF16 because the barrier-free `M>1` kernel unpacks quantized blocks per output row without shared-memory weight reuse.

### Complete Resident FFN (`Benchmark_FFNMixedMetal`, `-benchtime=5x -count=3`, median resident interval)

| Rows | FP32 (ms) | FP16 (ms) | BF16 (ms) | Q8_0 (ms) | Q4_0 (ms) | Q8_0 / Q4_0 Speedup vs FP32 | Weight Bytes (FP32 → Q8_0 → Q4_0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `M=1` | 3.256 | 1.723 | 1.701 | **1.605** | **1.533** | **2.03× / 2.12×** | 201,326,592 → **53,477,376** (-3.76×) → **28,311,552** (-7.11×) |
| `M=27` | 12.254 | 5.696 | 5.473 | **6.329** | **7.085** | **1.94× / 1.73×** | 201,326,592 → **53,477,376** (-3.76×) → **28,311,552** (-7.11×) |
| `M=28` | 12.157 | 5.749 | 5.592 | **6.554** | **7.272** | **1.85× / 1.67×** | 201,326,592 → **53,477,376** (-3.76×) → **28,311,552** (-7.11×) |
| `M=29` | 13.254 | 6.069 | 6.088 | **6.844** | **7.673** | **1.94× / 1.73×** | 201,326,592 → **53,477,376** (-3.76×) → **28,311,552** (-7.11×) |

## `GATE-E` Release Readiness Checklist (NN-09 through NN-12)

- [x] **NN-09** (`#149`, merged `1dc533a1`): Multi-dtype ADR (`docs/adr-gpu-reduced-precision.md`), `gpu.DType` (`Float32`, `Float16`, `BFloat16`), checked byte sizing, raw/converting host transfers, and `Scope.Copy`.
- [x] **NN-10** (`#150`, merged `0c24093b`): FP16/BF16 resident weights with FP32-accumulated `M=1` GEMV and `M>1` GEMM (`docs/gpu-nn-10-fp16-bf16-matmul.md`).
- [x] **NN-11** (`#151`, merged `3c40a385`): Parallel SIMD `RMSNorm` (26× faster across 33 dispatches), tied FP16/BF16 `GatherRows`, and optional `Float16` `KVCache` (`docs/gpu-nn-11-rmsnorm-and-ops.md`).
- [x] **NN-12** (`gpu-nn-12`, PR open): GGUF-compatible `Q8_0`/`Q4_0` block-quantized weight tensors (`docs/adr-gpu-quantization.md`), `UploadBytes`/`DownloadBytesInto`, `Quantize*`/`Dequantize*`, fused Metal `M=1` GEMV and `M>1` GEMM (`docs/gpu-nn-12-quantized-matmul.md`).
- [x] **Backward compatibility & portability:** Zero regressions to FP32 `gpu` APIs, ANN Metal (`Sequential`), default CPU, `purego`, `CGO_ENABLED=0`, and Linux cross-builds; all 14 matrix checks pass.
- [ ] **Owner action (`GATE-E`):** Review/merge NN-12, tag and publish immutable `neuralnetwork` release (e.g., `v0.12.0`), and record exact tag, commit SHA, and module checksums in `TODO.md` before dispatching `LL-06` and `ONNX-01`.
