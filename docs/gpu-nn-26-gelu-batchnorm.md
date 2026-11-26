# NN-26: `Scope.GELU` / `Scope.GELUTanh` & `Scope.BatchNorm` Report

**Task:** NN-26 (`gpu-nn-26`, based on `v0.15.0` @ `cbb1df9423a0b1569296382b0ead18de6584c27b`)  
**Hardware:** Apple M3 (`darwin/arm64`, Metal, `CGO_ENABLED=1`)  
**Raw benchmark logs:** `.benchmarks/gpu-nn-26/run1/bench.txt` (gitignored)

---

## Exported API Summary

- `(*Scope).GELU(input, destination *Tensor) error`  
  Exact-erf FP32 GELU `0.5 * x * (1 + erf(x / sqrt(2)))` evaluated via `erfc` in FP32 with `[options setFastMathEnabled:NO]`. Saturates cleanly for `x >= 10` (`x`) and `x <= -10` (`0`), preserving `+Inf -> +Inf`, `-Inf -> 0`, and `NaN -> NaN`.
- `(*Scope).GELUTanh(input, destination *Tensor) error`  
  Tanh-approximate FP32 GELU `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` evaluated via `x * sigmoid(2 * inner)` to prevent negative-side `1 + tanh(z)` cancellation, with `[options setFastMathEnabled:NO]` and clean `|x| >= 10` saturation (`+Inf -> +Inf`, `-Inf -> 0`, `NaN -> NaN`).
- `(*Scope).BatchNorm(input, scale, bias, mean, variance, destination *Tensor, epsilon float32) error`  
  Stored-statistics FP32 inference batch normalization over rank ≥ 2 `[N, C, ...]` matching `tensor.BatchNormalizationInto` and ONNX opset-15 `BatchNormalization`:  
  `y = scale[c] * (x - mean[c]) / sqrt(variance[c] + epsilon) + bias[c]`.  
  Requires matching `Float32` `input`/`destination` (`count <= math.MaxUint32`), rank-1 `[C]` `Float32` parameters, finite non-negative `epsilon` (`0` permitted), and disjoint `destination` (including via `Tensor.View`). Dispatches a 3D `(spatial/4, C, N)` `float4`-vectorized Metal kernel when `spatial % 4 == 0` and a 3D `(spatial, C, N)` scalar kernel otherwise, compiled lazily via `pthread_once` with `[options setFastMathEnabled:NO]`.

---

## Numerical Verification Against Independent FP64 References

All kernels were verified against independent FP64 references (and `tensor.BatchNormalizationInto` for `BatchNorm`), covering 1D–5D shapes, odd/prime widths, `epsilon == 0`, special values (`0`, `-0`, subnormals, `±10`, `±100`, `±1000`, `±MaxFloat32`, `±Inf`, `NaN`, zero-variance division), view-alias rejection, and 8-worker `-race -tags=metal` concurrency stress with zero native resource leaks.

| Operation | Test Set | Finite Points | p50 Abs Err | p95 Abs Err | p99 Abs Err | Max Abs Err | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `Scope.GELU` | Dense sweep `[-12, 12]` | 20,001 | — | — | — | `4.06e-07` | `< 1e-6` PASS |
| `Scope.GELU` | Random + edges `[4096]` | 4,094 | `2.32e-10` | `2.03e-07` | `2.88e-07` | `3.99e-07` | `< 1e-6` PASS |
| `Scope.GELUTanh` | Dense sweep `[-12, 12]` | 20,001 | — | — | — | `5.12e-07` | `< 1e-6` PASS |
| `Scope.GELUTanh` | Random + edges `[4096]` | 4,094 | `2.34e-14` | `1.49e-07` | `2.96e-07` | `4.67e-07` | `< 1e-6` PASS |
| `Scope.BatchNorm` | `[32, 64]` (`eps=1e-5`, scalar) | 2,048 | `2.58e-08` | `2.08e-07` | `4.33e-07` | `1.45e-06` | `< 1e-5` PASS |
| `Scope.BatchNorm` | `[2, 32, 7, 7]` (`eps=1e-5`, scalar) | 3,136 | `2.10e-08` | `1.59e-07` | `3.23e-07` | `9.86e-07` | `< 1e-5` PASS |
| `Scope.BatchNorm` | `[4, 16, 14, 14]` (`eps=1e-5`, vec4) | 12,544 | `2.53e-08` | `2.16e-07` | `4.48e-07` | `1.70e-06` | `< 1e-5` PASS |
| `Scope.BatchNorm` | `[1, 256, 14, 14]` (`eps=1e-5`, vec4) | 50,176 | `2.56e-08` | `2.34e-07` | `5.17e-07` | `3.77e-06` | `< 1e-5` PASS |

---

## Apple M3 Benchmark Results

Command:
```bash
CGO_ENABLED=1 GOWORK=off go test -run=^$ -bench='Benchmark_GELUMetal|Benchmark_BatchNormMetal' -benchtime=1s -count=3 -tags=metal ./gpu
```

Median of 3 runs (isolated 1-dispatch `NewScope -> Encode -> Submit -> Wait -> Close` for Metal; paired `tensor.BatchNormalizationInto` for CPU):

| Benchmark | Shape | Elements | Median Latency | Effective BW | Allocs / Op | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `GELU` (Metal) | `[32, 4096]` | 131,072 | `0.381 ms` | `2.76 GB/s` | `528 B / 8` | Submit/wait-dominated at 1 dispatch |
| `GELU` (Metal) | `[2, 64, 3072]` | 393,216 | `0.487 ms` | `6.46 GB/s` | `528 B / 8` | Submit/wait-dominated at 1 dispatch |
| `GELUTanh` (Metal) | `[32, 4096]` | 131,072 | `0.372 ms` | `2.82 GB/s` | `528 B / 8` | Submit/wait-dominated at 1 dispatch |
| `GELUTanh` (Metal) | `[2, 64, 3072]` | 393,216 | `0.439 ms` | `7.16 GB/s` | `528 B / 8` | Submit/wait-dominated at 1 dispatch |
| `BatchNorm` (Metal) | `[32, 64, 56, 56]` | 6,422,528 | `1.056 ms` | `48.67 GB/s` | `528 B / 8` | **8.7× faster than CPU** (`vec4` 3D grid) |
| `BatchNorm` (CPU) | `[32, 64, 56, 56]` | 6,422,528 | `9.194 ms` | `5.59 GB/s` | `112 B / 4` | `tensor.BatchNormalizationInto` |
| `BatchNorm` (Metal) | `[1, 256, 14, 14]` | 50,176 | `0.149 ms` | `2.70 GB/s` | `528 B / 8` | Isolated submit/wait floor; zero-copy inside resident scope |
| `BatchNorm` (CPU) | `[1, 256, 14, 14]` | 50,176 | `0.073 ms` | `5.52 GB/s` | `112 B / 4` | `tensor.BatchNormalizationInto` |

---

## Handoff Notes for NN-27 (`MaxPool2D` & `AvgPool2D`)

- `internal/device/batchnorm_metal_source.h` establishes the 3D `dispatchThreads` + adaptive `(tgX, tgY, tgZ)` pattern bounded by `maxThreadsPerThreadgroup` and `maxTotalThreadsPerThreadgroup`; `NN-27` can reuse the same pattern for 3D `(outW, outH, N*C)` pooling grids.
- All new pipelines compile lazily via `pthread_once` with `[options setFastMathEnabled:NO]` and zero impact on ANN Metal or non-Metal builds.
