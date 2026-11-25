# NN-14: Standard Activations (`ReLU`, `Sigmoid`, `Tanh`) & Parallel SIMD `Softmax` / `SoftmaxAxis`

**Date:** 2026-11-25  
**Hardware:** Apple M3 (8-core CPU, 24 GB unified memory, `darwin/arm64`)  
**Raw artifacts:** `.benchmarks/gpu-nn-14/run1/` (gitignored)

## Summary

NN-14 adds five FP32 Metal compute operations to `gpu.Scope`:

- `Scope.ReLU(input, destination *Tensor) error`
- `Scope.Sigmoid(input, destination *Tensor) error`
- `Scope.Tanh(input, destination *Tensor) error`
- `Scope.Softmax(input, destination *Tensor) error`
- `Scope.SoftmaxAxis(input, destination *Tensor, axis int) error`

All new Metal shaders compile lazily via `pthread_once` with
`[options setFastMathEnabled:NO]` (zero impact on ANN startup or existing
pipelines) and inherit `NN-13` storage-level lease and `Tensor.View` alias
rejection automatically.

## Kernel Architecture & Edge-Case Semantics

### Elementwise Activations (`internal/device/elementwise_metal_source.h`)

- **`ReLU(x)`:** `isnan(x) ? x : (x > 0.0f ? x : 0.0f)`. Propagates `NaN`,
  maps `-Inf` and negative values to `0.0`, preserves `+Inf` and positive normal
  values.
- **`Sigmoid(x)`:** `exponent = exp(-fabs(x)); (x < 0.0f ? exponent : 1.0f) / (1.0f + exponent)`.
  Numerically stable across all magnitudes (`+Inf → 1.0`, `-Inf → 0.0`, `NaN → NaN`).
- **`Tanh(x)`:** `fabs(x) >= 10.0f ? copysign(1.0f, x) : tanh(x)`. Saturates
  cleanly to `±1.0` for large finite `|x|` and `±Inf` without `Inf/Inf`
  overflow while preserving `NaN` propagation.

### Parallel SIMD Softmax (`internal/device/softmax_metal_source.h`)

- **Contiguous last-axis (`inner == 1`) fast path (`nn_tensor_softmax_last_axis`):**
  1 threadgroup per row (`32`–`256` threads) cooperating via `simd_max` /
  `simd_sum` and threadgroup reduction using disjoint broadcast slots
  (`shared[34]`: `shared[0..31]` for SIMD partials, `shared[32]` for `rowMax`,
  `shared[33]` for `rowSum`), preventing WAR threadgroup-memory races under
  concurrent GPU execution. Uses `float4` loads/stores when `axisSize % 4 == 0`
  and exact scalar loops otherwise.
- **Strided general-axis (`inner > 1`) path (`nn_tensor_softmax_strided`):**
  Maps each thread to one `(outer, inner)` slice of `[outer, axisSize, inner]`
  with coalesced access across adjacent `inner` lanes.
- **Edge-case behavior (both paths):**
  - `axisSize == 1`: returns `1.0` for finite inputs; `NaN` for `NaN` or `±Inf`.
  - Large constant offsets (`±1000`): subtracting `rowMax` prevents `exp`
    overflow/underflow.
  - Masked `-Inf` entries in finite rows: `exp(-Inf - rowMax) = 0.0` (matches
    causal/padding attention masking).
  - Rows containing `NaN`, `+Inf`, or all `-Inf`: produce `NaN` across the row
    via IEEE-754 `exp(x - rowMax) / sum` propagation.

## Independent FP64 Verification

Verified against independent FP64 CPU references across 1D/2D/3D/4D shapes,
odd/prime widths (`1, 3, 17, 31, 32, 33, 64, 128, 257, 1024, 4096, 128256`),
special values (`NaN`, `±Inf`, `±100`, `±1000`, `±MaxFloat32`, subnormals,
signed zero), view-aliased rejection, and 8-worker × 12-iteration concurrent
`-race -tags=metal` stress tests (zero WAR races, zero native leaks).

| Operation | Shape / Path | Finite Count | p50 Abs Err | p95 Abs Err | p99 Abs Err | Max Abs Err | Max Rel Err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ReLU` | `[4096]` (incl. edges) | 4,094 | `0` | `0` | `0` | `1e-40` (subnormal flush) | `0` |
| `Sigmoid` | `[4096]` (incl. edges) | 4,095 | `4.02e-09` | `5.55e-08` | `6.94e-08` | `8.60e-08` | `2.21e-07` |
| `Tanh` | `[4096]` (incl. edges) | 4,095 | `1.59e-09` | `5.76e-08` | `8.56e-08` | `1.19e-07` | `1.19e-07` |
| `Softmax` | `[4, 4096]` (`float4` last-axis) | 16,384 | `5.87e-14` | `8.83e-11` | `2.26e-10` | `5.45e-10` | — |
| `Softmax` | `[1, 128256]` (`float4` vocab) | 128,256 | `1.01e-15` | `1.77e-12` | `5.92e-12` | `1.97e-11` | — |
| `Softmax` | `[2, 257]` (scalar last-axis) | 514 | `9.62e-13` | `1.34e-09` | `4.16e-09` | `7.08e-09` | — |
| `SoftmaxAxis` | `[2, 5, 7, 13]` axis 1 (strided) | 910 | `1.44e-09` | `4.59e-08` | `8.38e-08` | `1.58e-07` | — |
| `SoftmaxAxis` | `[2, 5, 7, 13]` axis 2 (strided) | 910 | `5.44e-10` | `3.75e-08` | `8.58e-08` | `1.75e-07` | — |
| Resident MLP | `MatMulRT → ReLU → MatMulRT → Softmax` (`[8,64]→128→33`) | 264 | `1.32e-09` | `3.59e-09` | `4.39e-09` | `5.71e-09` | — |

## Apple M3 Benchmarks (`-count=3`, median)

Command:

```bash
CGO_ENABLED=1 GOWORK=off go test -run '^$' \
  -bench 'Benchmark_ActivationsMetal|Benchmark_SoftmaxMetal|Benchmark_ResidentMLPMetal' \
  -benchmem -count=3 -tags=metal ./gpu
```

### Isolated Operations

| Benchmark | Shape | Median Latency | Host Alloc |
| --- | --- | ---: | ---: |
| `ReLU` | `[1, 4096]` | `0.114 ms/op` | `528 B/op`, `8 allocs/op` |
| `ReLU` | `[32, 4096]` | `0.149 ms/op` | `528 B/op`, `8 allocs/op` |
| `ReLU` | `[1, 128256]` | `0.148 ms/op` | `528 B/op`, `8 allocs/op` |
| `Sigmoid` | `[1, 4096]` | `0.114 ms/op` | `528 B/op`, `8 allocs/op` |
| `Sigmoid` | `[32, 4096]` | `0.149 ms/op` | `528 B/op`, `8 allocs/op` |
| `Sigmoid` | `[1, 128256]` | `0.149 ms/op` | `528 B/op`, `8 allocs/op` |
| `Tanh` | `[1, 4096]` | `0.114 ms/op` | `528 B/op`, `8 allocs/op` |
| `Tanh` | `[32, 4096]` | `0.177 ms/op` | `528 B/op`, `8 allocs/op` |
| `Tanh` | `[1, 128256]` | `0.175 ms/op` | `528 B/op`, `8 allocs/op` |
| `Softmax` (last-axis SIMD) | `[1, 4096]` | `0.145 ms/op` | `528 B/op`, `8 allocs/op` |
| `Softmax` (last-axis SIMD) | `[32, 4096]` | `0.180 ms/op` | `528 B/op`, `8 allocs/op` |
| `Softmax` (last-axis SIMD) | `[1, 128256]` | `0.286 ms/op` | `528 B/op`, `8 allocs/op` |
| `SoftmaxAxis` (strided, axis 1) | `[32, 64, 64]` | `0.228 ms/op` | `528 B/op`, `8 allocs/op` |

### Full Single-Scope Resident MLP (`MatMulRightTranspose → ReLU → MatMulRightTranspose → Softmax`)

MLP dimensions: `in=1024`, `hidden=2048`, `classes=1000`, FP32 resident weights,
**1 `gpu.Scope` submission** and **0 intermediate host transfers** verified via
`internal/device.ResourceSnapshot`.

| Batch | Total Latency (incl. input upload + output readback) | Resident Scope Latency | Host Alloc |
| --- | ---: | ---: | ---: |
| `M=1` | `0.565 ms/op` | `0.563 ms/op` | `528 B/op`, `8 allocs/op` |
| `M=32` | `1.762 ms/op` | `1.744 ms/op` | `528 B/op`, `8 allocs/op` |
