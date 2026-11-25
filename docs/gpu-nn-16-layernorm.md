# NN-16: Parallel SIMD `LayerNorm` (Optional Bias, FP32/FP16/BF16 Weights)

**Date:** 2026-11-25  
**Hardware:** Apple M3 (8-core CPU, 24 GB unified memory, `darwin/arm64`)  
**Raw artifacts:** `.benchmarks/gpu-nn-16/run1/` (gitignored)

## Summary

NN-16 adds ONNX opset-17 `LayerNormalization` to `gpu.Scope`:

- `Scope.LayerNorm(input, weight, bias, destination *Tensor, epsilon float32) error`

Normalizes each row along the last axis `width = shape[len(shape)-1]` for any
rank ≥ 1 `Float32` `input` and matching `Float32` `destination`:

$$y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot w_i + b_i$$

- `weight` is required: rank-1 `[width]` with `Float32`, `Float16`, or `BFloat16`.
- `bias` is optional (`nil` omits bias addition): when non-nil, rank-1 `[width]`
  with `Float32`, `Float16`, or `BFloat16`, independently of `weight` dtype.
- `epsilon` must be finite and positive (`epsilon > 0 && epsilon <= math.MaxFloat32`).
- Rejects quantized weights/biases, non-`Float32` input/destination, or
  destination aliasing `input`, `weight`, or `bias` (including via `Tensor.View`)
  with `ErrInvalid`.

## Kernel Architecture (`internal/device/layernorm_metal_source.h`)

- Compiles lazily via `pthread_once` with `[options setFastMathEnabled:NO]` as a
  single compact Metal pipeline (`nn_tensor_layer_norm`) with uniform dtype tags
  for `weight` and optional `bias` (avoiding 12-pipeline combinatorial explosion
  while supporting all mixed `Float32`/`Float16`/`BFloat16` combinations).
- Dispatches 1 threadgroup per row (`32`–`256` threads) cooperating via
  `simd_sum` and **disjoint `shared[34]` threadgroup broadcast slots**
  (`shared[0..31]` for per-simdgroup partials, `shared[32]` for shifted mean,
  `shared[33]` for `invStd = 1 / sqrt(var + epsilon)`), eliminating WAR
  threadgroup-memory races under concurrent GPU execution.
- **Numerically stable `x0`-shifted two-pass FP32 reduction:**
  - Anchors each row at `x0 = input[start]` so Pass 1 accumulates
    `shiftedMean = sum(x_i - x0) / W` and Pass 2 accumulates
    `var = sum(((x_i - x0) - shiftedMean)^2) / W`.
  - Eliminates both `E[x^2] - E[x]^2` catastrophic cancellation and FP32 mean
    rounding drift on large-offset rows (`1e5 + delta`), and yields exact zero
    variance on constant rows.
- Uses vectorized `float4` (`half4`/`bfloat4` for affine params) loads/stores
  when `width % 4 == 0`, with an exact scalar fallback for arbitrary `width >= 1`.

## Independent FP64 Verification

Verified against independent FP64 CPU references across `M = 1, 6, 27, 28, 29, 32`,
widths `1, 3, 17, 32, 33, 64, 65, 128, 257, 768, 1024, 2048, 4096`, 1D–4D shapes,
`bias == nil` and non-nil, `Float32`/`Float16`/`BFloat16` (including mixed
weight/bias dtypes), constant rows, `1e5 + delta` cancellation, very small
(`1e-18`) and large (`1e15`) values, `NaN`/`±Inf`, `Tensor.View` execution and
alias rejection, and 8-worker × 16-iteration concurrent `-race -tags=metal`
stress tests (zero data races, zero native leaks).

| Shape | Weight / Bias DType | p50 Abs Err | p95 Abs Err | p99 Abs Err | Max Abs Err |
| --- | --- | ---: | ---: | ---: | ---: |
| `[29, 257]` (scalar odd) | `Float32` / `nil` | `1.85e-08` | `1.04e-07` | `1.64e-07` | `2.98e-07` |
| `[29, 257]` (scalar odd) | `Float16` / `Float32` | `2.32e-08` | `1.08e-07` | `1.78e-07` | `3.32e-07` |
| `[29, 257]` (scalar odd) | `BFloat16` / `BFloat16` | `2.27e-08` | `1.07e-07` | `1.71e-07` | `3.87e-07` |
| `[1, 768]` | `Float32` / `Float32` | `2.49e-08` | `1.07e-07` | `1.50e-07` | `2.47e-07` |
| `[32, 768]` | `Float32` / `Float32` | `2.12e-08` | `1.07e-07` | `1.59e-07` | `3.10e-07` |
| `[32, 768]` | `BFloat16` / `Float16` | `2.13e-08` | `1.07e-07` | `1.58e-07` | `3.16e-07` |
| `[1, 2048]` | `Float32` / `Float32` | `1.59e-08` | `7.02e-08` | `1.11e-07` | `1.81e-07` |
| `[32, 2048]` | `Float32` / `Float32` | `2.03e-08` | `9.70e-08` | `1.51e-07` | `3.76e-07` |
| `[32, 2048]` | `Float16` / `Float16` | `2.03e-08` | `9.70e-08` | `1.54e-07` | `3.41e-07` |
| `[32, 2048]` | `BFloat16` / `BFloat16` | `2.03e-08` | `9.67e-08` | `1.53e-07` | `3.55e-07` |
| `[32, 4096]` | `Float32` / `Float32` | `1.97e-08` | `9.22e-08` | `1.44e-07` | `3.77e-07` |
| `[32, 4096]` | `BFloat16` / `BFloat16` | `1.97e-08` | `9.20e-08` | `1.46e-07` | `4.05e-07` |
| `[4, 256]` (`1e5 + delta` vec) | `Float32` / `Float32` | `2.35e-08` | `1.22e-07` | `1.80e-07` | `2.83e-07` |
| `[4, 257]` (`1e5 + delta` scalar) | `Float32` / `Float32` | `3.04e-08` | `1.70e-07` | `2.53e-07` | `3.41e-07` |

All finite absolute errors across the entire sweep and cancellation tests are
`≤ 4.05e-07`.

## Apple M3 Benchmarks (`-count=3`, median)

Command:

```bash
CGO_ENABLED=1 GOWORK=off go test -run '^$' \
  -bench 'Benchmark_LayerNormMetal' \
  -benchmem -count=3 -tags=metal ./gpu
```

| Shape | Dispatches | `Float32` | `Float16` | `BFloat16` | `Float32` (No Bias) | Per-Dispatch (`Float32`) | Host Alloc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `[1, 768]` | 1 | `0.118 ms/op` | `0.124 ms/op` | `0.120 ms/op` | — | `118.3 µs` (submit-bound) | `528 B/op`, `8 allocs/op` |
| `[32, 768]` | 1 | `0.131 ms/op` | `0.139 ms/op` | `0.135 ms/op` | — | `130.9 µs` (submit-bound) | `528 B/op`, `8 allocs/op` |
| `[1, 768]` | 33 | `0.302 ms/op` | `0.304 ms/op` | `0.305 ms/op` | `0.297 ms/op` | `9.15 µs/disp` | `528 B/op`, `8 allocs/op` |
| `[32, 768]` | 33 | `0.316 ms/op` | `0.319 ms/op` | `0.321 ms/op` | `0.312 ms/op` | `9.58 µs/disp` | `528 B/op`, `8 allocs/op` |
| `[1, 2048]` | 1 | `0.145 ms/op` | `0.144 ms/op` | `0.144 ms/op` | — | `144.8 µs` (submit-bound) | `528 B/op`, `8 allocs/op` |
| `[32, 2048]` | 1 | `0.149 ms/op` | `0.157 ms/op` | `0.160 ms/op` | — | `148.6 µs` (submit-bound) | `528 B/op`, `8 allocs/op` |
| `[1, 2048]` | 33 | `0.324 ms/op` | `0.327 ms/op` | `0.329 ms/op` | `0.317 ms/op` | `9.83 µs/disp` | `528 B/op`, `8 allocs/op` |
| `[32, 2048]` | 33 | `0.376 ms/op` | `0.371 ms/op` | `0.377 ms/op` | `0.356 ms/op` | `11.39 µs/disp` | `528 B/op`, `8 allocs/op` |
