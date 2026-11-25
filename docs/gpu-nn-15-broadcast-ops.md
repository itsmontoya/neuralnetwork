# NN-15: Elementwise Arithmetic (`Sub`, `Div`, `Scale`) & Multidirectional Broadcast Binary Ops

**Date:** 2026-11-25  
**Hardware:** Apple M3 (8-core CPU, 24 GB unified memory, `darwin/arm64`)  
**Raw artifacts:** `.benchmarks/gpu-nn-15/run1/` and `.benchmarks/gpu-nn-15/run2/` (gitignored)

## Summary

NN-15 adds seven FP32 Metal compute operations to `gpu.Scope`:

- `Scope.Sub(left, right, destination *Tensor) error`
- `Scope.Div(left, right, destination *Tensor) error`
- `Scope.Scale(input, destination *Tensor, scale float32) error`
- `Scope.AddBroadcast(left, right, destination *Tensor) error`
- `Scope.SubBroadcast(left, right, destination *Tensor) error`
- `Scope.MulBroadcast(left, right, destination *Tensor) error`
- `Scope.DivBroadcast(left, right, destination *Tensor) error`

Existing `Scope.Add` and `Scope.Mul` remain 100% backward-compatible and continue
to require identical operand shapes. All new Metal shaders compile lazily via
`pthread_once` with `[options setFastMathEnabled:NO]` and inherit `NN-13`
storage-level lease and `Tensor.View` alias rejection automatically.

## Kernel Architecture & Broadcast Rules

### Same-Shape Arithmetic

- `Sub` (`nn_tensor_sub`) and `Div` (`nn_tensor_div`) extend
  `internal/device/elementwise_metal_source.h` using the contiguous 1D
  elementwise dispatch path.
- `Scale` (`nn_tensor_scale`) multiplies each FP32 element by an inline scalar
  `float32` parameter without allocating a constant device tensor.

### Multidirectional Broadcast (`internal/device/broadcast_metal_source.h`)

- Follows ONNX/NumPy right-aligned broadcasting rules for ranks `1..6`; rejects
  incompatible shapes, mismatched destination shape, or rank > 6 with
  `ErrInvalid`.
- Host-side stride coalescing (`buildBroadcastSpec`) drops unit output axes and
  merges adjacent axes with matching stride steps before dispatching across four
  specialized paths:
  1. **Fast path 1 (identical shapes / identical contiguous layout):** dispatches
     directly to `nn_tensor_{add,sub,mul,div}` with zero stride overhead.
  2. **Fast path 2 (last-axis broadcast, e.g., `[..., N] op [N]` or
     `[..., N] op [1, ..., 1, N]` in either operand order):** coalesces leading
     axes into `[rows, cols]` and dispatches `nn_tensor_broadcast_last_axis` on
     a 2D `(cols, rows, 1)` grid with coalesced memory access and zero
     intermediate scratch buffer.
  3. **Fast path 3 (scalar broadcast, `[1]` or `[1, ..., 1]` on either side):**
     dispatches `nn_tensor_broadcast_scalar` with a single scalar load.
  4. **General path (up to rank 6):** dispatches `nn_tensor_broadcast_general`
     using precomputed coalesced output and input strides.
- IEEE-754 FP32 arithmetic (`fastMathEnabled = NO`): `1/0 → +Inf`, `-1/0 → -Inf`,
  `0/0 → NaN`, `Inf - Inf → NaN`, `0 * Inf → NaN`, and `NaN` propagation.

## Independent FP64 Verification

Verified against independent FP64 CPU references across 1D–6D shapes, odd/prime
dimensions, non-commutative left/right broadcast ordering, special values
(`0/0`, `1/0`, `-1/0`, `Inf - Inf`, `0 * Inf`, `NaN`, signed zero, large
magnitudes), `Tensor.View` read sharing and destination alias rejection, and
8-worker × 12-iteration concurrent `-race -tags=metal` stress tests (zero data
races, zero native leaks).

| Operation | Shape / Path | Finite Count | p50 Abs Err | p95 Abs Err | p99 Abs Err | Max Abs Err | Max Rel Err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Sub` | `[4096]` (incl. edges) | 4,089 | `0` | `3.81e-06` | `3.81e-06` | `3.81e-06` | `5.95e-08` |
| `Div` | `[4096]` (incl. edges) | 4,083 | `1.76e-08` | `2.08e-07` | `8.46e-07` | `1.28e-05` | `5.89e-08` |
| `Scale` (`0.125`) | `[4096]` (incl. edges) | 4,090 | `0` | `0` | `0` | `0` | `0` |
| `Scale` (`-3.5`) | `[4096]` (incl. edges) | 4,088 | `4.77e-07` | `1.91e-06` | `2.86e-06` | `7.56e+22` (at `1e30`) | `5.95e-08` |
| `AddBroadcast` | `[32, 257] + [257]` (last-axis right) | 8,224 | `0` | `4.77e-07` | `9.54e-07` | `9.54e-07` | `5.96e-08` |
| `SubBroadcast` | `[257] - [32, 257]` (last-axis left) | 8,224 | `0` | `4.77e-07` | `9.54e-07` | `9.54e-07` | `5.96e-08` |
| `MulBroadcast` | `[3, 7, 5] * [1]` (scalar right) | 105 | `1.05e-07` | `4.29e-07` | `4.69e-07` | `4.74e-07` | `5.57e-08` |
| `DivBroadcast` | `[19, 33] / [19, 1]` (column bias) | 627 | `1.59e-08` | `1.11e-07` | `2.17e-07` | `3.67e-07` | `5.85e-08` |
| `DivBroadcast` | `[2, 1, 4, 1] / [1, 3, 1, 5]` (4D) | 120 | `2.01e-08` | `9.12e-08` | `2.02e-07` | `4.12e-07` | `5.72e-08` |
| `DivBroadcast` | `[2, 1, 3, 1, 2, 1] / [1, 2, 1, 2, 1, 3]` (6D) | 144 | `1.83e-08` | `1.84e-07` | `3.29e-07` | `6.73e-07` | `5.93e-08` |

All finite relative errors are within `≤ 5.96e-08` (half an FP32 ULP).

## Apple M3 Benchmarks (`-count=3`, median)

Command:

```bash
CGO_ENABLED=1 GOWORK=off go test -run '^$' \
  -bench 'Benchmark_RowBiasMetal|Benchmark_ArithmeticAndBroadcastMetal' \
  -benchmem -count=3 -tags=metal ./gpu
```

### Row Bias: `AddBroadcast` vs `ONNX-01` `GatherRows + Add` Workaround

| Workload | `AddBroadcast` Median | `GatherRows + Add` Median | Speedup | Device Scratch Saved |
| --- | ---: | ---: | ---: | ---: |
| `[1, 4096] + [4096]` (1 disp) | `0.113 ms/op` | `0.142 ms/op` | `1.26×` | `16 KB` → `0 B` |
| `[32, 4096] + [4096]` (1 disp) | `0.177 ms/op` | `0.167 ms/op` | `0.94×` (submit-bound) | `512 KB` → `0 B` |
| `[32, 4096] + [4096]` (32 dispatches / scope) | `0.445 ms/op` (`13.9 µs/disp`) | `4.217 ms/op` (`131.8 µs/disp`) | **`9.48×`** | `512 KB` → `0 B` |

Inside a multi-op resident `gpu.Scope`, eliminating the blit-to-compute encoder
transitions and intermediate `[M, N]` scratch traffic reduces per-dispatch
row-bias latency from `131.8 µs` to `13.9 µs` (**9.48× faster**) while freeing
the entire `[M, N]` FP32 scratch allocation (`512 KB` at `M=32, N=4096`).

### Isolated Operations

| Benchmark | Shape | Median Latency | Host Alloc |
| --- | --- | ---: | ---: |
| `Sub` | `[32, 4096]` | `0.170 ms/op` | `528 B/op`, `8 allocs/op` |
| `Div` | `[32, 4096]` | `0.169 ms/op` | `528 B/op`, `8 allocs/op` |
| `Scale` | `[32, 4096]` | `0.144 ms/op` | `528 B/op`, `8 allocs/op` |
| `AddBroadcast` (scalar) | `[32, 4096] + [1]` | `0.146 ms/op` | `576 B/op`, `9 allocs/op` |
| `AddBroadcast` (column) | `[32, 4096] + [32, 1]` | `0.185 ms/op` | `576 B/op`, `9 allocs/op` |
| `AddBroadcast` (4D) | `[2, 8, 32, 64] + [1, 8, 1, 64]` | `0.166 ms/op` | `576 B/op`, `9 allocs/op` |
