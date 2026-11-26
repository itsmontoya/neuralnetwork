# NN-27: 2D Spatial Pooling (`Scope.MaxPool2D` & `Scope.AvgPool2D`) Report

**Task:** NN-27 (`gpu-nn-27`, stacked on `gpu-nn-26` @ `a6226c60773dc96d24a8e8b90866056eb0dcb9e3`)  
**Hardware:** Apple M3 (`darwin/arm64`, Metal, `CGO_ENABLED=1`)  
**Raw benchmark logs:** `.benchmarks/gpu-nn-27/run1/bench.txt` (gitignored)

---

## Exported API Summary

- `type Pool2DConfig struct` (`gpu/pool2d_config.go`):
  `{Kernel [2]int64, Strides [2]int64, Pads [4]int64, Dilations [2]int64, CeilMode bool, CountIncludePad bool}` with pads ordered `[top, left, bottom, right]`.
- `Pool2DShape(input []int64, config Pool2DConfig) ([]int64, error)`:
  Validates rank-4 positive NCHW input, positive kernel/stride/dilation, nonnegative pads, floor/ceil geometry (including trailing ceil window removal matching `tensor.Pool2DShape` and ONNX), and strictly positive output spatial dimensions, returning `ErrInvalid` on invalid geometry or overflow.
- `(*Scope).MaxPool2D(input, destination *Tensor, config Pool2DConfig) error`:
  FP32 NCHW 2D spatial max pooling into a disjoint destination matching `Pool2DShape(input.Shape(), config)`. Excludes out-of-bounds padded positions (empty windows yield `-Inf`), propagates `NaN`, ignores `config.CountIncludePad` (matching `tensor.MaxPool2DInto` and ONNX `MaxPool`), and rejects direct or `Tensor.View` aliasing.
- `(*Scope).AvgPool2D(input, destination *Tensor, config Pool2DConfig) error`:
  FP32 NCHW 2D spatial average pooling into a disjoint destination matching `Pool2DShape(input.Shape(), config)` with FP32 accumulation. Divides by the valid in-bounds sample count when `CountIncludePad == false` (`0/0 -> NaN` on empty windows) or `Kernel[0] * Kernel[1]` when `CountIncludePad == true` (matching `tensor.AveragePool2DInto` and ONNX `AveragePool`). Supports arbitrary positive dilations (matching ONNX opset 19+ and `tensor.AveragePool2DInto` for `[1, 1]`).
- **Metal kernels (`internal/device/pool2d_metal_source.h`):**
  `nn_tensor_max_pool2d` and `nn_tensor_avg_pool2d`, compiled lazily via `pthread_once` with `[options setFastMathEnabled:NO]` and dispatched over a 3D `(outW, outH, N*C)` grid with adaptive spatial-tile threadgroups (`16×16×1` when `outH > 1`, channel-filled when `outH == 1`).

---

## Numerical Verification Against Independent FP64 References & `tensor`

All 10 geometry suites (2×2, 3×3, odd floor/ceil, asymmetric padding, dilated 2×3 and 2×2, `CountIncludePad=false/true`, global 7×7) matched `tensor.MaxPool2DInto` and `tensor.AveragePool2DInto` (for unit dilation) **bit-for-bit**, and matched independent FP64 references:

| Geometry | Output Elements | `MaxPool2D` vs FP64 | `AvgPool2D` p50 Abs Err | `AvgPool2D` p95 Abs Err | `AvgPool2D` p99 Abs Err | `AvgPool2D` Max Abs Err |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `2x2_stride2_even` `[2,4,8,8]` | 128 | `0` (exact) | `0.00e+00` | `0.00e+00` | `2.98e-08` | `2.98e-08` |
| `2x2_stride2_odd_floor` `[2,3,7,7]` | 54 | `0` (exact) | `0.00e+00` | `2.98e-08` | `2.98e-08` | `2.98e-08` |
| `2x2_stride2_odd_ceil_excl_pad` `[2,3,7,7]` | 96 | `0` (exact) | `0.00e+00` | `0.00e+00` | `2.98e-08` | `2.98e-08` |
| `2x2_stride2_odd_ceil_incl_pad` `[2,3,7,7]` | 96 | `0` (exact) | `0.00e+00` | `0.00e+00` | `2.98e-08` | `2.98e-08` |
| `3x3_stride2_pad1_resnet` `[2,16,28,28]` | 6,272 | `0` (exact) | `2.48e-09` | `2.65e-08` | `3.97e-08` | `7.29e-08` |
| `asymmetric_pad_excl_pad` `[2,5,9,10]` | 440 | `0` (exact) | `3.10e-10` | `1.99e-08` | `4.97e-08` | `5.96e-08` |
| `asymmetric_pad_incl_pad` `[2,5,9,10]` | 480 | `0` (exact) | `1.24e-09` | `9.93e-09` | `2.98e-08` | `3.97e-08` |
| `dilated_2x3_excl_pad` `[2,4,11,13]` | 360 | `0` (exact) | `2.48e-09` | `2.65e-08` | `3.31e-08` | `6.29e-08` |
| `dilated_2x2_ceil_incl_pad` `[2,3,10,9]` | 144 | `0` (exact) | `2.48e-09` | `1.99e-08` | `2.98e-08` | `3.97e-08` |
| `global_pool_7x7` `[4,32,7,7]` | 128 | `0` (exact) | `5.17e-09` | `2.22e-08` | `3.07e-08` | `3.56e-08` |

Special-value and concurrency tests verified all-negative `MaxPool2D` (no zero clamping), `NaN`/`±Inf` propagation, empty padded windows (`-Inf` / `NaN` / `0`), view-alias rejection on Metal, and 8-worker × 16-iteration `-race -tags=metal` stress with zero native buffer or scope leaks (`ResourceSnapshot`).

---

## Apple M3 Benchmark Results

Command:
```bash
CGO_ENABLED=1 GOWORK=off go test -run=^$ -bench='Benchmark_Pool2DMetal' -benchtime=1s -count=3 -tags=metal ./gpu
```

Median of 3 runs (isolated 1-dispatch `NewScope -> Encode -> Submit -> Wait -> Close` for Metal; paired `tensor.MaxPool2DInto` / `tensor.AveragePool2DInto` for CPU):

| Operation | Shape & Geometry | Metal Median | Metal BW | CPU Median | CPU BW | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `MaxPool2D` | `[32, 64, 112, 112]` `k=3 s=2 p=1` (ResNet-50 stem) | `2.013 ms` | `63.80 GB/s` | `196.5 ms` | `0.65 GB/s` | **97.6×** |
| `MaxPool2D` | `[32, 512, 7, 7]` `k=7 s=1 p=0` (ResNet-50 global) | `0.279 ms` | `11.74 GB/s` | `1.389 ms` | `2.36 GB/s` | **5.0×** |
| `MaxPool2D` | `[32, 32, 28, 28]` `k=2 s=2 p=0` (MNIST) | `0.450 ms` | `8.93 GB/s` | `1.579 ms` | `2.54 GB/s` | **3.5×** |
| `AvgPool2D` | `[32, 64, 112, 112]` `k=3 s=2 p=1` (ResNet-50 stem) | `1.903 ms` | `67.48 GB/s` | `62.66 ms` | `2.05 GB/s` | **32.9×** |
| `AvgPool2D` | `[32, 512, 7, 7]` `k=7 s=1 p=0` (ResNet-50 global) | `0.387 ms` | `8.46 GB/s` | `0.673 ms` | `4.87 GB/s` | **1.7×** |
| `AvgPool2D` | `[32, 32, 28, 28]` `k=2 s=2 p=0` (MNIST) | `0.489 ms` | `8.21 GB/s` | `1.512 ms` | `2.66 GB/s` | **3.1×** |

---

## Handoff Notes for NN-28 (`Scope.Conv2D`)

- `gpu/pool2d_config.go` provides `checkedAddInt64` and `checkedMulInt64` in package `gpu` that `gpu/conv2d_config.go` (`Conv2DConfig` / `Conv2DShape`) can reuse directly for overflow-safe spatial dimension validation.
- `internal/device/pool2d_metal_source.h` and `nn_metal_scope_encode_tensor_pool2d` demonstrate the 3D `(outW, outH, N*C)` adaptive-tile dispatch pattern (`16×16×1` spatial tiles for `outH > 1`, channel-filled when `outH == 1`), which can be reused for `NN-28`'s direct/depthwise `Conv2D` fallback alongside the `(BM=32, BN=64, BK=32)` `simdgroup_float8x8` implicit-GEMM fast path.
