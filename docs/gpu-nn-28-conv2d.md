# NN-28: NCHW `Scope.Conv2D` & `GATE-J` Release Readiness Report

**Task:** NN-28 (`gpu-nn-28`, stacked on `gpu-nn-27` @ `5bb7ea8f757004a2d3b031f2d2a6de7b040434ef`)  
**Hardware:** Apple M3 (`darwin/arm64`, Metal, `CGO_ENABLED=1`)  
**Raw benchmark logs:** `.benchmarks/gpu-nn-28/run1/bench.txt` (gitignored)

---

## Exported API Summary

- `type Conv2DConfig struct` (`gpu/conv2d_config.go`):
  `{Strides [2]int64, Pads [4]int64, Dilations [2]int64, Groups int64}` with pads ordered `[top, left, bottom, right]`.
- `Conv2DShape(input, weights []int64, config Conv2DConfig) ([]int64, error)`:
  Validates rank-4 positive NCHW input `[N, C, H, W]` and weights `[M, C/G, kH, kW]`, positive `Groups`/strides/dilations, nonnegative pads, group divisibility (`C % G == 0`, `M % G == 0`, `weights[1] == C/G`), positive output spatial dimensions, and `uint32` element-count limits, returning `ErrInvalid` on invalid geometry or overflow.
- `(*Scope).Conv2D(input, weights, bias, destination *Tensor, config Conv2DConfig) error`:
  Records NCHW 2D cross-correlation with `Float32` `input` and `destination`, `Float32`/`Float16`/`BFloat16` `weights`, and optional `bias` (`nil`, or rank-1 `[M]` in `Float32`/`Float16`/`BFloat16`). Accumulation is performed in FP32. Rejects quantized weights/bias and destination aliasing (direct or via `Tensor.View`) with `ErrInvalid`.
- **Metal kernels (`internal/device/conv2d_metal_source.h`):**
  Compiled lazily via `pthread_once` with `[options setFastMathEnabled:NO]` and FP32 accumulation:
  - **Implicit-GEMM `simdgroup_float8x8` fast path** (`nn_conv2d_implicit_gemm_{f32,f16,bf16}`): maps each group `g` as `[N*outH*outW, (C/G)*kH*kW] × [M/G, (C/G)*kH*kW]^T` using `(BM=32, BN=64, BK=32)`, 4 SIMD groups (128 threads), **zero materialized im2col buffer**, dual `k >= K` and out-of-bounds spatial zero-padding, fused optional bias addition, and coalesced NCHW scatter stores. Dispatched when `M/G >= 8 && N*outH*outW >= 8 && (C/G)*kH*kW >= 8`.
  - **Direct/depthwise fallback** (`nn_conv2d_direct_{f32,f16,bf16}`): 3D `(outW, outH, N*M)` adaptive-tile grid for depthwise (`G == C`), small-channel, and small-spatial shapes below the implicit-GEMM threshold.

---

## Numerical Verification Against Independent FP64 References & `tensor.Conv2DInto`

All 11 geometry suites × 3 weight/bias dtypes (`Float32`, `Float16`, `BFloat16`) were verified against independent FP64 references (and `tensor.Conv2DInto` for `Float32`), with all max absolute errors `< 3e-5`:

| Geometry | Path | Output Elems | `Float32` p50 / p95 / p99 / Max | `Float16` Max | `BFloat16` Max |
| --- | --- | ---: | --- | ---: | ---: |
| `1x1_direct_small` `[1,4,3,3] -> [1,5,3,3]` | Direct | 45 | `4.58e-8` / `1.83e-7` / `2.78e-7` / `2.78e-7` | `3.10e-7` | `3.03e-7` |
| `1x1_implicit_gemm` `[2,33,7,7] -> [2,65,7,7]` | Implicit-GEMM | 6,370 | `2.10e-7` / `8.86e-7` / `1.41e-6` / `3.44e-6` | `2.52e-6` | `2.91e-6` |
| `3x3_implicit_gemm_odd` `[3,11,9,10] -> [3,33,9,10]` | Implicit-GEMM | 8,910 | `4.92e-7` / `2.22e-6` / `3.61e-6` / `8.12e-6` | `7.41e-6` | `1.02e-5` |
| `3x3_no_bias_resnet_style` `[2,32,14,14] -> [2,64,14,14]` | Implicit-GEMM | 25,088 | `1.50e-6` / `6.63e-6` / `1.10e-5` / `2.64e-5` | `2.95e-5` | `2.61e-5` |
| `5x5_strided_asymmetric_pad` `[2,8,15,17] -> [2,16,7,6]` | Implicit-GEMM | 1,344 | `9.69e-7` / `3.89e-6` / `7.37e-6` / `1.11e-5` | `1.04e-5` | `1.68e-5` |
| `7x7_resnet_stem` `[2,3,28,28] -> [2,32,14,14]` | Implicit-GEMM | 12,544 | `7.52e-7` / `3.41e-6` / `5.54e-6` / `1.54e-5` | `1.20e-5` | `9.97e-6` |
| `dilated_3x3` `[2,10,15,13] -> [2,18,15,13]` | Implicit-GEMM | 7,020 | `4.15e-7` / `1.82e-6` / `2.96e-6` / `5.62e-6` | `1.45e-5` | `7.90e-6` |
| `grouped_g2_implicit_gemm` `[2,16,8,8] -> [2,24,8,8]` | Implicit-GEMM | 3,072 | `3.57e-7` / `1.53e-6` / `2.45e-6` / `4.62e-6` | `4.09e-6` | `4.45e-6` |
| `grouped_g4_direct_fallback` `[2,12,7,7] -> [2,16,7,7]` | Direct | 1,568 | `1.43e-7` / `6.25e-7` / `9.55e-7` / `1.64e-6` | `1.36e-6` | `3.04e-6` |
| `depthwise_g_eq_c` `[2,16,11,11] -> [2,16,11,11]` | Direct | 3,872 | `6.22e-8` / `2.59e-7` / `4.14e-7` / `9.56e-7` | `1.11e-6` | `9.64e-7` |
| `edge_1x1_spatial_out` `[1,12,3,3] -> [1,16,1,1]` | Direct | 16 | `5.78e-7` / `1.84e-6` / `1.84e-6` / `1.84e-6` | `2.24e-6` | `4.18e-6` |

Additional Metal verification:
- **Special values (`NaN`, `+Inf`, `-Inf`):** Passed on both `implicit_gemm_odd_K_padded` (`K=27`, not divisible by 32) and `direct_fallback_padded` with exact `NaN`/`±Inf` propagation and zero cross-lane contamination.
- **View-alias rejection:** Verified on real Metal (`ErrInvalid`).
- **1-`Scope` ResNet Bottleneck** (`Conv2D 1×1 → BatchNorm → ReLU → Conv2D 3×3 → BatchNorm → ReLU → Conv2D 1×1 → BatchNorm → Add → ReLU → MaxPool2D`): `0.000e+00` max abs err vs paired CPU `tensor` execution, zero im2col allocations, zero native leaks.
- **Concurrency:** 8-worker × 16-iteration `-race -tags=metal` stress test passed with zero races and zero native leaks (`ResourceSnapshot`).

---

## Apple M3 Benchmark Results

Command:
```bash
CGO_ENABLED=1 GOWORK=off go test -run=^$ -bench='Benchmark_Conv2DMetal|Benchmark_ResNetBottleneckMetal' -benchtime=1s -count=3 -tags=metal ./gpu
```

### Isolated `Conv2D` (Median of 3 runs, zero im2col buffer)

| Layer | Metal `Float32` | Metal `Float16` | Metal `BFloat16` | CPU `tensor.Conv2DInto` | Speedup (Metal FP32 vs CPU) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ResNet50_3x3` `[8,64,56,56] -> 64` | `2.081 ms` (`889.0 GFLOP/s`) | `2.085 ms` (`887.1 GFLOP/s`) | `2.092 ms` (`884.1 GFLOP/s`) | `35.56 ms` (`52.02 GFLOP/s`) | **17.1×** |
| `ResNet50_1x1` `[8,256,56,56] -> 64` | `1.100 ms` (`747.3 GFLOP/s`) | `1.085 ms` (`757.4 GFLOP/s`) | `1.088 ms` (`755.9 GFLOP/s`) | `19.68 ms` (`41.78 GFLOP/s`) | **17.9×** |
| `ViTPatch` `[8,3,224,224] -> 768` (`k=16 s=16`) | `2.053 ms` (`901.1 GFLOP/s`) | `2.049 ms` (`902.9 GFLOP/s`) | `2.058 ms` (`898.9 GFLOP/s`) | `250.4 ms` (`7.388 GFLOP/s`) | **122.0×** |
| `MNIST` `[32,1,28,28] -> 32` (`k=5 p=2`) | `0.366 ms` (`109.7 GFLOP/s`) | `0.362 ms` (`110.9 GFLOP/s`) | `0.364 ms` (`110.2 GFLOP/s`) | `0.999 ms` (`40.20 GFLOP/s`) | **2.7×** |

### Complete 11-Op ResNet-50 Bottleneck in 1 `gpu.Scope` (`[8, 256, 56, 56] -> 64 -> 64 -> 256 + Add + ReLU + MaxPool2D`)

| Execution Mode | Median Latency | Host Allocs / Op | Speedup vs CPU |
| --- | ---: | ---: | ---: |
| Metal `Float32` (1 `Scope`, 0 im2col) | `6.821 ms` | `1,592 B / 17` | **23.3×** |
| Metal `Float16` weights (1 `Scope`, 0 im2col) | `6.847 ms` | `1,592 B / 17` | **23.2×** |
| Metal `BFloat16` weights (1 `Scope`, 0 im2col) | `6.864 ms` | `1,592 B / 17` | **23.2×** |
| CPU `tensor` (`Float32`) | `159.0 ms` | `464 B / 16` | `1.0×` |

---

## `GATE-J` Release Readiness Checklist (`v0.16.0`: `NN-26`, `NN-27`, `NN-28`)

- [x] **NN-26 (`gpu-nn-26`, PR #167):** `Scope.GELU`, `Scope.GELUTanh`, and `Scope.BatchNorm` (`[N, C, ...]` stored-statistics inference). Verified against independent FP64 references (`< 1e-6` GELU/GELUTanh, `< 4e-6` BatchNorm), `BatchNorm` **8.7×** vs CPU at `[32, 64, 56, 56]`, 8-worker `-race -tags=metal` clean, report in `docs/gpu-nn-26-gelu-batchnorm.md`.
- [x] **NN-27 (`gpu-nn-27`, PR #168):** `Pool2DConfig`, `Pool2DShape`, `Scope.MaxPool2D`, and `Scope.AvgPool2D` (NCHW, floor/ceil, asymmetric pads, dilations, `CountIncludePad`). Bit-exact match against `tensor.MaxPool2DInto`/`AveragePool2DInto` and FP64 verified, **97.6×** (`MaxPool2D`) / **32.9×** (`AvgPool2D`) on ResNet-50 stem, 8-worker `-race -tags=metal` clean, report in `docs/gpu-nn-27-pool2d.md`.
- [x] **NN-28 (`gpu-nn-28`):** `Conv2DConfig`, `Conv2DShape`, and `Scope.Conv2D` (implicit-GEMM `simdgroup_float8x8` `(BM=32, BN=64, BK=32)` with zero im2col buffer + direct/depthwise fallback, `Float32`/`Float16`/`BFloat16` weights & optional bias). FP64/`tensor.Conv2DInto` verified (`max err < 3e-5`), **17.1×–122.0×** isolated M3 speedup (`~890–903 GFLOP/s`), **23.3×** full 1-`Scope` ResNet-50 bottleneck speedup (`6.82 ms` vs `159.0 ms`), 8-worker `-race -tags=metal` clean, report in `docs/gpu-nn-28-conv2d.md`.
- [x] **Portability & Backward Compatibility:** All 14 entries of the Required Test Matrix pass across default CPU, `purego`, `CGO_ENABLED=0`, Linux `amd64` cross-builds, and `CGO_ENABLED=1 -tags=metal` (`-race` included). All existing `v0.12.1`–`v0.15.0` `gpu` APIs and ANN Metal (`Sequential`) paths remain unchanged and passing.
- [x] **Numerics & Resource Safety:** All new Metal kernels compile lazily via `pthread_once` with `[options setFastMathEnabled:NO]` and FP32 accumulation; storage-level `Tensor.View` alias rejection and `ResourceSnapshot` zero-leak invariants verified across all new ops.
- [ ] **Owner Action (`GATE-J`):** Review and merge PRs #167 (`NN-26`), #168 (`NN-27`), and `NN-28`, tag and publish immutable `v0.16.0`, and record the commit SHA and module checksums in `TODO.md`.
