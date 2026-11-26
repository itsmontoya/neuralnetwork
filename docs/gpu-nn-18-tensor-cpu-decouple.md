# NN-18: Decouple `neuralnetwork/tensor` CPU `MatMul`/`Gemm` from Legacy Copying Metal Adapter

**Date:** 2026-11-25  
**Hardware:** Apple M3 (8-core CPU, 24 GB unified memory, `darwin/arm64`)  
**Raw artifacts:** `.benchmarks/gpu-nn-18/run1/` (gitignored)

## Summary

NN-18 removes the legacy `//go:build metal && !purego && darwin && cgo` copying
`matrix.Matrix` adapter in `tensor/matrix_product.go` and makes the host
`internal/float32product` implementation unconditional across all build tags:

- `tensor.MatMulInto`, `tensor.GemmInto`, and `tensor.PreparedRightTranspose`
  now always execute on the host CPU/NEON `internal/float32product` path,
  including under `CGO_ENABLED=1 -tags=metal`.
- `tensor.PrepareRightTranspose` now packs eligible ARM64 weights into
  eight-output NEON tiles under `-tags=metal` just as it does under default CPU
  builds.
- `matrix/`, `model/`, `internal/device/`, and `gpu/` are completely untouched;
  existing ANN Metal (`matrix`, `model.Sequential`) and explicit resident GPU
  execution (`neuralnetwork/gpu`) remain unchanged.

## Verification

- Added `Test_MatrixProduct_MetalUsesHostPath` in
  `tensor/matrix_product_metal_test.go` (`//go:build darwin && arm64 && cgo && metal && !purego`)
  and `Test_MatrixProduct_MetalPacksOnARM64` in
  `tensor/matrix_product_metal_packed_test.go` (`//go:build darwin && arm64 && cgo && metal && !purego && !race`),
  confirming via `internal/device.ExperimentalRuntime().ResourceSnapshot()` that
  `MatMulInto`, `GemmInto` (all four transpose combinations), and
  `PreparedRightTranspose.GemmInto` produce zero Metal buffer allocations,
  zero scopes, and zero command submissions while packing weights on ARM64
  (when `!race`) and matching dense `GemmInto` bit-for-bit.
- Updated `Test_Gemm_PreparedRightTranspose` in
  `tensor/prepared_right_transpose_test.go` to assert
  `weight.packed == float32product.CanPackRightTranspose(...)` unconditionally.

## Apple M3 Paired Benchmarks (`-count=3`, median)

Command:

```bash
CGO_ENABLED=1 GOWORK=off go test -run '^$' \
  -bench 'Benchmark_(MatMulInto|GemmInto)' \
  -benchmem -count=3 -tags=metal ./tensor
```

Reference CPU baseline (`CGO_ENABLED=0`, no `-tags=metal`):

```bash
CGO_ENABLED=0 GOWORK=off go test -run '^$' \
  -bench 'Benchmark_(MatMulInto|GemmInto)' \
  -benchmem -count=3 ./tensor
```

| Benchmark | Before (`-tags=metal`) | After (`-tags=metal`) | `CGO_ENABLED=0` Ref | Latency Delta | Allocation Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `MatMulInto/M1_1024x4096` | `2.794 ms/op` (`16.83 MB/op`, `10 allocs`) | `2.300 ms/op` (`16.5 KB/op`, `3 allocs`) | `2.411 ms/op` (`16.5 KB/op`, `3 allocs`) | **1.21× faster** | **1,021× less** (`-99.90%`) |
| `MatMulInto/M32_1024x4096` | `4.446 ms/op` (`18.48 MB/op`, `26 allocs`) | `73.390 ms/op` (`524 KB/op`, `3 allocs`) | `74.043 ms/op` (`524 KB/op`, `3 allocs`) | CPU path\* | **35.2× less** (`-97.16%`) |
| `GemmInto/Dense/M1_1024x4096` | `2.919 ms/op` (`16.82 MB/op`, `26 allocs`) | `0.189 ms/op` (`17.2 KB/op`, `12 allocs`) | `0.208 ms/op` (`17.2 KB/op`, `12 allocs`) | **15.4× faster** | **979× less** (`-99.90%`) |
| `GemmInto/Prepared/M1_1024x4096` | `2.980 ms/op` (`16.82 MB/op`, `26 allocs`) | `0.163 ms/op` (`1.02 KB/op`, `11 allocs`) | `0.172 ms/op` (`1.02 KB/op`, `11 allocs`) | **18.3× faster** | **16,422× less** (`-99.99%`) |
| `GemmInto/Dense/M32_1024x4096` | `4.925 ms/op` (`17.96 MB/op`, `26 allocs`) | `3.004 ms/op` (`525 KB/op`, `12 allocs`) | `3.253 ms/op` (`525 KB/op`, `12 allocs`) | **1.64× faster** | **34.2× less** (`-97.07%`) |
| `GemmInto/Prepared/M32_1024x4096` | `4.964 ms/op` (`17.96 MB/op`, `26 allocs`) | `1.364 ms/op` (`1.02 KB/op`, `11 allocs`) | `1.481 ms/op` (`1.02 KB/op`, `11 allocs`) | **3.64× faster** | **17,538× less** (`-99.99%`) |
| `GemmInto/Dense/M32_2048x8192` | `20.379 ms/op` (`69.47 MB/op`, `26 allocs`) | `13.597 ms/op` (`1.05 MB/op`, `12 allocs`) | `13.437 ms/op` (`1.05 MB/op`, `12 allocs`) | **1.50× faster** | **66.2× less** (`-98.49%`) |
| `GemmInto/Prepared/M32_2048x8192` | `18.826 ms/op` (`69.47 MB/op`, `26 allocs`) | `5.225 ms/op` (`1.02 KB/op`, `11 allocs`) | `5.569 ms/op` (`1.02 KB/op`, `11 allocs`) | **3.60× faster** | **67,842× less** (`-99.998%`) |

\*Note on `MatMulInto/M32_1024x4096`: previously, `-tags=metal` silently copied
the `[32, 1024]` and `[1024, 4096]` slices into temporary `matrix.Matrix`
buffers (`18.48 MB/op`) and dispatched them to Metal MPS. After decoupling,
`tensor.MatMulInto` consistently runs on the host CPU `float32product.Standard`
path matching `CGO_ENABLED=0` (`73.39 ms/op` vs `74.04 ms/op`, `524 KB/op`).
Callers seeking GPU acceleration for large matrix products should use explicit
resident `neuralnetwork/gpu` (`Scope.MatMul`, `Scope.MatMulRightTranspose`,
`Scope.BatchedMatMul`).

## Downstream Note (`onnx`)

Once a `neuralnetwork` release containing NN-18 is tagged and pinned in
`github.com/itsmontoya/onnx`, `onnx`'s `DeviceCPU` execution under `-tags=metal`
will automatically benefit from the host CPU/NEON path and packed
`PreparedRightTranspose` weights without copying through `matrix.Matrix`, and
`onnx`'s `numerical_product_metal.go` tolerance workaround can be simplified.
