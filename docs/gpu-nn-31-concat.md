# NN-31: `Scope.Concat` (Rank 1–6 Non-Quantized) & `GATE-K` Release Readiness Checklist

**Date:** 2026-11-27  
**Repository:** `github.com/itsmontoya/neuralnetwork`  
**Hardware:** Apple M3 (`darwin/arm64`, 8-core CPU / 10-core GPU, unified memory)  
**Base:** `gpu-nn-30` @ `d6021b99df6ab7728739f12a6901a14264d43b59` (PR #171)  
**Raw benchmark logs:** `.benchmarks/gpu-nn-31/run1/concat.txt` (gitignored)

---

## 1. Public API & Validation Contract

Added `Scope.Concat` in `gpu/scope.go`:

```go
func (s *Scope) Concat(inputs []*Tensor, destination *Tensor, axis int) (err error)
```

- **Supported types & ranks:** Non-quantized dense tensors (`Float32`, `Float16`, `BFloat16`) of rank `1..6`. All inputs and `destination` must belong to the same `Device` and share the exact same `DType` and rank.
- **Shape & bounds checks:** `len(inputs) >= 1`, `0 <= axis < rank`, every non-concat dimension matches `destination.Shape()[d]`, the sum of `inputs[i].Shape()[axis]` equals `destination.Shape()[axis]` (checked against overflow), and all operand element counts are `<= math.MaxUint32`.
- **Ownership, aliasing, and views:** `destination` must not alias any input (checked on shared `*tensorStorage`, rejecting direct and `Tensor.View` aliases with `ErrInvalid`). Duplicate read inputs (e.g., `[x, x]` or sibling views `[v1, v2]` of the same storage) are allowed and share one read lease cleanly.

---

## 2. Dispatch & Encoder Coalescing Design

Implemented in `internal/device/concat_spec.go`, `internal/device/concat_metal_source.h`, `internal/device/metal_backend.h`, `internal/device/metal_backend.m`, `internal/device/scope.go`, `internal/device/backend.go`, `internal/device/backend_default.go`, and `internal/device/backend_metal.go`:

- **Contiguous blit fast path (`outer == 1`, e.g., `axis == 0` or all leading dims `== 1`):**
  - Calls `nn_metal_scope_end_compute_encoder(scopeRecord)` before creating a single `MTLBlitCommandEncoder`, copies all `K` inputs sequentially at their destination byte offsets in one blit encoder, ends the blit encoder, and retains all source and destination buffers.
- **Strided compute path (`outer > 1`, e.g., NCHW channel/spatial concat with `N > 1`):**
  - Lazily compiles `nn_tensor_concat_u32` (`Float32`, 4-byte raw `uint` copy) and `nn_tensor_concat_u16` (`Float16`/`BFloat16`, 2-byte raw `ushort` copy) via `pthread_once` with `[options setFastMathEnabled:NO]`.
  - Acquires the scope's coalesced `MTLComputeCommandEncoder` via `nn_metal_scope_compute_encoder(scopeRecord, context)` without calling `[encoder endEncoding]`, binds the pipeline and destination buffer once, and dispatches each input over a 3D grid `(inner, srcAxisLen, outer)` with coalesced inner-axis threads (`tgX` up to 256). All `K` input dispatches remain inside the single active compute encoder.

---

## 3. Verification Summary

- **Bit-exact parity (`Test_MetalTensorConcatBitExact`):** Verified exact raw bits (`math.Float32bits` / `uint16`) across 57 combinations covering ranks 1D, 2D, 3D, 4D (NCHW `axis=0,1,2,3`), and 6D; 1, 2, 3, and 8 inputs; `Float32`, `Float16`, and `BFloat16`; odd/prime dimensions; both `outer == 1` blit and `outer > 1` strided compute paths; and special IEEE-754 bit patterns (`+0.0`, `-0.0`, `±Inf`, custom quiet `NaN` payloads, positive/negative subnormals).
- **Repeated inputs & `Tensor.View` (`Test_MetalTensorConcatRepeatedAndViews`):** Verified `[v1, v1, v2]` on both `axis=0` (blit) and `axis=1` (strided compute) across all 3 dense dtypes.
- **Encoder coalescing & discard safety (`Test_MetalTensorConcatEncoderCoalescingAndDiscard`):** Verified interleaved `Scale → Concat(outer>1 compute) → Concat(outer==1 blit) → Scale` in one `Scope`, plus uncommitted discard after both paths with zero `ResourceSnapshot` leaks.
- **Validation & concurrency (`Test_Scope_Concat`, `Test_MetalTensorConcatConcurrentStress`):** Verified all 22 validation/error cases (empty/nil inputs, axis bounds, dtype/rank/dim mismatch, axis sum overflow/underflow, quantized `Q8_0`/`Q4_0`/`Q4_1`, direct and view aliasing, uninitialized, closed, cross-device, uint32 overflow) and 8-worker `-race -tags=metal` concurrent stress under **Metal API Validation (`METAL_DEVICE_WRAPPER_TYPE=1`)** with zero warnings and zero leaks.

---

## 4. Apple M3 Benchmarks

Methodology: `CGO_ENABLED=1 GOWORK=off go test -tags=metal -run='^$' -bench=Benchmark_ConcatMetal -benchmem -benchtime=500ms -count=5 ./gpu`. Table reports median latency and effective memory bandwidth (`2 × dstBytes / latency`, covering read + write traffic) across 5 runs.

| Case | Path | `Float32` Median | `Float16` Median | `BFloat16` Median |
| --- | --- | ---: | ---: | ---: |
| `CNN_Channel [8, 64+128+64, 56, 56] axis=1` | Strided compute (`outer=8`) | `0.888 ms` (`57.85 GB/s`) | `0.563 ms` (`45.66 GB/s`) | `0.559 ms` (`45.98 GB/s`) |
| `Batch [4+4, 256, 56, 56] axis=0` | Blit (`outer=1`) | `0.877 ms` (`58.56 GB/s`) | `0.534 ms` (`48.14 GB/s`) | `0.533 ms` (`48.18 GB/s`) |
| `CNN_Channel [1, 64+128+64, 56, 56] axis=1` | Blit (`outer=1`) | `0.278 ms` (`23.09 GB/s`) | `0.221 ms` (`14.50 GB/s`) | `0.222 ms` (`14.47 GB/s`) |
| `CNN_Channel [8, 8×32, 56, 56] axis=1` | Strided compute (`outer=8`, 8 inputs) | `0.890 ms` (`57.76 GB/s`) | `0.572 ms` (`44.89 GB/s`) | `0.574 ms` (`44.73 GB/s`) |

Notice that 8-way channel concatenation (`[8, 8×32, 56, 56]`) matches 3-way channel concatenation (`0.890 ms` vs `0.888 ms`) because `NN-29` coalesces all 8 dispatches into one `MTLComputeCommandEncoder`.

---

## 5. `GATE-K` Release Readiness Checklist (`v0.17.0`: `NN-29`, `NN-30`, `NN-31`)

- [x] **NN-29 (`gpu-nn-29`, PR #170):** `MTLComputeCommandEncoder` coalescing per `Scope` in `internal/device/metal_backend.m` with zero public API changes. Verified with `METAL_DEVICE_WRAPPER_TYPE=1`, interleaved compute/blit chains, uncommitted discard/cancel/failure cleanup, and 8-worker `-race -tags=metal` stress. Delivers **1.92×–2.95×** on `CausalAttentionMetal` (16× dispatches), **1.14×–1.64×** on quantized `M=1` FFN decode, and **1.47×** on FP32 BERT-base encoder block (`docs/gpu-nn-29-encoder-coalescing.md`).
- [x] **NN-30 (`gpu-nn-30`, PR #171):** Rank-2 `Float32 × Float32` SIMD GEMV (`M == 1` `MatMulRightTranspose`) & `simdgroup_float8x8` `(BM=32, BN=64, BK=32)` tiled GEMM (`M >= 4` `MatMul` / `MatMulRightTranspose`), preserving `nn_matmul` tiny-shape fallback and leaving ANN `matrix` / `model.Sequential` untouched. Reaches **86.4 GB/s** FP32 GEMV (**1.20×–4.59×**), **1.66 TFLOP/s** FP32 GEMM (**2.19×–8.28×**), and **3.65×–3.92×** on `M=27..29` resident FP32 FFN / **3.88×** on FP32 Transformer block (`docs/gpu-nn-30-fp32-gemv-gemm.md`).
- [x] **NN-31 (`gpu-nn-31`):** `Scope.Concat` for rank 1–6 non-quantized `Float32`/`Float16`/`BFloat16` tensors with `outer == 1` single-encoder blit fast path (`nn_metal_scope_end_compute_encoder`) and `outer > 1` coalesced strided compute kernels (`nn_tensor_concat_u32` / `nn_tensor_concat_u16`). Bit-exact across all 57 shape/axis/dtype combinations + IEEE-754 special bit patterns + repeated/view inputs + compute/blit coalescing + 8-worker `-race -tags=metal` stress; **57.8–58.6 GB/s** on M3 (`docs/gpu-nn-31-concat.md`).
- [x] **Portability & Backward Compatibility:** All 14 entries of the Required Test Matrix pass across default CPU, `purego`, `CGO_ENABLED=0`, Linux `amd64` cross-builds, and `CGO_ENABLED=1 -tags=metal` (`-race` included). All existing `v0.12.1`–`v0.16.0` `gpu` APIs and ANN Metal (`Sequential`) paths remain unchanged and passing.
- [x] **Numerics & Resource Safety:** All new Metal kernels compile lazily via `pthread_once` with `[options setFastMathEnabled:NO]`; Metal API Validation (`METAL_DEVICE_WRAPPER_TYPE=1`) passes with zero warnings; storage-level `Tensor.View` alias rejection and `ResourceSnapshot` zero-leak invariants verified across all new ops.
- [ ] **Owner Action (`GATE-K`):** Review and merge PRs #170 (`NN-29`), #171 (`NN-30`), and `NN-31`, tag and publish immutable `v0.17.0`, and record the commit SHA and module checksums in `TODO.md` before dispatching `ONNX-04` and `LL-08`.
