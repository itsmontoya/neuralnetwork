# NN-17: General `Transpose`, `MatMulLeftTranspose`, & Batched 3D+ `MatMul`

**Date:** 2026-11-25  
**Hardware:** Apple M3 (8-core CPU, 24 GB unified memory, `darwin/arm64`)  
**Raw artifacts:** `.benchmarks/gpu-nn-17/run1/` (gitignored)

## Summary

NN-17 completes Phase 3A (`NN-13`–`NN-17`) of the generic GPU primitives roadmap in `gpu.Scope`:

- `Scope.Transpose(input, destination *Tensor, perm []int) error`
  - Permutes axes of any rank 1..6 non-quantized tensor (`Float32`, `Float16`, `BFloat16`, matching input/destination dtype).
  - Validates permutation length, uniqueness, range `0..rank-1`, exact destination shape, and non-aliasing destination (including via `Tensor.View`).
  - Fast-paths identity and size-1-only permutations directly to a Metal blit copy (`encodeCopy`).
- `Scope.MatMulLeftTranspose(left, right, destination *Tensor) error`
  - Computes rank-2 `destination[M, N] = left[K, M]^T × right[K, N]` without materializing a transposed copy of `left`.
  - Supports `Float32` `left`/`destination` and `Float32`, `Float16`, or `BFloat16` `right` with FP32 accumulation (`fastMathEnabled = NO`). Rejects quantized right weights with `ErrInvalid`.
  - Existing rank-2 `Scope.MatMul` and `Scope.MatMulRightTranspose` remain bit-for-bit untouched.
- `Scope.BatchedMatMul(left, right, destination *Tensor) error`
- `Scope.BatchedMatMulRightTranspose(left, right, destination *Tensor) error`
  - Computes `left[..., M, K] × right[..., K, N]` and `left[..., M, K] × right[..., N, K]^T` for rank 2..6 tensors with ONNX/NumPy right-aligned broadcasting over up to 4 batch dimensions.
  - Supports `Float32` `left`/`destination` and `Float32`, `Float16`, or `BFloat16` `right` with FP32 accumulation (`fastMathEnabled = NO`). Rejects quantized right weights with `ErrInvalid`.

## Kernel Architecture

- **`internal/device/transpose_metal_source.h`** (lazy `pthread_once`, `[options setFastMathEnabled:NO]`):
  - Host coalesces adjacent contiguous input axes and drops size-1 axes before encoding.
  - Coalesced rank ≤ 1 fast-paths to `encodeCopy` blit.
  - 2D transposes (`[R, C] → [C, R]`) dispatch a 16×16 tiled threadgroup-memory kernel (`tile[16][17]` with +1 padding to avoid bank conflicts) for coalesced global reads and writes (`nn_tensor_transpose_2d_u32` / `nn_tensor_transpose_2d_u16`).
  - General rank 2..6 permutations dispatch `nn_tensor_transpose_general_u32` (4-byte) / `nn_tensor_transpose_general_u16` (2-byte).
- **`internal/device/batched_matmul_metal_source.h`** (lazy `pthread_once`, `[options setFastMathEnabled:NO]`):
  - Six dtype-specialized pipelines (`nn_matmul_left_transpose_{f32,f16,bf16}` and `nn_batched_matmul_{f32,f16,bf16}`) with zero dtype branches inside the inner `K` loop.
  - Batched kernels dispatch a 3D grid `(N, M, totalBatches)` with host-coalesced matrix-unit batch strides (`batchRank <= 4`) and a 1D-batch fast path without integer division.
  - Right-transposed batched paths vectorize `float4`/`half4`/`bfloat4` loads with FP32 `fma` when `K % 4 == 0`, with exact scalar fallback for arbitrary `K >= 1`.

## Independent Verification & FP64 Error Distributions

All Metal tests ran under `-count=1 -race -tags=metal` with zero data races and zero native buffer/scope leaks (`internal/device` `ResourceSnapshot` verified).

- **`Transpose`:** Verified **bit-exact** (0 bit mismatches) across `Float32` (including `NaN`, `±Inf`, `-0.0`), `Float16`, and `BFloat16` for 1D identity, 2D (`1×1`, `7×13`, `16×16`, `17×33`, `64×128`), 3D (`[3,5,7]` perm `[2,0,1]`, `[4,1,9]` size-1 contiguous), 4D attention (`[2,17,8,32] ↔ [2,8,17,32]`), 6D arbitrary permutation, `Tensor.View`, and view-alias rejection.
- **`MatMulLeftTranspose` & `BatchedMatMul` / `BatchedMatMulRightTranspose`:** Verified against independent FP64 CPU references (using round-tripped `Float16`/`BFloat16` effective weights) across odd/even dimensions, 2D–6D shapes, broadcast batch dims, `NaN`/`±Inf`/`0*Inf`, `A^T × A` self-products, and 8-worker × 16-iteration concurrent `-race -tags=metal` stress tests.

| Operation | Shape / Case | Right DType | p50 Abs Err | p95 Abs Err | p99 Abs Err | Max Abs Err |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `MatMulLeftTranspose` | `[17, 19]^T × [17, 23]` | `Float32` | `5.45e-08` | `2.31e-07` | `3.56e-07` | `4.76e-07` |
| `MatMulLeftTranspose` | `[64, 32]^T × [64, 64]` | `Float16` | `1.83e-07` | `7.85e-07` | `1.20e-06` | `2.40e-06` |
| `MatMulLeftTranspose` | `[129, 33]^T × [129, 65]` | `Float32` | `3.80e-07` | `1.72e-06` | `2.87e-06` | `4.51e-06` |
| `MatMulLeftTranspose` | `[129, 33]^T × [129, 65]` | `BFloat16` | `3.87e-07` | `1.66e-06` | `2.38e-06` | `4.21e-06` |
| `BatchedMatMul` | `[4, 17, 33] × [4, 33, 19]` | `Float32` | `9.55e-08` | `4.22e-07` | `7.42e-07` | `1.32e-06` |
| `BatchedMatMulRightTranspose` | `[4, 17, 33] × [4, 19, 33]^T` | `Float16` | `9.55e-08` | `3.92e-07` | `7.72e-07` | `1.42e-06` |
| `BatchedMatMul` (shared wt) | `[3, 19, 64] × [64, 48]` | `Float32` | `1.95e-07` | `8.30e-07` | `1.39e-06` | `2.89e-06` |
| `BatchedMatMul` (shared wt) | `[3, 19, 64] × [64, 48]` | `BFloat16` | `1.88e-07` | `7.97e-07` | `1.31e-06` | `2.25e-06` |
| `BatchedMatMul` (attn) | `[2, 8, 32, 64] × [2, 8, 64, 32]` | `Float32` | `1.90e-07` | `8.15e-07` | `1.30e-06` | `3.03e-06` |
| `BatchedMatMulRightTranspose` (attn) | `[2, 8, 32, 64] × [2, 8, 32, 64]^T` | `Float32` | `1.33e-07` | `4.45e-07` | `6.23e-07` | `1.45e-06` |
| `BatchedMatMul` (4D multi-bcast) | `[2, 1, 13, 29] × [1, 3, 29, 17]` | `Float32` | `8.51e-08` | `3.76e-07` | `6.52e-07` | `1.14e-06` |
| `BatchedMatMul` (6D bcast) | `[2, 1, 2, 1, 5, 7] × [1, 2, 1, 3, 7, 4]` | `BFloat16` | `2.01e-08` | `9.24e-08` | `1.47e-07` | `1.78e-07` |
| **1-Scope Transformer Block** | `B=2, S=16, H=4, D=16, E=64, FFN=128` | `Float32` | `4.84e-08` | `1.46e-07` | `2.11e-07` | `3.15e-07` |
| **1-Scope Transformer Block** | `B=2, S=16, H=4, D=16, E=64, FFN=128` | `Float16` | `4.87e-08` | `1.45e-07` | `2.05e-07` | `3.06e-07` |
| **1-Scope Transformer Block** | `B=2, S=16, H=4, D=16, E=64, FFN=128` | `BFloat16` | `4.74e-08` | `1.44e-07` | `2.03e-07` | `2.95e-07` |

## Apple M3 Benchmarks (`-count=3`, median)

Command:

```bash
CGO_ENABLED=1 GOWORK=off go test -run '^$' \
  -bench 'Benchmark_(Transpose|MatMulLeftTranspose|BatchedMatMul|ONNXTransformerEncoderBlock)Metal' \
  -benchmem -count=3 -tags=metal ./gpu
```

| Benchmark | Shape / Config | DType | Median Latency | Host Alloc |
| --- | --- | --- | ---: | --- |
| `Transpose` (2D tiled) | `[1024, 1024]`, perm `[1, 0]` | `Float32` | `0.320 ms/op` | `528 B/op`, `8 allocs/op` |
| `Transpose` (2D tiled) | `[1024, 1024]`, perm `[1, 0]` | `Float16` | `0.262 ms/op` | `528 B/op`, `8 allocs/op` |
| `Transpose` (4D attn) | `[2, 64, 12, 64] → [2, 12, 64, 64]` | `Float32` | `0.201 ms/op` | `528 B/op`, `8 allocs/op` |
| `Transpose` (4D attn) | `[2, 64, 12, 64] → [2, 12, 64, 64]` | `Float16` | `0.194 ms/op` | `528 B/op`, `8 allocs/op` |
| `MatMulLeftTranspose` | `[768, 32]^T × [768, 768]` | `Float32` | `0.738 ms/op` | `528 B/op`, `8 allocs/op` |
| `MatMulLeftTranspose` | `[768, 32]^T × [768, 768]` | `Float16` | `0.687 ms/op` | `528 B/op`, `8 allocs/op` |
| `MatMulLeftTranspose` | `[768, 32]^T × [768, 768]` | `BFloat16` | `0.732 ms/op` | `528 B/op`, `8 allocs/op` |
| `BatchedMatMulRightTranspose` | `[2, 12, 64, 64] × [2, 12, 64, 64]^T` | `Float32` | `0.251 ms/op` | `528 B/op`, `8 allocs/op` |
| `BatchedMatMul` | `[2, 12, 64, 64] × [2, 12, 64, 64]` | `Float32` | `0.306 ms/op` | `528 B/op`, `8 allocs/op` |
| `BatchedMatMul` (shared wt) | `[2, 64, 768] × [768, 768]` | `Float32` | `1.429 ms/op` | `528 B/op`, `8 allocs/op` |
| `BatchedMatMul` (shared wt) | `[2, 64, 768] × [768, 768]` | `Float16` | `1.422 ms/op` | `528 B/op`, `8 allocs/op` |
| `BatchedMatMul` (shared wt) | `[2, 64, 768] × [768, 768]` | `BFloat16` | `1.411 ms/op` | `528 B/op`, `8 allocs/op` |
| **ONNX Transformer Encoder Block (1 Scope, 22 dispatches)** | `B=2, S=64, H=12, D=64, E=768, FFN=3072` | `Float32` | `14.50 ms/op` | `2920 B/op`, `22 allocs/op` |
| **ONNX Transformer Encoder Block (1 Scope, 22 dispatches)** | `B=2, S=64, H=12, D=64, E=768, FFN=3072` | `Float16` | `14.44 ms/op` | `2920 B/op`, `22 allocs/op` |
| **ONNX Transformer Encoder Block (1 Scope, 22 dispatches)** | `B=2, S=64, H=12, D=64, E=768, FFN=3072` | `BFloat16` | `14.28 ms/op` | `2920 B/op`, `22 allocs/op` |

---

## `GATE-F` Release Readiness Checklist (`v0.13.0` — `NN-13` through `NN-17`)

| Task | PR / Branch | Delivered APIs & Artifacts | FP64 / Verification Summary | Status |
| --- | --- | --- | --- | --- |
| **NN-13** | [#154](https://github.com/itsmontoya/neuralnetwork/pull/154) (`gpu-nn-13`) | `docs/adr-gpu-onnx-primitives.md`, `Tensor.View(shape []int64)`, `tensorStorage` lease/alias tracking | Zero added device bytes on views, view alias rejection, deferred native release, zero leaks | PR OPEN |
| **NN-14** | [#155](https://github.com/itsmontoya/neuralnetwork/pull/155) (`gpu-nn-14`) | `Scope.ReLU`, `Scope.Sigmoid`, `Scope.Tanh`, `Scope.Softmax`, `Scope.SoftmaxAxis`, `docs/gpu-nn-14-activations-softmax.md` | FP64 verified across 1D–4D & prime widths, disjoint `shared[34]` SIMD softmax, 1-scope MLP | PR OPEN |
| **NN-15** | [#156](https://github.com/itsmontoya/neuralnetwork/pull/156) (`gpu-nn-15`) | `Scope.Sub`, `Scope.Div`, `Scope.Scale`, `Scope.{Add,Sub,Mul,Div}Broadcast`, `docs/gpu-nn-15-broadcast-ops.md` | FP64 verified ranks 1..6, row-bias/scalar/contiguous fast paths, eliminates `GatherRows` bias scratch | PR OPEN |
| **NN-16** | [#157](https://github.com/itsmontoya/neuralnetwork/pull/157) (`gpu-nn-16`) | `Scope.LayerNorm(input, weight, bias, dst, eps)`, `docs/gpu-nn-16-layernorm.md` | Two-pass `x0`-shifted FP32 reduction, disjoint `shared[34]`, optional bias, FP32/FP16/BF16 params, max err `≤ 4.05e-07` | PR OPEN |
| **NN-17** | `gpu-nn-17` (this PR) | `Scope.Transpose`, `Scope.MatMulLeftTranspose`, `Scope.BatchedMatMul`, `Scope.BatchedMatMulRightTranspose`, `docs/gpu-nn-17-batched-matmul-transpose.md` | Bit-exact transpose, FP64 verified 2D–6D batched & left-T matmul, 1-scope BERT-base Transformer encoder block (`14.28–14.50 ms/op`) | PR OPEN |

### Owner `GATE-F` Pre-Release Checklist
- [ ] Review and merge PRs `#154` (`NN-13`), `#155` (`NN-14`), `#156` (`NN-15`), `#157` (`NN-16`), and the `NN-17` PR in order into `main`.
- [ ] Confirm all 14 entries of the Required Test Matrix pass on `main` after merge (including `-race -tags=metal` and Linux `amd64` cross-build).
- [ ] Confirm backward compatibility of all `v0.12.1` `gpu` APIs and zero regressions on `llama` / ANN `Sequential` paths.
- [ ] Tag and publish immutable release `v0.13.0` on `github.com/itsmontoya/neuralnetwork`.
- [ ] Record exact Git tag, commit SHA, module checksum, and `go.mod` checksum in `TODO.md` under `GATE-F` before dispatching `ONNX-02`.
