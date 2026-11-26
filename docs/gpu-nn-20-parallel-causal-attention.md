# NN-20: Parallel SIMD `CausalAttention` (FP32 & FP16 KV Cache)

**Task:** `NN-20`  
**Base commit:** `4aaa96122c0b1c5e9a08d54d84c3db8bef0f55d5` (`gpu-nn-19` / PR #160, stacked on `v0.13.1`)  
**Hardware:** Apple M3, 24 GiB unified memory, macOS 26.5.2 (`25F84`), `go1.27.3 darwin/arm64`  
**Raw benchmark logs:** `.benchmarks/gpu-nn-20/2026-11-25-m3/before.txt` and `after.txt` (gitignored)

## Summary

`nn_tensor_causal_attention` (FP32 KV) and `nn_tensor_causal_attention_f16` (FP16 KV) have been redesigned from single-thread-per-head loops (with per-KV-row global `result` read-modify-write) into **threadgroup-cooperative parallel SIMD online-softmax kernels** with register-resident queries and accumulators, zero intermediate global `result` traffic, and multi-SIMD split-KV reduction for longer contexts.

The public `gpu.Scope.CausalAttention` API, validation, GQA/MQA/MHA grouping, causal masking, empty-cache zeroing, and FP32 accumulation (`fastMathEnabled = NO`) remain completely unchanged.

## Kernel Design

Both kernels share a templated helper `nn_causal_attention_head<KV>` (`float` / `half`) in [`internal/device/attention_metal_source.h`](../internal/device/attention_metal_source.h), compiled lazily via `pthread_once(&nnMetalAttentionMathOnce, ...)` with `[options setFastMathEnabled:NO]` and dispatched as **1 threadgroup per `(queryRow, head)`** in [`internal/device/metal_backend.m`](../internal/device/metal_backend.m):

- **Adaptive threadgroup sizing (`metal_backend.m`):**
  - `headDim <= 128`:
    - `maxVisible < 64`: **1 SIMD group** (`32` threads) — zero threadgroup barriers and zero threadgroup memory traffic.
    - `64 <= maxVisible < 256`: **4 SIMD groups** (`128` threads) — split-KV across 4 stripes.
    - `maxVisible >= 256`: **8 SIMD groups** (`256` threads) — split-KV across 8 stripes.
  - `headDim > 128`: **1 SIMD group** (`32` threads) using a strided 32-lane parallel SIMD fallback (`simd_sum` dot product + coalesced 32-lane output update, zero stack spills).
- **Register-resident online softmax (`headDim <= 128`):**
  - Each of the 32 lanes in a SIMD group caches its coalesced strided slice of `Q` (`simdLane + 32*k`, `k = 0..3`) and unnormalized output accumulator `acc` in registers (`q0..q3`, `acc0..acc3`), with branch-free unrolled inner loops for `headDim == 64` and `headDim == 128` and exact bounds-guarded loops for arbitrary `headDim <= 128`.
  - Per KV row, lanes compute partial dot products → `simd_sum(dot)` → online softmax update (`nextMax`, `alpha = exp(localMax - nextMax)`, `beta = exp(score - nextMax)`) → register `acc = acc * alpha + beta * v`, eliminating per-row divisions and all intermediate global `result` traffic.
- **WAR-race-free multi-SIMD merge (`numSimdGroups > 1`):**
  - Before the single `threadgroup_barrier(mem_flags::mem_threadgroup)`, each SIMD group writes only to its own disjoint slots: `sharedMax[simdGroup]`, `sharedSum[simdGroup]`, and `sharedAcc[simdGroup * 128 + f]`.
  - After the barrier, only `simdGroup == 0` executes (read-only on `sharedMax`, `sharedSum`, `sharedAcc`), computing exact online-softmax rescaling (`globalMax = max_g(m_g)`, `w_g = exp(m_g - globalMax)` for active stripes with `s_g != 0`, `out = sum_g(w_g * acc_g) / sum_g(s_g * w_g)`) and writing coalesced `result[start + f]`. Because no thread writes to threadgroup memory after the barrier, WAR races are structurally impossible.

## Correctness Verification & FP64 Error Distribution

Verified on Apple M3 (`CGO_ENABLED=1 go test -count=1 -race -tags=metal ./gpu/...`):

- **Shape & width coverage (`Test_MetalCausalAttentionParallelShapesAndWidths`):** MHA (`4/4`), GQA (`32/8`, `8/2`, `6/2`), MQA (`8/1`), `M = 1, 6, 27, 28, 29, 65`, `visible = 1, 2, 6, 7, 27, 28, 29, 32, 33, 38, 65, 127, 128, 129, 256, 512, 1024`, `headDim = 1, 3, 17, 32, 33, 64, 65, 128, 129, 256`, both `Float32` and `Float16` `KVCache`, including multi-SIMD partial-stripe prefill (`queryRow = 0` with `visible = 1 < numSimdGroups`).
- **Overflow guard & empty cache (`Test_MetalCausalAttentionOverflowGuardAndEmptyCache`):** verified exact zero output on empty cache and finite FP64-matching outputs under extreme opposing score magnitudes across single-SIMD (`H=7`), 4-SIMD (`H=128`), 8-SIMD (`H=512`), and `headDim=256` fallback paths.
- **8-worker concurrent stress (`Test_MetalCausalAttentionConcurrentStress`):** 8 goroutines × 8 iterations × 16 dispatches/scope across `H=38` (1-SIMD), `H=128` (4-SIMD), and `H=512` (8-SIMD) for `Float32` and `Float16` KV caches under `-race -tags=metal` — zero races, zero accuracy deviations, zero leaked native buffers/scopes.
- **Existing tests:** all `attention_metal_test.go` and `rmsnorm_gather_kv_metal_test.go` tolerances pass unchanged.

### Llama 3.2 1B (`heads=32, kvHeads=8, headDim=64`) FP64 Absolute Error

| Shape | KV DType | `p50` | `p95` | `p99` | `max` |
| --- | --- | ---: | ---: | ---: | ---: |
| `M=1, H=38` (1-SIMD) | `Float32` | `3.68e-09` | `1.66e-08` | `2.54e-08` | `4.49e-08` |
| `M=1, H=38` (1-SIMD) | `Float16` | `3.91e-09` | `1.59e-08` | `2.58e-08` | `5.74e-08` |
| `M=1, H=128` (4-SIMD) | `Float32` | `2.21e-09` | `7.32e-09` | `1.10e-08` | `1.59e-08` |
| `M=1, H=128` (4-SIMD) | `Float16` | `2.14e-09` | `7.12e-09` | `9.59e-09` | `1.38e-08` |
| `M=1, H=512` (8-SIMD) | `Float32` | `1.55e-09` | `4.77e-09` | `7.04e-09` | `1.25e-08` |
| `M=1, H=512` (8-SIMD) | `Float16` | `1.44e-09` | `4.59e-09` | `6.41e-09` | `9.92e-09` |
| `M=1, H=1024` (8-SIMD) | `Float32` | `1.38e-09` | `4.16e-09` | `5.88e-09` | `1.05e-08` |
| `M=1, H=1024` (8-SIMD) | `Float16` | `1.34e-09` | `4.25e-09` | `5.56e-09` | `9.32e-09` |
| `M=29, H=29` (1-SIMD prefill) | `Float32` | `4.30e-09` | `1.77e-08` | `2.81e-08` | `7.59e-08` |
| `M=29, H=29` (1-SIMD prefill) | `Float16` | `4.31e-09` | `1.80e-08` | `2.90e-08` | `6.28e-08` |

## Apple M3 Paired Before vs After Benchmarks

Command (run identically before and after kernel changes, 3-run medians at `-benchtime=10x -count=3`):
```bash
CGO_ENABLED=1 GOWORK=off go test -tags=metal -run='^$' \
  -bench='^(Benchmark_CausalAttentionMetal|Benchmark_AttentionMetal|Benchmark_TransformerBlockMixedMetal)$' \
  -benchtime=10x -count=3 ./gpu
```

### 1. Isolated 16-Dispatch `CausalAttention` (`heads=32, kvHeads=8, headDim=64`)

| Shape | KV DType | Before (`v0.13.1`) | After (`NN-20`) | Speedup |
| --- | --- | ---: | ---: | ---: |
| `M=1, H=38` (×16) | `Float32` | `6.575 ms` | **`0.488 ms`** | **`13.47×`** |
| `M=1, H=38` (×16) | `Float16` | `6.345 ms` | **`0.418 ms`** | **`15.19×`** |
| `M=1, H=128` (×16) | `Float32` | `21.473 ms` | **`0.355 ms`** | **`60.44×`** |
| `M=1, H=128` (×16) | `Float16` | `20.531 ms` | **`0.349 ms`** | **`58.80×`** |
| `M=1, H=512` (×16) | `Float32` | `87.076 ms` | **`0.746 ms`** | **`116.73×`** |
| `M=1, H=512` (×16) | `Float16` | `80.788 ms` | **`0.729 ms`** | **`110.85×`** |
| `M=1, H=1024` (×16) | `Float32` | `182.368 ms` | **`1.098 ms`** | **`166.11×`** |
| `M=1, H=1024` (×16) | `Float16` | `165.683 ms` | **`1.197 ms`** | **`138.41×`** |
| `M=29, H=29` (×16 prefill) | `Float32` | `5.190 ms` | **`0.507 ms`** | **`10.24×`** |
| `M=29, H=29` (×16 prefill) | `Float16` | `5.031 ms` | **`0.519 ms`** | **`9.70×`** |

### 2. Full Attention Layer & Mixed-Precision Transformer Block (`heads=32, kvHeads=8, headDim=64`)

| Benchmark | Before (`v0.13.1`) | After (`NN-20`) | Speedup |
| --- | ---: | ---: | ---: |
| `Benchmark_AttentionMetal/M1-H511` | `6.733 ms` | **`1.709 ms`** | **`3.94×`** |
| `Benchmark_AttentionMetal/M29-H0` | `4.213 ms` | **`4.019 ms`** | **`1.05×`** |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/FP32_KV32` | `9.236 ms` | **`4.101 ms`** | **`2.25×`** |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/FP16_KV32` | `7.406 ms` | **`2.321 ms`** | **`3.19×`** |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/FP16_KV16` | `7.154 ms` | **`2.603 ms`** | **`2.75×`** |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/BF16_KV32` | `7.324 ms` | **`2.551 ms`** | **`2.87×`** |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/BF16_KV16` | `7.225 ms` | **`2.469 ms`** | **`2.93×`** |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/FP32_KV32` | `17.604 ms` | **`17.268 ms`** | **`1.02×`** |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/FP16_KV32` | `7.657 ms` | **`7.454 ms`** | **`1.03×`** |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/FP16_KV16` | `7.590 ms` | **`7.446 ms`** | **`1.02×`** |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/BF16_KV32` | `7.605 ms` | **`7.352 ms`** | **`1.03×`** |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/BF16_KV16` | `7.632 ms` | **`7.341 ms`** | **`1.04×`** |

---

## GATE-G Release Readiness Checklist (`v0.14.0` — `NN-19` & `NN-20`)

Prepared for owner review before merging `NN-19` (`#160`) and `NN-20` and tagging immutable release `v0.14.0` for downstream `llama` (`LL-07`):

- [x] **`NN-19` Quantized `GatherRows` (`gpu-nn-19`, PR #160):**
  - `Scope.GatherRows` supports `Q8_0 → Float32`, `Q4_0 → Float32` (on-the-fly GPU dequantization, `fastMathEnabled = NO`, 2-byte aligned block reads, 512-row `setBytes` chunking) and matching `Q8_0 → Q8_0` / `Q4_0 → Q4_0` raw-byte blit.
  - Cross-quantized and quantized↔16-bit conversions rejected with `ErrInvalid`; storage-level alias checks enforced across `Tensor.View`.
  - Bit-exact agreement with `DequantizeQ8_0` / `DequantizeQ4_0` across normal and special FP16 scales; single-allocation tied `[128256, 2048]` `GatherRows + MatMulRightTranspose` delivers **`1.83×`** (`Q8_0`, `-246.3 MB`, `-2.74 ms/tok`) and **`3.26×`** (`Q4_0`, `-377.6 MB`, `-4.18 ms/tok`) vs `BF16` ([`docs/gpu-nn-19-quantized-gather.md`](gpu-nn-19-quantized-gather.md)).
- [x] **`NN-20` Parallel SIMD `CausalAttention` (`gpu-nn-20`):**
  - Threadgroup-cooperative SIMD online-softmax kernels for both `Float32` and `Float16` `KVCache` (`fastMathEnabled = NO`), register-resident `Q`/`acc` for `headDim <= 128`, WAR-race-free multi-SIMD split-KV merge, and 32-lane SIMD fallback for `headDim > 128`.
  - 16-dispatch `M=1` decode drops from `6.35–182.37 ms` to **`0.35–1.20 ms`** (`13.5×–166.1×` speedup across `H=38..1024`); `M=29` prefill drops `9.7×–10.2×`; full `M1_H511` Transformer block drops `2.25×–3.19×`.
  - FP64 max abs error `<= 7.6e-8` across Llama 3.2 1B shapes; 8-worker `-race -tags=metal` concurrent stress passes with zero leaks.
- [x] **Portability & backward compatibility:**
  - All 14 entries of the Required Test Matrix pass (`default`, `-race`, `CGO_ENABLED=0`, `purego`, `metal`, `metal purego`, `GOOS=linux GOARCH=amd64`, and `CGO_ENABLED=1 -race -tags=metal`).
  - Zero public API breaking changes; all `v0.12.1` / `v0.13.x` contracts and ANN Metal paths preserved unchanged.
- [ ] **Owner actions for `GATE-G`:**
  1. Review and merge `NN-19` (`#160`) and `NN-20` into `main`.
  2. Tag and publish immutable release `v0.14.0` on `github.com/itsmontoya/neuralnetwork`.
  3. Record exact Git tag, commit SHA, and Go module checksums in `TODO.md` before dispatching `LL-07`.
