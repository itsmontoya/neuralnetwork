# NN-24: In-Place `KVCache` Append (`O(1)` Long-Context Decode & −50% KV Device Memory)

**Date:** 2026-11-26  
**Base:** `v0.14.0` (`8dae3930318ca42c0b7fc1fae59d43d1a5eaca02`)  
**Hardware:** Apple M3 (`darwin/arm64`, Metal, `CGO_ENABLED=1`)

## Summary

Replaced the double-buffered, prefix-copying `gpu.KVCache` implementation (`slots [2]*Tensor` + `active int`) with a single contiguous `[2, capacity, heads, headWidth]` device buffer and `O(1)` in-place `AppendKV` that writes only the newly appended rows at `[position..position+rows)` directly into the uncommitted tail.

- **Zero public API changes**: all `KVCache`, `KVAppend`, and `Scope` signatures remain unchanged.
- **Exact −50% KV device memory and native buffer count**: 1 `MTLBuffer` per `KVCache` instead of 2 (verified via `internal/device.ResourceSnapshot`).
- **True `O(1)` `AppendKV` scaling across `H = 38..4096`**: 16-layer `AppendKV` latency stays flat at **0.165–0.184 ms** (FP32) and **0.216–0.231 ms** (FP16) regardless of context length — up to **40.71× faster** (FP32) and **16.66× faster** (FP16) at `H = 4096`.
- **Safety & concurrency preserved**:
  - `AppendKV` acquires exclusive `writeAccess` on `cache.slot`, rejecting competing readers or writers with `ErrBusy`.
  - `KVCache.Reset()` checks `state.pending != nil || storage.readers > 0 || storage.writer != nil || storage.hostAccess` under `device.mutex`, returning `ErrBusy` if an append is pending or any scope holds a lease on the cache buffer.
  - Failed, discarded, or canceled appends never touch the committed prefix `[0..length)` and preserve `slot.storage.initialized = true` whenever `cache.length > 0`, keeping committed history readable and retryable at `position == length`.

## Paired Apple M3 Benchmarks (Before vs After)

Raw logs stored under `.benchmarks/gpu-nn-24/before/bench.txt` and `.benchmarks/gpu-nn-24/after/bench.txt` (gitignored). All numbers are medians of 3 runs (`-benchtime=500ms -count=3`).

### 16-Layer `AppendKV` (`Benchmark_AppendKVMetal`, `kvHeads=8, headDim=64`, 1 `Scope` Submission)

| Context (`H`) | KV DType | Before (ms/op) | After (ms/op) | Speedup | KV Device Memory (16 layers) | KV Buffers |
| ---: | :---: | ---: | ---: | ---: | ---: | :---: |
| 38 | `Float32` | 0.352 | 0.181 | **1.95×** | 5.11 MB → 2.56 MB (−50%) | 32 → 16 |
| 38 | `Float16` | 0.933 | 0.225 | **4.15×** | 2.56 MB → 1.28 MB (−50%) | 32 → 16 |
| 512 | `Float32` | 1.180 | 0.175 | **6.73×** | 67.24 MB → 33.62 MB (−50%) | 32 → 16 |
| 512 | `Float16` | 1.106 | 0.216 | **5.11×** | 33.62 MB → 16.81 MB (−50%) | 32 → 16 |
| 2048 | `Float32` | 3.413 | 0.184 | **18.56×** | 268.57 MB → 134.28 MB (−50%) | 32 → 16 |
| 2048 | `Float16` | 2.009 | 0.231 | **8.69×** | 134.28 MB → 67.14 MB (−50%) | 32 → 16 |
| 4096 | `Float32` | 6.706 | 0.165 | **40.71×** | 537.00 MB → 268.50 MB (−50%) | 32 → 16 |
| 4096 | `Float16` | 3.733 | 0.224 | **16.66×** | 268.50 MB → 134.25 MB (−50%) | 32 → 16 |

For `Float16`, removing the per-step prefix blit encoder also eliminates an entire blit→compute encoder transition per layer, yielding a **4.15×** speedup even at short context `H = 38`.

### Attention & Transformer Block (`H = 511` Decode & `M = 29` Prefill)

| Benchmark | Before (ms/op) | After (ms/op) | Speedup | KV Device Bytes |
| --- | ---: | ---: | ---: | ---: |
| `Benchmark_AttentionMetal/M1-H511` | 1.621 | 1.873 (min 1.475) | 0.87×* | 4,194,304 → 2,097,152 (`prefix-copy-B`: 2,093,056 → 0) |
| `Benchmark_AttentionMetal/M29-H0` | 8.784 | 3.956 | 2.22×* | 4,194,304 → 2,097,152 |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/FP32_KV32` | 3.872 | 3.816 | 1.01× | 4,194,304 → 2,097,152 |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/FP16_KV32` | 1.857 | 1.764 | 1.05× | 4,194,304 → 2,097,152 |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/FP16_KV16` | 1.879 | 1.847 | 1.02× | 2,097,152 → 1,048,576 |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/BF16_KV32` | 1.929 | 1.815 | 1.06× | 4,194,304 → 2,097,152 |
| `Benchmark_TransformerBlockMixedMetal/M1_H511/BF16_KV16` | 1.896 | 1.832 | 1.03× | 2,097,152 → 1,048,576 |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/FP32_KV32` | 16.922 | 16.868 | 1.00× | 4,194,304 → 2,097,152 |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/FP16_KV32` | 3.740 | 3.786 | 0.99× | 4,194,304 → 2,097,152 |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/FP16_KV16` | 3.718 | 3.688 | 1.01× | 2,097,152 → 1,048,576 |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/BF16_KV32` | 3.753 | 3.780 | 0.99× | 4,194,304 → 2,097,152 |
| `Benchmark_TransformerBlockMixedMetal/M29_H0/BF16_KV16` | 3.707 | 3.756 | 0.99× | 2,097,152 → 1,048,576 |

\* `Benchmark_AttentionMetal` runs first in the process while Apple M3 GPU clocks ramp up (`M1-H511` individual runs: before `1.621 / 2.475 / 1.583 ms`, after `1.873 / 1.879 / 1.475 ms`). On the steady-state `M1_H511` Transformer block, every mode shows a consistent 1–6% speedup at `H=511` alongside the 50% KV memory reduction.

## Numerical Verification (Independent FP64 Reference)

`Test_MetalKVCacheLongContextParity` verifies incremental `AppendKV` + `CausalAttention` at `H = 2048` and `H = 4096` (`heads=8, kvHeads=2, headDim=32`) against an independent FP64 reference:

| Context (`H`) | KV DType | p50 abs err | p95 abs err | p99 abs err | max abs err | Limit |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | `Float32` | `6.58e-10` | `1.77e-09` | `2.20e-09` | `3.24e-09` | `1e-5` |
| 2048 | `Float16` | `6.13e-10` | `1.95e-09` | `2.42e-09` | `3.34e-09` | `1e-5` |
| 4096 | `Float32` | `7.10e-10` | `1.87e-09` | `2.33e-09` | `2.98e-09` | `1e-5` |
| 4096 | `Float16` | `6.81e-10` | `1.67e-09` | `2.22e-09` | `2.71e-09` | `1e-5` |
