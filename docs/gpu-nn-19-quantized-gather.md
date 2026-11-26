# NN-19: Quantized `GatherRows` (`Q8_0` / `Q4_0` → `Float32` Dequantizing Gather & Matching Blit)

**Task:** `NN-19`  
**Base commit:** `2621db5ea830996a9c0ceaa2f32650788930b938` (`v0.13.1`)  
**Hardware:** Apple M3, 24 GiB unified memory, macOS 26.5.2 (`25F84`), `go1.27.3 darwin/arm64`  
**Raw benchmark log:** `.benchmarks/gpu-nn-19/2026-11-25-m3/bench.txt` (gitignored)

## Summary

`Scope.GatherRows` now supports block-quantized `Q8_0` and `Q4_0` rank-2 source
tables for both:

1. **On-the-fly dequantizing gather into `Float32`** (`Q8_0 → Float32`,
   `Q4_0 → Float32`), executed on GPU by `nn_tensor_gather_rows_q8_0_to_f32` and
   `nn_tensor_gather_rows_q4_0_to_f32` compiled lazily with
   `fastMathEnabled = NO`.
2. **Matching quantized raw-byte gather** (`Q8_0 → Q8_0`, `Q4_0 → Q4_0`),
   reusing the existing Metal blit `EncodeGatherRows` with
   `rowBytes = (columns / QuantBlockSize) * dtype.BlockBytes()`.

Cross-quantized and quantized/16-bit conversions (`Q8_0 → Q4_0`,
`Q4_0 → Q8_0`, `Q8_0 → Float16`, `Q8_0 → BFloat16`, `Q4_0 → Float16`,
`Q4_0 → BFloat16`, `Float32 → Q8_0`, `Float32 → Q4_0`, `Float16 → Q8_0`,
`BFloat16 → Q4_0`) remain rejected with `ErrInvalid`.

## Kernel Design

Both dequantizing kernels live in
[`internal/device/attention_metal_source.h`](../internal/device/attention_metal_source.h)
and are compiled lazily via `pthread_once(&nnMetalAttentionMathOnce, ...)` with
`[options setFastMathEnabled:NO]`:

- **`nn_tensor_gather_rows_q8_0_to_f32`**: treats `source` as
  `const device uchar *` (respecting the 2-byte alignment of 34-byte blocks with
  no misaligned 4-byte/16-byte casts), reads the 2-byte `half` scale at
  `(rows[outRow] * blocksPerRow + blockInRow) * 34`, reads
  `char(block[2 + elemInBlock])`, and writes `float(scale) * float(q)` to
  `destination`.
- **`nn_tensor_gather_rows_q4_0_to_f32`**: treats `source` as
  `const device uchar *` (respecting the 2-byte alignment of 18-byte blocks),
  reads the 2-byte `half` scale at
  `(rows[outRow] * blocksPerRow + blockInRow) * 18`, unpacks the GGUF
  split-half nibble (`elemInBlock < 16` → low nibble of `block[2 + elemInBlock]`,
  `elemInBlock >= 16` → high nibble of `block[2 + elemInBlock - 16]`), offsets
  by `-8`, and writes `float(scale) * float(int(nibble) - 8)` to `destination`.
- **Chunked index dispatch**: reuses the existing 512-row `setBytes` chunking
  loop in `nn_metal_scope_encode_gather_rows_mixed` (zero temporary buffers).

## Correctness Verification

Verified on Apple M3 (`CGO_ENABLED=1 go test -count=1 -race -tags=metal ./gpu/...`):

- **Bit-exact dequantization (`Q8_0 → Float32`, `Q4_0 → Float32`):** 100%
  bit-exact (`math.Float32bits`) match against host `DequantizeQ8_0` and
  `DequantizeQ4_0` across `M = 1, 6, 27, 28, 29, 600` (crossing the 512-row
  chunk boundary), widths `32, 64, 96, 128, 2048`, repeated indices, and
  boundary rows `0` and `R-1`.
- **Special FP16 scales:** bit-exact match for `+0.0`, `-0.0`, negative scale
  (`-1.5`), max finite FP16 (`65504`), smallest and largest FP16 subnormals,
  `+Inf`, `-Inf`, and `NaN` propagation (`NaN` verified via `math.IsNaN`).
- **Matching quantized blit (`Q8_0 → Q8_0`, `Q4_0 → Q4_0`):** bit-for-bit raw
  byte equality across all shapes and special scales.
- **Tied Llama `[128256, 2048]` embedding + classifier in 1 `gpu.Scope`:**
  verified single shared weight allocation (`279,085,056` bytes for `Q8_0`,
  `147,750,912` bytes for `Q4_0`), `3` created native buffers total (weight,
  embed, logits), `1` command buffer submission, exact embedding/logit outputs,
  and zero live buffer leaks on close.
- **Aliasing & concurrency:** verified storage-level alias rejection (including
  competing writes across distinct `Tensor.View` handles) and 8-worker
  concurrent read-sharing under `-race -tags=metal`.

## Apple M3 Benchmark Results — Tied `[128256, 2048]` Embedding / Classifier

Command:
```bash
CGO_ENABLED=1 GOWORK=off go test -tags=metal -run='^$' \
  -bench='^Benchmark_TiedEmbeddingClassifierMetal$' -benchtime=20x -count=5 ./gpu
```

All timings are 5-run medians (`-benchtime=20x -count=5`, `8 allocs/op`,
`528–533 B/op` host overhead per submission):

| DType | Weight Size (Decimal / Binary) | Memory Saved vs BF16 | `GatherRows` `M=1` | `GatherRows` `M=29` | `Classifier` `M=1` | Tied `Gather + Classifier` `M=1` (1 Scope) | Speedup vs BF16 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `BFloat16` | `525.3 MB` (`501.0 MiB`) | baseline | `0.191 ms` | `0.204 ms` | `6.025 ms` | `6.027 ms` | `1.00×` |
| `Q8_0` | `279.1 MB` (`266.2 MiB`) | **`-246.3 MB` (`-46.9%`)** | `0.184 ms` | `0.162 ms` | `3.388 ms` | **`3.288 ms`** | **`1.83×`** (`-2.74 ms/tok`) |
| `Q4_0` | `147.8 MB` (`140.9 MiB`) | **`-377.6 MB` (`-71.9%`)** | `0.207 ms` | `0.159 ms` | `1.838 ms` | **`1.851 ms`** | **`3.26×`** (`-4.18 ms/tok`) |

Standalone `GatherRows` (`M=1` and `M=29`) is dominated by Metal command buffer
submit/wait latency (~`0.16–0.21 ms`). When fused into the same `gpu.Scope` as
the classifier projection (`TiedGatherAndClassifier_M1`), the quantized gather
adds negligible overhead (`~0–13 µs`), unlocking the full `2.74 ms/tok` (`Q8_0`)
to `4.18 ms/tok` (`Q4_0`) savings and `246–378 MB` device memory reduction for
downstream `llama` (`LL-07`).
