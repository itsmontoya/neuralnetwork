# Device-resident embeddings and output projection

Upload one row-major FP32 weight `[N,K]` for a tied embedding and output
projection. `GatherRows` reads the embedding vectors from that allocation.
`MatMulRightTranspose` uses the same stored layout as a projection weight.
Untied models can pass a separate `[N,K]` projection tensor. Share read-only
weights through one logical device and give each concurrent session distinct
writable states and outputs.

| API | Shape and operation |
| --- | --- |
| `scope.GatherRows(source, rows, destination)` | Copies zero-based host indices `rows []int64` from rank-two `source[R,C]` to `destination[len(rows),C]`. The list is nonempty and can contain duplicates in any order. |
| `scope.MatMulRightTranspose(left, weight, logits)` | Reads `[M,K]` and `[N,K]` and writes row-major `[M,N]`. |
| `logits.DownloadInto(values)` | Copies all elements into an exactly sized caller-owned FP32 slice after a successful scope wait. |

Gather consumes the index slice during recording. It uses one Metal blit
encoder and one buffer copy per row, with no index tensor or host activation
transfer. It preserves the copied FP32 bits. Recording work scales with the
number of selected rows. All dimensions and native byte offsets fit the
allocated buffers. Projection uses 32-bit shader indices and supports at
most 2³²−1 elements per operand. Device allocation limits apply to both
operations. Projection uses the existing FP32 accumulation and nonfinite
arithmetic. Close scores can change order when upstream rounding changes.

Both destinations must be distinct from their sources, fully overwritten,
and on the same logical device as the inputs. A source can be initialized by
an earlier operation in the scope. Invalid rank, shape, device, alias or
index returns `ErrInvalid`. Closed, uninitialized and busy operands use
the existing scope errors. A recording error after earlier work requires
discarding the scope. A submitted command failure invalidates its writable
destinations until a later full write succeeds. Canceling a wait leaves the
leases outstanding until a later wait or draining close observes completion.

## LL-03 integration

After the owner publishes GATE-C, verify the release SHA, module checksum and
live Llama base. Start from the reviewed LL-02 integration and preserve its
[staged KV publication contract](gpu-attention.md#staging-and-publication).

1. Upload the tied FP32 weight once. For Llama 3.2 1B it occupies
   1,050,673,152 bytes (`[128256,2048]`). When the decoder owns that weight,
   reuse it for both call sites and close it once. Account for any host copy
   still retained by the caller.
2. Gather the initial IDs `[128000,791,6864,315,9822,374]` into `[6,2048]`
   and record the existing transformer operations and final RMSNorm in that
   scope. Gather row 5 of the normalized result into `[1,2048]` for cached
   prefill. Record `MatMulRightTranspose` with the tied weight and
   `[1,128256]` logits. Decode can project its already single-row normalized
   result directly. Full-prefix diagnostics can project every normalized row
   by allocating `[M,128256]` logits.
3. Submit and wait once. A full single-row download copies 513,024 bytes.
   Let Llama validate every value as finite before accepting the largest
   value, choosing the lowest index on an exact tie. Preserve Llama's output
   and generation policy. Publish pending layer appends and advance the
   session cursor only after successful output and context validation.
   Abandon pending updates and drain submitted scopes on earlier errors.
4. Verify both cached and full-prefix paths on the pinned checkpoint,
   including the close scores at position 11, full-logit references,
   independent sessions and recovery. Record weight and activation copies,
   device allocation, peak RSS, and the final readback after integration.

The [Llama LL-02 handoff](https://github.com/itsmontoya/llama/pull/33) records
the previous tagged classifier's per-call weight staging. The existing
[`gpu` package](../gpu) also provides explicit upload and readback, fill and
copy, standard matmul, Add, Mul, SiLU, SwiGLU, RMSNorm, RotatePairs,
CausalAttention and staged KV appends.

## Metal correctness

Run the geometry and fault tests on a native Metal device.

```sh
GOWORK=off CGO_ENABLED=1 go test -count=1 -v -tags=metal ./gpu \
  -run '^Test_(GatherRows|MetalGatherRows)'
```

The synthetic odd-width composition gathers first, last and repeated rows
bit-for-bit in caller order, then projects all gathered rows. The
`[128256,2048]` test shares one weight through two outstanding scopes and
compares all logits against independent FP64 dots. Maximum absolute errors
were `4.17e-6` and `3.53e-6`. Their p95 errors were `1.31e-6` and
`1.16e-6`. The controlled exact tie selected the lower ID in the host scan,
and the one-ULP near tie selected the higher logit. The tests also cover
invalid indices, logical-device and alias checks, discard, cancellation,
terminal failure after actual Metal writes, and a successful retry.

A standalone check used host-normalized activations from LL-02 on
neuralnetwork `v0.11.1` and the official Llama 3.2 1B checkpoint, revision
`4e20de362430cd3b72f300e6b0f18e50e7166e08`. Manifest, frozen reference, and
checkpoint SHA-256 digests were
`3404dfda97c69cf3a9a3cf15dc4db7017cb049da07462c52f2e3004f6e12ab64`,
`3a66ddc5cbdef45e397e47a6bd992676ab3d35234a8b389908f721b05466d361`,
and `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`.
In one scope the two blit encoders gathered the six embedding rows and
selected normalized row 11. Two projections wrote all 13 logit rows and
a standalone row 11. All 12,288 embedding values matched the host bits.
The maximum observed difference from the host over 1,667,328 logits was
zero with those same supplied activations. Row 11 agreed between the
single-row and full-prefix projections. All 344 sampled Transformers
logits passed the frozen `1e-3 + 1e-3*abs(reference)` bound, with maximum
error `4.96e-5`. The exact IDs were
`[12366,13,1102,374,279,1455,95551,3363]`. At position 11, ID 95551 led by
`0.04335022` versus the frozen `0.043340683`. The standalone check starts
from host activations. LL-03 will repeat the references through the
integrated GPU transformer and classifier.

## Apple M3 output boundary measurements

The matched M=1 study used the same position-11 activation and checkpoint
weights. Each run checked every output against the released host result and
verified the exact winning ID. Host NEON and tagged Matrix both used the
ordinary tensor `GemmInto` call. The tagged process confirmed one native
command per sample. Resident Metal allocated and uploaded weights once.
The explicit re-upload control used the same scope and destination but copied
the weight into the existing device buffer once per measured step.

Apple M3 Mac15,13 with 24 GiB unified memory, macOS 26.5.2 (25F84), Go
1.27.3, SDK 14.4, `GOWORK=off`, `GOARM64=v8.0`, `GOMAXPROCS=8`, cgo.
Runtime MSL compiled successfully. The offline `metal` compiler was
unavailable. The primary series alternated mode order within seven matched
groups of fresh processes. Each process ran a first call and seven later
calls with the same inputs. Warm values are medians and ranges of per-process
medians. Cold values are from the first call after model buffer setup.
The process starts cold and the system shader cache may persist.

| M=1 output boundary | Host NEON | Tagged Matrix | Resident Metal | Explicit re-upload |
| --- | ---: | ---: | ---: | ---: |
| First call wall, ms | 14.414 (14.151–16.540) | 177.149 (142.896–192.617) | 32.439 (26.763–39.270) | 59.640 (46.858–68.249) |
| Warm step wall, ms | 12.910 (12.755–15.206) | 154.607 (135.996–266.524) | 12.538 (12.376–12.664) | 38.827 (38.404–39.038) |
| Process wall, s | 0.360 (0.350–0.680) | 1.590 (1.380–3.800) | 0.400 (0.370–0.420) | 0.620 (0.580–0.660) |
| Peak RSS, GiB | 1.980 (1.980–1.980) | 7.354 (7.034–8.233) | 1.991 (1.991–1.991) | 1.991 (1.991–1.991) |

Resident warm steps took 0.969× the host NEON time, 0.081× the tagged
Matrix time and 0.324× the explicit re-upload time in the paired medians.
Resident Metal was faster on this isolated warm workload in all seven
pairs. Process wall includes checkpoint decoding, setup, checks and cleanup.
The resident process spent a median 27.767 ms opening the device, 73 µs
allocating its buffers and 66.275 ms on the first weight upload. Re-upload
took a median 25.604 ms per warm step. The ordinary tagged path also creates
host Matrix copies and per-call device allocations.

For a resident step, median warm scope creation, encode and submit took
2.000, 5.208 and 2.666 µs. The first encode took 259.625 µs. Warm wait
took 12.379 ms. A separate diagnostic binary recorded Metal command start
and end timestamps for five more fresh processes per mode. The isolated
resident command contained one projection dispatch. Its native interval was
17.376 ms (11.727–22.038) on the first call and 12.003 ms (11.884–12.230)
on warm calls. The first-use projection library and pipeline initialization
took 0.148 ms (0.128–0.222). CPU wait includes queueing and host scheduling
as well as GPU execution.

Full readback plus finite scan and selection cost 132.501 µs
(124.958–138.209), a median 1.064% of resident step wall. The 513,024-byte
copy alone took a median 28.958 µs. A pre-written eight-byte buffer took
0.125 µs to copy and establishes a small-transfer floor. Computing a result
on device would add a reduction and its encoding cost. The measured upper
bound on savings supports keeping full readback and host selection for
LL-03. A future reduction would need an integer index, lowest-ID tie
handling, and invalid status for any NaN or either infinity.

The resident model weight, single-row activation and logits requested
1,051,194,368 device bytes. The eight-byte transfer control was a separate
allocation. Four allocations were reused for all eight commands and native
live bytes returned to zero when they closed. LL-02's tagged classifier
stages 1,050,681,344 weight-plus-activation bytes per single-row call. The
matched tagged control peaked at 3,153,583,104 requested device bytes,
with 1,051,194,368–2,102,388,736 still live when sampled before ordinary
host-backed matrices were finalized. Peak RSS includes the host inputs,
references and runtime. It can overlap requested native bytes on unified
memory.

Local raw evidence and scripts are in
`.benchmarks/gpu-nn-07/2026-11-23-m3/`. `fixture/export.zsh` creates the
host reference input using a test overlay over Llama LL-02. `run.zsh`
builds and times the standalone projection harness from implementation
commit `cbff03336ef7224e14454bfd32674ffba3815036`. `summarize.py` aggregates
the raw JSON and Metal diagnostics. The benchmarks characterize only the
output boundary. LL-03 will measure the complete generation step.
