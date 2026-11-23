# Device-resident causal attention and KV

The experimental `gpu` package records FP32 normalization, projections,
pair rotation, causal attention and residual addition on one Metal scope. It
uses the same logical device, queue, tensor leases and buffers as the other
explicit GPU operations. It compiles the new kernels on their first use.
Build on macOS ARM64 with cgo and `-tags=metal`.

## Shapes and operations

| API | Shape and operation |
| --- | --- |
| `scope.RMSNorm(input, weight, destination, epsilon)` | Normalizes along the last axis with a matching rank-one weight and finite positive FP32 epsilon. Destination matches input. The reduction rescales the row before squaring. |
| `scope.MatMulRightTranspose(input, weights, projection)` | Reads `[M,K]` and `[N,K]` and writes `[M,N]`. |
| `scope.RotatePairs(input, cosine, sine, destination, headWidth, position, tableStart)` | Rotates each pair as `(x0*cos - x1*sin, x1*cos + x0*sin)`. Width is even. Identically shaped tables `[P,headWidth/2]` start at absolute position `tableStart` and broadcast over heads. The input positions through `position+M-1` must fit the tables. |
| `gpu.NewKVCache(device, capacity, kvHeads, headWidth)` | Allocates empty, private history with fixed dimensions. Each of two slots holds logical key and value planes `[capacity,kvHeads,headWidth]`. |
| `scope.AppendKV(cache, keys, values, position)` | Returns a `*gpu.KVAppend` for a positive consecutive block starting exactly at the committed length. Both tensors must describe the same row count and cache head geometry. |
| `scope.CausalAttention(queries, cache, destination, position)` | Uses positions `position+i` for each query row, with cache positions starting at zero. Each consecutive group of `queryHeads/kvHeads` query heads reads one KV head. The division must be exact. The destination matches the query shape. |
| `update.Publish()`, `update.Close()`, `cache.Len()`, `cache.Reset()`, `cache.Close()` | Publish or discard a staged append, observe the committed length, reset the length, or release the cache. |

Rotation, append and attention accept token-major `[M,heads*headWidth]` and
`[M,heads,headWidth]`. The byte order is identical, with the feature index
changing fastest, then head, then token. Pass projection results directly in
rank-two form and use the packed attention destination for the output
projection. Key and value sources may use different ranks. Rotation accepts
a coefficient window, or a full table with `tableStart = 0`. Cache and tensor
elements are checked against the device allocation and 32-bit shader index
limits. Absolute positions use nonnegative `int64` values and checked
intervals. Tensors require positive shapes. Destinations are disjoint and
fully written. Normal scope operations require no intermediate host transfer.

Causal attention computes `dot(query,key)/sqrt(headWidth)` and normalizes
over populated keys at or before the absolute query position. Empty history
produces an all-zero destination. MHA, GQA and MQA use the same grouped
mapping without copying KV heads. A single kernel keeps an online running
maximum, denominator and weighted mean for each query-head pair. The
destination holds the running weighted mean, so there is no score or
probability tensor. The kernel is a scalar correctness baseline, with one GPU
thread per query-head pair. Its work grows with the attended prefix.
Use finite data with representable FP32 products and running dot sums. FP32
reduction order, Metal intrinsics and possible subnormal flushing can affect
rounding.

## Staging and publication

The cache starts with length zero. `AppendKV` copies both committed prefixes
and appends the current rows into the inactive slot in one blit encoder.
The current slot is leased for reading, and the inactive slot is leased for
writing through completion. Later attention in that scope reads the staged
prefix, including its new rows. Independent scopes read a snapshot of
the committed prefix. They keep their slot leased until completion even if
publication or reset occurs in the meantime.

A scope must submit and successfully observe completion with `Wait` before
its append can publish. Keep the returned update and validate the enclosing
step first. `Publish` swaps slots and advances the committed length under
the device mutex. Publication performs no transfer, submit or wait. A second
append or reset returns `ErrBusy` while an update is outstanding. An older
reader still holding the next staging slot can also cause `ErrBusy`. Finish
or discard outstanding scopes before reuse.

If recording is poisoned, submission fails or is canceled, or the GPU
reports a command failure, the stage is discarded. Committed planes were
bound only for reading. If `Wait` is canceled, either wait again with a
live context or drain the scope with `Close`. The appends are discarded when
a draining `Close` observes completion first. If a successful `Wait` already
made an update publishable, call `update.Close()` to abandon it on a later
step error. Abandoning an update leaves the existing length and values intact.
Retry from the same committed length. The new append rewrites the prefix
and its new rows before attention can consume them. Closing a cache prevents
new uses and publication while in-flight scopes retain their leased slots.
Reset only changes the length. A read already recorded keeps its snapshot.
Recovery applies to failed commands on a usable device. A lost device
requires reconstructing its allocations and cache.

A single-layer command can use the staged rows directly. With initialized
inputs, coefficients and weights and caller-allocated scratch:

```go
// scope.RMSNorm and the Q/K/V projections have already been recorded.
if err = scope.RotatePairs(q, cos, sin, rotatedQ, width, position, tableStart); err != nil {
    return err
}
if err = scope.RotatePairs(k, cos, sin, rotatedK, width, position, tableStart); err != nil {
    return err
}
var update *gpu.KVAppend
if update, err = scope.AppendKV(cache, rotatedK, v, position); err != nil {
    return err
}
defer update.Close()
if err = scope.CausalAttention(rotatedQ, cache, attended, position); err != nil {
    return err
}
// Record the output projection and residual addition before submission.
if err = scope.Submit(ctx); err != nil {
    return err
}
if err = scope.Wait(ctx); err != nil {
    return err
}
if err = validateCompletedStep(ctx); err != nil {
    return err
}
return update.Publish()
```

The scope owner calls `Close` on every path. Put scope close in a defer
when recording begins, so cancellation while waiting drains the command and
leases. The [complete Metal composition](../gpu/attention_scratch_metal_test.go)
includes normalization, four projections, rotation, attention and residual.

## LL-02 integration

1. After GATE-B, pin and verify the released neuralnetwork version against
   the owner-recorded checksum. Verify the live Llama base and use the
   reviewed LL-01 contract for the existing FFN. [LL-01 attention handoff](https://github.com/itsmontoya/llama/blob/9b4f51e60677fb7f704ce6f2dc8bd3cc29a2de37/NEURALNETWORK_OPTIMIZATION_HANDOFF.md#generic-operations-and-storage-for-nn-06)
2. Reuse one device per decoder. Share read-only norm, projection and rotary
   coefficients. For the Llama 3.2 1B checkpoint, Q/O remain `[2048,2048]`,
   K/V `[512,2048]`, and the grouping is 32:8 at width 64. Keep Llama's
   checkpoint Q/K row conversion and scaled rotary policy. Upload one cosine
   and one sine coefficient per pair in `[P,32]` using the existing angle
   policy. Keep the coefficient’s original sign. The kernel applies the
   minus sign to the first-feature term.
3. Give each session independent scratch and one cache per layer. Within a
   session, serialize the full step, reset and close. Record prefill at
   position zero and decode at the session's last committed position. Keep
   each layer's append handle through final output and context validation.
   A layer can wait and close its scope while its update stays staged. Its
   activations can feed later commands. The caller controls the final
   readback boundary.
4. If any layer, validation or cancellation fails, close every pending
   append and drain all outstanding scopes. The session cursor and all layer
   lengths stay at the previous committed position. After all waits and
   final validation succeed, publish every layer's append under the same
   serialized step, then advance the session cursor. A publishable update
   stays valid during this sequence when the owner excludes concurrent
   close, reset and abandonment. `Publish` is per cache. The session makes
   model-wide publication observable as one step by keeping the serialized
   boundary until its own position is updated.
5. Compare the released integration against the frozen logits and actual
   greedy IDs, cached/full-prefix parity, the 27/28/29-row tails, sustained
   decode and independent sessions. The standalone upstream results
   establish the operation contracts. The downstream integration establishes
   end-to-end output and timing on the pinned checkpoint.

## Resource and measurement contract

One complete synthetic attention layer encodes nine compute dispatches and
one blit encoder in one command buffer. The blit emits two copies for a
fresh prefix or four when previous rows exist. A full step can perform one
input upload and one output readback. Multiple math operations or layers can
be recorded without waiting in between if the caller keeps their operands
resident and uses disjoint scratch where needed.

A cache reserves `16*capacity*kvHeads*headWidth` bytes across active and
staging slots. At capacity 512 with 8 heads of width 64, that is 4 MiB per
layer or 64 MiB across 16 layers in one session. At committed length `L`,
each append of `M` rows copies `8*L*kvHeads*headWidth` prefix bytes and
`8*M*kvHeads*headWidth` append bytes within GPU memory. These are logical
bytes copied. The example scratch is `4*M*(7*hidden+3*kvHeads*headWidth)`
bytes. With `hidden=2048`, its scratch is 63,488 bytes at M=1 and
1,841,152 bytes at M=29. Coefficients at capacity 512 occupy 131,072 bytes.
Projection and norm weights request 41,951,232 bytes. Host activation
input and output staging total 16,384 bytes at M=1 or 475,136 bytes at M=29.
The long-history benchmark seeds the cache once from separate 2,093,056-byte
host and device staging allocations. Its timed iterations reuse both slots.

Run the Metal correctness and benchmark entry points on Apple Silicon.

```sh
GOWORK=off CGO_ENABLED=1 go test -count=1 -v -tags=metal ./gpu \
  -run '^Test_(MetalAttention|MetalKV|KVCache)'
GOWORK=off CGO_ENABLED=1 GOMAXPROCS=8 go test -c -tags=metal -o "$bin" ./gpu
"$bin" -test.run '^$' -test.bench '^Benchmark_AttentionMetal$/^M1-H511$' \
  -test.benchtime=12x -test.count=1 -test.v
"$bin" -test.run '^$' -test.bench '^Benchmark_AttentionMetal$/^M29-H0$' \
  -test.benchtime=12x -test.count=1 -test.v
```

Each benchmark uses one set of deterministic dense FP32 weights, a fixed
input and a fixed synthetic history. It verifies every output against an
independent FP64 reference outside the measured interval. Seeds are recorded
in the [fixture](../gpu/attention_fixture_metal_test.go) and
[benchmark](../gpu/attention_metal_benchmark_test.go). Measurements include
one cold call and three warmups per process. The decode case appends the
512th row, and the prefill case appends 29 rows to empty history. Each sample
discards its completed stage to repeat at the same position.

### Apple M3 results

Measured with the standalone synthetic layer at
`12784cf59a5c291e4badfc462fedb5f69ee310e1` on Apple M3 Mac15,13, 24 GiB,
with 10 GPU cores, macOS 26.5.2 (25F84), SDK 14.4, Go 1.27.3, cgo, Metal,
`GOWORK=off`, `GOARM64=v8.0` and `GOMAXPROCS=8`. The offline
`metal` compiler was unavailable. The Metal driver compiled MSL at runtime.
The measurements are for one synthetic layer with weights already uploaded.

| FP64 comparison | Prefill maximum absolute error | Single-row decode maximum absolute error |
| --- | ---: | ---: |
| MHA 3:3, width 6, 27 rows | 1.30e-7 | 5.69e-8 |
| GQA 6:3, width 10, 28 rows | 2.04e-7 | 5.37e-8 |
| MQA 5:1, width 8, 29 rows | 1.40e-7 | 5.12e-8 |
| GQA 32:8, width 64, 6 rows | 2.20e-6 | 1.04e-6 |

Across those eight comparisons the median absolute error per workload
ranged from 1.64e-8 to 1.50e-7, p95 from 4.16e-8 to 5.36e-7 and
RMSE from 2.17e-8 to 2.67e-7. The composition limit is 1e-5 maximum
absolute error. The direct mask, adjacent-pair rotation and RMSNorm
checks use 1e-6. Their maximum error was 9.83e-8. Empty-history output
was exactly zero. The RMSNorm test includes a zero row and finite FP32
extremes. The causal mask test uses large positive and negative scores,
a nonzero query offset and an odd head width. Separate tests cover
absolute position overflow, mismatched devices and shapes, independent
concurrent sessions, cache closing while work is outstanding, and
retry after cancellation or a simulated terminal failure following real
Metal writes.

| Timing | M=1, committed length 511 | M=29, committed length 0 |
| --- | ---: | ---: |
| Open default device, ms | 24.67 (24.37–40.33) | 26.46 (24.34–30.24) |
| Allocate device tensors, ms | 0.120 (0.104–0.198) | 0.157 (0.134–0.166) |
| Upload shared weights and coefficients, ms | 3.30 (3.19–3.84) | 3.56 (3.07–3.89) |
| First encode, ms | 0.404 (0.370–31.44) | 0.673 (0.651–0.709) |
| First step wall, ms | 21.54 (8.27–51.46) | 9.38 (7.53–16.02) |
| Warm scope create / encode / submit, µs | 2.1 / 18.5 / 2.5 | 1.9 / 16.6 / 2.8 |
| Warm upload / readback, µs | 1.0 / 3.1 | 6.8 / 14.7 |
| Warm wait, ms | 7.29 (7.15–7.34) | 4.85 (4.65–4.92) |
| Warm wall, ms | 7.33 (7.19–7.37) | 4.89 (4.71–4.97) |
| Native warm GPU command interval, ms | 6.99 (6.94–10.52) | 4.41 (4.26–4.44) |
| Native first GPU command interval, ms | 10.88 (6.89–25.45) | 6.94 (4.83–16.85) |

The standard timings are medians and ranges of seven fresh processes per
shape, with alternating shape order and 12 measured steps per process.
Warm wall spans activation upload, scope creation, encode, submit, wait,
scope and update close, and output readback. The separate diagnostic build
logs Metal `GPUStartTime` and `GPUEndTime` after completion for five fresh
processes per shape, reporting whole command intervals. Instrumentation
also times first-use source library and pipeline creation within each lazy
initializer. For decode, medians were 0.161 ms for attention math, 0.041 ms
for tensor matmul and 0.101 ms for tensor elementwise. For prefill they
were 0.215, 0.045 and 0.099 ms. The process starts cold, while the Metal
driver may reuse its system shader cache. CPU wait includes queue time, GPU
execution and Go scheduling. Command timestamps include the blit and every
dispatch, so they do not identify individual kernel costs.

Warm steps allocated a median 1,656 Go bytes in 22 allocations for decode
and 1,624 bytes in 21 allocations for prefill. Median process peak RSS was
112.63 MiB and 111.22 MiB, respectively. RSS includes benchmark inputs,
CPU references and runtime overhead, and can overlap requested device bytes
on unified memory. Local reproduction scripts, diagnostic instrumentation,
raw samples, and test logs live under
`.benchmarks/gpu-nn-06/2026-11-23-m3/`. There is no matched CPU timing
series for this benchmark. LL-02 will measure the full checkpoint
integration and remaining host crossings.
