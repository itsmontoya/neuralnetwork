# NN-13 measurements (2026-11-09)

NN-13 is complete for the measured CPU tensor-product adapter. The working
runtime supplies the input missing in the November 6 assessment. The evidence
justifies sharing existing CPU product loops with private tensor storage and
one call-owned product workspace. Session storage reuse remains RT-16 work.
No public tensor/storage API changes.

## Exact sources and environment

- Runtime: [`4d5a000c20ffae3e053190a4cd04322529851800`](https://github.com/itsmontoya/onnx/commit/4d5a000c20ffae3e053190a4cd04322529851800), merged on `main`.
- Before backend: `v0.7.0`, `6048452c57a0a293db4591cd2e2febd7d6b743a3`.
- After backend: [`a64fa687e269c3a1bf3ac345434b46c8ed4a59fc`](https://github.com/itsmontoya/neuralnetwork/commit/a64fa687e269c3a1bf3ac345434b46c8ed4a59fc), fetched as `v0.7.1-0.20261109154142-a64fa687e269` with `go get`. Timing/profile/consumer checks use no local replacement.
- Go `go1.27.2 darwin/arm64`; macOS 26.5.2 (25F84); Apple M3.
- Performance samples: `GOMAXPROCS=1`, `CGO_ENABLED=0`, default and `purego` builds; ten 100 ms samples per case/phase. No other benchmark/test workload ran concurrently with those samples.
- Model/input/expected-output TensorProto paths and SHA-256 hashes are in [fixtures.json](fixtures.json), copied from the pinned runtime provenance. Both graphs run batches 1 and 3; existing fixture loaders check every artifact hash.
- Reference outputs are frozen checked-in fixtures. No reference runtime executes during Go measurements, so reference thread settings do not apply to these runs. Generation provenance remains in the pinned runtime manifest.

## Scope and ownership

`internal/float32product` contains the existing CPU loops, preserving float32
accumulation order. Matrix methods still synchronize host storage and choose
execution/fallback before calling these loops. The tensor CPU adapter directly
reads its private validated contiguous inputs and writes a disjoint call-owned
product buffer. Gemm stages one product; batched MatMul reuses disjoint blocks
of one output-sized workspace, then copies once to its caller-owned destination.
No input/initializer storage becomes writable through a new API, retained by a
kernel, or cached globally. Validation precedes destination writes. Outputs
retain their ownership/error contract.

The `darwin && cgo && metal && !purego` build keeps the copying Matrix adapter.
No residency/device representation or host/device boundary changed. Actual
Metal matmul, residency-coherence and CPU-fallback/resume tests passed on this
host; [metal.txt](metal.txt) records targeted tests. Constant packing and public
session workspaces are not justified by these small workloads and are not added.

## Latency and Go allocations

Latency cells show median [minimum, maximum] ns/op over ten samples. Allocation
cells show the median and include [minimum, maximum] when values varied. Run
allocation counts/B/op were stable; Load has small setup/map-allocation variation.
`Load` includes LoadBytes/Close; `First` includes NewSession/Run/Close on an
already-loaded model; `Warm` reuses a warmed session and includes owned output
copies. Fixture decoding/comparison stays outside timed loops. Raw samples:
[before-default](before-default.txt), [after-default](after-default.txt),
[before-purego](before-purego.txt), [after-purego](after-purego.txt).

### default

| Case / phase | Before ns/op | After ns/op | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| `mlp_batch_1/Load` | 184165 [182458, 224442] | 181160 [180372, 233533] | 156512 [156512, 156804] → 156513 [156512, 156825] | 1182 [1182, 1184] → 1182 [1182, 1185] |
| `mlp_batch_1/First` | 3594 [3561, 3644] | 3495 [3478, 3520] | 5192 → 4808 | 101 → 89 |
| `mlp_batch_1/Warm` | 3676.5 [3549, 3910] | 3472.5 [3434, 3518] | 5168 → 4784 | 100 → 88 |
| `mlp_batch_3/Load` | 182526 [181163, 221120] | 184254 [182153, 187560] | 156512 [156512, 156527] → 156512 [156512, 156527] | 1182 → 1182 |
| `mlp_batch_3/First` | 3871 [3792, 4527] | 3697 [3631, 3727] | 5632 → 5152 | 101 → 89 |
| `mlp_batch_3/Warm` | 3848 [3784, 3938] | 3677.5 [3637, 3749] | 5608 → 5128 | 100 → 88 |
| `shape_batch_1/Load` | 207179 [205936, 212788] | 211586 [208889, 212404] | 185651 [185651, 185737] → 185651 [185651, 185739] | 1508 → 1508 |
| `shape_batch_1/First` | 6027 [5924, 6174] | 6144 [6078, 6175] | 10920 → 10920 | 139 → 139 |
| `shape_batch_1/Warm` | 6040.5 [5976, 6384] | 6100.5 [6030, 6165] | 10896 → 10896 | 138 → 138 |
| `shape_batch_3/Load` | 210720 [208240, 215369] | 211051 [210475, 214442] | 185651 [185651, 185668] → 185651 [185651, 185667] | 1508 → 1508 |
| `shape_batch_3/First` | 6156.5 [6096, 6346] | 6285.5 [6188, 6598] | 11312 → 11312 | 139 → 139 |
| `shape_batch_3/Warm` | 6145 [6066, 6265] | 6291 [6245, 7355] | 11288 → 11288 | 138 → 138 |

### purego

| Case / phase | Before ns/op | After ns/op | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| `mlp_batch_1/Load` | 185024 [181471, 238036] | 191922 [186506, 238217] | 156513 [156512, 156826] → 156513 [156512, 156827] | 1182 [1182, 1185] → 1182 [1182, 1185] |
| `mlp_batch_1/First` | 3689 [3649, 3735] | 3600 [3578, 3631] | 5192 → 4808 | 101 → 89 |
| `mlp_batch_1/Warm` | 3700.5 [3611, 3940] | 3566 [3536, 3643] | 5168 → 4784 | 100 → 88 |
| `mlp_batch_3/Load` | 187703 [184366, 189950] | 192676 [190562, 193056] | 156512 [156512, 156528] → 156513 [156512, 156526] | 1182 → 1182 |
| `mlp_batch_3/First` | 3877 [3820, 4004] | 3742.5 [3691, 3759] | 5632 → 5152 | 101 → 89 |
| `mlp_batch_3/Warm` | 3896 [3843, 4132] | 3719.5 [3709, 3735] | 5608 → 5128 | 100 → 88 |
| `shape_batch_1/Load` | 212120 [209646, 215487] | 213315 [206808, 216475] | 185651 [185651, 185738] → 185651 [185651, 185734] | 1508 → 1508 |
| `shape_batch_1/First` | 6044.5 [5996, 6166] | 6214 [6173, 6243] | 10920 → 10920 | 139 → 139 |
| `shape_batch_1/Warm` | 6104.5 [5976, 6148] | 6175.5 [6084, 6247] | 10896 → 10896 | 138 → 138 |
| `shape_batch_3/Load` | 213880 [210716, 216112] | 213852 [212231, 214135] | 185651 [185651, 185668] → 185651 [185651, 185667] | 1508 → 1508 |
| `shape_batch_3/First` | 6247.5 [6185, 6372] | 6295 [6285, 6444] | 11312 → 11312 | 139 → 139 |
| `shape_batch_3/Warm` | 6206.5 [6129, 6403] | 6270 [6177, 6336] | 11288 → 11288 | 138 → 138 |

The stable retention justification is the MLP reduction of 12 allocations per
run, 384 B/run at batch 1 and 480 B/run at batch 3, plus payload metrics below.
Short samples suggest a small MLP latency benefit on this host, without proving
a cross-machine speedup. Shape is an unchanged control: its latency varies
despite identical allocation/copy/live-payload counts. Loading is unchanged.
No universal zero-allocation or speedup target is imposed.

## Executed copies and live numeric payload

These observations are separate serial instrumented runs, never timing or Go
allocation benchmarks. [before-payload.json](before-payload.json) and
[after-payload.json](after-payload.json) include source-site attribution.
The byte sizes below are observed numeric payload sizes, not allocator-size
estimates or unimplemented counters.

- Copied bytes count actual copied element counts of each payload `copy`, plus scalar assignments in explicit transpose materialization. Arithmetic/activation result writes, binary wire decoding, and shape-metadata copies are excluded.
- Peak payload counts simultaneously owned numeric buffers in nested function scopes: immutable constants and caller inputs, run-value/returned-output clones, destinations, Matrix input/output/value copies, product staging, integer shape-value staging, and numeric accessor buffers. Aliases count once. Strong references prevent addresses being reused while tracked.
- Allocations are released at function exit unless reachable through named results or caller-owned arguments/receivers. This is a reproducible **function-scoped upper bound** on logical live numeric storage. It can retain a local beyond compiler-determined last use; it is not exact GC liveness, RSS, or a hard memory cap. Wire bytes, protobuf/Go objects, allocator overhead, ordinary shape/stride metadata and instrumentation memory are excluded. These pinned fixture initializers all use raw data; generated protobuf typed-numeric allocation paths are not exercised or claimed covered.
- First includes existing model/input roots. Warm also retains one prior output set and checks it after another Run and Close. Its peak includes those retained outputs, exercising ownership. This differs from a timing loop where an unused prior output may become dead earlier.
- JSON `allocated_payload_bytes` is cumulative newly registered payload, excluding initial roots. It differs from simultaneous peak and Go benchmark B/op.

| Case / phase | Copied bytes before → after | Peak payload bytes before → after |
| --- | ---: | ---: |
| `mlp_batch_1/First` | 256 → 168 | 228 → 204 |
| `mlp_batch_1/Load` | 68 → 68 | 84 → 84 |
| `mlp_batch_1/Warm` | 256 → 168 | 236 → 212 |
| `mlp_batch_3/First` | 536 → 368 | 364 → 340 |
| `mlp_batch_3/Load` | 68 → 68 | 84 → 84 |
| `mlp_batch_3/Warm` | 536 → 368 | 388 → 364 |
| `shape_batch_1/First` | 312 → 312 | 328 → 328 |
| `shape_batch_1/Load` | 64 → 64 | 40 → 40 |
| `shape_batch_1/Warm` | 312 → 312 | 392 → 392 |
| `shape_batch_3/First` | 648 → 648 | 712 → 712 |
| `shape_batch_3/Load` | 64 → 64 | 40 → 40 |
| `shape_batch_3/Warm` | 648 → 648 | 872 → 872 |

MLP run copies fall from 256 to 168 bytes at batch 1 and from 536 to 368 bytes
at batch 3. The function-scoped peak falls by 24 bytes at both batches; unchanged
activation/facade copies become the peak. Shape copies remain, including its
explicit transpose. No zero-copy claim is made.

## Allocation profile cross-check

Rate-1 `alloc_space` profiles were collected separately for
`BenchmarkInference/mlp_batch_3/Warm` with one-second benchmark durations.
The [before](before-alloc.txt)/[after](after-alloc.txt) profile summaries and
[before](before-profile.txt)/[after](after-profile.txt) iteration counts are
published. Different counts mean whole-profile totals cannot be compared as
per-run metrics. Normalizing recurring sites by iterations gives approximately:

| Allocation site | Before B/run | After B/run |
| --- | ---: | ---: |
| Matrix.FromSlice | 408 | 96 |
| Matrix.New | 264 | 96 |
| Matrix.Values | 120 | 48 |
| CPU product workspace | 0 | 72 |

The net 480 B/run reduction agrees with the unprofiled benchmark. Remaining
Matrix sites belong to the unchanged activation adapter. These values include
allocator rounding/object overhead and are not payload copies or peak memory.
Profiling overhead makes these runs unsuitable as latency samples.

## Reproduce

Use a clean isolated checkout of the pinned runtime for timing/consumer checks.
First run with its released backend, then fetch the exact after revision:

```sh
git checkout --detach 4d5a000c20ffae3e053190a4cd04322529851800
GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkInference$' -benchtime=100ms -benchmem -count=10 .
GOMAXPROCS=1 CGO_ENABLED=0 go test -tags=purego -run '^$' -bench '^BenchmarkInference$' -benchtime=100ms -benchmem -count=10 .
GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkInference/mlp_batch_3/Warm$' -benchtime=1s -memprofile=before.pprof -memprofilerate=1 .
go get github.com/itsmontoya/neuralnetwork@a64fa687e269c3a1bf3ac345434b46c8ed4a59fc
# Repeat both ten-sample commands and the profile command (after.pprof).
go tool pprof -top -nodecount=30 -alloc_space before.pprof
go tool pprof -top -nodecount=30 -alloc_space after.pprof
```

For payload observations, run from this backend branch with both Git source
repositories available (paths below are examples):

```sh
python3 tools/nn13/measure.py --runtime /path/to/onnx --backend . --backend-revision 6048452c57a0a293db4591cd2e2febd7d6b743a3 --output before-payload.json
python3 tools/nn13/measure.py --runtime /path/to/onnx --backend . --backend-revision a64fa687e269c3a1bf3ac345434b46c8ed4a59fc --output after-payload.json
```

The tool creates isolated Git archives at those exact revisions, copies only
this branch's measurement tooling, instruments them with the Go AST rewriter,
and uses a temporary Go workspace to resolve those explicitly modified observer
copies. It deletes them afterward. No production checkout or module-cache
source is rewritten. This observation-only overlay differs from the unmodified
published dependency used for timings/profiles/tests. Python 3, Go, Git and tar
are required; Python ONNX/ORT is not needed. This is a bounded observer for
these pinned fixtures, not a general profiler.

## Validation and remaining work

- Backend gofmt, vet, normal, cgo-disabled, cgo-disabled purego, race, and `go test -tags=metal ./matrix ./tensor` passed. Independent product tests check all transpose combinations bit-for-bit, read-only inputs and zero-contraction clearing. Existing public product fixtures cover vectors, batching, broadcasting, empty tensors, alias errors and unchanged destinations on errors.
- Observer tests check exact copy counts, subslice alias deduplication, receiver retention, scope release, metadata exclusion and scalar transpose-copy accounting. Both pinned observations were reproduced successfully.
- The runtime fetched the published after backend and passed vet, normal, cgo-disabled, cgo-disabled purego and race checks. Existing `TestSessionOutputsSurviveAndConstantsStayPrivate`, `TestSessionConcurrentIndependentRuns`, `TestSessionBusyAndSeparateSessions`, `TestSessionCancellationAtEveryBoundaryAndRecovery`, `TestM2ReferenceGraphs`, `TestM2DynamicFailuresRecover` and dynamic-batch tests exercise ownership, immutable constants, integer precision, independence and cleanup.
- Actual Metal standard/left/right products, residency coherence and dependent CPU-fallback/resume tests passed. Their tensor execution path retains the copying adapter. No GPU speedup is claimed.
- RT-16 owns runtime liveness, session reuse, CNN measurements and consumer scratch-budget updates. Existing reservations remain conservative for this CPU change. It may consume the public source/pseudo-version above without waiting for a tag or merging this optimization PR.
