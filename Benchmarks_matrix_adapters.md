# Tensor arithmetic matrix adapters (Pass 16)

The equal-shape Add/Sub/Mul adapter now uses the existing
`Matrix.ValuesInto(destination.floats)` instead of allocating `Matrix.Values()`
and copying its result a second time. For `[1,10]` Add, this removes **48 B and
one allocation per call: 336 B / 7 objects → 288 B / 6 objects**. The input
matrix copies and private output matrix remain. MatMul is unchanged and retains
its complete output staging and planning allocations.

## Scope and contract

Base: `740a244bf56214bb3443a69287a2d800674620e4` (fetched `origin/main`, also
the v0.10.9 merge). Branch: `optimization-pass-16-matrix-adapters`.
The default branch was checked against the remote before branching; it had no
subsequent fix for this mechanism. The initial worktree was clean.

Commits: `2065cd6` adds contract tests and reusable baseline probes; `fa87db8`
implements the production adapter change. The following documentation commit
records the measurements and validation. No release, tag, or merge is performed.

Production changes are confined to `tensor/arithmetic.go`. Reusable benchmarks
are in `tensor/matrix_adapter_benchmark_test.go`; focused regressions are in
`tensor/matrix_adapter_test.go`. This report is the durable review summary.
No dependency, public API, matrix constructor/accessor copy contract, tensor
validation order, arithmetic dispatch, tolerance, or MatMul/Gemm implementation
changes. ONNX was not changed or used for integration.

The existing signatures remain:

```go
func AddInto(left, right, destination *Tensor) (err error)
func SubInto(left, right, destination *Tensor) (err error)
func MulInto(left, right, destination *Tensor) (err error)
func MatMulInto(left, right, destination *Tensor) (err error)
```

Inputs remain read-only; repeated left/right inputs are allowed. The destination
must remain disjoint, including pointer/shallow-copy aliases of empty tensors.
Its shape and backing storage are preserved and completely overwritten on
success. Actual shapes and values are validated anew on each call. No geometry
or payload is cached, borrowed, retained, or exposed. Arbitrary ranks, scalars,
empty outputs, broadcast fallbacks and architecture-selected matrix arithmetic
remain on their existing paths, including purego and Metal-tagged builds.

`ValuesInto` validates the source and destination length, then synchronizes host
values **before** copying. Every returned error precedes its write. The adapter
still computes into a private matrix, so an operation failure after writing that
matrix leaves the tensor destination untouched. The same wrapping context is
preserved. Public Add/Sub/Mul argument-error atomicity therefore survives; no
new failure can occur after tensor publication starts.

### Alternatives and MatMul failure audit

Using `ValuesInto` needs no new ownership API and preserves exactly the existing
matrix device/synchronization and SIMD selection. A shared slice kernel would
require moving or duplicating architecture dispatch and testing a broader
boundary. A public borrowed view would add overlap, mutation, panic, expiration,
and device ownership contracts. Neither is justified for this small change.
Pooling and caches would introduce persistent lifetime/concurrency obligations.

MatMul still measures 144 B / 3 allocations for `[1,256] × [256,10]`:
80 B plan object, 16 B result-shape array, 48 B output staging. Its host input
payloads already go directly to `internal/float32product`; matrix input views
would remove no host copies. No new MatMul optimization is claimed.

The returned-error sites were inspected in order:

1. Left/right tensor validation: initialization, exact type, storage lengths,
   dimension/product/byte/host-index checks.
2. Plan construction: both shapes, non-scalar ranks, contraction compatibility,
   batch broadcasting, checked result shape and ownership of copied dimensions.
3. Destination validation: initialization, type, exact shape and identities.
   Empty outputs return only afterward; empty contractions then clear to +0.
4. Batch preparation: checked left/right/output block products, contiguous batch
   strides, and both broadcast stride calculations. These all precede writes.
   Validated positive nonempty shapes bound the batch coordinate offsets and
   host conversions; empty dimensions avoid unused stride calculations.
5. Host products synchronously call float32product and return nil. A separate
   host-only destination path is plausible, but would require another backend
   capability split to remove only the 40-byte final copy and staging here.
6. The Metal build instead constructs matrices and can fail during allocation,
   matrix validation, runtime/execution selection, buffer preparation, command
   encoding, execution finish/abort, or host synchronization/observation. A later
   batch can fail after earlier private products completed. Gemm shares these
   helpers, including transpose paths.

Keeping staging avoids that additional implementation and proof surface:
**all batches must succeed before the single destination copy**. The unchanged
code structurally preserves atomicity for every returned product error, including
a late device error. A separate host-only test overlay injects an error after
the second batch's kernel has written its private output; destination and input
bits remain unchanged in default, purego, race and purego+race runs. The overlay
and its source generator are retained with the evidence and add no production
hook. Existing Metal runtime fault tests were also run; a tensor-level injected
later-batch GPU failure was not added or tested. This does not prove device
products cannot fail. Planning stays unchanged as well; this is an arithmetic
adapter change, not a planning rewrite.

## Measurement policy and evidence

Measured 2026-11-12: Apple M3, Mac15,13, 8 cores, 24 GiB; macOS 26.5.2
(25F84), Darwin 25.5.0; Go 1.27.2, darwin/arm64. Module declares Go 1.26.1;
dependencies remain simd v1.3.0 and x/sys v0.45.0.

Both binaries use the same benchmark source, toolchain and host. Build policy:
`GOWORK=off CGO_ENABLED=0 GOARM64=v8.0`, default optimization, no gcflags;
the purego pair additionally uses `-tags=purego`. Latency uses `GOMAXPROCS=1`,
200 ms per benchmark, eight repetitions **per version per mode**, sequential
before/after pairs with order alternating AB / BA each round. No profiling,
race instrumentation, other test suite or compilation runs during this series.
All valid samples are retained. Tables use medians, full min–max ranges and
population CV (standard deviation / mean). Setup, destination creation and
initial values are outside timing. Inputs use
`float32((index+offset)%29-14)*0.125`, offsets 3 and 7; existing channel Add
benchmarks retain their own documented value formulas and both operand orders.

The complete public-call rows include validation and dispatch. Adapter, matrix
method, plan, staged-batch and kernel rows are separate scopes, not interchangeable
public-call implementations. No stripped-kernel or diagnostic-ceiling saving is
added to another saving. Initial seven-repetition baseline probes are preserved
separately from the balanced comparison.

Raw samples, executable binaries, scripts, allocation profiles, compiler output,
disassembly, and verification logs are together at:
`.benchmarks/matrix-adapters/20261112-Or0as6/` (gitignored).
`measure.py` reproduces the balanced series; `summary.json` retains every
sample and derived statistics. `validate.py` records individual check commands,
environment overrides, logs and exit statuses in `validation.json`.

### default build

Each timing cell is **median [minimum–maximum] µs/op; CV%**. B/op and objects are
constant across the eight samples unless shown otherwise.

| Scope | Before µs/op; CV | After µs/op; CV | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| Public broadcast Add/MNISTFirst/Forward | 1.944000 [1.908000–1.973000]; 1.11% | 1.985500 [1.956000–2.146000]; 3.19% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTFirst/Reverse | 1.955000 [1.864000–2.113000]; 3.37% | 1.973500 [1.862000–1.999000]; 2.03% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTSecond/Forward | 1.199000 [1.147000–1.226000]; 1.98% | 1.222500 [1.176000–1.247000]; 1.72% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTSecond/Reverse | 1.202000 [1.151000–1.221000]; 1.86% | 1.222000 [1.185000–1.295000]; 2.36% | 0 → 0 | 0 → 0 |
| Public broadcast Add/Batch/Forward | 0.826000 [0.805400–0.877000]; 3.19% | 0.817050 [0.777900–0.877200]; 3.12% | 0 → 0 | 0 → 0 |
| Public broadcast Add/Batch/Reverse | 0.899350 [0.860200–0.956100]; 2.75% | 0.895750 [0.860400–0.951900]; 3.27% | 0 → 0 | 0 → 0 |
| Public Add/0 | 0.054005 [0.050580–0.054500]; 2.25% | 0.053945 [0.052790–0.054440]; 0.85% | 0 → 0 | 0 → 0 |
| Public Add/1 | 0.165550 [0.156600–0.168200]; 1.97% | 0.161650 [0.159800–0.163800]; 0.67% | 160 → 156 | 7 → 6 |
| Public Add/10 | 0.179900 [0.173300–0.181800]; 1.40% | 0.170750 [0.168400–0.172500]; 0.69% | 336 → 288 | 7 → 6 |
| Public Add/15 | 0.183900 [0.175500–0.185500]; 1.64% | 0.174200 [0.171700–0.174800]; 0.58% | 400 → 336 | 7 → 6 |
| Public Add/16 | 0.183250 [0.175900–0.186400]; 1.57% | 0.173850 [0.173100–0.175500]; 0.46% | 400 → 336 | 7 → 6 |
| Public Add/17 | 0.190100 [0.183100–0.191400]; 1.31% | 0.178150 [0.177200–0.180800]; 0.72% | 464 → 384 | 7 → 6 |
| Public Add/4096 | 4.690500 [4.553000–4.862000]; 1.73% | 3.640000 [3.489000–3.696000]; 1.87% | 65680 → 49296 | 7 → 6 |
| Public Add/65536 | 63.380000 [61.136000–70.514000]; 4.01% | 50.997500 [49.728000–51.637000]; 1.09% | 1048720 → 786576 | 7 → 6 |
| Public Sub/0 | 0.053900 [0.051850–0.057860]; 2.94% | 0.053770 [0.053460–0.054180]; 0.35% | 0 → 0 | 0 → 0 |
| Public Sub/1 | 0.166200 [0.158800–0.201600]; 7.73% | 0.161250 [0.158600–0.162900]; 0.96% | 160 → 156 | 7 → 6 |
| Public Sub/10 | 0.180350 [0.173400–0.221100]; 7.56% | 0.171000 [0.167200–0.172100]; 1.00% | 336 → 288 | 7 → 6 |
| Public Sub/15 | 0.184750 [0.178300–0.267700]; 14.13% | 0.173550 [0.172500–0.175000]; 0.48% | 400 → 336 | 7 → 6 |
| Public Sub/16 | 0.183150 [0.178200–0.189000]; 1.71% | 0.173250 [0.170800–0.174900]; 0.83% | 400 → 336 | 7 → 6 |
| Public Sub/17 | 0.189550 [0.186200–0.198900]; 1.86% | 0.178900 [0.176000–0.192400]; 2.63% | 464 → 384 | 7 → 6 |
| Public Sub/4096 | 4.740000 [4.508000–5.014000]; 2.83% | 3.625000 [3.568000–3.690000]; 1.29% | 65680 → 49296 | 7 → 6 |
| Public Sub/65536 | 63.300000 [62.062000–65.236000]; 1.46% | 51.060000 [49.924000–52.383000]; 1.61% | 1048720 → 786576 | 7 → 6 |
| Public Mul/0 | 0.054010 [0.052640–0.054400]; 0.94% | 0.053840 [0.053530–0.054500]; 0.55% | 0 → 0 | 0 → 0 |
| Public Mul/1 | 0.166950 [0.162400–0.169000]; 1.13% | 0.163650 [0.162400–0.166700]; 0.76% | 160 → 156 | 7 → 6 |
| Public Mul/10 | 0.179550 [0.176500–0.182600]; 1.00% | 0.170100 [0.169100–0.172200]; 0.65% | 336 → 288 | 7 → 6 |
| Public Mul/15 | 0.183150 [0.179000–0.187400]; 1.37% | 0.174750 [0.171800–0.177000]; 0.92% | 400 → 336 | 7 → 6 |
| Public Mul/16 | 0.181550 [0.177200–0.184500]; 1.12% | 0.173000 [0.170600–0.174500]; 0.62% | 400 → 336 | 7 → 6 |
| Public Mul/17 | 0.188900 [0.183100–0.189700]; 1.10% | 0.177850 [0.176500–0.238500]; 10.80% | 464 → 384 | 7 → 6 |
| Public Mul/4096 | 4.719000 [4.566000–4.916000]; 2.00% | 3.647500 [3.529000–3.769000]; 1.71% | 65680 → 49296 | 7 → 6 |
| Public Mul/65536 | 63.264500 [62.290000–67.004000]; 2.55% | 51.188000 [48.828000–52.361000]; 1.84% | 1048720 → 786576 | 7 → 6 |
| Matrix scope/Adapter | 0.102850 [0.100800–0.103500]; 0.77% | 0.093400 [0.091240–0.093790]; 0.85% | 336 → 288 | 7 → 6 |
| Matrix scope/FromSlice | 0.023735 [0.023050–0.024170]; 1.26% | 0.023880 [0.023230–0.024590]; 1.51% | 96 → 96 | 2 → 2 |
| Matrix scope/New | 0.022470 [0.021710–0.022710]; 1.36% | 0.022560 [0.021790–0.022800]; 1.26% | 96 → 96 | 2 → 2 |
| Matrix scope/MatrixAdd | 0.019570 [0.019340–0.019700]; 0.53% | 0.019605 [0.019520–0.019830]; 0.45% | 0 → 0 | 0 → 0 |
| Matrix scope/Values | 0.015960 [0.015720–0.016180]; 0.80% | 0.015960 [0.015450–0.017420]; 3.34% | 48 → 48 | 1 → 1 |
| Matrix scope/ValuesInto | 0.005053 [0.005019–0.005063]; 0.26% | 0.005052 [0.005035–0.005157]; 0.79% | 0 → 0 | 0 → 0 |
| Matrix scope/Copy | 0.001868 [0.001856–0.001886]; 0.44% | 0.001871 [0.001865–0.002041]; 2.97% | 0 → 0 | 0 → 0 |
| MatMul/MNIST/Complete | 1.687500 [1.675000–1.708000]; 0.57% | 1.691000 [1.669000–1.700000]; 0.58% | 144 → 144 | 3 → 3 |
| MatMul/MNIST/Plan | 0.062135 [0.060980–0.066810]; 3.27% | 0.061140 [0.060530–0.066070]; 2.75% | 96 → 96 | 2 → 2 |
| MatMul/MNIST/Staging | 1.580000 [1.532000–1.592000]; 1.08% | 1.576500 [1.566000–1.590000]; 0.46% | 48 → 48 | 1 → 1 |
| MatMul/MNIST/Kernel | 1.578500 [1.504000–1.677000]; 3.32% | 1.564000 [1.551000–1.571000]; 0.47% | 0 → 0 | 0 → 0 |
| MatMul/Dot/Complete | 0.414900 [0.394300–0.758100]; 26.87% | 0.413100 [0.407900–0.444000]; 2.86% | 96 → 96 | 2 → 2 |
| MatMul/Dot/Plan | 0.051745 [0.050700–0.052840]; 1.26% | 0.051190 [0.050060–0.062280]; 7.12% | 96 → 96 | 2 → 2 |
| MatMul/Dot/Staging | 0.313500 [0.296800–0.731900]; 38.69% | 0.310600 [0.306500–0.340100]; 3.23% | 0 → 0 | 0 → 0 |
| MatMul/Dot/Kernel | 0.246050 [0.238700–0.439300]; 26.87% | 0.245850 [0.242000–0.284800]; 5.21% | 0 → 0 | 0 → 0 |
| MatMul/Batch/Complete | 2.777000 [2.685000–2.802000]; 1.25% | 2.788000 [2.768000–3.341000]; 6.36% | 1200 → 1200 | 7 → 7 |
| MatMul/Batch/Plan | 0.085835 [0.084080–0.093530]; 3.10% | 0.086080 [0.083910–0.090360]; 2.25% | 128 → 128 | 3 → 3 |
| MatMul/Batch/Staging | 2.632000 [2.558000–2.661000]; 1.09% | 2.653500 [2.619000–2.677000]; 0.67% | 1072 → 1072 | 4 → 4 |

Public Add/10 improves in 8/8 paired rounds; median paired saving 0.010350 µs (not an ONNX saving).

Add/10 before samples (µs, round order): 0.173300, 0.180100, 0.180600, 0.179100, 0.179500, 0.179700, 0.181700, 0.181800.
Add/10 after samples (µs, round order): 0.170800, 0.172500, 0.169700, 0.169300, 0.168400, 0.171100, 0.170700, 0.170800.

### purego build

Each timing cell is **median [minimum–maximum] µs/op; CV%**. B/op and objects are
constant across the eight samples unless shown otherwise.

| Scope | Before µs/op; CV | After µs/op; CV | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| Public broadcast Add/MNISTFirst/Forward | 1.979000 [1.969000–2.186000]; 3.60% | 1.984500 [1.960000–2.172000]; 4.01% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTFirst/Reverse | 1.984000 [1.971000–2.110000]; 2.14% | 1.988500 [1.971000–2.011000]; 0.61% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTSecond/Forward | 1.231000 [1.204000–1.281000]; 1.72% | 1.217500 [1.200000–1.227000]; 0.70% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTSecond/Reverse | 1.225500 [1.210000–1.298000]; 2.13% | 1.234500 [1.224000–1.240000]; 0.38% | 0 → 0 | 0 → 0 |
| Public broadcast Add/Batch/Forward | 0.829700 [0.823000–0.939500]; 4.35% | 0.893850 [0.877300–0.905400]; 1.04% | 0 → 0 | 0 → 0 |
| Public broadcast Add/Batch/Reverse | 0.834300 [0.829200–0.842400]; 0.49% | 0.822750 [0.818200–0.826500]; 0.34% | 0 → 0 | 0 → 0 |
| Public Add/0 | 0.054335 [0.053940–0.054960]; 0.48% | 0.054265 [0.054040–0.054650]; 0.37% | 0 → 0 | 0 → 0 |
| Public Add/1 | 0.165650 [0.163400–0.172000]; 1.52% | 0.160350 [0.158700–0.161800]; 0.68% | 160 → 156 | 7 → 6 |
| Public Add/10 | 0.179350 [0.177400–0.183700]; 1.22% | 0.168750 [0.167200–0.182600]; 2.75% | 336 → 288 | 7 → 6 |
| Public Add/15 | 0.183700 [0.181400–0.207900]; 4.33% | 0.172650 [0.170800–0.174900]; 0.72% | 400 → 336 | 7 → 6 |
| Public Add/16 | 0.186300 [0.182300–0.204200]; 3.62% | 0.173500 [0.171600–0.176600]; 0.87% | 400 → 336 | 7 → 6 |
| Public Add/17 | 0.192600 [0.188400–0.195500]; 1.12% | 0.179050 [0.176200–0.204700]; 4.83% | 464 → 384 | 7 → 6 |
| Public Add/4096 | 5.873000 [5.714000–6.059000]; 1.76% | 4.775000 [4.691000–5.537000]; 5.34% | 65680 → 49296 | 7 → 6 |
| Public Add/65536 | 85.400500 [82.295000–86.943000]; 1.60% | 71.157000 [69.296000–71.780000]; 1.05% | 1048720 → 786576 | 7 → 6 |
| Public Sub/0 | 0.054175 [0.053840–0.054690]; 0.47% | 0.054290 [0.054150–0.054520]; 0.21% | 0 → 0 | 0 → 0 |
| Public Sub/1 | 0.164700 [0.163000–0.226000]; 11.75% | 0.161150 [0.159400–0.181600]; 4.24% | 160 → 156 | 7 → 6 |
| Public Sub/10 | 0.181050 [0.177300–0.184800]; 1.27% | 0.169100 [0.168100–0.172900]; 1.02% | 336 → 288 | 7 → 6 |
| Public Sub/15 | 0.184150 [0.181200–0.222400]; 6.85% | 0.172200 [0.171100–0.174800]; 0.76% | 400 → 336 | 7 → 6 |
| Public Sub/16 | 0.185100 [0.181800–0.198500]; 2.68% | 0.173000 [0.171300–0.175400]; 0.84% | 400 → 336 | 7 → 6 |
| Public Sub/17 | 0.191300 [0.188700–0.203400]; 2.26% | 0.178650 [0.176700–0.194000]; 2.92% | 464 → 384 | 7 → 6 |
| Public Sub/4096 | 5.889500 [5.845000–6.007000]; 1.04% | 4.762000 [4.700000–4.866000]; 1.19% | 65680 → 49296 | 7 → 6 |
| Public Sub/65536 | 85.088500 [82.016000–86.628000]; 1.58% | 71.081500 [70.348000–73.152000]; 1.23% | 1048720 → 786576 | 7 → 6 |
| Public Mul/0 | 0.054090 [0.053890–0.054530]; 0.38% | 0.054280 [0.054190–0.058100]; 2.30% | 0 → 0 | 0 → 0 |
| Public Mul/1 | 0.165150 [0.163300–0.167600]; 0.87% | 0.160750 [0.158600–0.161300]; 0.50% | 160 → 156 | 7 → 6 |
| Public Mul/10 | 0.179800 [0.178100–0.205700]; 5.02% | 0.169000 [0.167700–0.170000]; 0.51% | 336 → 288 | 7 → 6 |
| Public Mul/15 | 0.183750 [0.181900–0.187700]; 1.12% | 0.174050 [0.171400–0.206300]; 6.12% | 400 → 336 | 7 → 6 |
| Public Mul/16 | 0.184950 [0.182200–0.187200]; 0.84% | 0.173500 [0.171400–0.175300]; 0.76% | 400 → 336 | 7 → 6 |
| Public Mul/17 | 0.190650 [0.187900–0.192600]; 0.79% | 0.178550 [0.176700–0.179500]; 0.55% | 464 → 384 | 7 → 6 |
| Public Mul/4096 | 5.878500 [5.824000–5.985000]; 0.76% | 4.801000 [4.706000–4.866000]; 0.96% | 65680 → 49296 | 7 → 6 |
| Public Mul/65536 | 85.526500 [84.271000–86.096000]; 0.79% | 71.628000 [69.957000–71.928000]; 0.92% | 1048720 → 786576 | 7 → 6 |
| Matrix scope/Adapter | 0.101800 [0.099560–0.125300]; 7.58% | 0.092185 [0.089990–0.094320]; 1.35% | 336 → 288 | 7 → 6 |
| Matrix scope/FromSlice | 0.024085 [0.023210–0.024430]; 1.40% | 0.023830 [0.022950–0.023980]; 1.30% | 96 → 96 | 2 → 2 |
| Matrix scope/New | 0.022490 [0.021590–0.023220]; 1.84% | 0.022400 [0.021540–0.022790]; 1.53% | 96 → 96 | 2 → 2 |
| Matrix scope/MatrixAdd | 0.018540 [0.018420–0.018870]; 0.80% | 0.018540 [0.018410–0.019150]; 1.21% | 0 → 0 | 0 → 0 |
| Matrix scope/Values | 0.015930 [0.015340–0.016020]; 1.31% | 0.015995 [0.015580–0.017400]; 3.14% | 48 → 48 | 1 → 1 |
| Matrix scope/ValuesInto | 0.005061 [0.005037–0.005087]; 0.29% | 0.005050 [0.005038–0.005621]; 3.68% | 0 → 0 | 0 → 0 |
| Matrix scope/Copy | 0.001874 [0.001864–0.001885]; 0.35% | 0.001866 [0.001859–0.001881]; 0.41% | 0 → 0 | 0 → 0 |
| MatMul/MNIST/Complete | 1.698500 [1.688000–1.718000]; 0.55% | 1.702000 [1.691000–1.732000]; 0.70% | 144 → 144 | 3 → 3 |
| MatMul/MNIST/Plan | 0.061635 [0.060910–0.062500]; 0.67% | 0.061445 [0.060990–0.062250]; 0.59% | 96 → 96 | 2 → 2 |
| MatMul/MNIST/Staging | 1.589500 [1.577000–1.600000]; 0.43% | 1.590500 [1.587000–1.596000]; 0.22% | 48 → 48 | 1 → 1 |
| MatMul/MNIST/Kernel | 1.571500 [1.542000–1.585000]; 0.75% | 1.571000 [1.540000–1.592000]; 0.92% | 0 → 0 | 0 → 0 |
| MatMul/Dot/Complete | 0.412050 [0.397000–0.423200]; 2.05% | 0.410800 [0.397900–0.421900]; 1.70% | 96 → 96 | 2 → 2 |
| MatMul/Dot/Plan | 0.051890 [0.051250–0.052590]; 0.81% | 0.051640 [0.050940–0.051990]; 0.59% | 96 → 96 | 2 → 2 |
| MatMul/Dot/Staging | 0.308200 [0.306100–0.312500]; 0.66% | 0.307250 [0.305800–0.823300]; 45.87% | 0 → 0 | 0 → 0 |
| MatMul/Dot/Kernel | 0.247850 [0.246200–0.306400]; 7.62% | 0.255050 [0.245900–0.728300]; 49.79% | 0 → 0 | 0 → 0 |
| MatMul/Batch/Complete | 2.796000 [2.784000–3.033000]; 2.79% | 2.804000 [2.789000–2.820000]; 0.35% | 1200 → 1200 | 7 → 7 |
| MatMul/Batch/Plan | 0.086810 [0.086120–0.087960]; 0.67% | 0.085845 [0.084960–0.087220]; 0.90% | 128 → 128 | 3 → 3 |
| MatMul/Batch/Staging | 2.653500 [2.642000–2.674000]; 0.46% | 2.654500 [2.649000–2.713000]; 0.75% | 1072 → 1072 | 4 → 4 |

Public Add/10 improves in 8/8 paired rounds; median paired saving 0.010150 µs (not an ONNX saving).

Add/10 before samples (µs, round order): 0.183700, 0.179100, 0.179500, 0.179400, 0.183400, 0.177800, 0.179300, 0.177400.
Add/10 after samples (µs, round order): 0.182600, 0.171200, 0.169400, 0.169100, 0.168400, 0.168300, 0.168200, 0.167200.


### Regression results and limitation

Public Add/10 improves in every paired round in both modes; the default median falls
5.09% and purego 5.91%. All nonempty equal-shape Add/Sub/Mul sizes remove exactly one
payload allocation. Empty results remain allocation-free. MatMul allocations and medians
remain essentially unchanged (MNIST complete +0.21% default / +0.21% purego). Both
channel Adds retain zero B/op and zero allocations; their ranges overlap before/after.

**An extra purego broadcast Batch/Forward probe is slower in these test binaries.** The
broad series rises from 0.829700 to 0.893850 µs (+7.73%). Two additional isolated,
balanced eight-round comparisons reproduce the difference:

| Purego Batch/Forward series | Before median [range] µs/op; CV | After median [range] µs/op; CV | B/op; allocs/op |
| --- | ---: | ---: | ---: |
| Isolated repeat | 0.820950 [0.782000–0.853800]; 2.28% | 0.883650 [0.879900–0.897600]; 0.72% | 0; 0 |
| Identical correctness-test sources | 0.785900 [0.774800–0.845400]; 2.64% | 0.839800 [0.827000–0.862000]; 1.24% | 0; 0 |

The matched baseline uses the original arithmetic source through a Go overlay while
compiling the exact same correctness tests as the candidate. `nm` confirms the same
production addresses as the original baseline. Generated purego
`broadcastAddFloat32Into` has 252 instructions in both versions, with only four
data-relocation operand differences; its function entry shifts from `0x100180440` to
`0x100180420` because the adapter shrinks by 32 text bytes. Its arithmetic and branches
are unchanged. This is consistent with a code-layout effect, but causation is not
established. The repeatable ~0.054–0.063 µs increase is retained as a review limitation,
not discarded as random noise. There is no blanket claim that every supported path
avoids regression. No extra production changes were added to tune this unrelated
broadcast loop or its alignment.

The purego adapter scope also has one high baseline sample (0.125300 µs, CV 7.58%).
Purego dot-product diagnostic scopes include high samples with CV up to 49.79%, so their
latency differences do not establish a kernel change. All samples remain in the table.
CPU scheduling/frequency and binary layout are not controlled by affinity. Initial
probes, follow-ups, and the main balanced series are kept separate. These backend
results establish neither ONNX whole-Run speedup nor retained-memory reduction.

### Allocation and copying ledger

Separate 10,000-call profiles use `-memprofilerate=1`; their instrumented timings
are excluded from latency results. Filter stacks through `matrixBinaryInto` or
`MatMulInto` to exclude benchmark setup (the MatMul fixture itself constructs
one extra plan outside timing). Before Add: `FromSlice` 40,000 objects / 1,920,000 B,
`New` 20,000 / 960,000 B, `Values` 10,000 / 480,000 B. After Add retains the first
two sites and eliminates the `Values` site: 60,000 objects / 2,880,000 B.
Before/after MatMul retains 20,000 plan/shape objects / 960,000 B and 10,000
staging objects / 480,000 B through the complete-call stack.

Adapter disassembly switches the `Values` call and extra `runtime.memmove` to
`ValuesInto`; constructor and operation calls remain. Runtime-sized output
staging still allocates even if compiler escape output calls it non-escaping.

For ten floats, source-derived copying falls from 160 to 120 logical bytes
(320 → 240 B copy read/write traffic): two 40-B input copies and one 40-B
output-to-tensor copy remain. This excludes zeroing, metadata and useful
arithmetic traffic; it is not measured DRAM traffic. The isolated copy benchmark
cannot be subtracted from public-call latency as an independent saving.

## Correctness and build validation

Focused tests compare Add/Sub/Mul output bits to their existing matrix kernels
at lengths 1, 10, 15, 16, 17 and 4096, plus scalar, zero dimensions and rank nine.
They include signed zeros, infinities, a NaN payload, a subnormal and maximum
finite float; verify exact read-only input bits, prefilled destinations and
unchanged destination shape/backing storage; repeat calls with changed current
weights and repeated inputs; verify output lifetime after dropping sources and
GC; and exercise concurrent shared readers with private outputs. An injected
arithmetic error after writing the private output verifies wrapped error and
unchanged tensor destination bits.

Existing tensor suites additionally cover numerical references, invalid
arguments and storage, dimension overflow, exact/shallow/empty aliases,
bidirectional/arbitrary-rank broadcasting, stride carries, the two channel Adds,
MatMul vector promotion/dot products, batch broadcasting, zero contractions and
empty outputs, shape mismatches, dynamic weights, concurrency and Gemm variants.
Matrix tests retain their public copy, row-view, allocation and device contracts.
No numerical tolerance was changed.

All **29 correctness/build checks passed** (fresh test execution with `-count=1`).

- `gofmt` on changed Go files; repository `go fmt ./...` (no formatting changes); `git diff --check`.
- `go vet ./...`, `go mod tidy -diff` (no diff), `go mod verify`, `go build ./...`.
- `go test ./...`, `CGO_ENABLED=0 go test ./...`, `CGO_ENABLED=0 go test -tags=purego ./...`.
- `go test -race ./...`, `go test -race -tags=purego ./...`.
- `CGO_ENABLED=1 go test -tags=metal ./...`; verbose matrix/internal-device Metal tests passed on this Apple M3 with no skips, including existing device fault/resource tests.
- The test-only second-batch failure overlay passed under CGO-disabled default/purego and race default/purego (four separate checks).

| CGO-disabled cross-build target | Default | purego | Execution |
| --- | --- | --- | --- |
| darwin/arm64 | pass | pass | native suites above |
| darwin/amd64 | pass | pass | compilation only |
| linux/arm64 | pass | pass | compilation only |
| linux/amd64 | pass | pass | compilation only |
| windows/amd64 | pass | pass | compilation only |
| linux/386 | pass | pass | compilation only |

Each cross-build runs `CGO_ENABLED=0 GOOS=<os> GOARCH=<arch> go build ./...`, repeated
with `-tags=purego`. No nonnative execution or cross-platform Metal runtime coverage is
claimed. The repository contains no checked-in CI workflow definitions beyond its PR
template; the listed checks include the README Metal verification. Tensor-level
later-batch GPU fault injection and ONNX integration/Go–ORT validation remain unexecuted
as described above.

One initial baseline test-binary invocation from the repository root failed to locate
package-relative fixtures. It was corrected by running `go test ./tensor` in the normal
Go harness before production edits; that suite and all fresh final suites passed. The
failed harness log is preserved separately and is not a numerical failure or latency
sample.

## Memory and lifetime accounting

No added or removed persistent state: **0 retained bytes, 0 retained objects,
0 retained payload bytes** by design. Inputs, destination and constructor
behavior are unchanged; the eliminated allocation was invocation-lived. This
does not establish a reduction in retained heap, RSS or model/session memory.
Added constructor/initialization work and allocations: **zero**; there is no
new constructor, cache, setup step, warmup or amortization requirement. No new
device resources, asynchronous work, lifetime obligations or ownership rules.

Equal-shape nonempty Add/Sub/Mul logically need three output-sized scratch
payload buffers rather than four (two copied inputs, one matrix output): 120 B
rather than 160 B for `[1,10]`. Matrix metadata is still allocated. General
high-rank broadcast metadata costs are unchanged. Host MatMul still requires
one complete output-sized staged buffer, 40 B here, and none for empty
contractions. Metal product allocation and scratch requirements are unchanged.

## Later ONNX integration

After maintainer review, merge and a separate real backend publication, a
**dependency-only update to that published version** is sufficient. No API use
changes in `numerical_call.go`, prepared descriptors, or product adapters are
needed. The next ONNX agent must use a real published version with its checksum;
do not use local replacements or an unreleased checkout.

Preserve actual per-call concrete shapes/current values, fixed prepared-signature
guards, symbolic/dynamic fallbacks, binding and produced-value checks, input
overrides, destination disjointness and invocation ownership. Preserve every
existing context check around execution/copies, arguments/nodes and publication,
failed-Run recovery and synchronous completion. This backend API adds no context
argument and does not poll or own cancellation.

The existing four-output-buffer equal-shape Add workspace charge is still sound
and may remain conservative. A separately validated integration could reduce
that charge to three payload buffers for builds using this adapter, preserving
all empty-result and high-rank accounting. Do not relax quotas before checking
the released source/build path. **Do not reduce host MatMul staging charges in
`numerical_product_host.go` or Metal product workspace accounting** on this
change; neither product implementation was optimized. There is no persistent
payload or initialization charge to add.

Transfer validation for that later ONNX agent:

- [ ] Measure fresh equal Add and MatMul adapters on real intermediates, including
  complete component µs/op, B/op and allocs/op; retain repeated raw samples.
- [ ] Measure complete ordinary Run B/op and allocs/op. The backend predicts only
  one removed 48-B allocator-class object at the `[1,10]` Add, conditionally
  92,826 B / 72 objects from the supplied historical 92,874 B / 73 baseline.
  This is **not measured ONNX behavior** or a whole-Run latency claim.
- [ ] Repeat unchanged fixed-policy numerical checks and Go/ORT timings; preserve
  tolerances and distinguish scopes, toolchains and sampling from historical runs.
- [ ] Check output and initializer bits, input immutability, independently owned
  outputs, output survival across Runs and model/session closure, shared readers,
  mutable/current overrides, dynamic shapes and destination alias rejection.
- [ ] Exercise cancellation/deadlines, failed-Run recovery, Busy/Close lifecycle,
  load/live/workspace quotas and empty-result accounting before any charge change.
- [ ] Run races, purego and CGO-disabled suites, purego+races, module verification,
  supported cross-builds and available Metal validation; report unavailable paths.

This change was not released or tagged, and no ONNX whole-Run, fixed-policy,
initialization or retained-memory improvement was measured.
