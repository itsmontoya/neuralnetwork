# Tensor arithmetic matrix adapters (Pass 16)

The equal-shape Add/Sub/Mul adapter now uses the existing
`Matrix.ValuesInto` with the captured destination slice instead of allocating `Matrix.Values()`
and copying its result a second time. For `[1,10]` Add, this removes **48 B and
one allocation per call: 336 B / 7 objects → 288 B / 6 objects**. The input
matrix copies and private output matrix remain. MatMul is unchanged and retains
its complete output staging and planning allocations.

The adapter captures the three invocation-local slice headers before creating
the matrices. These headers reference existing tensor storage for this call;
they add no payload buffers or retained aliases. The final measurements below
use this form of the adapter.

## Scope and contract

Base: `740a244bf56214bb3443a69287a2d800674620e4` (fetched `origin/main`, also
the v0.10.9 merge). Branch: `optimization-pass-16-matrix-adapters`.
The default branch was checked against the remote before branching; it had no
subsequent fix for this mechanism. The initial worktree was clean.

Commits `2065cd6`, `fa87db8` and `718560b` establish the tests, initial adapter
change and initial report. The follow-up records the final slice-header form
and fresh measurements/validation. The complete commit list is in
[PR #128](https://github.com/itsmontoya/neuralnetwork/pull/128).
No release, tag, or merge is performed.

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
or payload is cached or retained across calls, and no aliases are exposed.
Arbitrary ranks, scalars, empty outputs, broadcast fallbacks and
architecture-selected matrix arithmetic remain on their existing paths,
including purego and Metal-tagged builds.

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

Final comparison measured 2026-11-13, following exploratory runs on 2026-11-12:
Apple M3, Mac15,13, 8 cores, 24 GiB; macOS 26.5.2
(25F84), Darwin 25.5.0; Go 1.27.2, darwin/arm64. Module declares Go 1.26.1;
dependencies remain simd v1.3.0 and x/sys v0.45.0.

Both binaries use the same benchmark and correctness-test source, toolchain and
host. A Go overlay supplies the original base's `tensor/arithmetic.go` for the
baseline; all other production source is unchanged from that base. Build policy:
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
`.benchmarks/matrix-adapters/20261112-Or0as6/20261113-final/` (gitignored).
There, `measure.py` reproduces the balanced series and `summary.json` retains
every sample and derived statistic. `validate.py` records individual check
commands, environment overrides, logs and exit statuses in `validation.json`.
The parent run directory preserves the original baseline, source overlays,
fault-injection generator and rejected adapter experiments; its `final/`
subdirectory is an earlier return-form experiment, not the final comparison.

### default build

Each timing cell is **median [minimum–maximum] µs/op; CV%**. B/op and objects are
constant across the eight samples unless shown otherwise.

| Scope | Before µs/op; CV | After µs/op; CV | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| Public broadcast Add/MNISTFirst/Forward | 1.893000 [1.860000–2.137000]; 4.69% | 1.883500 [1.860000–2.116000]; 4.11% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTFirst/Reverse | 1.898500 [1.860000–2.042000]; 3.41% | 1.890000 [1.874000–1.927000]; 0.94% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTSecond/Forward | 1.158500 [1.146000–1.194000]; 1.30% | 1.159500 [1.139000–1.177000]; 1.07% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTSecond/Reverse | 1.165500 [1.147000–1.258000]; 2.93% | 1.162500 [1.146000–1.174000]; 0.76% | 0 → 0 | 0 → 0 |
| Public broadcast Add/Batch/Forward | 0.808000 [0.768600–0.852300]; 3.01% | 0.798100 [0.765400–0.896400]; 4.60% | 0 → 0 | 0 → 0 |
| Public broadcast Add/Batch/Reverse | 0.869650 [0.855700–0.904000]; 1.53% | 0.873050 [0.862900–0.930100]; 2.28% | 0 → 0 | 0 → 0 |
| Public Add/0 | 0.053330 [0.050790–0.054190]; 1.78% | 0.053315 [0.051800–0.054340]; 1.53% | 0 → 0 | 0 → 0 |
| Public Add/1 | 0.163800 [0.155700–0.165200]; 1.78% | 0.161600 [0.157200–0.163300]; 1.29% | 160 → 156 | 7 → 6 |
| Public Add/10 | 0.178700 [0.169800–0.180300]; 1.77% | 0.170000 [0.168700–0.180900]; 2.22% | 336 → 288 | 7 → 6 |
| Public Add/15 | 0.182000 [0.173600–0.183800]; 1.65% | 0.173350 [0.170100–0.181900]; 1.94% | 400 → 336 | 7 → 6 |
| Public Add/16 | 0.182050 [0.173600–0.188200]; 2.08% | 0.173300 [0.169500–0.175300]; 0.96% | 400 → 336 | 7 → 6 |
| Public Add/17 | 0.189200 [0.180800–0.196000]; 2.19% | 0.177350 [0.176600–0.190100]; 2.36% | 464 → 384 | 7 → 6 |
| Public Add/4096 | 4.644000 [4.568000–4.785000]; 1.65% | 3.609000 [3.371000–3.664000]; 2.59% | 65680 → 49296 | 7 → 6 |
| Public Add/65536 | 62.571000 [62.071000–63.088000]; 0.51% | 50.183500 [49.400000–51.064000]; 1.00% | 1048720 → 786576 | 7 → 6 |
| Public Sub/0 | 0.053465 [0.050790–0.054510]; 1.90% | 0.053505 [0.052290–0.054080]; 1.16% | 0 → 0 | 0 → 0 |
| Public Sub/1 | 0.164550 [0.155300–0.181300]; 4.20% | 0.160700 [0.157600–0.161900]; 0.82% | 160 → 156 | 7 → 6 |
| Public Sub/10 | 0.178500 [0.170500–0.182200]; 1.87% | 0.169350 [0.165800–0.185700]; 3.39% | 336 → 288 | 7 → 6 |
| Public Sub/15 | 0.182550 [0.174000–0.185300]; 1.81% | 0.171950 [0.167000–0.173400]; 1.11% | 400 → 336 | 7 → 6 |
| Public Sub/16 | 0.181150 [0.172400–0.231900]; 9.29% | 0.172100 [0.168500–0.195900]; 4.71% | 400 → 336 | 7 → 6 |
| Public Sub/17 | 0.188650 [0.183900–0.191200]; 1.32% | 0.176500 [0.174200–0.178300]; 0.71% | 464 → 384 | 7 → 6 |
| Public Sub/4096 | 4.696500 [4.481000–4.851000]; 2.19% | 3.585000 [3.544000–3.636000]; 0.87% | 65680 → 49296 | 7 → 6 |
| Public Sub/65536 | 62.448500 [60.687000–64.469000]; 1.77% | 50.241500 [48.658000–50.765000]; 1.23% | 1048720 → 786576 | 7 → 6 |
| Public Mul/0 | 0.053345 [0.050730–0.054170]; 1.82% | 0.053465 [0.051810–0.053930]; 1.38% | 0 → 0 | 0 → 0 |
| Public Mul/1 | 0.165450 [0.156700–0.167300]; 1.88% | 0.161750 [0.159000–0.163200]; 0.83% | 160 → 156 | 7 → 6 |
| Public Mul/10 | 0.177950 [0.176900–0.185400]; 1.42% | 0.168550 [0.167500–0.170700]; 0.68% | 336 → 288 | 7 → 6 |
| Public Mul/15 | 0.181150 [0.175500–0.183700]; 1.31% | 0.172100 [0.169500–0.174200]; 0.88% | 400 → 336 | 7 → 6 |
| Public Mul/16 | 0.180250 [0.175000–0.183200]; 1.30% | 0.170300 [0.167900–0.171400]; 0.75% | 400 → 336 | 7 → 6 |
| Public Mul/17 | 0.187700 [0.178900–0.195800]; 2.33% | 0.175850 [0.174500–0.178900]; 0.83% | 464 → 384 | 7 → 6 |
| Public Mul/4096 | 4.761500 [4.508000–4.878000]; 2.64% | 3.605000 [3.515000–3.820000]; 2.31% | 65680 → 49296 | 7 → 6 |
| Public Mul/65536 | 62.857500 [61.528000–63.223000]; 0.86% | 50.364000 [49.300000–53.211000]; 2.18% | 1048720 → 786576 | 7 → 6 |
| Matrix scope/Adapter | 0.102000 [0.098940–0.110400]; 3.30% | 0.093320 [0.092640–0.100800]; 2.68% | 336 → 288 | 7 → 6 |
| Matrix scope/FromSlice | 0.023360 [0.022710–0.023870]; 1.65% | 0.023820 [0.023710–0.024270]; 0.81% | 96 → 96 | 2 → 2 |
| Matrix scope/New | 0.021830 [0.021470–0.022580]; 1.96% | 0.022465 [0.022300–0.022800]; 0.62% | 96 → 96 | 2 → 2 |
| Matrix scope/MatrixAdd | 0.019455 [0.018850–0.021140]; 3.15% | 0.019435 [0.019290–0.019780]; 0.75% | 0 → 0 | 0 → 0 |
| Matrix scope/Values | 0.015850 [0.015410–0.016090]; 1.16% | 0.015815 [0.015440–0.015970]; 0.97% | 48 → 48 | 1 → 1 |
| Matrix scope/ValuesInto | 0.005055 [0.004937–0.005505]; 3.17% | 0.005034 [0.004998–0.005061]; 0.37% | 0 → 0 | 0 → 0 |
| Matrix scope/Copy | 0.001862 [0.001809–0.001874]; 1.01% | 0.001861 [0.001845–0.001867]; 0.35% | 0 → 0 | 0 → 0 |
| MatMul/MNIST/Complete | 1.676000 [1.603000–1.686000]; 1.60% | 1.677000 [1.621000–1.701000]; 1.63% | 144 → 144 | 3 → 3 |
| MatMul/MNIST/Plan | 0.061020 [0.058730–0.062070]; 1.59% | 0.061175 [0.060270–0.067380]; 3.47% | 96 → 96 | 2 → 2 |
| MatMul/MNIST/Staging | 1.562500 [1.477000–1.589000]; 2.17% | 1.559500 [1.502000–1.669000]; 3.20% | 48 → 48 | 1 → 1 |
| MatMul/MNIST/Kernel | 1.531000 [1.494000–1.570000]; 1.45% | 1.544000 [1.505000–1.578000]; 1.60% | 0 → 0 | 0 → 0 |
| MatMul/Dot/Complete | 0.402350 [0.400200–0.811500]; 29.72% | 0.407950 [0.392800–0.777500]; 32.32% | 96 → 96 | 2 → 2 |
| MatMul/Dot/Plan | 0.051225 [0.049860–0.051570]; 0.98% | 0.051470 [0.050140–0.054280]; 2.23% | 96 → 96 | 2 → 2 |
| MatMul/Dot/Staging | 0.306500 [0.300900–0.809600]; 44.42% | 0.316950 [0.299900–0.663100]; 35.83% | 0 → 0 | 0 → 0 |
| MatMul/Dot/Kernel | 0.241750 [0.233900–0.390300]; 18.99% | 0.244600 [0.234400–0.619600]; 43.59% | 0 → 0 | 0 → 0 |
| MatMul/Batch/Complete | 2.724000 [2.658000–2.744000]; 0.98% | 2.721500 [2.635000–2.796000]; 1.72% | 1200 → 1200 | 7 → 7 |
| MatMul/Batch/Plan | 0.083820 [0.081380–0.085770]; 1.53% | 0.083880 [0.082390–0.085020]; 1.12% | 128 → 128 | 3 → 3 |
| MatMul/Batch/Staging | 2.579500 [2.491000–2.954000]; 5.06% | 2.578000 [2.504000–2.631000]; 1.64% | 1072 → 1072 | 4 → 4 |

Public Add/10 improves in 7/8 paired rounds; median paired saving 0.008700 µs (not an ONNX saving).

Add/10 before samples (µs, round order): 0.169800, 0.178200, 0.177400, 0.179200, 0.178400, 0.180000, 0.179000, 0.180300.
Add/10 after samples (µs, round order): 0.180900, 0.172900, 0.171900, 0.169000, 0.170100, 0.168700, 0.169900, 0.168900.

### purego build

Each timing cell is **median [minimum–maximum] µs/op; CV%**. B/op and objects are
constant across the eight samples unless shown otherwise.

| Scope | Before µs/op; CV | After µs/op; CV | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| Public broadcast Add/MNISTFirst/Forward | 1.955500 [1.928000–2.130000]; 3.09% | 1.936500 [1.912000–2.172000]; 4.03% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTFirst/Reverse | 1.934500 [1.879000–1.955000]; 1.17% | 1.933500 [1.892000–1.968000]; 1.09% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTSecond/Forward | 1.195000 [1.164000–1.383000]; 5.34% | 1.194500 [1.177000–1.224000]; 1.23% | 0 → 0 | 0 → 0 |
| Public broadcast Add/MNISTSecond/Reverse | 1.201500 [1.184000–1.265000]; 1.94% | 1.194500 [1.185000–1.297000]; 2.90% | 0 → 0 | 0 → 0 |
| Public broadcast Add/Batch/Forward | 0.821300 [0.794500–0.879900]; 3.20% | 0.821300 [0.806500–0.887400]; 2.83% | 0 → 0 | 0 → 0 |
| Public broadcast Add/Batch/Reverse | 0.821400 [0.805900–0.831300]; 0.85% | 0.827400 [0.816200–0.864800]; 1.72% | 0 → 0 | 0 → 0 |
| Public Add/0 | 0.054165 [0.053420–0.054690]; 0.75% | 0.053955 [0.053540–0.054150]; 0.41% | 0 → 0 | 0 → 0 |
| Public Add/1 | 0.163850 [0.162900–0.165600]; 0.53% | 0.161350 [0.159300–0.163500]; 0.71% | 160 → 156 | 7 → 6 |
| Public Add/10 | 0.178350 [0.177600–0.179500]; 0.31% | 0.170650 [0.168100–0.203400]; 6.28% | 336 → 288 | 7 → 6 |
| Public Add/15 | 0.183000 [0.181900–0.183900]; 0.36% | 0.173700 [0.171900–0.177300]; 0.94% | 400 → 336 | 7 → 6 |
| Public Add/16 | 0.182650 [0.181900–0.186500]; 0.78% | 0.173850 [0.171400–0.176000]; 0.75% | 400 → 336 | 7 → 6 |
| Public Add/17 | 0.189050 [0.188200–0.190600]; 0.35% | 0.178400 [0.177500–0.179500]; 0.35% | 464 → 384 | 7 → 6 |
| Public Add/4096 | 5.773500 [5.679000–6.545000]; 4.50% | 4.671000 [4.632000–4.705000]; 0.57% | 65680 → 49296 | 7 → 6 |
| Public Add/65536 | 81.980500 [80.674000–83.967000]; 1.37% | 69.470500 [68.760000–71.255000]; 1.14% | 1048720 → 786576 | 7 → 6 |
| Public Sub/0 | 0.053790 [0.053540–0.054380]; 0.47% | 0.054050 [0.053660–0.056020]; 1.28% | 0 → 0 | 0 → 0 |
| Public Sub/1 | 0.163400 [0.163100–0.165600]; 0.60% | 0.161400 [0.160000–0.175200]; 3.21% | 160 → 156 | 7 → 6 |
| Public Sub/10 | 0.178450 [0.177800–0.181300]; 0.70% | 0.169750 [0.168700–0.173200]; 0.80% | 336 → 288 | 7 → 6 |
| Public Sub/15 | 0.182850 [0.181400–0.185400]; 0.76% | 0.174300 [0.171900–0.177300]; 0.90% | 400 → 336 | 7 → 6 |
| Public Sub/16 | 0.183000 [0.181800–0.185200]; 0.56% | 0.174550 [0.172000–0.188400]; 2.73% | 400 → 336 | 7 → 6 |
| Public Sub/17 | 0.190300 [0.188600–0.191600]; 0.43% | 0.178250 [0.176900–0.180100]; 0.60% | 464 → 384 | 7 → 6 |
| Public Sub/4096 | 5.808500 [5.775000–5.869000]; 0.56% | 4.673000 [4.631000–4.748000]; 0.83% | 65680 → 49296 | 7 → 6 |
| Public Sub/65536 | 83.787000 [80.400000–84.509000]; 1.73% | 69.610000 [69.314000–69.736000]; 0.20% | 1048720 → 786576 | 7 → 6 |
| Public Mul/0 | 0.054035 [0.053650–0.060000]; 3.68% | 0.053765 [0.053680–0.054400]; 0.48% | 0 → 0 | 0 → 0 |
| Public Mul/1 | 0.163850 [0.163300–0.170300]; 1.37% | 0.161050 [0.160000–0.163800]; 0.72% | 160 → 156 | 7 → 6 |
| Public Mul/10 | 0.178750 [0.178500–0.188800]; 1.83% | 0.169600 [0.168300–0.170600]; 0.40% | 336 → 288 | 7 → 6 |
| Public Mul/15 | 0.182950 [0.181600–0.185300]; 0.65% | 0.173550 [0.172600–0.174800]; 0.45% | 400 → 336 | 7 → 6 |
| Public Mul/16 | 0.183350 [0.182600–0.187500]; 0.80% | 0.174650 [0.172500–0.176400]; 0.86% | 400 → 336 | 7 → 6 |
| Public Mul/17 | 0.189450 [0.188700–0.193000]; 0.83% | 0.178700 [0.176800–0.180100]; 0.55% | 464 → 384 | 7 → 6 |
| Public Mul/4096 | 5.819000 [5.792000–5.877000]; 0.48% | 4.697500 [4.626000–4.759000]; 0.97% | 65680 → 49296 | 7 → 6 |
| Public Mul/65536 | 83.882500 [81.961000–84.824000]; 1.26% | 69.743500 [68.436000–70.688000]; 0.94% | 1048720 → 786576 | 7 → 6 |
| Matrix scope/Adapter | 0.101700 [0.101300–0.104500]; 0.96% | 0.092365 [0.091980–0.092950]; 0.33% | 336 → 288 | 7 → 6 |
| Matrix scope/FromSlice | 0.023860 [0.023780–0.025970]; 2.91% | 0.023860 [0.023770–0.023910]; 0.24% | 96 → 96 | 2 → 2 |
| Matrix scope/New | 0.022490 [0.022360–0.022650]; 0.38% | 0.022480 [0.022400–0.022820]; 0.65% | 96 → 96 | 2 → 2 |
| Matrix scope/MatrixAdd | 0.018430 [0.018400–0.018810]; 0.71% | 0.018515 [0.018370–0.021910]; 5.99% | 0 → 0 | 0 → 0 |
| Matrix scope/Values | 0.016040 [0.015970–0.016230]; 0.50% | 0.016060 [0.015950–0.016580]; 1.31% | 48 → 48 | 1 → 1 |
| Matrix scope/ValuesInto | 0.005050 [0.005037–0.005100]; 0.39% | 0.005047 [0.005034–0.005711]; 4.26% | 0 → 0 | 0 → 0 |
| Matrix scope/Copy | 0.001868 [0.001857–0.001873]; 0.25% | 0.001867 [0.001854–0.001873]; 0.29% | 0 → 0 | 0 → 0 |
| MatMul/MNIST/Complete | 1.695000 [1.678000–1.701000]; 0.44% | 1.684000 [1.678000–1.785000]; 2.00% | 144 → 144 | 3 → 3 |
| MatMul/MNIST/Plan | 0.061185 [0.060850–0.067030]; 3.15% | 0.061265 [0.060890–0.065710]; 2.41% | 96 → 96 | 2 → 2 |
| MatMul/MNIST/Staging | 1.573000 [1.570000–1.586000]; 0.31% | 1.572000 [1.568000–1.609000]; 0.82% | 48 → 48 | 1 → 1 |
| MatMul/MNIST/Kernel | 1.563500 [1.555000–1.681000]; 2.51% | 1.556500 [1.550000–1.652000]; 2.01% | 0 → 0 | 0 → 0 |
| MatMul/Dot/Complete | 0.409900 [0.403900–0.414200]; 0.76% | 0.409550 [0.403400–0.416500]; 0.92% | 96 → 96 | 2 → 2 |
| MatMul/Dot/Plan | 0.051080 [0.050790–0.051580]; 0.49% | 0.051650 [0.050890–0.058880]; 4.81% | 96 → 96 | 2 → 2 |
| MatMul/Dot/Staging | 0.307450 [0.306100–0.311900]; 0.63% | 0.310350 [0.307100–0.311100]; 0.40% | 0 → 0 | 0 → 0 |
| MatMul/Dot/Kernel | 0.245350 [0.243600–0.280800]; 5.52% | 0.245000 [0.242700–0.285600]; 5.45% | 0 → 0 | 0 → 0 |
| MatMul/Batch/Complete | 2.778000 [2.757000–2.798000]; 0.51% | 2.774500 [2.762000–2.798000]; 0.44% | 1200 → 1200 | 7 → 7 |
| MatMul/Batch/Plan | 0.085980 [0.085530–0.087170]; 0.68% | 0.086530 [0.085420–0.087460]; 0.73% | 128 → 128 | 3 → 3 |
| MatMul/Batch/Staging | 2.628500 [2.613000–2.649000]; 0.42% | 2.635500 [2.617000–2.658000]; 0.48% | 1072 → 1072 | 4 → 4 |

Public Add/10 improves in 7/8 paired rounds; median paired saving 0.007200 µs (not an ONNX saving).

Add/10 before samples (µs, round order): 0.177900, 0.178500, 0.178600, 0.179500, 0.178400, 0.178300, 0.177600, 0.177800.
Add/10 after samples (µs, round order): 0.170900, 0.168100, 0.173300, 0.169300, 0.174400, 0.168900, 0.203400, 0.170400.

### Regression results and remaining limits

Public Add/10 improves in 7/8 paired rounds in each mode. Its median falls
4.87% in default and 4.32% in purego. Every nonempty equal-shape Add/Sub/Mul
fixture removes exactly one payload allocation; empty results remain
allocation-free. Both channel Adds remain at zero B/op and zero allocations.
MatMul planning/staging allocations are unchanged. Its MNIST complete median
is 1.676000 → 1.677000 µs in default and 1.695000 → 1.684000 µs in purego.

No material supported-path regression was observed in the final fixtures.
The largest complete-public-call median increase is 1.39% (default dot
product); the largest increase among all diagnostic scopes is 3.41% (default
dot staging), with overlapping ranges. These are observations for this host,
toolchain, fixtures and binaries, not a guarantee about all downstream layouts.

All samples are retained. Purego Add/10 has one high after sample of 0.203400
µs (CV 6.28%). Dot-product scopes have much wider variation, up to 44.42% CV;
their small median differences do not establish a kernel change. Scheduling,
CPU frequency and core placement were not pinned. The complete MNIST MatMul
and private adapter scopes have their own ranges/CVs in the tables above.

Exploratory equivalent adapter forms shifted subsequent function addresses and
produced repeatable 7–11% increases in an untouched broadcast probe; those
forms were rejected. Capturing the three call-local slice headers keeps the
adapter's generated size at 816 text bytes, equal to baseline, and preserves
the broadcast function's entry in both final native test binaries: default
`0x100182eb0`, purego `0x100180440`. No padding, new public API or broadcast
kernel change was added. Layout is consistent with the earlier differences,
but its causal role was not proved. The parent directory preserves every
exploratory sample and overlay, separate from the final series. Downstream
ONNX timings must still be measured independently.

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
In these Go 1.27.2 binaries, the retained MatMul staging uses a 32-byte stack
buffer for at most eight floats; larger outputs call `runtime.makeslice`.
This explains zero heap allocations in the scalar dot-product staging scope,
while the ten-float fixture still allocates 48 B. Zero B/op does not mean that
staging or its final copy was removed.

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
The checks were repeated on the final slice-header implementation on 2026-11-13;
all final logs and exit statuses are in `20261113-final/validation.json`.

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
Added constructor/model/session initialization work and allocations: **zero**;
there is no new constructor, cache, setup step or explicit warmup/amortization
phase. Existing matrix constructor costs are measured separately above. No new
device resources, asynchronous work, lifetime obligations or ownership rules.

The invocation-local slice headers increase the compiled ARM64 adapter frame
from **160 to 208 bytes (+48 stack bytes)** in both CGO-disabled ARM64 latency
builds, measured from the prologues. This is distinct from the removed 48-B heap
allocation and does not add a heap object. Stack size depends on compiler/build/architecture;
automatic runtime stack growth was not separately measured. No reduction in
total peak memory or RSS is claimed.

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
all empty-result and high-rank accounting. This would describe numeric payload
only: account for the additional native stack frame separately wherever the
quota definition includes stack metadata. Do not relax quotas before checking
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
