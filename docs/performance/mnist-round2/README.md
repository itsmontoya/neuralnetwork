# Convolution and broadcast optimization (2026-11-09)

The two MNIST optimizations prototyped in the ONNX repository are integrated
without changing public APIs or numeric storage ownership. On this Apple M3,
the median of five default Go benchmark samples fell from 703.94 to 436.32 µs
(38.0%). The separate 19-case ORT comparison measured 693.92 to 436.71 µs for
Go, with all Go outputs bit-identical to the baseline and the archived prototype.

## Implementation and regression coverage

`tensor/conv2d.go` computes four adjacent output columns together when width
dilation is one and all four horizontal windows fit within the input. Four
scalar accumulators share weights and geometry while retaining each output's
input-channel, kernel-row, kernel-column order. Bias is added after accumulation.
Horizontal padding, tails and width dilation use the existing scalar kernel;
height padding still skips invalid rows without evaluating padded products.

`tensor/arithmetic.go` advances broadcast coordinates and source offsets from
the trailing axis, resetting offsets on carries. Broadcast axes have zero
source stride. Scalar results require no axis advances; empty results return
after validation as before. This removes division/modulo from the element loop.

Shape, type and destination alias checks still complete before any writes.
Inputs remain immutable. Convolution adds no buffers, packed weights or retained
state. Broadcast replaces output-stride metadata with equally ranked coordinate
metadata; it adds no numeric payload or caller workspace requirement. Existing
product workspaces, tensor ownership and concurrency contracts are unchanged.

The prototype's 15 convolution cases are retained. Five more scalar comparisons
cover depthwise groups, empty batches, empty output height/width, and rounded
products. Four explicit expected-output cases cover bias after cancellation,
skipping nonfinite padded weights, signed-zero products and nonfinite bias.
All 20 scalar comparisons also check input, weight and bias immutability by bits.
The rounded-product oracle uses the original scalar kernel because Go permits
fused multiplication/addition; it preserves the existing compiler behavior.
Finite values, infinities and signed zeros compare by bits; NaNs compare by class.

Eleven broadcast shapes run through all four arithmetic operations (44 cases),
including alternating axes, leading and trailing carries, singleton axes, rank
promotion, scalars and empties. Each reference decodes its flat index independently
with division/modulo, and the tests check source bits after execution. Existing
reference, invalid-argument, alias, destination-preservation and concurrency tests
remain in place.

## Sources and measurement policy

- Backend base: `743e049ca9e4877de3c33c0969b17119b6f1491d`, whose tree matches
  `v0.8.0`. The baseline benchmark binary was compiled before the changes.
- ONNX: `8823861325e3218a8b6dea78332d38c6d8ed2df5`. Its `go.mod`, `go.sum`
  and Python comparison driver have unchanged SHA-256 hashes. The dependency pin
  remains `neuralnetwork v0.8.0`.
- Prototype: the ONNX checkout's
  `docs/performance/mnist-round2/neuralnetwork.patch`, SHA-256
  `98abc44818df84a0a64ef5449d7f57c815b2aedd51aff7b6ad179e6d3e05c612`.
  It applied cleanly. Integration differences are grouped local declarations,
  explanatory comments and additional tests; the two algorithms are unchanged.
- Apple M3, macOS 26.5.2 (25F84), Go 1.27.2, darwin/arm64.
  Timings use `GOMAXPROCS=1`, `CGO_ENABLED=0` and precompiled binaries.
- A temporary Go workspace selects this checkout and the ONNX checkout.
  Broadcast-only and convolution-only binaries use Go overlays restoring the
  other production file from the base. A separate overlay applies the original
  prototype for a direct timing control. No module-cache source is edited.

Five default variants ran sequentially in rotating order for five repetitions,
500 ms per MNIST sample. Purego baseline/combined and default small-fixture
baseline/combined pairs alternated order for five repetitions; small fixtures
used 200 ms per case. Model/session setup and warmup are outside timing; each
inference returns owned outputs. Tests, builds and profiling did not run
concurrently with timings. No profiles were collected for this integration.

[results.json](results.json) records identities, hashes, settings, individual
samples, run order, comparison outputs and checks. [benchmarks.txt](benchmarks.txt)
contains raw benchmark output; [build-info.txt](build-info.txt) records every
binary's Go build metadata. Full comparison reports, including raw call latencies,
and the temporary scripts are also retained locally in the ignored
`.tmp/mnist-round2/`. The original workspace and binaries remain in
`/private/tmp/neuralnetwork-round2-1ax_txuw/` for inspection.

## Repeated benchmark results

Times are median [minimum, maximum] µs/op across five samples. Allocations are
constant within each row.

| Build / variant | MNIST µs/op | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Default baseline | 703.94 [694.55, 715.53] | 179504 | 176 |
| Default broadcast only | 687.65 [669.06, 692.04] | 179440 | 174 |
| Default convolution only | 461.66 [450.41, 468.43] | 179504 | 176 |
| Default combined | 436.32 [426.87, 441.56] | 179440 | 174 |
| Original prototype, rebuilt | 436.44 [432.01, 442.54] | 179440 | 174 |
| Purego baseline | 725.81 [721.60, 744.78] | 179504 | 176 |
| Purego combined | 442.02 [434.76, 450.60] | 179440 | 174 |

The combined default median is 1.61× faster; purego reduces latency by 39.1%.
The rebuilt prototype and integrated implementation differ by 0.03% in these
short samples, providing no evidence of a meaningful implementation difference.

| Small fixture / Warm | Baseline median ns/op | Combined median ns/op |
| --- | ---: | ---: |
| mlp_batch_1 | 3461 | 3475 |
| mlp_batch_3 | 3625 | 3617 |
| shape_batch_1 | 6495 | 6412 |
| shape_batch_3 | 6563 | 6538 |
| cnn_batch_1 | 6349 | 6317 |
| cnn_batch_3 | 7519 | 7515 |

Small-fixture allocations are unchanged. These short samples show little
movement and do not establish a speedup for models with narrow interiors.

## Repeated ORT comparison

The unmodified Python driver measures the published v0.8.0 baseline. The second
run uses a temporary driver copy with exactly two replacements: `ROOT` points
to the original ONNX checkout and its explicit `GOWORK="off"` selects the
temporary workspace. An environment-only workspace override would be discarded
by the original driver. The runner, comparison and timing logic are unchanged.

Both runs use the original upstream vector, zeros, ones and 16 uniform random
inputs with seed 0, all `[1,1,28,28]`. Every input and model hash matches the
archived prototype. Settings remain `atol=1e-4`, `rtol=1e-5`, ten warmups and
100 measured calls per case (1,900 per runtime). ORT uses the CPU provider,
sequential execution, one intra/inter-op thread and disabled graph optimization.
Python 3.9.6, NumPy 1.26.4, ONNX 1.16.2, ORT 1.19.2 and protobuf 4.25.3 match
the prototype environment.

| Variant | Go median / p95 µs | ORT median / p95 µs | Go / ORT median |
| --- | ---: | ---: | ---: |
| Published v0.8.0 | 693.92 / 770.63 | 42.42 / 45.38 | 16.36× |
| Integrated changes | 436.71 / 510.44 | 42.08 / 47.29 | 10.38× |

All 19 Go outputs match the baseline and archived prototype bit-for-bit.
Go-versus-ORT and all top-1 checks pass. Original-versus-converted ORT outputs
agree exactly; the upstream expected-output check passes. The maximum absolute
Go-versus-ORT error remains `1.9073486328125e-5`.

The repeated Go comparison median improves by 37.1%. The prototype reported
709.81 → 440.83 µs and a 16.09× → 9.86× median gap. Here, both Go and ORT ran
faster; ORT's lower median makes the resulting gap larger despite a slightly
lower optimized Go time. The independent repeated Go benchmarks support the
latency improvement. These are local measurements of one model, not guarantees
for other models or machines.

## Validation and reproduction

[validation.txt](validation.txt) records successful checks:

```sh
GOWORK=off go test ./...
GOWORK=off CGO_ENABLED=0 go test -tags=purego ./tensor
GOWORK=off go vet ./...
GOWORK=off CGO_ENABLED=1 go test -race ./...
# From the ONNX checkout, selecting the temporary workspace:
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go test -tags=purego ./...
CGO_ENABLED=0 go vet ./...
```

The new regression tests also pass with an overlay restoring the original two
kernels. Formatting and `git diff --check` pass. Validation was repeated after
the final rounded-product regression was added.

To reproduce the workspace checks and combined timings without editing ONNX's
dependency pin, create a temporary workspace from this backend checkout:

```sh
backend_root=$PWD
onnx_root=/path/to/onnx
measurement_dir=$(mktemp -d /tmp/neuralnetwork-round2.XXXXXX)
(cd "$measurement_dir" && go work init "$backend_root" "$onnx_root")
cd "$onnx_root"
GOWORK="$measurement_dir/go.work" CGO_ENABLED=0 go test ./...
GOWORK="$measurement_dir/go.work" CGO_ENABLED=0 go test -tags=purego ./...
GOWORK="$measurement_dir/go.work" GOMAXPROCS=1 CGO_ENABLED=0 go test -c -o "$measurement_dir/after.test" .
GOWORK=off GOMAXPROCS=1 CGO_ENABLED=0 go test -c -o "$measurement_dir/before.test" .
```

Alternate those binaries, five repetitions, from the ONNX root:

```sh
GOMAXPROCS=1 "$measurement_dir/before.test" -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=500ms -test.benchmem
GOMAXPROCS=1 "$measurement_dir/after.test" -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=500ms -test.benchmem
```

Prepare the optional model/Python cache using ONNX's
`tools/compare_runtime/README.md`. For the 19-case comparison, copy
`tools/compare_runtime.py` to the temporary directory, replace only `ROOT` and
the explicit `GOWORK` assignment as described above, and run it with the pinned
Python environment. Use the original driver for the baseline. Keep their
`--cache-dir` and `--report` outputs in the temporary directory. Explicit options:

```text
--warmup 10 --iterations 100 --seed 0 --random-cases 16
--atol 1e-4 --rtol 1e-5 --ort-optimizations disabled
```
