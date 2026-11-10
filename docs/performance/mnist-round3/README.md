# Conv2D channel blocking (2026-11-09)

MNIST inference improves from **444.36 to 320.22 µs**, a **27.9% latency
reduction**, across seven alternating measurements against published v0.9.0.
Allocations remain **179440 B/op, 174 allocs/op**. All 19 standard ORT cases
pass with bit-identical Go outputs; the measured Go/ORT median gap is **7.36×**.

## Change and review decisions

`tensor/conv2d.go` shares each input sample and clipped geometry across four
output filters at padding boundaries and remainder columns. Interior blocks
still call the existing four-column helper. Channel blocks stay within groups;
remaining filters retain both the spatial fast path and scalar fallback.

The experimental patch's scalar-only leftover-filter loop was replaced by a
shared traversal supporting one or four filters. An initial version used that
traversal for every shape: three alternating 80 ms samples showed +5.2% for
depthwise and +3.9% for an exact four-column output. The final dispatch keeps
the original traversal for fewer than four filters per group and for unit width
dilation with no horizontal padding and output width divisible by four.
The corresponding tuning rerun reduced the depthwise difference to +0.1%.
[Tuning logs](tuning.txt) and [results](results.json) retain both sweeps.

Each output keeps its input-channel/kernel-row/kernel-column accumulation order
and float32 expressions; bias is added last. Padded positions are skipped before
reading weights, including nonfinite weights. The public API, validation and
alias checks are unchanged. There are no new buffers, packed weights, caches,
goroutines, cgo paths or dependencies. The [generated helper assembly](four-channels-objdump.txt)
retains four scalar `FMADDS` accumulators on this machine.

## Repeated measurements

Apple M3, macOS arm64, Go 1.27.2, `GOARM64=v8.0`, `GOMAXPROCS=1`,
`CGO_ENABLED=0`. All binaries were precompiled; pairs ran sequentially with
order reversed each repetition. Profiling, tests and builds ran separately.
End-to-end calls return owned outputs; loading, session setup and warmup are
outside timing. Kernel benchmarks reuse a disjoint destination and include
validation; tensor construction is outside timing.

MNIST uses seven repetitions at 700 ms; purego uses five at 500 ms; the real
Conv nodes use five at 300 ms. Times below are **median [min, max] µs/op**.
Bytes and allocations are unchanged for every row.

| Benchmark | v0.9.0 | Change | Latency change | B/op / allocs/op |
| --- | ---: | ---: | ---: | ---: |
| Default MNIST | 444.36 [431.49, 449.05] | 320.22 [318.11, 322.24] | -27.9% | 179440 / 174 |
| Purego MNIST | 449.03 [448.17, 450.01] | 322.38 [321.65, 332.66] | -28.2% | 179440 / 174 |
| First real Conv node | 62.29 [61.86, 62.64] | 51.18 [50.94, 59.62] | -17.8% | 32 / 1 |
| Second real Conv node | 292.63 [291.87, 293.61] | 182.05 [180.71, 182.97] | -37.8% | 32 / 1 |

All seven changed default MNIST samples beat all seven baseline samples.
Node timings use actual intermediate tensors and ONNX activation correction;
they are diagnostic, not an additive latency decomposition.

The 33 kernel shapes below use five alternating 200 ms repetitions. Definitions
are in [conv2d_benchmark_test.go](../../../tensor/conv2d_benchmark_test.go).
Times are **median [min, max] ns/op**; all rows retain **32 B/op, 1 alloc/op**
from checked output-shape metadata. `Groups4/5/7/8` names count filters *per
group* and use two groups, two batches and three input channels per group.

| Shape | v0.9.0 | Change | Latency change |
| --- | ---: | ---: | ---: |
| MNISTFirst | 64167.00 [62637.00, 88848.00] | 52408.00 [51917.00, 54277.00] | -18.33% |
| MNISTSecond | 299801.00 [294580.00, 325399.00] | 184421.00 [183991.00, 186941.00] | -38.49% |
| Narrow1 | 1784.00 [1757.00, 1839.00] | 748.80 [745.00, 873.10] | -58.03% |
| Narrow3 | 5123.00 [5085.00, 5183.00] | 2011.00 [1988.00, 2046.00] | -60.75% |
| Exact4 | 2518.00 [2475.00, 2622.00] | 2501.00 [2476.00, 2550.00] | -0.68% |
| Tail1 | 4160.00 [4100.00, 4210.00] | 3105.00 [3050.00, 3721.00] | -25.36% |
| Tail2 | 5803.00 [5681.00, 5895.00] | 3694.00 [3626.00, 3788.00] | -36.34% |
| Tail3 | 7575.00 [7428.00, 8625.00] | 4376.00 [4298.00, 4827.00] | -42.23% |
| WideExact | 376144.00 [373565.00, 467086.00] | 376242.00 [372242.00, 400537.00] | +0.03% |
| WideTail | 381700.00 [379971.00, 431019.00] | 380955.00 [378610.00, 394958.00] | -0.20% |
| WidePadded | 493606.00 [489311.00, 502305.00] | 477840.00 [468447.00, 482590.00] | -3.19% |
| Filters4 | 6568.00 [6492.00, 6632.00] | 5234.00 [5143.00, 5404.00] | -20.31% |
| Filters5 | 8229.00 [8070.00, 8313.00] | 6957.00 [6790.00, 7062.00] | -15.46% |
| Filters7 | 11410.00 [11187.00, 11530.00] | 10253.00 [10079.00, 10374.00] | -10.14% |
| Filters8 | 12810.00 [12687.00, 13017.00] | 10245.00 [10118.00, 10391.00] | -20.02% |
| Groups4 | 25830.00 [25173.00, 28939.00] | 20222.00 [20070.00, 20717.00] | -21.71% |
| Groups5 | 32629.00 [31657.00, 46770.00] | 26892.00 [26654.00, 27246.00] | -17.58% |
| Groups7 | 44877.00 [44512.00, 55452.00] | 39876.00 [39576.00, 41153.00] | -11.14% |
| Groups8 | 50919.00 [50481.00, 51188.00] | 40391.00 [39732.00, 43656.00] | -20.68% |
| Depthwise | 9167.00 [9033.00, 9231.00] | 9077.00 [8987.00, 9284.00] | -0.98% |
| DepthwiseMultiplier4 | 35637.00 [35368.00, 35846.00] | 27619.00 [27507.00, 27779.00] | -22.50% |
| Fallback3 | 4870.00 [4842.00, 4897.00] | 4923.00 [4846.00, 5027.00] | +1.09% |
| Strided | 19693.00 [19591.00, 19796.00] | 13646.00 [13539.00, 13822.00] | -30.71% |
| Dilated | 49481.00 [49215.00, 49937.00] | 15905.00 [15694.00, 16157.00] | -67.86% |
| DilatedStrided | 40246.00 [39027.00, 41943.00] | 12756.00 [12618.00, 13107.00] | -68.30% |
| HeavyPadding | 8740.00 [8497.00, 9145.00] | 2634.00 [2586.00, 2674.00] | -69.86% |
| Batches | 53694.00 [50169.00, 59454.00] | 40251.00 [39982.00, 41566.00] | -25.04% |
| Pointwise | 6428.00 [6396.00, 6557.00] | 6513.00 [6330.00, 9702.00] | +1.32% |
| TinyPointwise | 168.90 [165.60, 170.50] | 150.80 [149.80, 159.30] | -10.72% |
| EmptyHeight | 961.90 [940.60, 976.10] | 823.30 [814.40, 829.00] | -14.41% |
| EmptyWidth | 827.90 [825.00, 975.30] | 366.10 [350.20, 628.10] | -55.78% |
| EmptyBatch | 113.20 [110.60, 114.80] | 114.10 [110.10, 306.30] | +0.80% |
| EmptyOutput | 119.70 [116.30, 170.60] | 119.00 [115.20, 132.40] | -0.58% |

Wide exact and wide remainder outputs are effectively neutral in this sweep;
wide padded output improves 3.2%. Reported regressions are +1.32% for Pointwise,
+1.09% for Fallback3, +0.80% for EmptyBatch and +0.03% for WideExact. Their ranges
overlap, and these cases retain the original kernel/empty-output path. This
does not establish zero dispatch overhead. The six small ONNX warm fixtures
range from -1.06% to +0.97% in median latency, with identical allocations.
All samples and ranges, including outliers, are retained in
[benchmarks.txt](benchmarks.txt) and [results.json](results.json).

## Standard Go-versus-ORT comparison

Both runs use the unchanged upstream vector, zeros, ones and 16 NumPy uniform
inputs with seed 0. Input/model hashes, `atol=1e-4`, `rtol=1e-5`, ten warmups
and 100 measured calls per case match: 1,900 calls per runtime. Python 3.9.6,
NumPy 1.26.4, ONNX 1.16.2, ORT 1.19.2 and protobuf 4.25.3 use the existing
pinned environment. ORT uses sequential CPU execution, one intra/inter-op
thread and disabled graph optimization. These runtimes are timed sequentially.

| Backend | Go median / p95 µs | ORT median / p95 µs | Median gap |
| --- | ---: | ---: | ---: |
| Published v0.9.0 | 452.77 / 566.76 | 44.62 / 47.54 | 10.15× |
| Channel blocks | 325.48 / 386.63 | 44.25 / 45.96 | 7.36× |

All 19 cases and top-1 checks pass. Go outputs are bit-identical between
backends; maximum Go-versus-ORT error remains `1.9073486328125e-5`.
Original/converted ORT outputs agree exactly, and the upstream expected-output
check passes. The 7.36× gap is measured; the earlier projected 7.5× was not used.

The original driver explicitly overrides `GOWORK=off`. The experimental run
uses a temporary driver copy with exactly two substitutions: `ROOT` points at
the sibling ONNX checkout, and that explicit `GOWORK` assignment selects the
temporary workspace. Inputs, tolerances, runner code and timing logic are
unchanged. Driver hashes and the substitutions are in [results.json](results.json).
The sibling driver and dependency pin remain unchanged.

The 131 archived round-three requests also produce bit-identical Go outputs:
upstream, zeros, ones and 128 seeded uniform probes, including [-10,10] inputs.
These supplement correctness checks; they are not ORT timing samples.
[Comparison evidence](comparison.txt) records the request hash and seed.

## Correctness and profiling

There are 432 new scalar-reference cases: 18 explicit geometries × filters
4/5/7/8 per group × one/two groups × finite/rounded/nonfinite payloads. They
cover tails, batches, padding, strides, dilation, depthwise multipliers and empty
dimensions, checking output bits and source immutability. NaNs compare by class;
finite values, infinities and signed zeros compare by bits. Five explicit
expected-output cases exercise bias after cancellation, skipping infinite padded
weights, signed zero, infinite bias and NaN bias across a four-filter block and
three leftover filters in each group. The dynamic-weight/concurrency test now
also exercises a four-filter block plus a filter tail.

[Validation logs](validation.txt) record successful full backend default/race
suites, purego tensor tests, full backend vet, convolution tests with the original
kernel restored, and full ONNX default/purego suites plus vet against the workspace.
Existing alias rejection, destination-preservation and argument tests pass.

Separate five-second CPU profiles, focused on `Session.Run` stacks with relative
percentages, attribute 75.41% to convolution before and 69.12% after. The scalar
helper accounts for 37.70% before and has no samples after on this model; the
channel helper accounts for 27.94% after. These short profiles are attribution
evidence, not wall-clock percentages or speedup measurements:
[baseline](base-cpu.txt), [changed](final-cpu.txt).
Separate allocation profiles still attribute about 71.2% of bytes to tensor
construction and 22.7% to float snapshots:
[baseline](base-alloc-space.txt), [changed](final-alloc-space.txt).
Their different iteration counts prevent comparing raw allocation totals.

## Identity and reproduction

- Backend base: `d0e5c475b750fea5c453f2ab0437f6c3e708448f`; baseline is published
  `github.com/itsmontoya/neuralnetwork v0.9.0`, checksum
  `h1:4WNDbwFxz1AVZCVDk60KqBDRtzqFzAgNa+euNttE+2M=`.
- ONNX: `b3e2cc8078ece856bfd8296e07f8b383458bfb63`, still pinned to v0.9.0.
- ONNX baseline binaries select the published module with `GOWORK=off`;
  changed binaries select this checkout via an isolated workspace. Both default
  binaries include the same diagnostic benchmark via a Go overlay. Kernel
  baseline binaries restore only `conv2d.go`, byte-identical to v0.9.0, via an
  overlay; both include the same benchmark/test sources.
- Workspace `go 1.26.1` / `toolchain go1.27.2` matches both modules and runtime
  defaults. End-to-end timings were rebuilt/repeated after aligning the directive;
  the earlier sweep is retained locally and excluded from reported results.
- [Build metadata](build-info.txt) and [results.json](results.json) record source
  hashes, module identity, settings, run order, samples and comparison outputs.
  Binaries, full per-call ORT reports, profiles and scripts remain locally in
  `/private/tmp/neuralnetwork-conv3.vEnp2v/`.

[Reproduction commands](REPRODUCE.md) cover the workspace, validation, binaries
and unchanged comparison settings. The temporary scripts retain the executed
benchmark and profiling commands.

These measurements cover one Apple M3 and a finite shape matrix. They do not
establish cross-architecture performance or a general ORT gap for other models.
