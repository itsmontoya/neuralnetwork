# ARM64 direct-convolution interior tile

This change computes four adjacent columns for four output filters together.
The ARM64 kernel shares one four-value input load across four broadcast weights
and four vector accumulators. SIMD lanes represent independent outputs. The
input-channel, kernel-row, kernel-column reduction order stays unchanged, and
bias is added after reduction.

The public API and destination validation are unchanged. Unit width stride and
unit dilation select the tile; height stride can vary. Padding rows are clipped
before reading input or weights. The Go caller bounds complete payload spans;
each assembly load stays inside a valid row. Tiles stay within a group. Edges,
spatial tails, filter tails, width strides and dilation retain the existing
helpers, including the spatial helper for leftover filters.

## Portable alternative and numerical decision

A checked Go 4×4 tile was measured separately. Its complete row slices moved
payload checks outside the reduction loop, and it reduced the two MNIST kernel
medians to 25.46 / 114.32 µs in five alternating 200 ms samples. However, an
additional exact-bit probe found different NaN payload selection when both
multiplicands contain NaNs. The compiler commuted some scalar multiplies;
reversing the source expressions did not restore the released operand priority.
The [candidate source](checked-go-candidate.go.txt) and
[diagnostic evidence](numerical-diagnostics.txt) are retained for review.

That Go tile is **not shipped**. Non-ARM64 and `purego` builds retain the original
Go traversal. A build-tagged helper declines the tile and is inlined away.
ARM64 SIMD explicitly uses the released spatial helper's weight-first operand
priority. New ARM64 tests compare output bits, including quiet/signaling NaN
payloads, subnormals, overflow/cancellation and signed zeros, against the scalar
reference across two groups, four-filter tiles and leftover filters. These
probes also pass with the released convolution restored. Existing numerical
tests and tolerances were preserved.

The change adds four explicit geometries to the existing channel-block matrix:
96 additional scalar-reference combinations of filter/group counts and
finite/rounded/nonfinite payloads. They cover the final valid payload load,
multiple row-ending tiles, height stride and entirely padded rows. Existing
spatial/filter tails, asymmetric padding, batches, groups, empty dimensions,
immutability and destination rejection coverage remains. The concurrent dynamic
weights test now includes a spatial tile and a leftover filter.

## Allocation and workspace accounting

All direct Into benchmarks retain 32 B/op and one allocation from the checked
four-element output shape. The tile adds a 64-byte **stack accumulator block**
for transferring sixteen results from assembly; no heap scratch, packing,
constant cache, tensor-sized workspace, initialization phase or global mutable
storage is introduced. Escape analysis confirms that the tile arguments do not
escape. Tensor construction and fixture generation stay outside timing. The
ONNX zero-scratch contract and owned-output behavior remain unchanged.

## Measurement method and source identity

The starting checkout is exactly `v0.10.0`, commit
`1ea5263a7e7910aafbfe91695393f8b3382c6536`. Its convolution matches the cached
published source, checksum
`h1:kT9R36MEq7+MhM8/d4XHFlmwCJx6xrtyITrIZJPp12M=`. ONNX remains pinned to that
release. The original `.gitignore` edit and untracked `neuralnetwork.patch`
were preserved. The sibling ONNX source, dependency files and comparison driver
were not changed.

Apple M3, darwin/arm64, Go 1.27.2, `GOARM64=v8.0`, `CGO_ENABLED=0`,
`GOWORK=off`, `GOMAXPROCS=1`. The original benchmark was remeasured before edits:
MNISTFirst 48.48 µs, MNISTSecond 173.30 µs, each 32 B/op and one allocation.
The required initial broad baseline is also retained in the raw logs.

Final before/after binaries use the same tests and benchmark sources. A Go
overlay restores only the released `conv2d.go` for the kernel baseline. ONNX
baseline binaries select the published release; changed binaries select this
checkout through temporary `-modfile` files. Both use the unchanged ONNX Go
version directive. The standard comparison driver is run unchanged with that
same `GOFLAGS=-modfile=...` selection and a temporary model cache.

Every pair runs sequentially, with order reversed each repetition. No tests,
builds or profiles run concurrently with benchmark measurements. MNIST kernel
cases (bias and nil bias) and warm inference use seven 700 ms samples per
binary. The 33-shape default/purego sweeps use five 200 ms samples, real Conv
nodes five 400 ms samples, and purego inference five 500 ms samples. CPU profiles
are collected separately. The original optional-bias workload is unchanged;
the nil-bias benchmark is an additional matching workload.

[Reproduction commands](REPRODUCE.md), [measurement harness](measure.py),
[build identity](identity.json), [all samples and ranges](results.json), and
[raw benchmark logs](benchmarks.txt) provide the executed settings and evidence.
Binaries, full ORT reports and exploratory artifacts remain in
`/private/tmp/neuralnetwork-conv4/`.

## Validation and limits

[Validation logs](validation.txt) cover the full library default, purego and
race suites; purego tensor race tests; vet; convolution tests with the released
kernel restored; and full ONNX default/purego suites against the local backend.
Linux arm64 (default and purego), amd64 and 386 tensor tests cross-compile.
Runtime and performance validation is on this single darwin/arm64 machine;
other architectures were not executed. Go race instrumentation does not inspect
assembly instructions; the concurrent-call test exercises separate destinations
with shared immutable inputs and parameters.

The profile comparisons normalize only to sampled stacks containing
`onnx.(*Session).Run`, using `-focus` and `-relative_percentages`. They establish
where sampled cost moved, not exact wall-time attribution. SIMD, shared input
loads, fewer geometry calculations and removal of reduction-loop bounds checks
change together; their individual contributions were not measured. Boundary
and leftover-filter kernels are intentionally unchanged.

## Final alternating measurements

Times are median [min, max] µs/op. Allocation counts are unchanged in every row.
The real-node diagnostics reuse prepared tensors and include validation/context
checks; they are not an additive end-to-end decomposition.

| Workload | v0.10.0 | Final | Change | B/op / allocs/op |
| --- | ---: | ---: | ---: | ---: |
| MNISTFirst, bias | 51.101 [50.572, 51.930] | 18.539 [18.404, 19.081] | -63.72% | 32 / 1 |
| MNISTSecond, bias | 179.986 [178.066, 181.024] | 100.149 [99.474, 101.084] | -44.36% | 32 / 1 |
| MNISTFirst, nil bias | 50.393 [49.972, 50.678] | 18.107 [17.956, 18.152] | -64.07% | 32 / 1 |
| MNISTSecond, nil bias | 179.061 [177.955, 180.042] | 99.961 [99.429, 102.246] | -44.17% | 32 / 1 |
| Warm MNIST | 319.257 [317.486, 327.679] | 205.676 [204.533, 213.008] | -35.58% | 179440 / 174 |
| Warm MNIST, purego | 320.701 [319.451, 332.473] | 321.037 [319.604, 332.757] | +0.10% | 179440 / 174 |
| First prepared real Conv | 50.901 [50.832, 50.995] | 18.367 [18.213, 19.417] | -63.92% | 32 / 1 |
| Second prepared real Conv | 182.431 [181.709, 183.825] | 101.179 [100.475, 106.702] | -44.54% | 32 / 1 |

All seven final MNISTSecond samples beat all seven baseline samples, as do all
seven warm MNIST samples. The kernel and end-to-end acceptance targets are met.
Purego keeps its original path and is effectively neutral.

The following broad cases use five samples per binary. Default times are median
[min, max] ns/op. Every default and purego case retains **32 B/op, 1 alloc/op**;
complete purego ranges and all samples are in [results.json](results.json). No
case exceeds a 3% slowdown: the largest is +1.64% default and +1.44% purego.
[Benchstat comparisons](benchstat.txt) retain the statistical output.

| Shape | Default v0.10.0 | Default final | Default change | Purego change |
| --- | ---: | ---: | ---: | ---: |
| MNISTFirst | 51409.000 [51130.000, 51799.000] | 18538.000 [18420.000, 18676.000] | -63.94% | +0.48% |
| MNISTSecond | 181824.000 [181207.000, 183709.000] | 101353.000 [100369.000, 101947.000] | -44.26% | -0.05% |
| Narrow1 | 744.700 [737.900, 751.000] | 747.600 [742.900, 752.500] | +0.39% | +0.38% |
| Narrow3 | 1984.000 [1971.000, 1990.000] | 1991.000 [1977.000, 2016.000] | +0.35% | +0.00% |
| Exact4 | 2447.000 [2427.000, 2756.000] | 593.200 [586.500, 596.200] | -75.76% | +0.33% |
| Tail1 | 3035.000 [2997.000, 3057.000] | 1236.000 [1223.000, 1252.000] | -59.28% | -0.13% |
| Tail2 | 3611.000 [3567.000, 3615.000] | 1816.000 [1806.000, 1835.000] | -49.71% | +0.39% |
| Tail3 | 4247.000 [4214.000, 4270.000] | 2466.000 [2459.000, 2472.000] | -41.94% | +0.45% |
| WideExact | 366674.000 [363594.000, 373665.000] | 57729.000 [57677.000, 57949.000] | -84.26% | -0.41% |
| WideTail | 370993.000 [367808.000, 372509.000] | 60909.000 [60741.000, 61474.000] | -83.58% | -0.18% |
| WidePadded | 461199.000 [460651.000, 475833.000] | 82451.000 [82318.000, 82653.000] | -82.12% | -1.63% |
| Filters4 | 5075.000 [5006.000, 5180.000] | 1597.000 [1588.000, 1633.000] | -68.53% | -0.10% |
| Filters5 | 6678.000 [6655.000, 7126.000] | 3235.000 [3188.000, 3256.000] | -51.56% | -0.87% |
| Filters7 | 9892.000 [9802.000, 9929.000] | 6428.000 [6362.000, 6451.000] | -35.02% | -0.51% |
| Filters8 | 9986.000 [9868.000, 10227.000] | 3073.000 [3054.000, 3083.000] | -69.23% | +1.01% |
| Groups4 | 19729.000 [19600.000, 20072.000] | 5997.000 [5940.000, 6073.000] | -69.60% | -0.60% |
| Groups5 | 26408.000 [26099.000, 27026.000] | 12416.000 [12392.000, 12482.000] | -52.98% | +0.12% |
| Groups7 | 39391.000 [38685.000, 39749.000] | 25122.000 [25106.000, 25468.000] | -36.22% | +0.92% |
| Groups8 | 39958.000 [39399.000, 40370.000] | 11802.000 [11712.000, 13217.000] | -70.46% | -0.65% |
| Depthwise | 8935.000 [8831.000, 8995.000] | 8900.000 [8858.000, 9665.000] | -0.39% | +0.54% |
| DepthwiseMultiplier4 | 27277.000 [27186.000, 27406.000] | 11842.000 [11671.000, 11952.000] | -56.59% | -0.17% |
| Fallback3 | 4811.000 [4799.000, 4835.000] | 4825.000 [4766.000, 5345.000] | +0.29% | -0.12% |
| Strided | 13435.000 [13390.000, 13585.000] | 13406.000 [13334.000, 13468.000] | -0.22% | +0.65% |
| Dilated | 15848.000 [15472.000, 16008.000] | 15785.000 [15728.000, 15855.000] | -0.40% | +0.73% |
| DilatedStrided | 12688.000 [12410.000, 12724.000] | 12677.000 [12593.000, 12735.000] | -0.09% | +0.66% |
| HeavyPadding | 2670.000 [2616.000, 2796.000] | 2611.000 [2601.000, 2653.000] | -2.21% | +0.41% |
| Batches | 40003.000 [39864.000, 41071.000] | 11726.000 [11647.000, 11876.000] | -70.69% | +1.25% |
| Pointwise | 6328.000 [6284.000, 7488.000] | 1839.000 [1825.000, 1841.000] | -70.94% | -0.84% |
| TinyPointwise | 146.200 [145.400, 147.300] | 148.600 [144.500, 150.500] | +1.64% | +1.44% |
| EmptyHeight | 804.100 [797.000, 807.100] | 402.200 [392.900, 411.100] | -49.98% | +1.09% |
| EmptyWidth | 349.400 [348.000, 352.000] | 348.500 [347.100, 351.600] | -0.26% | +0.14% |
| EmptyBatch | 112.000 [110.900, 112.200] | 110.600 [108.800, 113.000] | -1.25% | +1.07% |
| EmptyOutput | 114.600 [113.500, 115.600] | 114.600 [112.800, 115.900] | +0.00% | +0.61% |

## Unchanged ORT comparison

Both runs use the original driver, 19 identical inputs, ten warmups and 100
measured calls/input, `atol=1e-4`, `rtol=1e-5`, one thread per runtime and disabled
ORT graph optimization. Python 3.9.6, NumPy 1.26.4, ONNX 1.16.2 and ORT 1.19.2
come from the existing pinned environment. Go model load/session initialization,
serialization and tensor preparation are outside inference timing.

All 19 numerical and top-1 checks pass. **Go outputs match the baseline bits
exactly.** Maximum Go-versus-ORT absolute error remains
`1.9073486328125e-5`. Input/model hashes, original-model/conversion/upstream
checks and output bits are retained in the [baseline report](ort-before-summary.json)
and [final report](ort-after-summary.json). Full per-call reports remain locally.

| Comparison | Go median / p95 µs | ORT median / p95 µs | Measured median ratio |
| --- | ---: | ---: | ---: |
| v0.10.0 | 323.896 / 396.631 | 44.875 / 46.291 | 7.22× |
| Final tile | 205.875 / 278.110 | 44.292 / 46.083 | 4.65× |

These ORT comparisons run sequentially and are separate from the alternating
Go benchmark. The reported ratio is measured from full inference, not inferred
from kernel timing. ORT timing variation is not attributed to the backend change.

## Profiles and generated code

Separately collected twelve-second benchmark profiles contain 4.42 s of sampled
Run stacks before and 1.95 s after. Different iteration/sample counts prevent
using their raw totals as a latency comparison. The fractions below use only
sampled Run stacks; rows overlap.

| Path | Before | Final |
| --- | ---: | ---: |
| Conv2DInto, cumulative | 73.76% | 57.44% |
| Old four-spatial helper | 42.08% | no samples |
| New checked tile wrapper, cumulative | absent | 16.92% |
| New SIMD reduction, included in wrapper | absent | 11.79% |
| Existing four-filter boundary helper | 26.70% | 38.46% |

The spatial work moves into the joint tile; the unchanged boundary helper now
accounts for the largest remaining convolution share. The [filtered baseline](cpu-before.txt)
and [filtered final](cpu-after.txt), [unfiltered baseline](cpu-before-full.txt)
and [unfiltered final](cpu-after-full.txt) are retained, along with raw
[baseline](cpu-before.pb.gz) and [final](cpu-after.pb.gz) CPU profiles.

[Baseline disassembly](spatial-before-objdump.txt) has four scalar `fmadd s...`
operations and repeated input index checks. The [final reduction](tile-arm64-objdump.txt)
has four `fmla.4s` operations, one shared input vector load and four weight
broadcast loads per iteration, with no payload checks inside reduction.
[Go wrapper assembly](tile-wrapper-objdump.txt) and
[BCE before](bce-before.txt)/[after](bce-after.txt) show the complete-span checks
outside that loop. Existing helper checks remain for fallback work.
[Escape diagnostics](escape.txt) confirm the stack-only tile arguments.
[Portable disassembly comparison](portable-assembly-check.txt) confirms that its
channel traversal is unchanged after normalizing relocated symbol addresses.
