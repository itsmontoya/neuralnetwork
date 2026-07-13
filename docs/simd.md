# SIMD Design

Captured on July 7, 2026.
Updated on July 12, 2026 for `float32` matrix storage,
`github.com/tphakala/simd` integration, and `metal` tag interaction.

## Decision

SIMD work uses `github.com/tphakala/simd/f32` for `float32` dot product and
elementwise kernels on `arm64` and `amd64`. That dependency provides runtime CPU
feature dispatch and pure Go fallbacks behind its API. Pure Go matrix kernels
remain the local portable baseline for unsupported platforms and for builds
using this repository's `purego` opt-out tag.

The optional `metal` tag excludes SIMD wrappers and is documented separately in
[`docs/metal.md`](metal.md). In that mode, large matrix multiplication may use
Metal on Darwin/cgo builds, while slice kernels use local scalar fallbacks
instead of CPU SIMD.

The SIMD boundary stays inside the `matrix` package. Public matrix methods keep
owning validation, destination shape checks, alias checks, and ownership
contracts. Private kernels receive already-validated `[]float32` storage and do
not expose mutable matrix data outside the package.

Checked-in architecture-specific assembly should be added only where benchmark
evidence shows it beats the external SIMD dependency or the dependency cannot
cover the operation cleanly.

## Supported GOARCH Values

| GOARCH | v2 SIMD status | Notes |
| --- | --- | --- |
| `arm64` | Supported for SIMD candidates | Primary development and benchmark path. First kernels should target baseline arm64 NEON available on supported Go arm64 platforms. |
| `amd64` | Supported for SIMD candidates | Kernels may use baseline amd64 SIMD instructions unless a runtime CPU feature guard is added. |
| Other | Pure Go fallback | No SIMD work is planned until benchmark evidence justifies it. |

Optional CPU-specific kernels such as amd64 AVX2/FMA variants must use runtime
feature detection and keep a safe architecture fallback. They should not replace
the baseline architecture path unless every supported CPU for that `GOARCH` can
execute the instructions.

## Maintenance Cost

Architecture-specific assembly increases review and release cost. Prefer the
existing `tphakala/simd` boundary for operations where it performs well and
matches the required semantics. Any new in-repo assembly kernel must include:

* A checked-in generator with a pinned tool dependency when generated assembly
  is used.
* Checked-in `.s` and stub files.
* A `go:generate` command that recreates generated files deterministically.
* Tests comparing the active kernel with the pure Go implementation.
* Benchmarks proving a stable win before integration into matrix operations.

Generated assembly files must not be edited manually. Review should focus on the
generator source and the generated diff together. Hand-written assembly should
stay small, private, and tied to benchmark evidence.

## Fallback Strategy

Each SIMD candidate has a pure Go implementation that remains available on all
architectures. The public method validates inputs before calling the private
kernel, so fallback and SIMD paths share the same public error behavior.

The active private kernel may choose a scalar fallback for small inputs or
unsupported CPU features. When using `tphakala/simd`, runtime CPU dispatch stays
inside the dependency.

Floating-point reductions such as dot products and sums may not be bit-for-bit
identical if SIMD changes accumulation order. Correctness tests should compare
within the existing matrix test tolerance unless a kernel preserves scalar
order exactly.

## Build Tags and File Layout

Use `purego` as this repository's explicit opt-out tag for external,
architecture-specific SIMD wrappers, and Metal. Use `metal` as an opt-in tag
that excludes SIMD wrappers.

```go
//go:build arm64 && !purego && !metal
//go:build amd64 && !purego && !metal
//go:build (amd64 || arm64) && !purego && !metal
```

Architecture-specific SIMD wrapper files use `arm64 && !purego` or
`amd64 && !purego`, with `!metal` included. Shared SIMD wrapper files use
`(amd64 || arm64) && !purego && !metal`. Pure Go fallback files use:

```go
//go:build purego || (!metal && !arm64 && !amd64) || (metal && (!darwin || !cgo))
```

Current layout:

```text
matrix/elementwise_pure.go        pure Go helpers available to tests
matrix/elementwise_arm64.go       f32 SIMD elementwise wrappers
matrix/elementwise_amd64.go       f32 SIMD elementwise wrappers
matrix/elementwise_default.go     local pure Go fallback wrappers
matrix/elementwise_metal.go       scalar wrappers for metal builds
matrix/dot_product.go             pure Go dot product helper
matrix/dot_product_simd.go        f32 SIMD dot product wrapper
matrix/dot_product_default.go     local pure Go dot product wrapper
matrix/dot_product_metal.go       scalar wrapper for metal builds
```

Generator files should use `//go:build ignore` and stay out of normal package
builds.

## First Kernel Order

Attempt kernels in this order:

1. Dot product for contiguous `[]float32` inputs.
2. Elementwise destination and in-place operations.
3. Reductions only if later profiling shows they remain hot.

The dot product is the first candidate because it is a small, isolated kernel
and can be tested against a pure Go reference directly. Matrix multiplication
should use it only when benchmarks show a stable win. The current
`MatMulInto` right-hand column access is strided, so dot product integration is
most likely to help `MatMulRightTransposeInto` first. Avoid adding packing or
tiling to ordinary `MatMulInto` unless benchmark evidence justifies the extra
complexity.

Elementwise candidates are:

* `AddInto`
* `SubtractInto`
* `MultiplyElementsInto`
* `AddScaledInPlace`
* `AddScalarInto`
* `MultiplyScalarInto`
* `MultiplyScalarInPlace`

Division candidates should wait because per-element zero validation and divide
latency make the expected win less clear.

Reduction candidates are:

* `RowSumsInto`
* `ColumnSumsInto`
* `AccumulateColumnSumsInto`

Reduction work should start only after matrix and dense-layer benchmarks or
profiles show these paths remain release-relevant.

## Benchmark Shapes

Each SIMD section must record the exact command, raw output, and interpretation
in `Benchmarks_v1.md`.

Dot product benchmarks:

| Case | Lengths |
| --- | --- |
| Small | `1`, `2`, `3`, `4` |
| Medium | `64`, `257` |
| Large | `4096`, `65537` |
| Uneven tail | `5`, `31`, `33`, `4099` |

Matrix multiplication benchmarks:

| Case | Shapes |
| --- | --- |
| Small | `2x2 * 2x2`, `4x4 * 4x4` |
| Medium | existing `64x64 * 64x64` |
| Large | `128x256 * 256x128` |
| Uneven | `17x33 * 33x19`, `63x65 * 65x31` |

Elementwise benchmarks:

| Case | Shapes |
| --- | --- |
| Small | `1x1`, `1x2`, `1x3`, `2x2` |
| Medium | existing `256x256` |
| Large | `1024x1024` |
| Uneven | `17x19`, `255x257` |

Reduction benchmarks:

| Case | Shapes |
| --- | --- |
| Small | `1x1`, `1x3`, `3x1` |
| Medium | `64x64`, existing `128x256` where applicable |
| Large | `512x512` |
| Uneven | `17x257`, `257x17` |

## Correctness Coverage

Kernel tests must compare the active kernel with the pure Go reference.

Cover:

* Empty inputs for private slice kernels.
* Invalid public matrix shapes and destination aliases through existing public
  methods.
* Sizes below vector width.
* Sizes exactly at vector width.
* Multiple vector-loop iterations.
* Scalar tails after vector loops.
* Positive, negative, zero, `Inf`, and `NaN` values where the public operation
  already permits them.
* Pure Go opt-out builds with `-tags=purego` when architecture-specific
  assembly is added.

Public matrix methods must keep the existing validation behavior and ownership
rules. SIMD code must not add public low-level APIs for v2.
