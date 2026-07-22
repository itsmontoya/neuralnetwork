# Metal Design

Captured on July 11, 2026.
Updated on July 21, 2026 for hybrid CPU SIMD and Metal selection.

## Decision

The `metal` build tag enables direct Metal access for large matrix
multiplication on Darwin/cgo builds. The public matrix API remains unchanged and
uses `float32` matrix values.

Metal kernels use `float` storage and accumulation. Because Go matrix storage is
also `float32`, the cgo bridge passes matrix buffers directly to Metal-backed
shared buffers without per-element precision conversion.

## Scope

The active Metal path covers:

* `MatMul`
* `MatMulInto`
* `MatMulLeftTransposeInto`
* `MatMulRightTransposeInto`

Only operations with at least `1 << 20` multiply-add work items are sent to
Metal. Smaller shapes use the scalar matrix multiplication helper to avoid
dispatch, conversion, and copy overhead.

Dot product and elementwise slice kernels do not dispatch to Metal. Benchmark
samples showed that moving O(n) slice kernels through Metal dispatch and buffer
copies was slower than CPU SIMD. On `arm64` and `amd64`, `metal` builds retain
the existing CPU SIMD wrappers for these operations. Unsupported architectures
and `purego` builds retain the local scalar fallbacks.

## Build Tags

Use:

```sh
go test ./... -tags=metal
```

The Metal implementation is compiled only when all of these are true:

* `darwin`
* `cgo`
* `metal`
* not `purego`

If `metal` is used on an unsupported platform, matrix multiplication uses its
portable fallback while eligible `arm64` and `amd64` CPU operations can still
use SIMD. The `purego` tag remains the explicit opt-out from both SIMD and
Metal, including when combined with `metal`.

## File Layout

```text
matrix/metal.go                 Go cgo bridge and Metal dispatch policy
matrix/metal_backend.h          C interface for the Objective-C backend
matrix/metal_backend.m          Metal device, pipeline, buffer, and dispatch code
matrix/matmul_metal.go          matrix multiplication wrappers for metal builds
matrix/matmul_default.go        scalar matrix multiplication wrappers
matrix/matmul_pure.go           scalar matrix multiplication helpers
matrix/metal_internal_test.go   Metal integration tests
internal/metaltest/counters.go  private synchronous bridge counters
```

The private counters record buffer creation, input upload, result download,
command submission, wait, and failure activity only while explicitly enabled
by repository tests. They do not add a public diagnostics API or change the
synchronous dispatch contract.

## Correctness Notes

Public validation, shape checks, and destination alias checks remain in
`Matrix` methods before private kernels are called.

Metal integration tests compare GPU matrix multiplication against the scalar
`float32` reference with a tolerance that accounts for GPU accumulation order.
If no Metal device is available, Metal integration tests skip. Shader
compilation or command failures are treated as test failures.
