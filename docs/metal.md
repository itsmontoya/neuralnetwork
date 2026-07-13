# Metal Design

Captured on July 11, 2026.

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
copies was slower than CPU SIMD. In `metal` builds, the SIMD files are excluded
and those slice kernels use scalar fallbacks instead.

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

If `metal` is used on unsupported platforms, or with `purego`, the package
builds with scalar fallbacks. The `purego` tag remains the explicit opt-out from
both SIMD and Metal.

## File Layout

```text
matrix/metal.go                 Go cgo bridge and Metal dispatch policy
matrix/metal_backend.h          C interface for the Objective-C backend
matrix/metal_backend.m          Metal device, pipeline, buffer, and dispatch code
matrix/matmul_metal.go          matrix multiplication wrappers for metal builds
matrix/matmul_default.go        scalar matrix multiplication wrappers
matrix/matmul_pure.go           scalar matrix multiplication helpers
matrix/metal_internal_test.go   Metal integration tests
```

## Correctness Notes

Public validation, shape checks, and destination alias checks remain in
`Matrix` methods before private kernels are called.

Metal integration tests compare GPU matrix multiplication against the scalar
`float32` reference with a tolerance that accounts for GPU accumulation order.
If no Metal device is available, Metal integration tests skip. Shader
compilation or command failures are treated as test failures.
