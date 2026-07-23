# Device-Resident Metal Design

Approved architecture for the first device-resident milestone. The original
synchronous matrix-multiplication backend was captured on July 11, 2026. This
document freezes the next implementation boundary before matrix storage,
command lifetime, or model execution changes.

Updated on July 21, 2026 after establishing hybrid CPU SIMD and Metal
selection, private synchronous bridge counters, and end-to-end baselines.

Updated on July 22, 2026 after introducing the persistent runtime, buffer, and
command-scope primitives.

Updated on July 22, 2026 after adding transparent matrix residency, revision
tracking, staged standalone writes, host-observation barriers, coherent CPU
fallbacks, device copies, and scratch-eviction cleanup.

Updated on July 22, 2026 after adding bounded sequential executions, explicit
matrix binding and propagation, dependent command batching, selective CPU
fallback barriers, top-level completion, and failure cleanup. Device kernels
beyond multiplication and copy remain later sections.

Updated on July 23, 2026 after adding resident dense-forward bias, ReLU, and
stable row-wise Softmax kernels, retaining all multiplication variants, and
keeping the complete supported prediction chain in one command scope.

Updated on July 23, 2026 after adding resident dense and activation backward
kernels, parameter-gradient accumulation and reset, column reductions, and
selective backward observation boundaries.

Updated on July 23, 2026 after profiling the complete dense path, adding
cold/warm dispatch policy, deterministic fit-scratch release, cleanup-failure
atomicity, resource and allocation gates, and mixed-workload stress coverage.

## Decision Summary

The `metal` build tag remains an implementation opt-in. Existing matrix,
activation, layer, loss, optimizer, and model APIs remain unchanged. No public
device, stream, execution-context, synchronization, or diagnostics API is
required.

On a supported Darwin/cgo build with an available Metal device, the first
device-resident path supports this dense classification graph:

```text
matrix rows -> Dense -> ReLU -> Dense -> Softmax
```

It supports prediction, backward propagation, and categorical-cross-entropy
training with SGD through the existing `Sequential.Predict`, `Backward`,
`TrainBatch`, and `Fit` methods. Eligible parameters, gradients, layer caches,
and intermediate matrices remain device-resident. A caller-provided input and
target may each be uploaded at most once per `TrainBatch`; a warmed parameter
is not uploaded again until a host mutation makes its host value newer.

The implementation uses the Apple Metal framework for device, command, and
resource management and repository-owned Metal Shading Language kernels for
the approved operation vocabulary. Metal Performance Shaders, MPSGraph, and
third-party runtimes are not part of this milestone. A tiled custom
multiplication kernel may replace the current naive custom kernel only after
the end-to-end measurements required below show a useful improvement.

Metal absence, an unsupported build, an unsupported operation, or a workload
below the measured dispatch threshold is a normal CPU/SIMD path. These cases
do not change results or surface a device error.

## Current Batched Execution Boundary

The matrix-facing Metal path currently accelerates `MatMul`, `MatMulInto`,
`MatMulLeftTransposeInto`, and `MatMulRightTransposeInto`, plus resident
matrix copies, row-vector bias addition, built-in ReLU forward, and stable
row-wise Softmax forward. Dense backward additionally keeps both transpose
multiplications, built-in ReLU and Softmax derivatives, matrix gradient
addition, and accumulated column sums resident. Eligible standalone
multiplication, copy, and device-current gradient reset
operations remain synchronous: they create a private execution, encode,
commit, wait, publish completed staging, and detach before returning success.
Inside a model execution, dependent forward operations share bounded command
buffers, and a supported backward pass from resident forward caches uses one
command buffer. Completion alone does not download results. The path retains
the custom naive multiplication shader. A successfully initialized runtime
uses a `1 << 22` multiplication threshold. Before that runtime is ready, a
`1 << 26` cold threshold prevents smaller calls from paying device discovery
and pipeline-compilation cost.

The shared runtime initializes one default device, queue, library, fill
pipeline, and multiplication pipeline. On an available Metal build, a model
execution lazily attaches private build-neutral residency records while
binding matrices, but creates no Metal buffer or command scope until an
eligible operation encodes. Each record owns at most one committed buffer plus
staging during a proposed write. Unsupported shapes, unavailable devices, and
a graph whose multiplications never reach the threshold select CPU/SIMD before
encoding. Once a qualifying multiplication activates an execution, smaller
dependent multiplications and the approved forward kernels remain on the
device to avoid a transfer boundary. Initialization,
allocation, upload, encoding, command, synchronization, and download failures
after an eligible attempt are returned instead of being silently replayed on
CPU.

Host observations wait only when their matrix has a pending producer. Host
mutations also wait when the current command scope reads that matrix. CPU
destinations inherit the execution binding, so later eligible operations
upload the CPU-written revision lazily. Unsupported and custom implementations
continue through their existing public methods; the model does not bypass
them with type switches.

The current implementation is located in:

```text
internal/device/runtime.go      process runtime and resource construction
internal/device/buffer.go       opaque buffer ownership and host transfers
internal/device/scope.go        command-scope state and encoding API
internal/device/execution.go    bounded multi-command execution lifecycle
internal/device/execution_adapter.go immutable opaque-value adapter registry
internal/device/execution_snapshot.go per-execution batching diagnostics
internal/device/residency.go    revisions, committed buffers, and staged writes
internal/device/residency_snapshot.go private coherence diagnostics
internal/device/backend.go      build-neutral backend boundary
internal/device/backend_metal.go Go/cgo Metal adapter
internal/device/metal_backend.h C interface for the Objective-C backend
internal/device/metal_backend.m persistent Metal resources and command scopes
matrix/residency.go             host/device coherence hooks
matrix/execution.go             matrix binding, propagation, and barriers
matrix/metal.go                 resident batched multiplication adapter
matrix/copy_metal.go            independent device-newer matrix copies
matrix/forward_device_metal.go  resident bias, ReLU, and Softmax adapters
matrix/backward_device_metal.go resident derivatives, reductions, and accumulation
matrix/matmul_metal.go          Metal-aware multiplication wrappers
matrix/matmul_default.go        portable multiplication wrappers
matrix/matmul_pure.go           pure-Go multiplication reference
matrix/metal_internal_test.go   Metal integration tests
internal/metaltest/counters.go  private matrix transfer and command counters
model/sequential.go             top-level execution ownership and propagation
```

The private counters record buffer creation, input upload and bytes, result
download and bytes, command submission, wait, and failure activity only while
explicitly enabled by repository tests. Per-matrix snapshots additionally
record revisions, proposals, publications, discarded publications, avoided
uploads, downloads, and device copies. Neither mechanism adds a public
diagnostics API.

Current `metal` builds retain architecture-specific CPU SIMD for dot-product
and elementwise work on `arm64` and `amd64`. Unsupported architectures and
`purego` builds retain local scalar fallbacks. This hybrid selection does not
change the standalone synchronization boundary described above.

## Persistent Runtime Boundary

The build-neutral `internal/device` package now owns the private `Runtime`,
`Buffer`, `Scope`, `Residency`, operation identifiers, coherence snapshots,
and aggregate resource snapshots.
The Darwin/cgo/`metal` backend implements those types with opaque C handles;
other build combinations compile a backend that reports normal unavailability.
No public neural-network package exposes a device handle.

The process runtime initializes the default device, one command queue, one
shader library, and immutable fill and multiplication pipelines once. A mutex
protects initialization and its cached result. Independent scopes create and
submit distinct command buffers through the thread-safe shared queue, while
CPU fallback paths do not enter a runtime-wide execution lock. Bridge errors
use thread-local storage so concurrent scopes do not overwrite one another's
diagnostics.

A `Buffer` owns exactly one shared-storage `MTLBuffer` and validates complete
float32 uploads and downloads against its overflow-checked byte length.
Release is explicit and idempotent. There is no idle buffer cache. Encoding a
buffer retains its Objective-C resource in the scope until the command finishes,
so releasing the Go buffer owner after encoding cannot invalidate pending GPU
work.

A `Scope` begins in the encoding state. It can encode ordered copy, float32
fill/zero, and multiplication commands, then commit once, poll completion, wait,
and release. Encoding or committing after submission and waiting before commit
are state errors. Release waits for a submitted command when needed and is
idempotent. Every Objective-C entry that creates temporary objects has an
autorelease pool, and partial runtime, buffer, or scope construction releases
the resources it acquired before reporting an error.

Aggregate private diagnostics count live, peak, created, and released buffers
and scopes, live and peak buffer bytes, and submitted and completed commands.
Tests may reset these counters only with no live buffer or scope. The Section 3
stress test performs 512 allocate/encode/wait/release cycles and requires live
resources to return to zero, created and released counts to balance, and peaks
to remain at one buffer, one scope, and 64 bytes.

The matrix adapter retains committed input and result buffers through each
matrix's residency record. A full device destination uses staging so a command
failure cannot corrupt its last committed value. After a successful wait, the
record swaps staging into the committed slot and releases the replaced buffer.
The returned matrix may remain device-newer. The numerical kernel remains
unchanged. Model-level calls share an outer execution while standalone matrix
calls retain their synchronous boundary.

## Compatibility Decision

Build-tag-transparent execution can preserve the existing correctness and
error contracts, so this milestone does not add a public execution API.

The private design has three parts:

* A build-neutral package under `internal/device` owns opaque executions,
  command scopes, buffers, residency records, operation identifiers, private
  snapshots, and one immutable adapter registry. It imports none of the public
  neural-network packages.
* `matrix` holds an optional private residency pointer, registers the only
  checked adapter, and owns the hooks that bind executions, classify host
  observations and mutations, propagate CPU destinations, and invoke
  build-specific device operations without exposing a handle or public API.
* `model` creates executions and binds inputs and targets through the internal
  adapter. Matrix operations propagate the bound execution to destinations.
  `activation`, `loss`, and `optimizer` request the few specialized private
  operations they own through `internal/device`. `layer` continues to call its
  existing matrix and activation APIs.

`Sequential` uses private `predict` and `backward` helpers that accept an
execution. Public standalone calls create and finish one execution;
`TrainBatch` creates an outer execution and reuses it for its private predict,
loss, backward, and update phases. Nested calls detect and borrow the execution
bound to their input, while only the outer owner finishes or aborts it. This
makes propagation explicit without changing
`layer.Layer`, storing a current scope on a goroutine, or using package-global
per-call state. The internal adapter uses checked opaque values to avoid an
import cycle; a missing or mismatched adapter is an internal error, not a CPU
fallback.

The following alternatives are rejected:

* Public `Device`, `Context`, `Stream`, `ToMetal`, or `Synchronize` APIs would
  expose a backend choice that callers do not need to make.
* Adding context arguments to `layer.Layer`, `loss.Loss`, or
  `optimizer.Optimizer` would break accepted interfaces and custom
  implementations.
* A package-global current command, goroutine identifier lookup, or goroutine
  local emulation would make nesting and concurrency implicit and unsafe.
* Model-level type switches that bypass existing layer methods would duplicate
  validation and fail to preserve private forward caches.
* A generic tensor, automatic-differentiation graph, or replacement `Matrix`
  is larger than the operation and ownership problem being solved.

## Coherence Model

A `Matrix` always represents one logical row-major `float32` value set. Shape,
host storage, revisions, device storage, pending work, and failures are private.
Host storage remains allocated at its existing length even when its values are
stale, so shape validation never needs a download.

Each matrix has a monotonically increasing logical revision. Revision zero
means that a location has no current copy. Host and device revisions identify
the last successfully published value in each location. A pending device write
has a proposed revision that is not published until its command succeeds. If a
revision counter would overflow, the matrix synchronizes while exclusively
owned and rebases the current revision to one.

The canonical states and transitions are:

| State | Meaning | Permitted transition |
| --- | --- | --- |
| New | A constructor produced current host values and no device buffer. | First eligible device read uploads once and becomes synchronized; a host mutation becomes host-newer. |
| Host-newer | The host revision is current and the device is absent or stale. | A device read uploads once; another host mutation advances only the host revision. |
| Synchronized | Completed host and device copies have the same revision. | A host write becomes host-newer; a device write becomes pending. |
| Device-newer | A completed device revision is current and host storage is stale. | A host observation downloads once; a device consumer reuses the buffer without transfer. |
| Pending | A scope owns an uncommitted or incomplete device write and its proposed revision. | Successful completion publishes device-newer; failure enters failed without publishing the proposed value. |
| Failed | The proposed device value was not published and the scope retains a contextual error. | The top-level boundary reports the error, discards staging resources, and restores the last committed state; poisoned scratch is quarantined until cleanup completes. |
| Pooled | An ownership overlay for an idle scratch matrix, never an independently current value. | Reuse requires no pending work and a full logical overwrite before old data can be observed; eviction releases device storage. |
| Released | Device storage has been detached and released. Host storage remains usable if it is current. | A later eligible device use allocates and uploads lazily; destruction may discard a completed device-only value because no observer remains. |

`Copied` is an operation, not a shared state. `Clone` and `CopyFrom` read the
source's latest logical revision and create or overwrite an independent
destination. Inside a compatible device scope they use a device-to-device copy
without downloading. Outside such a scope they may still perform a synchronous
device copy when measurement permits. They never share writable buffers.

### Host observations and mutations

A host observation waits for a relevant pending producer, reports its failure,
and downloads exactly once when the device revision is newer. `Values`,
`ValuesInto`, `At`, serialization, Go callbacks, CPU metrics, and CPU fallback
reads are observations. Returning a `*Matrix` from a parameter or layer
accessor is not itself an observation; observing that matrix through one of its
methods is.

A partial host mutation such as `Set` first obtains the latest logical value so
untouched elements are preserved. A full overwrite such as `Fill` or
`CopyValuesFrom` waits before reclaiming an in-use buffer but need not download
the value it replaces. Every successful host write advances the host revision
and makes device content stale. Validation failure does not change revisions.

A device destination uses a separate buffer when an existing alias contract
forbids aliasing. Where the public method permits destination aliasing, the
kernel must be demonstrably safe for that exact alias or use staging. A result
revision is published only after successful completion. Existing validation,
shape, alias, ownership, callback order, and partial-error behavior remain the
reference contract.

### Public matrix method classification

The initial classification is exhaustive for the current public `Matrix` API.
"Device operation" means eligible for the initial private vocabulary; it does
not promise dispatch for a small or standalone call. "CPU fallback" performs a
barrier before reading host data and makes any destination host-newer.

| Methods | Class | Coherence and destination behavior |
| --- | --- | --- |
| `New`, `FromSlice`, `NewRandom`, `NewUniform`, `NewNormal`, `NewXavierUniform`, `NewHeNormal` | Construction | Produce independent, host-current matrices with no device allocation. |
| `Rows`, `Cols`, `Shape`, `Validate` | Shape-only | Inspect shape and logical storage metadata without waiting or downloading. |
| `Values`, `ValuesInto`, `At` | Host observation | Wait, report pending failure, and download if device-newer; returned values remain copies where currently documented. |
| `Set` | Host mutation | Preserve untouched current values, write one host element, and invalidate the device revision. |
| `Fill`, `CopyValuesFrom` | Full host mutation | Wait for safe buffer reuse, overwrite host storage without downloading replaced values, and invalidate device content. Private device zero/fill used for gradient reset does not change the public classification. |
| `Clone`, `CopyFrom` | Device copy | Preserve deep-copy independence. `CopyFrom` fully overwrites its destination; no source/destination buffer alias is introduced. |
| `SelectRows`, `SelectRowsInto` | CPU fallback | Download the source if needed. `SelectRowsInto` preserves its no-input-alias rule and fully overwrites the destination. |
| `Add`, `AddInto`, `AddInPlace`, `AddScaledInPlace` | Device operation | Support gradient accumulation and SGD. Existing permitted elementwise destination aliasing remains valid; allocating forms own independent storage. |
| `AddMappedInPlace`, `AdamUpdateInPlace` | CPU fallback | Go callbacks and unsupported optimizer state require current host values before mutation. Existing multi-matrix alias checks remain unchanged. |
| `MultiplyScalarInPlace` | CPU fallback | Not needed by the approved graph; mutates the current host value after a barrier. |
| `Subtract`, `SubtractInto`, `MultiplyElements`, `MultiplyElementsInto`, `DivideElements`, `DivideElementsInto` | CPU fallback | Preserve allocating/destination ownership, permitted elementwise aliasing, zero checks, and existing error timing. |
| `AddScalar`, `AddScalarInto`, `MultiplyScalar`, `MultiplyScalarInto`, `DivideScalar`, `DivideScalarInto` | CPU fallback | Preserve allocating/destination ownership, permitted input aliasing, and division validation. |
| `SoftmaxRowsInto`, `SoftmaxRowsBackwardInto` | Device operation | Preserve stable row-wise semantics. Forward destination may alias input; backward destination may alias input but not output gradient, using staging when required. |
| `MatMul`, `MatMulInto`, `MatMulLeftTransposeInto`, `MatMulRightTransposeInto` | Device operation | Preserve all shape and no-destination-alias checks. Allocating and destination forms may finish device-newer after successful synchronization. |
| `Transpose`, `TransposeInto` | CPU fallback | Download the input and preserve the no-input-alias destination contract. |
| `RowSums`, `RowSumsInto` | CPU fallback | Row reduction is outside the approved vocabulary; slice-returning `RowSums` necessarily observes host values. |
| `ColumnSums` | Host observation and CPU fallback | Returns a host slice and therefore downloads before the existing reduction. |
| `ColumnSumsInto`, `AccumulateColumnSumsInto` | Device operation | Support dense bias gradients. Destinations cannot alias the input; overwrite versus accumulation semantics remain distinct. |
| `AddRowVectorInPlace` | Device operation | Supports dense bias addition and preserves the `[1, Cols]` validation and receiver ownership. |
| `Apply`, `ApplyInto`, `Pairwise`, `PairwiseInto` | CPU fallback | Arbitrary Go callbacks are never assumed to be shaders. Callback order, error propagation, and currently permitted destination aliasing remain unchanged. Built-in ReLU and categorical loss use separate private typed operations. |

This classification describes the complete milestone vocabulary. At the
current Section 8 boundary, the supported graph also encodes categorical
target validation, scalar loss reduction, prediction gradients, SGD updates,
and gradient reset. A warmed training step synchronizes once for its combined
loss and diagnostic result, then publishes backward gradients, every staged
parameter update, and every gradient reset atomically after a second command
scope completes. Unsupported losses and optimizers continue through explicit
CPU fallback barriers.

## Device Operation Vocabulary

Only these operations may be encoded for the first milestone:

* Allocate, upload, download, copy, zero, fill, and release a row-major
  `float32` buffer.
* Standard, left-transposed, and right-transposed matrix multiplication.
* Matrix copy for layer caches.
* Row-vector bias addition.
* ReLU forward and backward.
* Numerically stable row-wise Softmax forward and Jacobian-vector backward.
* Column-sum overwrite and accumulation.
* Elementwise addition and scaled addition for gradient accumulation and SGD.
* Categorical-cross-entropy one-hot validation, mean loss reduction, and mean
  prediction gradient using the existing clamp semantics.
* SGD parameter staging, atomic publication, and gradient zeroing.

The supported layer path recognizes only built-in `Dense` layers and built-in
`ReLU` and `Softmax` activations. Both value and pointer forms of the stateless
activation and loss types remain semantically equivalent. The supported
training path additionally requires built-in `CategoricalCrossEntropy`, plain
`SGD`, and no regularizer wrapper.

`Conv2D`, `MaxPool2D`, `BatchNormalization`, `Dropout`, `Flatten`,
`SimpleRNN`, `LastStep`, other built-in activations, other losses, Momentum,
Adam, regularized optimizers, learning-rate work beyond the existing scalar
setter, and all custom implementations remain CPU operations. Flatten may
eventually be represented as a metadata-only view, but it is a CPU boundary in
this milestone so no new alias contract is introduced.

Unsupported built-in and custom operations are never skipped. Before a custom
layer, activation, loss, optimizer, regularizer, metric, or Go callback reads a
matrix, the scope waits for its required producers and downloads only its
device-newer inputs. CPU destinations become host-newer and upload lazily only
if a later supported operation consumes them. An all-CPU graph never creates a
Metal scope or buffer.

## Command Scopes and Synchronization

Standalone matrix operations preserve their synchronous error contract. An
eligible standalone operation creates a private scope, encodes, commits, and
waits before returning success. Its matrix result may be device-newer after the
wait; synchronous means command completion and error delivery, not mandatory
host materialization.

Top-level behavior is:

| Boundary | Required behavior |
| --- | --- |
| `Sequential.Predict` | Create a scope, propagate it through all layers and fallback barriers, commit bounded command buffers, and wait before success. The returned output may remain device-newer. |
| `Sequential.Backward` | Create a scope that can consume completed resident forward caches, traverse layers in reverse, and wait before success. The returned input gradient may remain device-newer. |
| `Sequential.TrainBatch` | Create one outer scope. Reuse it through private predict and backward helpers. The pre-update loss scalar and target diagnostics force the first required synchronization. Backward and an atomic staged update complete before success. Input and target upload at most once in the scope. |
| `Sequential.Fit` | Use one bounded scope per `TrainBatch` and one per evaluation prediction. Never retain a command buffer across batches or epochs. Learning-rate schedules run before the batch scopes for an epoch. |
| CPU/custom boundary | Wait for required producers, surface errors, download only matrices read by CPU code, perform the existing operation, and mark written destinations host-newer. |
| Metrics and accuracy | Scalar device loss may be read directly. Existing accuracy functions are Go callbacks and require current host predictions and targets. |
| Fit callback and early stopping | Run only after batch and evaluation scopes have completed. They receive host scalar metrics, so they do not retain device work. |
| Serialization | Wait and download each serialized device-newer matrix through `Values`; do not serialize scopes or handles. |

Nested private calls reuse an explicitly supplied outer scope. Public calls do
not discover scopes through goroutine identity. Repeated or alternating calls
use distinct scopes. The existing concurrency contract is unchanged: callers
must not concurrently mutate or execute the same matrix, layer, optimizer, or
model. Distinct matrices and models may execute concurrently, and shared
runtime initialization, pipelines, counters, and command-queue access must be
race-free. Sharing parameters between concurrent models remains caller-owned
synchronization, as it is on CPU.

One command buffer is capped at 64 encoded kernels or 64 MiB of newly owned
transient staging, whichever is reached first. A single larger operation is
allowed and then forces a boundary. Resident operand buffers do not count as
transient staging. The scalar loss/diagnostic read, CPU fallback, top-level
return, and staged parameter publication are mandatory boundaries. These caps
bound long `Fit` calls without forcing a wait per matrix method and may be
tuned only with recorded end-to-end evidence.

## Failure Model and Atomic Publication

Backend selection distinguishes normal capability decisions from failures:

* Unsupported tags or platform, cgo disabled, no default Metal device, a shape
  outside backend limits, and an ineligible workload select CPU/SIMD before
  encoding. Device absence is cached as a normal capability result.
* Device initialization other than absence, shader-library or pipeline
  compilation, buffer allocation, upload, encoding, commit, execution,
  synchronization, and download failures are operational errors once an
  eligible Metal attempt begins. They are returned with stage and operation
  context and are not silently retried on CPU.
* CPU fallback is safe only before device work begins or after a successful
  fallback barrier. A failed command is never replayed because its partial GPU
  writes cannot be proven harmless.

Encoding errors return immediately. Execution and synchronization errors are
stored on the scope and returned at the next mandatory boundary. No top-level
method returns success while its scope has a pending error. A failed proposed
destination revision is not published, transient buffers are released after
Metal no longer references them, and scratch touched by a failed command is
quarantined until cleanup finishes.

Completed command-scope cleanup occurs before staged publications become
logical matrix values. A cleanup failure therefore discards the whole staged
batch and is reported at the same atomic boundary as allocation, upload,
encoding, submission, execution, synchronization, or download failure.
`TrainBatch` never exposes only part of a parameter update because cleanup
failed after the GPU completed.

`TrainBatch` preserves the existing phase order: prediction, scalar loss,
loss gradient, reverse traversal, parameter-order optimizer update, and
gradient reset. It also preserves the pre-update returned loss. Shape, alias,
target, layer-state, and optimizer validation occurs before the phase that
could mutate logical values.

Parameter updates are transactionally published. The GPU writes every updated
parameter and reset gradient to scope-owned staging buffers while the previous
parameter and gradient revisions remain committed. Only a successfully
completed update command swaps all parameter and gradient device handles and
revisions into their matrices, in stable parameter order. Failure discards all
staging and leaves the previous logical parameters and accumulated gradients
current. This avoids a partially applied Metal training step even if a command
fails after writing one staging buffer.

## Runtime and Resource Ownership

One lazily initialized process runtime owns the default `MTLDevice`, one
thread-safe command queue, the shader library, and one immutable pipeline per
operation. Device discovery and fixed resources initialize once. Pipeline
creation may be lazy per operation but is synchronized and its success or
failure is cached. The milestone does not select multiple devices or create a
queue per model.

Each execution scope exclusively owns its command buffers, encoders, pending
errors, transient buffers, retained operand references, and per-scope counters.
An encoder is ended on every path. Command and staging resources remain
retained until completion, then are explicitly released when the scope closes.
Every Objective-C bridge entry uses an autorelease pool. Independent scopes may
submit concurrently through the shared queue without sharing mutable command
state.

A matrix residency record owns at most one committed Metal buffer of exactly
`Rows*Cols*sizeof(float32)` bytes. A pending full write owns separate staging
until publication. Replacing or invalidating a buffer releases it after its
last command user completes. The initial milestone has no idle Metal buffer
cache: maximum idle pooled Metal bytes is zero. Existing `MatrixPool` instances
may retain up to four logical matrices per scratch role; eviction explicitly
detaches their device storage. Thus retained data-buffer memory is the exact
sum of live resident matrices plus the bounded active-scope staging described
above, not an unbounded backend cache.

Fit-owned batch and evaluation pools are explicitly emptied on every return,
including validation, callback, early-stopping, and operational-error paths.
Their retained matrices detach device residency before the local fit scratch
becomes unreachable, so repeated `Fit` calls do not depend on finalizer timing
to return Metal buffers.

Scope close, failed construction, failed encoding, scratch eviction, buffer
replacement, and explicit test shutdown all release their owned resources.
Unreachable caller-owned matrices use documented Go runtime cleanup as a final
safety net, but routine command, scratch, and replacement cleanup does not rely
on it. The fixed shared runtime lives for the process; a private test-only close
requires no live scopes or buffers and verifies balanced resource counts. No
public `Close` or shutdown commitment is introduced.

## Private Diagnostics

Build-neutral test doubles and Metal integration tests can take a private
per-scope snapshot containing:

* Host and device revision advances, proposed revisions, publications,
  discarded publications, and avoided redundant uploads.
* Buffer allocations, releases, current live count and bytes, peak live count
  and bytes, and transient staging bytes.
* Upload and download counts and bytes, plus device-to-device copy counts and
  bytes.
* Kernel encodes by operation, command-buffer creation, submission,
  completion, and failure counts.
* Wait and synchronization counts, including the reason for each mandatory
  boundary.
* CPU fallback barrier counts, matrices and bytes downloaded at a barrier, and
  subsequent lazy re-uploads.
* Initialization, compilation, allocation, upload, encoding, execution,
  synchronization, download, and cleanup errors by stage.

Per-scope counters prevent concurrent models from contaminating transfer and
command assertions. Runtime aggregate live/peak resource counters support leak
and stress tests. Tests may reset aggregates only while the private runtime is
idle. These types and snapshots stay under `internal`; benchmark output may
report their values, but no public diagnostics API or stability promise is
created.

## Numeric and Determinism Contract

The pure-Go path remains the correctness reference. Parity tests first verify
shape, aliasing, ownership, finite outputs, and diagnostics, then compare
finite values with `abs(got-want) <= atol + rtol*abs(want)`. NaN and infinity
in finite test fixtures always fail before a tolerance comparison.

Initial tolerances are:

| Operation | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| Copy, fill, zero, ReLU forward/backward | `0` | `0` |
| Elementwise add, bias add, scaled add, SGD | `1e-6` | `1e-6` |
| Softmax forward/backward | `2e-5` | `2e-5` |
| Column reduction | `max(1e-6, 8*reductionLength*eps*sumAbs)` | `2e-5` |
| Categorical gradient | `2e-5` | `2e-5` |
| Categorical scalar loss | `2e-5 + 8*batchRows*eps*sumAbs/batchRows` | `2e-5` |
| Matrix multiplication variants | `max(1e-6, 8*innerSize*eps*sumAbsProducts)` | `8*eps` |
| One complete training step | `2e-4` | `2e-4` |

Here `eps` is the `float32` machine epsilon, `reductionLength` is the number of
summed values, `innerSize` is the multiplication inner dimension, `sumAbs` is
the absolute sum reduced into the result, and `sumAbsProducts` is the absolute
dot-product sum for that output element. Tests with deliberate non-finite CPU
semantics are separate and compare classification and propagation explicitly
rather than letting a tolerance hide them.

Repeated supported executions with the same runtime, device, pipeline,
dispatch shape, input, and initial parameters must be bit-identical. CPU/Metal
and cross-device comparisons use the table above because accumulation and
transcendental order may differ. Random initialization and data shuffling stay
on their existing deterministic CPU paths.

## Dispatch and Benchmark Evidence

Section 2's synchronous baseline used a `1 << 20` multiplication threshold.
Section 9 replaces it with two process-readiness thresholds derived from full
model measurements:

| Runtime state | Minimum multiplication work | Behavior |
| --- | ---: | --- |
| Cold | `1 << 26` | Smaller graphs stay entirely on CPU/SIMD and do not initialize Metal. A qualifying graph initializes the shared runtime and uses Metal. |
| Ready | `1 << 22` | A qualifying multiplication activates the model execution; smaller dependent work stays in that execution. |

Model preflight computes each dense multiplication's
`rows*inputSize*outputSize` with saturating arithmetic before requesting the
runtime. This avoids residency records, buffer allocation, and pipeline work
for a cold ineligible graph. Runtime readiness becomes true only after a
usable shared Metal runtime has initialized successfully.

The cutoffs include initialization, upload, staging-buffer creation, command
submission, mandatory waits, and result observation. On the recorded Apple M3,
a warmed and observed `1 << 20` prediction was slower on Metal than CPU
(about 867 versus 730.9 microseconds). After dispatch hardening that workload
uses no Metal commands and measures 729.7 microseconds under the `metal` tag
versus 730.5 microseconds by default. At the ready cutoff, an observed
prediction with two `1 << 22` multiplications measures 1.126 milliseconds on
Metal versus 5.346 milliseconds on CPU. In fresh processes, the same
ready-cutoff graph stays on CPU and measures 11.801 milliseconds under the
`metal` tag versus 11.665 milliseconds by default. The representative cold
large `TrainBatch` crosses `1 << 26` and measures 58.347 milliseconds on Metal
including initialization versus 163.452 milliseconds on CPU.

Profiling confirms that CPU time is dominated by the three multiplication
variants. In the Metal profile, external GPU work dominates, followed by
buffer creation/release bridge calls; command commit, copy encoding, download,
and upload are smaller. Scheduler semaphore samples are negligible and the
distinct-model race/stress tests show no runtime-wide execution-lock
contention. No buffer cache was added: isolated bridge samples do not prove an
end-to-end win large enough to justify retained idle memory, and the current
failure-atomic staging lifetime is intentionally bounded. Command batching is
already at the observable minimum: one command/wait for prediction or
backward, two for `TrainBatch` because the pre-update loss must be observed,
and three for bounded `Fit` including evaluation. Kernel, elementwise, and
reduction implementations therefore remain unchanged.

Every performance change records hardware, OS, Go version, architecture, cgo
setting, tags, power mode, benchmark command, raw output, counters,
allocations, and interpretation in `Benchmarks_gpu.md`.

Measurements use at least ten samples and compare default CPU/SIMD, `purego`,
the synchronous `metal` baseline, and the resident `metal` path with identical
logical models. `benchstat` or an equivalent recorded median comparison is used
after one untimed correctness warm-up. Cold cases include runtime and pipeline
first use, buffer creation, transfers, submission, required waits, and host
observation. Warm cases reuse runtime, pipelines, parameters, and eligible
buffers and report both unobserved matrix returns and required observation
boundaries.

Representative dense-classification shapes are:

| Case | Batch | Input | Hidden | Classes | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| Small | 16 | 32 | 64 | 10 | Below dispatch thresholds; must remain CPU/SIMD. |
| Observed below threshold | 64 | 128 | 128 | 16 | First multiplication is exactly `1 << 20`; exposes observation and boundary overhead. |
| Warm threshold | 256 | 128 | 128 | 128 | Both multiplications are exactly `1 << 22`; validates ready-runtime eligibility. |
| Uneven | 127 | 257 | 263 | 19 | Exercises non-tile dimensions and a mixed large/small multiplication chain. |
| Large | 256 | 512 | 512 | 64 | Representative resident prediction, backward, and training workload. |

Benchmarks cover `Predict`, `Backward`, `TrainBatch`, and a bounded one-epoch,
one-batch `Fit` followed by full-dataset evaluation for each applicable shape.
They report uploads, downloads, bytes, buffer allocations, commands, kernels, waits,
synchronizations, fallback barriers, and Go allocations in addition to time.

The resident training backend is retained only if the warmed Large
`TrainBatch` median is at least 1.5x faster than default CPU/SIMD and the
bounded Large `Fit` median is at least 1.25x faster. A Small CPU/SIMD case is a
material regression only when its median is both more than 10% and more than 5
microseconds slower. Prediction and backward results must also be reported even
when they are not the limiting gate. If these gates are not met, independently
useful hybrid SIMD or runtime work may remain, but device-resident training is
stopped or redesigned rather than broadening the graph or weakening
correctness.

The final Section 9 medians meet these gates. Warmed Large prediction is 9.8x
faster, backward is 51.1x faster, `TrainBatch` is 18.6x faster, and bounded
`Fit` is 27.9x faster than default CPU/SIMD. Small `metal` medians differ from
default by less than two microseconds and create no Metal buffers, commands,
or waits.

## Build-Tag Contract

The intended hybrid selection is:

| Build | Darwin/cgo with device | Unsupported platform, cgo disabled, or no device |
| --- | --- | --- |
| Default | CPU SIMD on `amd64`/`arm64`; pure-Go multiplication | Same architecture selection; no Metal code compiled. |
| `purego` | Pure Go for multiplication, dot products, and elementwise work; Metal excluded | Pure Go. |
| `metal` | Metal is eligible for approved work; CPU fallbacks still use SIMD on `amd64`/`arm64` | Same CPU SIMD selection as default on supported architectures, otherwise pure Go. |
| `metal purego` | `purego` wins: no Metal and no SIMD | Pure Go. |

Objective-C and Metal framework files compile only for
`darwin && cgo && metal && !purego`. The `metal` tag alone must remain portable.
Section 2 removed the former scalar-only Metal wrappers, so enabling Metal no
longer disables SIMD for CPU-resident operations. Backend stubs preserve the
same private interfaces everywhere else.

## Serialization, Parameters, and Scratch

`Sequential.Save` remains format `neuralnetwork.sequential`, version `1`.
`Values` synchronizes and downloads device-newer parameters and documented
layer state before JSON encoding. Devices, buffers, revisions, scopes,
pipelines, counters, pending errors, and scratch are never encoded. Equivalent
logical models therefore retain their existing bytes regardless of residency.

Loading constructs matrices through existing host constructors. Loaded
parameters and gradients start host-resident with no device buffer, pending
command, or cached runtime state. Their first eligible use uploads once.
`Parameter.Values` and `Gradient` continue returning their mutable matrix
pointers; shape access does not download, host observation does, and any host
mutation invalidates a stale device copy. `NewParameter`, `Clone`, and
serialization copies remain independent.

Scratch matrices preserve their existing dirty-reuse and ownership contracts.
They may keep compatible completed device storage while retained for the same
logical role, but reuse cannot reveal an earlier value, pending command, or
failure. A changed shape uses a distinct matrix, eviction releases residency,
and failed-scope scratch is not returned until cleanup is complete.

## Explicit Non-Goals

This milestone does not add public asynchronous execution, device selection,
multiple-device scheduling, mixed precision, quantization, a generic tensor,
an automatic-differentiation graph, a replacement matrix type, or serialized
runtime state. It does not refactor or accelerate CNN or RNN layers, decide new
RNN state semantics, make arbitrary Go callbacks shader-compatible, or change
custom layer contracts. It does not require a public API addition to preserve
correctness.

CNN and RNN kernels remain follow-up milestones. The accepted v1, CNN, and RNN
public surfaces continue unchanged.
