# Gradient Clipping Design

Status: implemented and verified additive optimizer and training contract.

This document freezes the additive public and behavioral contract for
[ROADMAP Item 2](../ROADMAP.md#2-add-opt-in-gradient-clipping-and-recurrent-training-controls).
The declarations below are implemented in package `optimizer` and recorded in
the additive post-v1 API inventory in
[v1-api-review.md](v1-api-review.md). Focused optimizer and model tests cover
the arithmetic, validation, observation, composition, allocation,
deterministic recurrent updates, ordinary and length-aware fitting, and Metal
fallback behavior frozen here.

The design adds one opt-in optimizer wrapper. It transforms accumulated
`optimizer.Parameter` gradients immediately before its base optimizer consumes
them. It does not change the stable `optimizer.Optimizer`, `layer.Layer`,
`loss.Loss`, `model.Sequential`, `model.FitConfig`, or
`model.SequenceFitConfig` contracts.

## Decision Summary

The accepted boundary is:

```text
layer backward
    -> ordered accumulated parameter gradients
    -> optional value clipping
    -> optional global-norm clipping
    -> base optimizer update state
    -> base-owned gradient reset
```

Value clipping uses one symmetric positive finite magnitude. Norm clipping
uses one L2 norm across every gradient element in the received parameter order.
When both controls are enabled, value clipping runs first and the global norm
is calculated from the value-clipped candidates. Norm accumulation uses
`float64` and an ordered `math.Hypot` fold so squaring an extreme finite
`float32` cannot overflow an intermediate.

Clipping remains strictly opt-in. No layer, model operation, built-in
optimizer, regularizer, or recurrent path constructs this wrapper
automatically.

## Public API

Package `optimizer` adds exactly these declarations:

```go
type GradientClippingConfig struct {
	MaxValue float32
	MaxNorm  float32
}

type GradientClippingObservation struct {
	ValueClipped        bool
	GlobalNorm          float64
	Scale               float64
	BaseUpdateCompleted bool
}

func NewGradientClipping(
	base Optimizer,
	config GradientClippingConfig,
) (out *GradientClipping, err error)

type GradientClipping struct { /* unexported fields */ }

func (c *GradientClipping) Update(
	parameters []*Parameter,
) (err error)
func (c *GradientClipping) LearningRate() (learningRate float32)
func (c *GradientClipping) SetLearningRate(
	learningRate float32,
) (err error)
func (c *GradientClipping) Base() (base Optimizer)
func (c *GradientClipping) Config() (config GradientClippingConfig)
func (c *GradientClipping) Observation() (
	observation GradientClippingObservation,
	available bool,
)
```

The field names and types are contractual. The implementation must include a
compile-time assertion that `*GradientClipping` implements `Optimizer`.

`GradientClippingConfig` is a value configuration:

* `MaxValue == 0` disables value clipping.
* `MaxNorm == 0` disables norm clipping.
* A nonzero enabled value must be positive and finite.
* Both fields cannot be zero because a wrapper with no enabled control is an
  invalid configuration.

The public fields make value-only, norm-only, and combined construction
explicit without adding focused constructors that can drift semantically:

```go
valueClipping, err := optimizer.NewGradientClipping(
	base,
	optimizer.GradientClippingConfig{MaxValue: 1},
)

normClipping, err := optimizer.NewGradientClipping(
	base,
	optimizer.GradientClippingConfig{MaxNorm: 5},
)

combinedClipping, err := optimizer.NewGradientClipping(
	base,
	optimizer.GradientClippingConfig{
		MaxValue: 1,
		MaxNorm:  5,
	},
)
```

Each example must handle its returned error. The repeated `err` declarations
are omitted only to keep the construction forms focused.

`NewGradientClipping` copies `config` into the wrapper. `Config` returns that
value copy. Mutating the caller's original configuration or a returned copy
does not reconfigure the wrapper. There are no clipping setters; callers
construct a distinct wrapper for a different clipping policy.

`Base` returns the same optimizer reference supplied at construction and does
not transfer ownership. `LearningRate` and `SetLearningRate` delegate directly
to that base. This lets existing `Fit` and `FitWithLengths` learning-rate
schedules use the unchanged `Optimizer` interface.

The configuration and observation contain only scalars. Their value accessors
cannot expose mutable internal slices, matrices, or scratch storage.

## Why the Optimizer Boundary Is General

Every current trainable layer exposes accumulated gradients through
`optimizer.Parameter`. `Sequential` already discovers those parameters in
stable layer order and passes one ordered slice to exactly one
`Optimizer.Update` call after backward. Dense, CNN, fixed-length RNN, and
explicit-length RNN training therefore share the same boundary.

Clipping needs no batch, layer, sequence, or loss information. It sees only
the same row-major `float32` value and gradient matrices that SGD, Momentum,
Adam, regularizers, and custom optimizers already receive. Consequently:

* `Optimizer` does not gain another method.
* `Layer` and layer backward signatures do not change.
* Loss value or gradient contracts do not change.
* `Sequential`, `FitConfig`, `SequenceFitConfig`, and training metrics do not
  gain clipping fields.
* No RNN type switch or recurrent-only default is introduced.

This keeps custom optimizers source-compatible. A custom optimizer can be used
as the base without implementing a clipping-specific interface.

## Arithmetic Contract

### Ordered logical input

The wrapper receives the exact `[]*Parameter` supplied to `Update`. Parameters
are traversed by increasing slice index. Each gradient is traversed by
increasing row and then increasing column, which is its physical row-major
order.

The wrapper passes the original parameter slice to the base exactly once and
in the same order. It does not sort, group, filter, or retain it. An empty
slice is valid: it represents no gradient elements, still delegates once to
the base, and produces a no-op observation if the base returns.

Repeated references in the parameter slice are not rejected or deduplicated.
Each occurrence contributes its gradient elements to the global norm in its
received position, matching the ordered logical input presented to the base.
Transformation uses the pre-mutation snapshot for each occurrence, so the
same physical gradient is not multiplied repeatedly merely because its
parameter pointer occurs more than once. The base retains ownership of any
update behavior caused by repeated references.

### Value clipping

For enabled `MaxValue = V`, every finite input gradient `g` has the candidate:

```text
valueCandidate(g) = max(-V, min(g, V))
```

The closed interval is `[-V, V]`. Values exactly equal to either boundary are
unchanged. Positive zero, negative zero, and every in-bound finite value are
preserved. `ValueClipped` is true when at least one original value was strictly
less than `-V` or strictly greater than `V`.

Non-finite values are rejected before this formula is used. Value clipping
never treats clamping as a repair for NaN or infinity.

### Global norm

Norm clipping has one global scope across all parameter occurrences:

```text
G = sqrt(sum(candidate * candidate))
```

`candidate` is `valueCandidate(g)` when value clipping is enabled and is `g`
otherwise. The batch dimension, layer boundary, parameter matrix shape, and
parameter kind do not create separate norm groups.

The implementation computes `G` by initializing `norm = 0` and folding each
ordered candidate as:

```go
norm = math.Hypot(norm, float64(candidate))
```

This defines `float64` accumulation precision and avoids an intermediate
`candidate*candidate` overflow for finite `float32` values. The ordered fold,
not a parallel or map-based reduction, is the CPU reference algorithm.

For enabled `MaxNorm = M`, the scale is:

```text
scale = 1                 when G == 0 or G <= M
scale = float64(M) / G    when G > M
```

The wrapper checks that `G` is finite after every fold and at completion. It
also checks that `scale` is finite, greater than zero, and no greater than one
before mutating a gradient.

An all-zero norm is valid and uses scale one. A norm exactly equal to the
limit is a no-op. If norm clipping is disabled, the observation reports
`GlobalNorm == 0` and `Scale == 1`.

When `scale < 1`, each stored result is:

```go
float32(float64(candidate) * scale)
```

This conversion, including ordinary `float32` rounding or underflow of an
individual result, is part of the contract. The pre-scaling `GlobalNorm` and
the `float64` scale are reported. Tests compare the resulting global bound
with `float32`-appropriate tolerance rather than promising exact real-number
arithmetic.

### Combined order

When both controls are enabled:

1. Reject any non-finite original gradient.
2. Clamp every value candidate to `[-MaxValue, MaxValue]`.
3. Compute the global norm of those clamped candidates.
4. Apply the single norm scale to every clamped candidate.

For example, gradients `[6, 8]`, `MaxValue = 5`, and `MaxNorm = 5` first
become `[5, 5]`. Their observed global norm is `sqrt(50)`, the scale is
`5/sqrt(50)`, and both final gradients are approximately `3.535534`.

Norm-first behavior would instead scale the original norm ten to produce
`[3, 4]`, after which value clipping would do nothing. That different result
is deliberately not the accepted order.

### Global-scope example

Given two ordered parameter gradients `[3, 4]` and `[0, 12]`, the global norm
is thirteen. With `MaxNorm = 6.5`, the scale is exactly one half and the base
receives `[1.5, 2]` and `[0, 6]`. A per-parameter rule would leave the first
gradient unchanged and is therefore not compatible with this contract.

## Validation and Errors

All wrapper-originated errors use `optimizer:` context. Constructor and update
validation use this order:

1. Receiver when applicable.
2. Base optimizer.
3. Clipping configuration.
4. Every parameter, value matrix, gradient matrix, and shape pair.
5. Total scratch element count.
6. Every gradient value and the complete norm/scale calculation.

No gradient is mutated and the base is not called until all six stages
succeed.

### Constructor and receiver validation

`NewGradientClipping` rejects a nil `Optimizer` interface with:

```text
optimizer: base optimizer is nil
```

Go interfaces that contain a typed nil pointer are not nil interfaces. The
wrapper does not use reflection to reinterpret custom interface values. A
typed nil built-in base reports its existing nil-receiver error when invoked;
a custom typed nil base owns its own method behavior.

Configuration errors identify the field and value:

* Both controls disabled:
  `optimizer: gradient clipping requires at least one enabled limit`.
* Negative `MaxValue`, NaN, or either infinity:
  `optimizer: gradient clipping max value must be positive and finite when enabled: maxValue=<value>`.
* Negative `MaxNorm`, NaN, or either infinity:
  `optimizer: gradient clipping max norm must be positive and finite when enabled: maxNorm=<value>`.

Zero disables one control and is not independently an error when the other
control is enabled. Positive subnormal `float32` limits are valid.

Calling `Update` on a nil receiver returns
`optimizer: gradient clipping optimizer is nil`. A non-nil zero value is
invalid because its base is nil and its configuration is empty; validation
reports the nil base first.

For nil or zero-value receivers:

* `LearningRate` returns zero.
* `Base` returns nil.
* `Config` returns the zero configuration.
* `Observation` returns the zero observation and `available == false`.
* `SetLearningRate` returns a contextual error and does not panic.

For a valid wrapper, `SetLearningRate` forwards the supplied value without
duplicating validation. The base's normal validation and exact error are
authoritative.

### Parameter and shape validation

For every parameter index, validation requires:

* A non-nil `*Parameter`.
* A non-nil, structurally valid value matrix.
* A non-nil, structurally valid gradient matrix.
* Identical value and gradient shapes.

The wrapper validates both matrices with `Matrix.Validate`, not only their
reported dimensions. Errors include the parameter index, for example:

```text
optimizer: gradient clipping parameter 2 invalid: <underlying error>
```

The wrapper does not reject finite or non-finite numeric parameter values
because clipping does not read them. A regularizer that uses parameter values
owns its own validation and arithmetic. If that regularizer is outside the
clipping wrapper, its resulting gradient is subsequently subject to the
clipping non-finite check.

The total number of gradient elements across parameter occurrences must fit
in `int`. An overflow error includes the parameter index whose element count
would exceed the limit. This check precedes scratch growth or gradient reads.

### Gradient and numeric validation

Every gradient value is copied into wrapper-owned scratch and inspected before
any gradient write. NaN, positive infinity, and negative infinity each return
an error containing:

* the parameter index;
* the zero-based row and column; and
* the offending value.

The error form is:

```text
optimizer: gradient clipping non-finite gradient: parameter=<p> row=<r> col=<c> value=<value>
```

If copying or synchronizing a gradient fails, the error includes its parameter
index and wraps the matrix error. Earlier scratch copies are private and no
gradient has changed.

The ordered `math.Hypot` fold is overflow-safe for the public finite
`float32` input domain. The implementation nevertheless checks every
intermediate and the final norm defensively. A non-finite result returns
`optimizer: gradient clipping global norm is non-finite` with the parameter
and element position at which it became non-finite when applicable.

A computed scale that is NaN, infinite, zero, negative, or greater than one
returns `optimizer: gradient clipping scale is invalid` with the norm, limit,
and scale values. This defensive branch is not expected to be reachable from
a valid positive finite `float32` norm limit and a finite public-domain norm,
but it remains an explicit tested invariant.

No threshold, non-finite-gradient, zero-norm, overflow, or scale case silently
repairs invalid input:

* all-zero norm succeeds with scale one;
* finite extreme gradients succeed because the norm fold is overflow-safe;
* non-finite gradients fail;
* non-finite norm or invalid scale fails.

## Transformation and Failure Phases

The wrapper owns reusable `[]float32` scratch large enough for one flattened
snapshot of every parameter occurrence in the current call. The update phases
are:

1. Validate the wrapper, configuration, and complete parameter structure.
2. Size scratch and copy every gradient into its ordered segment.
3. Validate all copied values and calculate value-clipping and norm
   diagnostics.
4. Transform only the private scratch.
5. Copy transformed segments back to the corresponding gradient matrices in
   parameter order, unless the complete transformation is a no-op.
6. Call `base.Update(parameters)` exactly once.
7. Publish the observation described below.

Configuration, parameter, gradient-copy, non-finite, norm, and scale failures
occur before phase five. They leave all gradients unchanged, do not call the
base, and cannot advance Momentum, Adam, or custom optimizer state.

Matrix mutation can still return an operational coherence error after complete
numeric prevalidation. A failure while writing parameter occurrence `i`
leaves every successfully written earlier occurrence transformed and does not
write the failing or later occurrence. The base is not called and the prior
observation is preserved. Repeated parameter references expose the physical
matrix state produced by any completed earlier occurrence. The wrapper does
not claim an atomic multi-matrix host mutation or an infallible rollback for a
device/runtime error.

Under the documented non-concurrent CPU lifecycle, the preceding complete
host snapshot and the model's explicit fallback barrier make such a
transformation error exceptional. The distinction remains contractual rather
than being hidden.

After all transformations succeed, the base owns the update:

* A nil error from the base means the base update completed for observation
  purposes.
* The base remains responsible for parameter changes, optimizer state, and
  successful gradient reset.
* The wrapper does not reset gradients before delegation and never calls the
  base a second time.
* If the base returns an error, the wrapper does not restore transformed
  gradients, parameter values, or optimizer state.

Built-in optimizers validate before consuming gradients, and clipping
prevalidation failures never call them. Their Momentum velocity and Adam
moment/step state therefore cannot advance on a clipping failure. Once a base
call begins, only that base can define and enforce its error atomicity.
Arbitrary custom optimizers may mutate some parameters, gradients, or private
state before returning an error; the wrapper cannot roll those effects back.

A panic from a custom base is not converted to an error. It follows the
repository's existing panic behavior, and no observation publication is
promised for the interrupted call.

## Observation Contract

`Observation` reports the most recent attempt whose clipping transformation
phase completed and whose base call returned normally. The `available` result
distinguishes the zero observation from recorded state.

Fields mean:

* `ValueClipped` reports whether at least one value candidate was clamped.
  It is false when value clipping is disabled.
* `GlobalNorm` is the norm after hypothetical value clipping and before norm
  scaling. It is zero when norm clipping is disabled.
* `Scale` is the norm scale. It is one when norm clipping is disabled, the norm
  is zero, or the norm does not exceed `MaxNorm`.
* `BaseUpdateCompleted` is true exactly when `base.Update` returned nil. It
  does not independently inspect a custom base's parameter or reset behavior.

The lifecycle is:

| State or result | Observation |
| --- | --- |
| Construction, nil receiver, or zero value before an update | Zero observation, `available == false`. |
| Successful no-op clipping phase and successful base | Current diagnostics, `Scale == 1`, `BaseUpdateCompleted == true`, and `available == true`. |
| Applied value and/or norm clipping and successful base | Current diagnostics, `BaseUpdateCompleted == true`, and `available == true`. |
| Constructor failure | No wrapper is returned. |
| Wrapper, configuration, parameter, gradient-copy, non-finite, norm, or scale failure | Previous observation and availability remain unchanged. |
| Transformation write failure before the base | Previous observation and availability remain unchanged. |
| Clipping succeeds and the base returns an error | Current clipping diagnostics replace the previous observation, `BaseUpdateCompleted == false`, and `available == true`. |

Publishing after the base returns ensures an observer never sees a current
attempt while its base call is still running. A failed clipping attempt is
reported by `Update` and cannot masquerade as a new successful observation:
the last published value remains unchanged. A base failure is distinguishable
because it publishes the current diagnostics with `BaseUpdateCompleted`
false.

No error, callback, timestamp, counter, parameter identity, or gradient slice
is retained in the observation. Callers already receive the current error
from `Update`; adding it to persistent observation would complicate ownership
without improving the required clipping evidence.

## Regularization Composition

Wrapper nesting is explicit and is never detected or reordered internally.
Both orders are supported because they answer different caller questions.

### Canonical: regularize, then clip

The canonical bounded-update path places `Regularized` outside
`GradientClipping`:

```go
clipping, err := optimizer.NewGradientClipping(
	base,
	optimizer.GradientClippingConfig{MaxValue: 3},
)
if err != nil {
	return err
}

optimizerRule, err := optimizer.NewRegularized(clipping, l1, l2)
if err != nil {
	return err
}
```

Calls flow from the outside inward. `Regularized.Update` applies `l1` and then
`l2` to the accumulated data gradient before `GradientClipping.Update`
inspects it. The final gradient presented to `base` is therefore bounded after
the supported regularization additions.

For parameter values `[2, -3]`, data gradients `[3, 4]`, L1 coefficient `0.5`,
L2 coefficient `0.25`, value limit `3`, and SGD learning rate `0.1`:

```text
data gradient           [3, 4]
L1 gradient             [0.5, -0.5]
L2 gradient             [0.5, -0.75]
combined gradient       [4, 2.75]
clipped base gradient   [3, 2.75]
final parameter values  [1.7, -3.275]
```

This is the documented construction when a caller wants clipping to bound the
complete gradient consumed by optimizer update state.

If a regularizer returns an error before delegation, the clipping wrapper is
not called. `Regularized` owns any gradient additions already made by earlier
regularizers, and the clipping observation remains unchanged.

### Alternate: clip, then regularize

Placing `GradientClipping` outside `Regularized` clips only the incoming data
gradient:

```go
regularized, err := optimizer.NewRegularized(base, l1, l2)
if err != nil {
	return err
}

optimizerRule, err := optimizer.NewGradientClipping(
	regularized,
	optimizer.GradientClippingConfig{MaxValue: 3},
)
if err != nil {
	return err
}
```

With the same values:

```text
data gradient           [3, 4]
clipped data gradient   [3, 3]
L1 gradient             [0.5, -0.5]
L2 gradient             [0.5, -0.75]
base gradient           [4, 1.75]
final parameter values  [1.6, -3.175]
```

The regularization additions occur after clipping and may take the base
gradient outside the clipping limits. `GradientClippingObservation` describes
the data-gradient clipping phase, while `BaseUpdateCompleted` covers the
complete nested `Regularized.Update` return.

No constructor accepts regularizers directly, no wrapper is unwrapped, and no
combination is silently canonicalized. The same nesting rules apply to L1,
L2, multiple regularizers, and custom regularizers.

## Training Integration

### Ordinary dense and CNN training

`TrainBatch` performs prediction, reports the pre-update loss, calculates the
loss gradient, traverses layers backward, rebuilds the stable parameter slice,
and calls the configured optimizer once. A `GradientClipping` instance works
there without model changes. `Fit` reuses `TrainBatch` for every mini-batch.

Dense, convolution, and batch-normalization parameters retain their existing
layer and within-layer order. Activations, pooling, dropout, flattening,
losses, and metrics need no clipping awareness.

### Fixed-length and explicit-length RNN training

`SimpleRNN.Backward` remains full backpropagation through every configured
physical step. It continues to sum input-weight, recurrent-weight, and bias
gradients across rows and steps in that parameter order. `LastStep` remains the
fixed physical-final-step adapter.

`TrainBatchWithLengths` uses the same optimizer boundary after
`GatherLastValid` has routed the direct output gradient to each selected
last-valid step and the unchanged recurrent backward pass has propagated it.
`FitWithLengths` and the later opt-in `FitWithLengthViews` reuse that operation
for mini-batches. `FitWithViews` likewise reuses `TrainBatch`; scoped data
ownership does not change clipping order or observations.

Clipping therefore occurs after complete recurrent backward and before
Momentum velocity, Adam moments, SGD, or custom optimizer state consumes the
gradient. It does not truncate the recurrence, detach a graph, mask recurrent
computation, change a logical length, or carry hidden state.

### Learning-rate schedules

`Fit`, `FitWithLengths`, `FitWithViews`, and `FitWithLengthViews` call
`SetLearningRate` on the configured `Optimizer` before the epoch's
mini-batches. `GradientClipping` forwards that call to its base. `Regularized`
also forwards through its base, so either documented nesting order reaches
SGD, Momentum, Adam, or a custom base.

### Built-in and custom optimizers

SGD, Momentum, and Adam receive the transformed gradients through their
unchanged `Update` method. Clipping necessarily precedes velocity or moment
state because the wrapper delegates only after transformation.

A custom optimizer receives the same ordered parameter slice once. Its
learning-rate behavior, state allocation, error atomicity, parameter update,
and gradient reset remain its responsibility. No type assertion is required
for correctness.

## Ownership, Allocation, and Concurrency

`GradientClipping` may retain only:

* its base optimizer reference;
* its copied scalar configuration;
* one reusable private `[]float32` scratch snapshot; and
* one scalar observation plus its availability flag.

It does not retain the caller's parameter slice, parameter pointers, matrix
pointers beyond the synchronous base call, or matrix storage. It does not
replace a parameter's value or gradient matrix.

The first update may grow scratch to the total number of gradient elements
across parameter occurrences. Later updates with the same or a smaller total
reuse that capacity. Scratch may retain the largest observed capacity for the
wrapper's lifetime. After warm-up:

* value-only, norm-only, and combined clipping add zero allocations per update;
* `Config`, `Base`, and `Observation` allocate no storage; and
* either regularization nesting adds no clipping-owned steady-state
  allocation.

These are incremental wrapper guarantees. A custom base or regularizer may
allocate independently.

The ordered slice and row-major fold make clipping deterministic. No map,
goroutine, random source, seed, global mutable state, parameter type switch, or
layer type influences arithmetic. Equivalent ordered gradients,
configuration, base state, and parameter values produce equivalent
observations and updates under the repository's current CPU guarantees.

The wrapper follows the repository's existing non-concurrent object
lifecycle. Callers must not concurrently call, inspect, or mutate the same
wrapper, base, parameter, matrix, layer, or model. Distinct wrapper and model
instances may be used concurrently subject to their bases' normal contracts.

## Serialization and Reconstruction

`Sequential.Save` stores layer configuration and parameter values only.
Gradient clipping configuration, scratch, observations, base optimizer state,
regularizers, accumulated gradients, learning-rate schedules, and training
history are not part of format `neuralnetwork.sequential`, version `1`.

After loading, a caller reconstructs the desired optimizer stack:

```go
base, err := optimizer.NewAdam(0.001)
if err != nil {
	return err
}

optimizerRule, err := optimizer.NewGradientClipping(
	base,
	optimizer.GradientClippingConfig{MaxNorm: 5},
)
if err != nil {
	return err
}

// Use optimizerRule in Fit, FitWithLengths, or TrainBatch after loading.
```

No model format field, version change, migration, or load-time default is
introduced. Equivalent model values retain their existing serialized bytes.

## Metal Fallback

The initial implementation has no device-resident clipping kernel.
`model.supportsResidentUpdate` recognizes only a direct `*optimizer.SGD`.
A `*optimizer.GradientClipping` around SGD is therefore not eligible for the
resident plain-SGD update.

On an active Metal execution, existing model orchestration performs the
`BoundaryCPUFallback` barrier after backward and before invoking the wrapper.
The barrier completes the backward producer. Clipping then observes the
current gradients on the host, transforms them through host matrix writes, and
delegates. Those writes make the gradients host-current rather than pending
device update inputs, so a nested SGD follows its CPU implementation.

This path must:

* preserve the existing contextual model error if the barrier or update fails;
* download only the matrices required by the fallback;
* never masquerade as a direct `*SGD`;
* never bypass the synchronization boundary;
* leave direct unwrapped SGD's resident eligibility unchanged; and
* match the CPU clipping observation and parameter update within the existing
  Metal training tolerances.

The optimizer implementation does not change `supportsResidentUpdate`.
Metal-tagged integration tests supply the end-to-end fallback proof. A future
device clipping kernel or broader resident optimizer stack requires separate
measured design work.

## Compatibility and Non-Goals

Without a caller-constructed `GradientClipping` wrapper, all existing
behavior remains unchanged, including:

* dense, CNN, fixed-length RNN, and explicit-length RNN forward and backward
  behavior;
* summed gradient accumulation and loss-controlled scaling;
* SGD, Momentum, Adam, custom optimizer, and regularizer behavior;
* parameter enumeration and successful gradient reset;
* learning-rate schedule timing;
* pre-update training loss, callbacks, history, and early stopping;
* caller-controlled randomness and determinism;
* version `1` serialization; and
* resident direct-SGD eligibility and CPU fallback behavior.

The accepted addition does not include:

* hidden clipping, damping, or recurrent-only defaults;
* per-parameter, per-layer, parameter-group, or adaptive norm scopes;
* truncated backpropagation, graph detachment, accumulation-window scaling, or
  recurrent state carry;
* masking, ragged storage, changed `LastStep`, or changed `GatherLastValid`;
* per-layer learning rates or general parameter groups;
* mixed precision, loss scaling, quantization, or non-finite repair;
* callbacks, logging, printing, or additions to `TrainMetrics`;
* optimizer, clipping, observation, or scratch serialization; or
* device-resident clipping kernels or broader resident optimizer support.

## Implementation File Map

The implemented boundary and integration proof are located in:

```text
optimizer/gradient_clipping_config.go
    GradientClippingConfig

optimizer/gradient_clipping_observation.go
    GradientClippingObservation

optimizer/gradient_clipping.go
    constructor, wrapper, Optimizer methods, validation, arithmetic,
    scratch lifecycle, and observation publication

optimizer/gradient_clipping_test.go
    public construction, arithmetic, ordering, observations, failures,
    delegation, built-in optimizers, regularization, and determinism

optimizer/gradient_clipping_internal_test.go
    package-private defensive norm-overflow and invalid-scale branches

optimizer/nil_receiver_test.go
    nil and zero-value accessor/setter coverage

optimizer/allocation_test.go
    warmed value, norm, combined, and regularized wrapper allocations

optimizer/optimizer_benchmark_test.go
    representative unclipped and clipped update costs

model/gradient_clipping_test.go
model/rnn_allocation_internal_test.go
    ordinary, scheduled, recurrent, explicit-length, error, and allocation
    integration

model/metal_training_test.go
    Metal-tagged clipping fallback, transfer, observation, and CPU parity

README.md
docs/rnn.md
docs/rnn-design.md
docs/sequence-lengths-design.md
docs/metal.md
docs/v1-api-review.md
    implemented construction guidance and additive API inventory

examples/rnn/main.go
examples/rnn/main_test.go
    deterministic fixed-length construction and training example
```

No production model or layer change was required because every training method
already shares the ordered optimizer boundary described above.

## Test Matrix

The completed test matrix covers:

| Area | Required proof |
| --- | --- |
| Constructors | Nil base; both controls disabled; zero-disabled counterpart; negative, NaN, and both infinities for each enabled limit; copied configuration. |
| Receivers | Nil receiver and invalid zero value for update, learning-rate forwarding, base, configuration, and observation access. |
| Parameter validation | Empty list; nil parameter; nil/invalid values; nil/invalid gradient; shape mismatch; total element-count overflow; parameter-index context. |
| Numeric failures | Gradient NaN and both infinities with row/column context; overflow-safe extreme finite values; defensive non-finite norm and invalid scale; all failures before mutation and base invocation where promised. |
| Value arithmetic | Positive, negative, zero, negative zero, exact bounds, inside bounds, and outside bounds across multiple rows and parameters. |
| Norm arithmetic | Zero, below limit, exact limit, above limit, global multi-parameter scope, ordered row-major fold, and hand-calculated scale. |
| Combined arithmetic | A case such as `[6, 8]` that distinguishes value-then-norm from norm-then-value. |
| Base order | Exactly one base call with the original ordered slice and transformed gradients; successful base-owned reset; injected base error and current failed-base observation. |
| Built-in state | Hand-calculated first SGD, Momentum, and Adam steps proving clipping precedes parameter, velocity, and moment updates; clipping failures do not advance state. |
| Regularization | L1, L2, multiple regularizers, both explicit wrapper orders, canonical combined-gradient bounds, and reversed-order results that may exceed a limit. |
| Observations | Before update; successful no-op; value clip; norm clip; combined clip; prevalidation failure preservation; transformation error preservation; base failure publication; scalar-copy ownership. |
| Learning rates | Direct forwarding and `Fit`/`FitWithLengths` schedule changes through both wrapper orders. |
| Determinism | Repeated equivalent ordered gradients, wrapper state, and bases produce equivalent observations and updates without hidden randomness or maps. |
| Allocation | Zero incremental warmed allocations for value, norm, combined, duplicate-reference, and both regularized compositions. |
| Model integration | `TrainBatch`, `Fit`, `TrainBatchWithLengths`, `FitWithLengths`, `FitWithViews`, and `FitWithLengthViews` use the shared optimizer boundary and retain contextual errors and training-mode cleanup. |
| Recurrent acceptance | One deterministic full-BPTT `SimpleRNN` update is directly bounded; matching unwrapped training retains the existing update; mixed logical lengths route gradients unchanged before clipping. |
| Metal | Wrapped SGD takes the explicit CPU fallback, observes/downloads current gradients, matches CPU results within tolerance, and does not change direct-SGD residency. |
| Persistence | Existing ANN, CNN, RNN, and gather version `1` bytes and load behavior remain unchanged; callers reconstruct clipping after load. |

Focused arithmetic tests use hand calculations and
`internal/testutil` with `float32`-appropriate tolerances. The recurrent
acceptance test asserts the one-step parameter update and observation directly;
it does not use a loss curve as evidence of clipping.

## Alternatives Rejected

### Model or fit configuration

Adding clipping fields to `FitConfig`, `SequenceFitConfig`, or `Sequential`
would duplicate behavior already available at the optimizer boundary and risk
changing stable structures or unkeyed literals. It would also leave direct
custom calls to `Optimizer.Update` without the same control.

### Changing Optimizer

Adding a clipping method to `Optimizer` would break every custom
implementation. The wrapper composes through the accepted three-method
interface instead.

### Layer or RNN clipping

Clipping inside `SimpleRNN.Backward`, another layer, or model reverse traversal
would make order depend on layer boundaries, miss non-RNN parameters, and
change gradients observable before optimizer update. Full layer gradients
remain unchanged.

### Per-parameter norm

Per-parameter clipping does not bound the norm consumed across the complete
model update and makes the result depend on parameter partitioning. Global
norm is the accepted initial training control. Parameter groups remain a
separate roadmap decision.

### Norm before value

Norm-first composition makes large outliers determine the shared scale before
the explicit value bound is applied. Value-first composition gives each
enabled control a clear phase and is frozen by the `[6, 8]` example.

### Automatic regularizer reordering

Inspecting wrapper types and moving regularization internally would make
custom wrappers ambiguous and change caller-authored update order. Explicit
nesting supports both meanings without hidden type-specific behavior.

### Callback observations

A callback would add lifetime, error, reentrancy, and allocation behavior for
data that fits in four scalar fields. The value observation accessor satisfies
the required caller-owned observability without changing model metrics.

### Silent non-finite repair

Clamping infinity or replacing NaN can conceal an invalid loss, activation, or
gradient source. Failing before mutation is the accepted behavior. General
non-finite training recovery and reduced-precision loss scaling remain
separate milestones.
