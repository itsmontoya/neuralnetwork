# RNN Architecture and Public Contract

Status: implemented additive post-v1 contract.

This document records the public and behavioral contract for the initial
recurrent neural network path. The milestone is additive to the reviewed
dense-network v1 API and the implemented post-v1 CNN API. It does not change
`layer.Layer`, `model.Sequential`, `data.Dataset`, losses, metrics, optimizers,
or the physical representation of `matrix.Matrix`.

## Milestone

The supported model shape is:

```text
flattened fixed-length sequence rows -> SimpleRNN -> LastStep -> Dense -> output activation
```

Callers construct, train, evaluate, save, and load that model through the
existing APIs. The recurrent implementation is a clear pure-Go CPU reference
path. Correctness and deterministic behavior take priority over kernel
optimization.

## Logical and Physical Layout

A matrix row is one complete fixed-length sequence. Sequences use logical
time-major `TF` order, so a batched matrix is logically `NTF`, with the matrix
row serving as `N`. The physical input shape is:

```text
[batch, steps*featureSize]
```

Within a row, the flattened column for `(step, feature)` is:

```text
step*FeatureSize + feature
```

The batch dimension is never included in this calculation. `data.Dataset`
continues to store one sample per row, so its existing batching behavior keeps
each sequence intact without knowing about logical sequence dimensions.
Callers provide already flattened, fixed-length rows; this milestone adds no
sequence-specific dataset or batching API.

Steps, feature sizes, and hidden sizes are positive. Every flattened size must
fit in an `int`, and validation detects overflow before multiplication or
allocation. Matrix inputs must have exactly `SequenceShape.Size()` columns.
The flattened offset calculation remains private; the formula above is the
public indexing contract.

## Public API

All new symbols are in package `layer`. Constructors return errors with `layer`
context when dimensions, dependencies, or derived shapes are invalid. The zero
values of the new shape, configuration, and layer types are invalid. Valid
shape and configuration values are immutable after construction.

### SequenceShape

```go
func NewSequenceShape(steps, featureSize int) (shape SequenceShape, err error)

type SequenceShape struct { /* unexported fields */ }

func (s SequenceShape) Steps() (steps int)
func (s SequenceShape) FeatureSize() (featureSize int)
func (s SequenceShape) Size() (size int)
```

`NewSequenceShape` validates positive dimensions and computes
`steps*featureSize` with overflow checks. `SequenceShape` contains only
comparable value fields, so callers use Go's `==` and `!=` operators for exact
shape equality. No public offset helper is added.

### SimpleRNNConfig and SimpleRNN

```go
func NewSimpleRNNConfig(
	inputShape SequenceShape,
	hiddenSize int,
) (config SimpleRNNConfig, err error)

type SimpleRNNConfig struct { /* unexported fields */ }

func (c SimpleRNNConfig) InputShape() (shape SequenceShape)
func (c SimpleRNNConfig) OutputShape() (shape SequenceShape)
func (c SimpleRNNConfig) HiddenSize() (hiddenSize int)

func NewSimpleRNN(
	config SimpleRNNConfig,
	inputInitializer WeightInitializer,
	recurrentInitializer WeightInitializer,
) (out *SimpleRNN, err error)

type SimpleRNN struct { /* unexported fields */ }

func (r *SimpleRNN) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error)
func (r *SimpleRNN) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error)
func (r *SimpleRNN) Config() (config SimpleRNNConfig)
func (r *SimpleRNN) InputShape() (shape SequenceShape)
func (r *SimpleRNN) OutputShape() (shape SequenceShape)
func (r *SimpleRNN) InputWeights() (weights *optimizer.Parameter)
func (r *SimpleRNN) RecurrentWeights() (weights *optimizer.Parameter)
func (r *SimpleRNN) Biases() (biases *optimizer.Parameter)
func (r *SimpleRNN) Parameters() (parameters []*optimizer.Parameter)
func (r *SimpleRNN) AppendParameters(parameters []*optimizer.Parameter) (out []*optimizer.Parameter)
func (r *SimpleRNN) ResetGradients() (err error)
```

`NewSimpleRNNConfig` revalidates the input shape, requires a positive hidden
size, and derives the equivalent of
`NewSequenceShape(inputShape.Steps(), hiddenSize)` as the output shape. It
rejects an output flattened size that cannot be represented.
`NewSimpleRNN` revalidates the configuration because callers can pass its
invalid zero value. It rejects nil initializers and initializer results with
invalid or unexpected shapes.

`Parameters` and `AppendParameters` use stable input-weight,
recurrent-weight, bias order. `AppendParameters` does not retain the supplied
slice. `ResetGradients` clears all three accumulated parameter gradients.

### LastStep

```go
func NewLastStep(inputShape SequenceShape) (out *LastStep, err error)

type LastStep struct { /* unexported fields */ }

func (l *LastStep) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error)
func (l *LastStep) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error)
func (l *LastStep) InputShape() (shape SequenceShape)
func (l *LastStep) OutputSize() (size int)
```

`OutputSize()` equals `InputShape().FeatureSize()`. `LastStep` exposes no
general slice, gather, reshape, or sequence-to-sequence adapter. It has no
parameters and no training mode.

## Recurrent Forward Semantics

`SimpleRNN` implements a plain Elman recurrence with a fixed tanh activation.
For batch row `n`, step `t`, and hidden value `j`:

```text
hidden[n, -1, j] = 0

hidden[n, t, j] = tanh(
    biases[0, j]
    + sum(feature, input[n, t, feature] * inputWeights[feature, j])
    + sum(previous, hidden[n, t-1, previous] * recurrentWeights[previous, j])
)
```

Input and output sequence values use the time-major flattened formula from the
layout section. The input matrix shape is `[batch, inputShape.Size()]`. The
output matrix shape is `[batch, outputShape.Size()]`, and its flattened column
for `(step, hidden)` is:

```text
step*HiddenSize + hidden
```

Every valid `Forward` starts every batch row from an all-zero hidden state.
State is not shared between rows and is not carried between samples, batches,
training and evaluation, or successive forward calls. Training and evaluation
therefore use the same stateless recurrence. `SimpleRNN` always returns every
hidden step so another recurrent layer can consume the output and a gradient
can reach any emitted step.

The tanh operation is intrinsic to `SimpleRNN`; it is not a configurable
`activation.Activation` and callers do not add a second tanh to obtain the
documented recurrence.

## Parameters, Initialization, and Determinism

Trainable values use these matrix layouts:

```text
inputWeights     [inputFeatureSize, hiddenSize]
recurrentWeights [hiddenSize, hiddenSize]
biases           [1, hiddenSize]
```

Each input-weight row corresponds to one input feature, and each column
corresponds to one hidden value. Each recurrent-weight row corresponds to a
previous hidden value, and each column corresponds to the current hidden value.
The bias row is shared across batch rows and time steps.

On successful construction, `NewSimpleRNN` invokes the initializers exactly
once each and in this order:

```text
inputInitializer(inputFeatureSize, hiddenSize)
recurrentInitializer(hiddenSize, hiddenSize)
```

Those arguments are both the existing `WeightInitializer` dimensions and the
required stored matrix shapes. Biases start at zero. The layer does not seed,
retain, or read any hidden random source. All randomness remains controlled by
the caller through the two initializers, so equivalent caller-seeded
initializers produce equivalent parameters and predictions.

## Full Backpropagation Through Time

`Backward` traverses every configured step in reverse order. At step `t`, the
hidden gradient is the sum of the direct gradient for emitted output `t` and
the recurrent hidden gradient propagated from step `t+1`. The tanh derivative
is applied to that combined gradient before calculating the input gradient,
parameter gradients, and recurrent gradient for step `t-1`.

For each step, the layer:

* Adds the direct and recurrent hidden gradients.
* Multiplies by `1-hidden[t]*hidden[t]` for the tanh derivative.
* Accumulates the outer product of `input[t]` and that result into the input
  weight gradient.
* Accumulates the outer product of `hidden[t-1]` and that result into the
  recurrent weight gradient; the zero initial state contributes zero at the
  first step.
* Accumulates the result into the bias gradient.
* Produces the input gradient through the input weights and propagates the
  prior-hidden gradient through the recurrent weights.

Parameter gradients sum across every batch row and time step. They also
accumulate across successful `Backward` calls until `ResetGradients` is called.
The layer performs no implicit mean scaling, truncation, graph detachment, or
gradient clipping. Existing losses control scaling through the gradient passed
to `Backward`.

`Backward` before a valid `Forward` returns a descriptive error. After a valid
forward pass, the output gradient must have the same batch size and exactly
`OutputShape().Size()` columns. A later valid forward pass replaces the cached
input and hidden history used by backward. The returned input gradient has
physical shape `[batch, InputShape().Size()]`.

## Last-Step Semantics

`LastStep.Forward` accepts `[batch, inputShape.Size()]` and copies the final
contiguous feature segment from each row into `[batch, OutputSize()]`. For row
`n`, its values are:

```text
output[n, feature] = input[n, (Steps-1)*FeatureSize + feature]
```

It does not reorder feature values or merge batch rows. `Backward` requires a
preceding valid forward pass and an output gradient matching the most recent
batch size and `OutputSize()`. It returns `[batch, inputShape.Size()]`, fills
every earlier step with zero, and copies the supplied gradient into the final
step.

For a one-step input, the only step is also the final step. Forward and backward
therefore preserve every numeric value while still returning independent
layer-owned matrices; they are validating copies, not aliases or special-case
views.

## Ownership and Scratch Results

`SimpleRNN` copies the caller input needed by backward and caches the hidden
values from the most recent valid forward pass. It does not retain
caller-owned matrix storage, so mutating the input after `Forward` cannot alter
the subsequent gradient calculation. `LastStep` retains only the most recent
valid batch size needed to validate backward and does not retain caller-owned
matrix storage.

Valid forward outputs and backward input gradients from both layers do not
alias their arguments. Results are layer-owned scratch matrices, so a later
call on the same layer may reuse and overwrite a previously returned result.
Callers that require longer ownership must clone it. Invalid calls do not
establish a valid backward cache.

## Interoperability

The existing `layer.Layer` matrix contract remains unchanged. A
`SequenceShape` supplies logical interpretation without introducing another
runtime container. `SimpleRNN` output can feed another `SimpleRNN` configured
with its `OutputShape()`. `LastStep` provides the explicit transition from an
all-steps recurrent result to an existing `Dense` layer.

Existing elementwise activation and dropout layers can operate on flattened
sequence values, although `SimpleRNN` already applies its internal tanh.
Existing batch normalization treats every flattened step-feature column as a
separate feature; it is not recurrent or temporal batch normalization.
Existing `Dense` transforms the complete physical row and is not a
time-distributed projection.

The initial data path is `data.Dataset` populated with pre-flattened sequence
rows. Existing batching selects complete rows and therefore complete
sequences. Losses and metrics see the final matrix output and require no
RNN-specific changes. `SimpleRNN` exposes `optimizer.Parameter` values and
participates in the private sequential parameter-enumeration interfaces;
`LastStep` contributes no parameters.

## Serialization Compatibility

RNN support extends the existing `neuralnetwork.sequential` JSON format at
version `1`; it does not introduce version `2`. The additive layer type names
are `simple_rnn` and `last_step`.

A `simple_rnn` layer record stores `steps`, `feature_size`, `hidden_size`,
`input_weights`, `recurrent_weights`, and `biases`. A `last_step` layer record
stores `steps` and `feature_size`. These are constructor inputs and trainable
values only. Output shapes are derived and validated rather than serialized.

Serialization does not store accumulated gradients, optimizer state, input
caches, hidden histories, scratch storage, the most recent batch size, carried
state, or other forward-pass state. A loaded `SimpleRNN` has zero gradients and
requires a new forward pass before backward, like a newly constructed layer.
A loaded `LastStep` likewise requires a new forward pass before backward.

Compatibility is:

* Existing ANN-only and CNN-only version `1` documents remain byte-stable when
  saved and continue to load with the extended reader.
* Existing ANN and CNN programs and public APIs remain source-compatible.
* An older version `1` reader can still read documents containing only layer
  types it knows. When given an RNN document, it rejects the unknown RNN layer
  type; it cannot load that model and must not silently substitute or skip the
  layer.
* Unsupported custom layers and unknown serialized layer types continue to
  fail with layer-index context.

Keeping version `1` treats the new layer variants as additive vocabulary while
preserving the meaning and encoding of every existing field and layer. New RNN
fields are omitted from non-RNN records. A future incompatible schema or
changed meaning requires a new format version and explicit migration handling.

## Supported and Deferred Features

The initial milestone supports:

* Batched, fixed-length, time-major sequence rows in row-major `float32`
  matrices.
* A stateless, zero-initialized Elman `SimpleRNN` with fixed tanh activation and
  all-steps output.
* Full backpropagation through every configured step and summed parameter
  gradients.
* A validating `LastStep` adapter for many-to-one models.
* Stacking recurrent layers and composition with existing dense layers,
  elementwise layers, datasets, losses, metrics, optimizers, training, and
  sequential serialization.
* A clear pure-Go CPU reference implementation with caller-controlled
  initialization randomness.

The following remain deferred:

* Variable-length, ragged, padded-and-masked, or packed sequence inputs.
  Padding supplied by a caller is treated as ordinary input in this milestone.
* Learned or caller-provided initial state, returned final state, state carry
  between calls, stateful training, and streaming inference.
* Truncated backpropagation through time, explicit graph detachment, and
  unbounded sequences.
* Configurable recurrent activations and recurrent or variational dropout.
* LSTM, GRU, bidirectional recurrence, encoder-decoder models, attention, and
  transformer layers.
* Learned embeddings, tokenization, vocabulary management, sequence-specific
  data loaders, and external sequence dataset integrations.
* Time-distributed dense projection, masked sequence losses and metrics, and
  richer sequence-to-sequence output adapters.
* A general automatic-differentiation graph.
* SIMD, Metal, goroutine-parallel, distributed, or third-party recurrent
  kernels.

These requirements do not justify a generic tensor abstraction, a replacement
sequence container, or any change to the stable `layer.Layer` matrix contract.
The explicit flattened fixed-length representation supplies all structure the
initial recurrent path needs. A broader container would be a repository-wide
API decision and requires separate maintainer direction.
