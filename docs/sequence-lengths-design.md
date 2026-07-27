# Explicit Sequence Lengths and Last-Valid Selection

Status: implemented.

This document freezes the additive contract for associating one positive
logical length with every padded, time-major sequence row and selecting the
last valid emitted step. It is the implementation contract for
[ROADMAP Item 1](../ROADMAP.md#1-add-explicit-sequence-lengths-and-safe-last-valid-selection).
The `data.SequenceLengths`, `data.SequenceDataset`, `data.SequenceBatch`,
`layer.GatherLastValid`, and length-aware `model.Sequential` APIs in this
document are implemented, including version `1` persistence. The complete
caller workflow is documented in [rnn.md](rnn.md) and demonstrated by the
runnable [explicit-length RNN example](../examples/rnn_lengths/main.go).

The design preserves the existing `matrix.Matrix`, `layer.Layer`,
`data.Dataset`, `data.Batch`, `model.FitConfig`, and `model.Sequential`
contracts. Existing callers continue to treat padding as ordinary input.
Length-aware behavior is available only through the additive APIs below.

## Supported Flow

The supported many-to-one graph is:

```text
padded flattened rows
    -> SimpleRNN
    -> zero or more additional SimpleRNN layers
    -> GatherLastValid
    -> Dense
    -> output activation
```

`SimpleRNN` remains unchanged. It still evaluates every configured physical
step from a zero initial hidden state and emits every hidden step.
`GatherLastValid` is the first boundary that receives logical lengths. It
selects one emitted step per row and routes the direct backward gradient only
to that selected step. The unchanged recurrent backward pass then propagates
through the corresponding valid prefix.

This contract does not mask recurrent computation. Padded suffix values may
affect hidden values in the suffix, but those suffix hidden values are not
selected and receive no direct gradient from `GatherLastValid`.

## Logical and Physical Layout

For a configured sequence shape with `S` steps and feature size `F`, values
remain a matrix with physical shape:

```text
[N, S*F]
```

The flattened column for `(step, feature)` remains:

```text
step*F + feature
```

Row `n` has one logical length `L[n]`. Valid values are the prefix at steps
`0..L[n]-1`. Values at steps `L[n]..S-1` are a padded suffix and remain
ordinary caller-owned matrix values.

The selected output has shape `[N, F]`:

```text
output[n, feature] =
    input[n, (L[n]-1)*F+feature]
```

Backward accepts `[N, F]`, returns `[N, S*F]`, places the supplied row gradient
at step `L[n]-1`, and writes zero at every other step.

## Public API

### Validated lengths

Package `data` owns the validated integer carrier because it already owns row
alignment, batching, shuffling, and splitting:

```go
func NewSequenceLengths(
	steps int,
	values []int,
) (out *SequenceLengths, err error)

type SequenceLengths struct { /* unexported fields */ }

func (l *SequenceLengths) Validate() (err error)
func (l *SequenceLengths) Steps() (steps int)
func (l *SequenceLengths) SampleCount() (samples int)
func (l *SequenceLengths) Values() (values []int, err error)
func (l *SequenceLengths) ValuesInto(destination []int) (err error)
func (l *SequenceLengths) SelectRowsInto(
	indexes []int,
	destination []int,
) (err error)
```

`NewSequenceLengths` copies `values`. `Values` returns a copy.
`ValuesInto` and `SelectRowsInto` fully overwrite caller-owned destinations
after validating every argument and retain neither destination nor indexes.
Those destination forms let length-aware fitting reuse scratch storage.

`SequenceLengths` is immutable after construction. The zero value and a nil
receiver are invalid. Returning or retaining a `*SequenceLengths` therefore
does not expose mutable internal storage.

### Aligned supervised data

Package `data` adds parallel, focused supervised types rather than changing
`Dataset` or `Batch`:

```go
func NewSequenceDataset(
	inputs,
	targets *matrix.Matrix,
	lengths *SequenceLengths,
) (out *SequenceDataset, err error)

type SequenceDataset struct { /* unexported fields */ }

func (d *SequenceDataset) Validate() (err error)
func (d *SequenceDataset) Inputs() (inputs *matrix.Matrix, err error)
func (d *SequenceDataset) Targets() (targets *matrix.Matrix, err error)
func (d *SequenceDataset) Lengths() (lengths *SequenceLengths, err error)
func (d *SequenceDataset) InputsInto(inputs *matrix.Matrix) (err error)
func (d *SequenceDataset) TargetsInto(targets *matrix.Matrix) (err error)
func (d *SequenceDataset) LengthsInto(lengths []int) (err error)
func (d *SequenceDataset) SelectRowsInto(
	indexes []int,
	inputs,
	targets *matrix.Matrix,
	lengths []int,
) (err error)
func (d *SequenceDataset) SampleCount() (samples int)
func (d *SequenceDataset) InputSize() (features int)
func (d *SequenceDataset) TargetSize() (values int)
func (d *SequenceDataset) Steps() (steps int)
func (d *SequenceDataset) Batches(
	batchSize int,
	random *rand.Rand,
) (batches []*SequenceBatch, err error)
func (d *SequenceDataset) Split(
	testFraction float32,
	random *rand.Rand,
) (train, test *SequenceDataset, err error)
```

`SequenceDataset` owns copies of all three aligned inputs. Its matrix
accessors return matrix copies, and `Lengths` returns an independent validated
length value. The `Into` methods copy into caller-owned destinations without
retaining them. The input width must be evenly divisible by `lengths.Steps()`,
so the stored rows describe a positive feature count at every configured
step.

`SequenceBatch` has no public constructor. `SequenceDataset.Batches` creates
it with owned aligned values:

```go
type SequenceBatch struct { /* unexported fields */ }

func (b *SequenceBatch) Inputs() (inputs *matrix.Matrix, err error)
func (b *SequenceBatch) Targets() (targets *matrix.Matrix, err error)
func (b *SequenceBatch) Lengths() (lengths *SequenceLengths, err error)
func (b *SequenceBatch) InputsInto(inputs *matrix.Matrix) (err error)
func (b *SequenceBatch) TargetsInto(targets *matrix.Matrix) (err error)
func (b *SequenceBatch) LengthsInto(lengths []int) (err error)
func (b *SequenceBatch) SampleCount() (samples int)
func (b *SequenceBatch) Steps() (steps int)
```

The sequence-specific types do not embed, expose, or widen `Dataset` and
`Batch`. They are supervised row-alignment types, not a general runtime
sequence container.

### Last-valid boundary

Package `layer` adds one concrete parameter-free adapter:

```go
func NewGatherLastValid(
	inputShape SequenceShape,
) (out *GatherLastValid, err error)

type GatherLastValid struct { /* unexported fields */ }

func (g *GatherLastValid) Forward(
	input *matrix.Matrix,
) (output *matrix.Matrix, err error)
func (g *GatherLastValid) Backward(
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error)
func (g *GatherLastValid) ForwardWithLengths(
	input *matrix.Matrix,
	lengths []int,
) (output *matrix.Matrix, err error)
func (g *GatherLastValid) BackwardWithLengths(
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error)
func (g *GatherLastValid) InputShape() (shape SequenceShape)
func (g *GatherLastValid) OutputSize() (size int)
```

`GatherLastValid` implements `layer.Layer` so it can remain in an ordinary
`Sequential` graph. Its ordinary `Forward` and `Backward` methods return
contextual errors directing callers to the length-aware methods. They never
fall back to the physical last step. `Forward` also invalidates any earlier
length-aware forward state.

`ForwardWithLengths` takes a slice so model fitting can pass its reusable
selected-length scratch without allocating a carrier for every mini-batch. It
validates and copies the slice before using or retaining it. Direct callers
may pass `lengths.Values()` or another integer slice; caller mutation after
success cannot affect backward.

The model integration recognizes this concrete adapter through a private
interface. No general metadata-bearing `Layer`, side-input registry, or
length-aware layer hierarchy is added.

### Length-aware model operations

`Sequential` adds explicit operations and leaves every existing method
unchanged:

```go
func (s *Sequential) PredictWithLengths(
	input *matrix.Matrix,
	lengths *data.SequenceLengths,
) (output *matrix.Matrix, err error)

func (s *Sequential) BackwardWithLengths(
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error)

func (s *Sequential) TrainBatchWithLengths(
	input,
	targets *matrix.Matrix,
	lengths *data.SequenceLengths,
	lossFunc loss.Loss,
	optimizerRule optimizer.Optimizer,
) (metrics TrainMetrics, err error)

func (s *Sequential) FitWithLengths(
	trainingData *data.SequenceDataset,
	config SequenceFitConfig,
) (history TrainingHistory, err error)
```

`SequenceFitConfig` mirrors the existing fit controls while giving validation
data the aligned sequence-specific type:

```go
type SequenceFitConfig struct {
	Epochs               int
	BatchSize            int
	Shuffle              bool
	Random               *rand.Rand
	Optimizer            optimizer.Optimizer
	LearningRateSchedule optimizer.LearningRateSchedule
	EarlyStopping        *EarlyStopping
	Loss                 loss.Loss
	ValidationData       *data.SequenceDataset
	Accuracy             AccuracyFunc
	Callback             FitCallback
}
```

The fields have the same meaning, validation, ordering, and callback behavior
as their `FitConfig` counterparts. A distinct type avoids changing
`FitConfig`, including unkeyed composite-literal compatibility, and makes it
impossible to attach a matrix-only validation dataset accidentally.

The length-aware model methods require exactly one `GatherLastValid` layer.
They reject graphs with no selector or more than one selector before running a
layer. The carrier's `Steps()` must equal the selector input shape's
`Steps()`. The same one-length-per-row value reaches the selector after any
number of same-step-count `SimpleRNN` layers.

## Construction and Invocation Examples

### Length construction and prediction

```go
lengths, err := data.NewSequenceLengths(3, []int{2, 3})
if err != nil {
	return err
}

inputs, err := matrix.FromSlice(2, 6, []float32{
	1, 0, 2, 0, 99, 99,
	3, 0, 4, 0, 5, 0,
})
if err != nil {
	return err
}

predictions, err := network.PredictWithLengths(inputs, lengths)
if err != nil {
	return err
}

outputGradient, err := matrix.New(predictions.Rows(), predictions.Cols())
if err != nil {
	return err
}

_, err = network.BackwardWithLengths(outputGradient)
if err != nil {
	return err
}
```

The first row selects recurrent step one even though its physical row has a
third padded step. The second row selects step two.

### One training batch

```go
metrics, err := network.TrainBatchWithLengths(
	inputs,
	targets,
	lengths,
	loss.MeanSquaredError{},
	optimizerRule,
)
if err != nil {
	return err
}
_ = metrics
```

The method owns the prediction/backward association for the complete training
operation. Callers do not set lengths on the model or layer before invoking
it.

### Training and validation data

```go
trainingData, err := data.NewSequenceDataset(
	trainingInputs,
	trainingTargets,
	trainingLengths,
)
if err != nil {
	return err
}

validationData, err := data.NewSequenceDataset(
	validationInputs,
	validationTargets,
	validationLengths,
)
if err != nil {
	return err
}

history, err := network.FitWithLengths(trainingData, model.SequenceFitConfig{
	Epochs:         20,
	BatchSize:      8,
	Shuffle:        true,
	Random:         rand.New(rand.NewSource(101)),
	Optimizer:      optimizerRule,
	Loss:           loss.MeanSquaredError{},
	ValidationData: validationData,
	Accuracy:       metric.MeanSquaredError{}.Value,
})
if err != nil {
	return err
}
_ = history
```

Training mini-batches, full training evaluation, and full validation
evaluation each receive their corresponding aligned lengths.

### Stacked recurrence

```go
firstConfig, err := layer.NewSimpleRNNConfig(inputShape, 6)
if err != nil {
	return nil, err
}
first, err := layer.NewSimpleRNN(
	firstConfig,
	layer.XavierUniformWeights(random),
	layer.XavierUniformWeights(random),
)
if err != nil {
	return nil, err
}

secondConfig, err := layer.NewSimpleRNNConfig(first.OutputShape(), 4)
if err != nil {
	return nil, err
}
second, err := layer.NewSimpleRNN(
	secondConfig,
	layer.XavierUniformWeights(random),
	layer.XavierUniformWeights(random),
)
if err != nil {
	return nil, err
}

gather, err := layer.NewGatherLastValid(second.OutputShape())
if err != nil {
	return nil, err
}
output, err := layer.NewDense(
	gather.OutputSize(),
	2,
	layer.XavierUniformWeights(random),
)
if err != nil {
	return nil, err
}

network, err := model.NewSequential(first, second, gather, output)
if err != nil {
	return nil, err
}
```

Both recurrent layers preserve the configured step count. Lengths are supplied
once to `PredictWithLengths`, `TrainBatchWithLengths`, or `FitWithLengths` and
are consumed only at `gather`.

### Save and load

```go
var document bytes.Buffer
if err := network.Save(&document); err != nil {
	return err
}

restored, err := model.LoadSequential(&document)
if err != nil {
	return err
}

predictions, err := restored.PredictWithLengths(inputs, lengths)
if err != nil {
	return err
}
_ = predictions
```

Lengths are invocation and dataset data. They are not stored in the model
document and must be supplied on the first length-aware call after loading.

## Validation Contract

### Integer and range rules

Lengths and the maximum step count use Go `int`.

* `steps` must be positive.
* The values slice must contain at least one value.
* Every value must be in the inclusive range `[1, steps]`.
* Zero length is rejected. A defined zero-length output or gradient belongs to
  the later fully masked/ragged milestone.
* Non-integral, NaN, and infinite values are unrepresentable by the API.
* Values parsed from external text must be converted to `int` by the caller
  without truncation before construction; this milestone adds no float or CSV
  length parser.
* Offset multiplication occurs only after `SequenceShape` and length bounds
  are validated. `SequenceShape.Size()` already proves `steps*featureSize`
  fits in `int`, so `(length-1)*featureSize` cannot overflow.

`NewSequenceLengths` errors use `data: sequence lengths` context and include
the offending row and value. Shape-dependent selector errors use
`layer: gather last valid` context. Model errors wrap those errors with the
operation and layer index where applicable.

### Count and shape rules

`NewSequenceLengths` validates values but has no matrix row count to compare.
The exact count check happens at the first association boundary:

* `NewSequenceDataset` requires `lengths.SampleCount() == inputs.Rows()` after
  validating the input/target row pairing and before copying any argument.
* `PredictWithLengths` and `TrainBatchWithLengths` require the same count as
  the direct input rows before beginning forward work.
* `GatherLastValid.ForwardWithLengths` independently requires the same count
  as its current input rows before indexing or establishing backward state.
* Selection, batching, splitting, fit evaluation, and validation preserve the
  count by applying one row-index list to inputs, targets, and lengths.

The data association also requires `inputs.Cols()%lengths.Steps() == 0`.
The model association requires `lengths.Steps()` to equal the configured
selector step count. `GatherLastValid` independently requires input width
`InputShape().Size()`.

Matrices require positive row and column counts, and `SequenceLengths` rejects
an empty value slice. Therefore no reviewed public path can construct or
produce an empty dataset, batch, split, prediction request, or selector call.
The final mini-batch may be partial but contains at least one row.

Top-level matrix validity and all length, count, carrier-step, destination,
and graph-compatibility validation precede indexing, output writes, layer
forward work, parameter-gradient work, or optimizer updates. Individual
layers and losses retain ownership of their existing width and target-column
validation as traversal reaches them. Preflight validation errors return no
public result and do not mutate caller arguments, parameters, accumulated
gradients, or the random source.

## Forward, Backward, and Failure State

`GatherLastValid.ForwardWithLengths` validates its receiver, input, input
width, length count, and every length before indexing or marking forward state
valid. It snapshots the selected lengths into adapter-owned storage. A
successful later call replaces the row count and length snapshot.

At the start of any ordinary `Forward` or `ForwardWithLengths` attempt,
`GatherLastValid` invalidates its prior backward state. An invalid forward
therefore cannot leave an earlier request available to backward. A failed
forward returns no output and requires another successful
`ForwardWithLengths`.

`BackwardWithLengths` before a matching successful forward fails. It validates
the output-gradient shape before writing scratch or accumulated parameter
gradients. Because the selector has no parameters, a failed backward changes
no gradient. An invalid direct selector backward leaves its matching valid
forward snapshot intact so the caller may correct the gradient shape and
retry.

At the model boundary:

* Starting either `Predict` or `PredictWithLengths` invalidates the model's
  previous length-aware backward association.
* A successful `PredictWithLengths` establishes the association used by
  `BackwardWithLengths`.
* A later successful `PredictWithLengths` replaces it.
* Any intervening `Predict` attempt, failed `PredictWithLengths`, or failed
  `BackwardWithLengths` invalidates it.
* `BackwardWithLengths` may be called repeatedly after one successful
  prediction, matching existing gradient-accumulation behavior, until one of
  the invalidating events above occurs.
* Ordinary `Predict` and `Backward` preflight a graph containing
  `GatherLastValid` and fail before traversing any layer, with directions to
  use the additive operations. Direct ordinary calls on the adapter also fail.
  Neither path reuses cached lengths or silently selects the physical last
  step.

`TrainBatchWithLengths` binds its forward and backward within one operation.
Every success, validation failure, layer failure, loss failure, optimizer
failure, panic cleanup, and execution abort clears the top-level association
before returning. `FitWithLengths` uses that operation for every mini-batch and
uses fresh length-aware predictions for training and validation evaluation.

These objects retain the existing concurrency contract: callers must not
concurrently execute or mutate the same dataset, batch, layer, optimizer, or
model. Distinct objects may be used concurrently. The design does not claim
that one `Sequential` or `GatherLastValid` instance supports concurrent
requests.

## Ownership and Result Lifetime

`NewSequenceLengths` copies the caller slice. `NewSequenceDataset` copies input
and target matrices and copies the validated lengths into independently owned
storage. Batches and splits own their selected copies. Public allocating
accessors return copies; destination accessors retain nothing.

`PredictWithLengths` copies carrier values into model-owned reusable scratch
before traversal. `GatherLastValid` then validates and snapshots the values
needed by backward. Mutating any original caller slice, matrix, accessor
result, or destination after a successful call cannot change the later
backward calculation.

Selector outputs and input gradients do not alias their current arguments.
Like existing built-in layers, they may use layer-owned scratch matrices, so a
later call on the same adapter may overwrite an earlier returned result.
Callers clone results that must outlive later calls.

## Row Alignment and Determinism

`SequenceDataset` applies exactly one index list to all three owned values.
Ordered selection, repeated indexes, seeded shuffling, mini-batching, partial
final batches, and splitting therefore preserve input/target/length alignment.
Validation of all indexes and destinations occurs before any destination is
written.

`FitWithLengths` builds one identity index slice at each epoch, shuffles it
only with `SequenceFitConfig.Random` when requested, and passes the same
`indexes[start:end]` to all three batch destinations. Reusable `[]int` scratch
holds selected lengths without a per-batch carrier allocation. Full training
evaluation uses all training lengths in dataset order. Full validation
evaluation uses all validation lengths in validation-dataset order.

No new random source exists. Given equivalent parameters, inputs, targets,
lengths, configuration, and caller-owned random sources, sequence batches,
predictions, gradients, updates, histories, callbacks, and early-stopping
decisions are equivalent.

## Parameters and Training Behavior

`SequenceLengths`, `SequenceDataset`, `SequenceBatch`, and
`GatherLastValid` expose no `optimizer.Parameter`. Sequential parameter
discovery remains in layer order. `SimpleRNN` retains input-weight,
recurrent-weight, bias order, and later layers retain their existing order.

Length-aware fitting preserves existing:

* summed layer gradient accumulation and loss-controlled scaling;
* gradient reset and optimizer update order;
* training-mode restoration;
* learning-rate schedule timing;
* callback timing and partial-history behavior;
* early-stopping monitoring;
* execution finish, abort, and panic cleanup; and
* pre-update `TrainMetrics.Loss`.

An invalid length, count, graph, input, or target association is rejected
before forward work and therefore before any gradient or parameter update.

## Serialization

`GatherLastValid` is part of the graph and uses the additive version `1` layer
name:

```text
gather_last_valid
```

Its record stores only:

```json
{
  "type": "gather_last_valid",
  "steps": 3,
  "feature_size": 4
}
```

Loading derives and validates the input shape through
`layer.NewSequenceShape`, then constructs the adapter with
`layer.NewGatherLastValid`. The document does not store lengths, datasets,
row counts, forward-valid flags, selected-length snapshots, output or gradient
scratch, optimizer state, accumulated gradients, random state, execution
state, or device residency.

A loaded model has fresh forward state. `BackwardWithLengths` fails until the
loaded model successfully runs `PredictWithLengths`.

The format remains `neuralnetwork.sequential`, version `1`. Existing ANN, CNN,
`simple_rnn`, and `last_step` documents contain no new fields and retain their
current bytes and load behavior. Older readers reject
`gather_last_valid` as an unknown additive layer instead of skipping or
substituting it.

## CPU and Metal Behavior

`GatherLastValid` uses the pure-Go CPU path. This milestone adds no Metal
gather kernel and no public device selection.

On a Metal-enabled model execution, the adapter is an ordinary unsupported CPU
boundary: it observes only its required input after a selective barrier,
produces a host-current output, and allows a later eligible dense operation to
upload that output lazily. Unsupported builds, unavailable devices, and small
workloads follow their existing CPU/SIMD behavior. The lack of a gather kernel
does not surface a device-selection error.

## Alternatives Considered

### Selected: aligned data types plus explicit model calls

The selected design combines:

* a validated owned integer carrier in `data`;
* additive supervised dataset and batch types that own row alignment;
* explicit `Sequential` operations that bind lengths to one call; and
* one concrete adapter method that receives the call's copied lengths.

This fits the stable matrix surface because values remain matrices and only the
new orchestration path knows about the side input. It is narrower than a
general sequence container or a new layer-wide context contract.

### Immutable request/container passed through every layer

A new request containing values and lengths could be passed through parallel
model and layer APIs. That would either require every layer to understand the
request, introduce adapters around every existing layer, or create a second
general layer contract. ROADMAP Items 4 and 31 own those broader container and
layer-boundary decisions. The current milestone needs lengths only at one
many-to-one boundary, so the additional abstraction is not justified.

### General additive side-input layer interface

A public interface accepting arbitrary metadata or masks would make future
propagation semantics look settled before ROADMAP Items 4, 5, and 15 decide
them. The model instead recognizes the concrete `GatherLastValid` capability
privately. This does not prevent a later reviewed interface from replacing the
private orchestration.

### Mutable layer or model setter

`SetLengths`, a mutable lengths field, or a setter called separately from
prediction is rejected. The association could survive a failed call, apply to
the wrong batch, race with another request, or become ambiguous across
training evaluation and validation.

### Global or goroutine-local state

Package-global current lengths, goroutine identity lookup, and goroutine-local
emulation are rejected. They make nesting, cleanup, and concurrency implicit.

### Encoding lengths in existing matrices

Target-encoded lengths and a silently appended or consumed input feature are
rejected. They change documented input/target meaning, can train on metadata
accidentally, and make shape and row alignment implicit.

### Expanding existing stable types

Adding lengths to `Dataset`, `Batch`, `FitConfig`, or existing
`Sequential` method signatures is rejected. It would alter stable behavior and
could break unkeyed `FitConfig` literals. Parallel additive types and methods
keep fixed-length and non-sequence callers source-compatible.

### Premature general containers or masking

A tensor, ragged tensor, packed representation, general sequence value,
metadata map, automatic mask propagation, or mask inside `SimpleRNN` is
rejected for this milestone. Those choices belong to ROADMAP Items 4, 5, 15,
and 31 and require broader consumers than last-valid selection.

## Compatibility and Non-Goals

The approved addition does not change:

* `LastStep`, including its fixed physical-final-step meaning;
* `SimpleRNN`, including zero initialization, stateless calls, fixed tanh,
  all-steps output, and full backpropagation through configured steps;
* caller-provided padding in existing APIs;
* `layer.Layer`, `data.Dataset`, `data.Batch`, `model.FitConfig`, or existing
  `Sequential` method signatures and behavior;
* matrix and dataset ownership;
* parameter order, gradient scaling, or optimizer behavior;
* caller-controlled random sources; or
* existing version `1` document bytes and load behavior.

This milestone does not add zero-length sequences, left padding, interior
holes, arbitrary masks, multiple valid spans, ragged or packed storage,
masking within recurrence, masked sequence-to-sequence losses or metrics,
state input/output or carry, streaming, truncation, detachment, clipping,
recurrent dropout, general sequence adapters, a Metal gather kernel, or a
general container/layer redesign.

The later opt-in optimizer-level clipping boundary is now frozen in
[gradient-clipping-design.md](gradient-clipping-design.md), with production
implementation pending. It consumes the complete accumulated gradients after
this document's last-valid routing and unchanged recurrent backward pass; it
does not revise the explicit-length contract.
