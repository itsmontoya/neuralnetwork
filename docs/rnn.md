# Recurrent Neural Networks

The recurrent neural network (RNN) path supports both fixed-length many-to-one
models and padded rows with an explicit positive logical length per sample.
Both paths use flattened time-major matrix rows and the same stateless
`SimpleRNN`.

Use `LastStep` with the existing model and dataset methods when every physical
step is valid. Use `GatherLastValid`, `data.SequenceLengths`, and the
length-aware model methods when rows have padded suffixes:

```text
fixed length:
    flattened rows -> SimpleRNN -> LastStep -> Dense -> output activation

explicit length:
    padded flattened rows -> SimpleRNN -> zero or more SimpleRNN layers
        -> GatherLastValid -> Dense -> output activation
```

For the complete behavioral contract, including failure-state and ownership
details, see [rnn-design.md](rnn-design.md) and
[sequence-lengths-design.md](sequence-lengths-design.md). Opt-in clipping
semantics are recorded in
[gradient-clipping-design.md](gradient-clipping-design.md).

## Input and Length Layout

Each `matrix.Matrix` row is one physical sequence. Values use time-major `TF`
order, so a batch is logically `NTF` and physically:

```text
[batch, steps*featureSize]
```

The flattened column for `(step, feature)` is:

```text
step*FeatureSize + feature
```

For example, a batch of ten sequences with four physical steps and three
features per step is a `[10, 12]` matrix. Step two occupies columns 6, 7, and
8. The batch row is never part of the flattened-column calculation.

For a padded row `n`, its length `L[n]` identifies the valid prefix at steps
`0..L[n]-1`. Steps `L[n]..steps-1` form the padded suffix. The physical row
always retains all `steps*featureSize` values:

```text
row 0: [valid valid pad]  length 2
row 1: [valid valid valid] length 3
```

Construct the shape and validated lengths explicitly:

```go
inputShape, err := layer.NewSequenceShape(3, 2)
if err != nil {
	return err
}

lengths, err := data.NewSequenceLengths(3, []int{2, 3})
if err != nil {
	return err
}
```

Steps and feature sizes must be positive, and their product must fit in an
`int`. Length values use Go `int`, must be non-empty, and must be in the
inclusive range `[1, steps]`. Zero-length rows, left padding, interior holes,
and arbitrary masks are not supported.

The length count must equal the associated matrix row count. A
`SequenceDataset` also requires matching input and target row counts and an
input width evenly divisible by the physical step count. These checks happen
before copying, model traversal, or parameter updates.

## Recurrence and Shapes

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

The parameter shapes are:

```text
inputWeights     [inputFeatureSize, hiddenSize]
recurrentWeights [hiddenSize, hiddenSize]
biases           [1, hiddenSize]
```

`SimpleRNN` emits every hidden step in time-major order. With `hiddenSize`
hidden values, its output is `[batch, steps*hiddenSize]`. Its fixed tanh is
intrinsic; do not add another tanh merely to obtain the documented recurrence.

Backward propagation traverses every configured physical step in reverse and
sums parameter gradients across rows and steps. The layer does not apply mean
scaling, truncation, clipping, or masking.

## Constructing a Fixed-Length RNN

This model maps three-step sequences with two features per step to two classes:

```go
inputShape, err := layer.NewSequenceShape(3, 2)
if err != nil {
	return err
}

recurrentConfig, err := layer.NewSimpleRNNConfig(inputShape, 6)
if err != nil {
	return err
}

recurrent, err := layer.NewSimpleRNN(
	recurrentConfig,
	layer.XavierUniformWeights(random),
	layer.XavierUniformWeights(random),
)
if err != nil {
	return err
}

lastStep, err := layer.NewLastStep(recurrent.OutputShape())
if err != nil {
	return err
}

output, err := layer.NewDense(
	lastStep.OutputSize(),
	2,
	layer.XavierUniformWeights(random),
)
if err != nil {
	return err
}

softmax, err := layer.NewActivation(activation.Softmax{})
if err != nil {
	return err
}

network, err := model.NewSequential(recurrent, lastStep, output, softmax)
if err != nil {
	return err
}
```

`LastStep` always selects physical step `Steps()-1`. Its backward path places
the supplied direct gradient at that physical final step and writes zero at
earlier output steps. It has no length dependency, and its behavior is
unchanged.

Train this graph with `data.Dataset`, `model.FitConfig`, and the existing
`Predict`, `Backward`, and `TrainBatch` methods. Any padding passed to those
APIs remains ordinary input.

## Constructing a Last-Valid RNN

Replace `LastStep` with `GatherLastValid` when rows have padded suffixes:

```go
inputShape, err := layer.NewSequenceShape(3, 2)
if err != nil {
	return err
}

recurrentConfig, err := layer.NewSimpleRNNConfig(inputShape, 6)
if err != nil {
	return err
}

recurrent, err := layer.NewSimpleRNN(
	recurrentConfig,
	layer.XavierUniformWeights(random),
	layer.XavierUniformWeights(random),
)
if err != nil {
	return err
}

gather, err := layer.NewGatherLastValid(recurrent.OutputShape())
if err != nil {
	return err
}

output, err := layer.NewDense(
	gather.OutputSize(),
	2,
	layer.XavierUniformWeights(random),
)
if err != nil {
	return err
}

softmax, err := layer.NewActivation(activation.Softmax{})
if err != nil {
	return err
}

network, err := model.NewSequential(recurrent, gather, output, softmax)
if err != nil {
	return err
}
```

The length-aware model graph must contain exactly one `GatherLastValid`. Its
configured step count must match the supplied carrier. Ordinary model methods
reject this graph with directions to use their length-aware counterparts; the
adapter never silently falls back to `LastStep`.

For a selector input shape `[N, S*F]`, gather returns `[N, F]`:

```text
output[n, feature] =
    input[n, (L[n]-1)*F+feature]
```

Backward returns `[N, S*F]`, places each row gradient at `L[n]-1`, and writes
zero at every other step. `SimpleRNN` still evaluates the padded suffix during
forward, but the selected earlier hidden state cannot depend on later values.
Gather backward introduces no direct suffix gradient, so the unchanged
recurrent backward path follows the corresponding valid prefix.

## Prediction and Direct Training

Inputs retain their full physical width, while lengths are supplied separately:

```go
inputs, err := matrix.FromSlice(2, 6, []float32{
	1, 0, 0, 1, 99, 99,
	0, 1, 1, 0, 0, 0,
})
if err != nil {
	return err
}

lengths, err := data.NewSequenceLengths(3, []int{2, 3})
if err != nil {
	return err
}

predictions, err := network.PredictWithLengths(inputs, lengths)
if err != nil {
	return err
}
```

The first prediction uses hidden step one and ignores the padded physical step
two. The second uses physical step two.

`BackwardWithLengths` uses the association established by the most recent
successful `PredictWithLengths`:

```go
outputGradient, err := matrix.New(predictions.Rows(), predictions.Cols())
if err != nil {
	return err
}

_, err = network.BackwardWithLengths(outputGradient)
if err != nil {
	return err
}
```

One supervised update binds forward and backward inside one call:

```go
metrics, err := network.TrainBatchWithLengths(
	inputs,
	targets,
	lengths,
	loss.CategoricalCrossEntropy{},
	optimizerRule,
)
if err != nil {
	return err
}
_ = metrics
```

Do not store lengths in targets, append them as feature columns, or set them on
a layer separately from the operation.

## Aligned Datasets and Fit

`data.SequenceDataset` owns aligned copies of inputs, targets, and lengths:

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
```

Its selection, batching, shuffling, partial final batches, and splitting use
one row permutation for all three values. A nil batching/splitting random
source preserves order; a caller-provided source makes shuffling reproducible.

Train with the parallel length-aware configuration:

```go
adam, err := optimizer.NewAdam(0.03)
if err != nil {
	return err
}

history, err := network.FitWithLengths(
	trainingData,
	model.SequenceFitConfig{
		Epochs:         80,
		BatchSize:      8,
		Shuffle:        true,
		Random:         rand.New(rand.NewSource(101)),
		Optimizer:      adam,
		Loss:           loss.CategoricalCrossEntropy{},
		ValidationData: validationData,
		Accuracy:       metric.CategoricalAccuracy{}.Value,
	},
)
if err != nil {
	return err
}
_ = history
```

`SequenceFitConfig` has the same optimizer, learning-rate schedule, early
stopping, accuracy, and callback controls as `FitConfig`, but its validation
data is a `*data.SequenceDataset`. Training batches, full training evaluation,
and full validation evaluation each receive their aligned lengths.

## Gradient Clipping

Gradient clipping is opt-in at the general optimizer boundary. The same
wrapper works with ordinary and length-aware training and with SGD, Momentum,
Adam, or a custom optimizer:

```go
// Choose one configuration.
config := optimizer.GradientClippingConfig{MaxValue: 1} // value only
// config := optimizer.GradientClippingConfig{MaxNorm: 5} // norm only
// config := optimizer.GradientClippingConfig{MaxValue: 1, MaxNorm: 5} // combined

base, err := optimizer.NewAdam(0.001)
if err != nil {
	return err
}

optimizerRule, err := optimizer.NewGradientClipping(base, config)
if err != nil {
	return err
}
```

For an RNN, `SimpleRNN.Backward` first completes full backpropagation through
the configured physical steps. `GatherLastValid`, when present, has already
routed each direct gradient to its selected last-valid step. The wrapper then
applies value clipping followed by global-norm clipping across every current
parameter gradient in stable model order. Only after that does the base
optimizer consume the gradient for SGD updates, Momentum velocity, or Adam
moments.

Clipping does not truncate or detach recurrence, mask recurrent computation,
carry hidden state, repair a non-finite activation or gradient, or change
`LastStep` or `GatherLastValid`. A non-finite accumulated gradient returns an
`optimizer:` error before any clipping mutation or base call. The enclosing
model method preserves its existing contextual optimizer-update error and
restores training state.

Wrapper nesting determines regularization order. The canonical construction
below applies L1 and then L2 additions before clipping, so the complete
data-plus-regularization gradient is bounded:

```go
base, err := optimizer.NewAdam(0.001)
if err != nil {
	return err
}

clipping, err := optimizer.NewGradientClipping(
	base,
	optimizer.GradientClippingConfig{MaxNorm: 5},
)
if err != nil {
	return err
}

l1, err := optimizer.NewL1(0.0001)
if err != nil {
	return err
}

l2, err := optimizer.NewL2WeightDecay(0.0001)
if err != nil {
	return err
}

optimizerRule, err := optimizer.NewRegularized(clipping, l1, l2)
if err != nil {
	return err
}
```

Putting `GradientClipping` outside `Regularized` instead clips only the
incoming data gradient; later regularization additions may exceed a clipping
limit. No wrapper silently changes caller-authored nesting.

Inspect the most recent completed clipping phase without adding a model
callback:

```go
observation, available := clipping.Observation()
if available {
	fmt.Printf(
		"value clipped=%t global norm=%g scale=%g base completed=%t\n",
		observation.ValueClipped,
		observation.GlobalNorm,
		observation.Scale,
		observation.BaseUpdateCompleted,
	)
}
```

The wrapper forwards `LearningRate` and `SetLearningRate`, so `Fit` and
`FitWithLengths` schedules reach the base through either documented
regularization order. It owns reusable scratch after its first update and
meets the ordinary and length-aware warmed allocation guarantees under the
repository's non-concurrent model and optimizer lifecycle.

Clipping configuration, scratch, observations, regularizers, and optimizer
state are not saved in `neuralnetwork.sequential` version `1`. Reconstruct the
stack after loading. Under the optional Metal build, wrapped SGD takes the
explicit CPU fallback after backward and matches the CPU clipping contract;
plain unwrapped SGD retains the resident fast path.

## Stacked Recurrence

Every `SimpleRNN` preserves the configured physical step count. A stacked
length-aware graph places one gather boundary after the final recurrent layer:

```go
firstConfig, err := layer.NewSimpleRNNConfig(inputShape, 6)
if err != nil {
	return err
}
first, err := layer.NewSimpleRNN(
	firstConfig,
	layer.XavierUniformWeights(random),
	layer.XavierUniformWeights(random),
)
if err != nil {
	return err
}

secondConfig, err := layer.NewSimpleRNNConfig(first.OutputShape(), 4)
if err != nil {
	return err
}
second, err := layer.NewSimpleRNN(
	secondConfig,
	layer.XavierUniformWeights(random),
	layer.XavierUniformWeights(random),
)
if err != nil {
	return err
}

gather, err := layer.NewGatherLastValid(second.OutputShape())
if err != nil {
	return err
}
output, err := layer.NewDense(
	gather.OutputSize(),
	2,
	layer.XavierUniformWeights(random),
)
if err != nil {
	return err
}

network, err := model.NewSequential(first, second, gather, output)
if err != nil {
	return err
}
```

Supply lengths once to `PredictWithLengths`, `TrainBatchWithLengths`, or
`FitWithLengths`; model orchestration consumes them only at `gather`.

## Statelessness, Ownership, and Determinism

Every `SimpleRNN.Forward` begins from an all-zero hidden state for every row.
State is not shared between rows or carried between samples, batches, training
and evaluation, or successive calls. Length-aware calls do not change this
behavior.

`NewSequenceLengths` copies its integer slice. `NewSequenceDataset` copies its
matrices and lengths. Allocating accessors return copies, and `Into` accessors
copy into caller-owned destinations without retaining them. The model and
gather adapter copy lengths needed for traversal and backward. Caller mutation
after a successful operation cannot retarget its later backward calculation.

Layer outputs and input gradients do not alias their current arguments, but
they may use layer-owned scratch and be overwritten by a later call on the
same object. Clone a result that must remain stable.

The length-aware path creates no random source. Initializers, data generation,
and shuffling remain caller-controlled. Equivalent parameters, inputs, targets,
lengths, configurations, and seeded random sources produce equivalent
batches, histories, updates, and predictions. Callers must not concurrently
execute or mutate the same model, layer, dataset, batch, or optimizer.

## Serialization

RNN layers use the existing `neuralnetwork.sequential` JSON format at version
`1`:

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
```

The additive layer names are `simple_rnn`, `last_step`, and
`gather_last_valid`. A gather record stores only `steps` and `feature_size`.
Logical lengths belong to each invocation or dataset and are not serialized.
The first operation on a loaded length-aware graph must therefore supply
lengths. `BackwardWithLengths` fails until a successful
`PredictWithLengths`.

Loading restores architecture and parameter values with zero accumulated
gradients and fresh forward state. Optimizer state, input caches, recurrent
hidden histories, gathered-length snapshots, scratch, training history, and
random source state are not stored.

Existing ANN-, CNN-, `simple_rnn`-, and `last_step`-containing version `1`
documents retain their encoding and load behavior. Older readers reject the
unknown additive `gather_last_valid` type instead of skipping or substituting
it.

## Migrating from LastStep

Keep `LastStep` and the existing APIs when all physical steps are valid. To
make a padded many-to-one graph length-aware:

1. Keep the same flattened matrix width and valid-prefix layout.
2. Construct one `SequenceLengths` value with the graph's physical step count.
3. Replace `LastStep` with `GatherLastValid` after the final recurrent layer.
4. Use `SequenceDataset` and `SequenceFitConfig` for multi-epoch fitting.
5. Call `PredictWithLengths`, `BackwardWithLengths`,
   `TrainBatchWithLengths`, or `FitWithLengths` as appropriate.
6. Supply lengths again after loading a saved model.

This migration changes only the explicit graph and invocation path. It does
not make `LastStep` length-aware, change `SimpleRNN`, or reinterpret padding
passed to existing APIs.

## Runnable Examples

The [fixed-length RNN example](../examples/rnn/main.go) learns whether event A
or B occurred first, uses `LastStep`, and trains through Adam with global-norm
clipping:

```sh
go run ./examples/rnn
```

The [explicit-length RNN example](../examples/rnn_lengths/main.go) trains with
mixed lengths, aligned validation data, shuffled partial batches, and
deliberately non-zero suffix padding:

```sh
go run ./examples/rnn_lengths
```

Both examples are synthetic, deterministic with their checked-in seeds, and
independent of downloads. Exact printed floating-point values are not
compatibility guarantees.

## Supported and Deferred Features

| Area | Supported | Deferred |
| --- | --- | --- |
| Data layout | Batched fixed-length rows and padded rows with one positive valid-prefix length | Ragged or packed storage, zero-length rows, left padding, interior holes, and arbitrary masks |
| Selection | Physical final step through `LastStep`; explicit last-valid step through `GatherLastValid` | General masking and richer sequence adapters |
| Recurrence | Stateless zero-initialized Elman `SimpleRNN` with fixed tanh and all-steps output | Configurable activations, LSTM, GRU, bidirectional recurrence, attention, and transformers |
| Gradients | Full recurrent backpropagation; direct last-valid gradient routed only to the selected step; opt-in value and global-norm clipping at the optimizer boundary | Truncation, graph detachment, and masked sequence losses |
| State | Independent zero state for every row and call | Initial/final state APIs, state carry, stateful training, and streaming |
| Runtime | Pure-Go recurrent and gather reference paths with transparent CPU fallback | Optimized or accelerator-specific recurrent and gather kernels |
| Containers | Existing matrices plus focused aligned supervised length types | General sequence containers, tensors, ragged tensors, and automatic-differentiation graphs |

Existing activation and dropout layers operate elementwise on flattened
sequence values. Existing batch normalization treats each flattened
step-feature column independently; it is not recurrent or temporal batch
normalization. Existing `Dense` transforms the complete physical row and is not
a time-distributed projection.
