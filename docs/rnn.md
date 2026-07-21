# Recurrent Neural Networks

The initial recurrent neural network (RNN) path composes with the same matrix,
dataset, model, loss, metric, optimizer, and serialization APIs as dense
artificial neural networks (ANNs) and convolutional neural networks (CNNs). It
is a pure-Go CPU reference implementation for pre-flattened, fixed-length
sequence datasets.

For the complete behavioral and public API contract, see
[rnn-design.md](rnn-design.md). This guide focuses on constructing and training
the supported many-to-one model shape:

```text
flattened sequence rows -> SimpleRNN -> LastStep -> Dense -> Softmax
```

## Input and Output Layout

Each `matrix.Matrix` row is one complete fixed-length sequence. A sequence uses
time-major `TF` order, so a batch is logically `NTF` and physically has shape:

```text
[batch, steps*featureSize]
```

The matrix column for a sequence coordinate is:

```text
step*FeatureSize + feature
```

For example, a batch of ten sequences with four steps and three features per
step is a `[10, 12]` matrix. The columns for step two are 6, 7, and 8. The
batch row is not part of the flattened-column calculation.

`data.Dataset` already treats each matrix row as one sample, so batching keeps
every sequence intact. Callers prepare fixed-length flattened rows before
constructing a dataset; there is no sequence-specific dataset or padding API.
Any caller-provided padding is treated as ordinary input.

`layer.NewSequenceShape` validates positive dimensions and rejects flattened
sizes that overflow `int`:

```go
inputShape, err := layer.NewSequenceShape(4, 3)
if err != nil {
	return nil, err
}
```

Input matrices must have exactly `inputShape.Size()` columns.

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

`SimpleRNN` returns every hidden step in time-major order. With `hiddenSize`
hidden values, its physical output shape is `[batch, steps*hiddenSize]`, and
the output column for `(step, hidden)` is:

```text
step*HiddenSize + hidden
```

The tanh is part of `SimpleRNN`; do not add another tanh layer merely to obtain
the documented recurrence. Backward propagation visits every step in reverse,
combines each direct output gradient with the recurrent gradient from the next
step, and sums parameter gradients across batch rows and time steps. It does
not apply mean scaling, truncation, or clipping.

## Constructing an RNN

The following model maps three-step sequences with two features per step to
two output classes:

```go
inputShape, err := layer.NewSequenceShape(3, 2)
if err != nil {
	return nil, err
}

recurrentConfig, err := layer.NewSimpleRNNConfig(inputShape, 6)
if err != nil {
	return nil, err
}

recurrent, err := layer.NewSimpleRNN(
	recurrentConfig,
	layer.XavierUniformWeights(random),
	layer.XavierUniformWeights(random),
)
if err != nil {
	return nil, err
}

lastStep, err := layer.NewLastStep(recurrent.OutputShape())
if err != nil {
	return nil, err
}

output, err := layer.NewDense(
	lastStep.OutputSize(),
	2,
	layer.XavierUniformWeights(random),
)
if err != nil {
	return nil, err
}

softmax, err := layer.NewActivation(activation.Softmax{})
if err != nil {
	return nil, err
}

network, err := model.NewSequential(recurrent, lastStep, output, softmax)
if err != nil {
	return nil, err
}
```

`random` is a caller-owned `*rand.Rand`. `NewSimpleRNN` calls the input
initializer first with `[inputFeatureSize, hiddenSize]`, then the recurrent
initializer with `[hiddenSize, hiddenSize]`. Biases begin at zero. Using
equivalent caller-seeded sources gives equivalent initialization and
predictions.

`LastStep` is the explicit sequence-to-dense boundary. It copies the final
contiguous hidden vector from each batch row into `[batch, hiddenSize]`.
Backward returns the full sequence shape, places the supplied gradient at the
final step, and leaves every earlier step at zero. For a one-step sequence,
forward and backward preserve all values while still returning independent
matrices.

Because `SimpleRNN` emits all steps, another `SimpleRNN` can consume its
`OutputShape()` before `LastStep`. An existing `Dense` layer consumes the
complete physical row, so it is not a time-distributed projection.

## Training and Evaluation

RNNs train through `model.Sequential.Fit` without an RNN-specific training
loop. Inputs contain flattened sequence rows and targets follow the selected
loss. For the two-class Softmax model above, targets are one-hot rows and
training can use categorical cross entropy:

```go
adam, err := optimizer.NewAdam(0.03)
if err != nil {
	return err
}

_, err = network.Fit(trainingData, model.FitConfig{
	Epochs:         80,
	BatchSize:      8,
	Shuffle:        true,
	Random:         rand.New(rand.NewSource(101)),
	Optimizer:      adam,
	Loss:           loss.CategoricalCrossEntropy{},
	ValidationData: validationData,
	Accuracy:       metric.CategoricalAccuracy{}.Value,
})
if err != nil {
	return err
}

predictions, err := network.Predict(inputs)
```

Batch size may be greater than one, and a final partial batch remains valid
because a complete sequence always occupies one row. Prediction, validation,
losses, and metrics use the existing matrix APIs unchanged.

The runnable [RNN example](../examples/rnn/main.go) learns whether event A or B
occurred first, then predicts after a blank final step. The synthetic task
makes temporal order necessary while keeping the command fast, reproducible,
and independent of downloads. Run it with:

```sh
go run ./examples/rnn
```

With the checked-in seeds, the meaningful stable expectation is successful
training and classification of the canonical `A then B` and `B then A`
sequences. Exact printed floating-point values and timing are not compatibility
guarantees.

## Statelessness and Determinism

Every valid `SimpleRNN.Forward` starts from an all-zero hidden state for every
batch row. Hidden state is not shared between rows and is not carried between
samples, batches, training and evaluation, or successive calls. Training and
prediction therefore use the same stateless recurrence.

The layer creates no random source. Initialization randomness comes only from
the two caller-provided weight initializers. Dataset generation and shuffled
training are deterministic when their callers also use equivalent seeded
sources. A repeated run with all seeds held constant produces the same model
output.

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
```

The additive layer names are `simple_rnn` and `last_step`. A `simple_rnn`
record stores its input sequence shape, hidden size, input weights, recurrent
weights, and biases. A `last_step` record stores its input sequence shape.

Loading restores architecture and parameter values with zero accumulated
gradients and fresh forward state. Optimizer state, input caches, hidden
histories, scratch storage, training history, and random source state are not
stored. A loaded recurrent model must run forward before backward.

Existing ANN- and CNN-only version `1` documents retain their encoding and
continue to load. Older readers that do not know the additive RNN layer names
reject RNN documents instead of skipping or substituting layers.

## Matrix Ownership

`SimpleRNN` copies the most recent valid input needed by backward and caches
its hidden history without retaining caller-owned storage. `LastStep` retains
only the most recent valid batch size. Mutating a caller input after forward
does not change the next backward calculation.

Valid forward outputs and backward input gradients do not alias their current
arguments. Results use layer-owned scratch matrices, so a later call on the
same layer may reuse and overwrite an earlier result. Clone a result that must
remain stable across later calls. Invalid calls do not establish a valid
backward cache, and backward before forward returns an error.

## Supported and Deferred Features

| Area | Initial support | Deferred |
| --- | --- | --- |
| Data layout | Batched fixed-length time-major `NTF`/`TF` `float32` matrix rows | Variable-length, ragged, padded-and-masked, and packed sequences |
| Recurrence | Stateless zero-initialized Elman `SimpleRNN` with fixed tanh and all-steps output | Configurable activations, LSTM, GRU, bidirectional recurrence, attention, and transformers |
| Gradients | Full backpropagation through every configured step with summed parameter gradients | Truncation, graph detachment, gradient clipping, and masked sequence losses |
| State | Independent zero state for every row and call | Learned or caller-provided state, returned state, state carry, stateful training, and streaming inference |
| Composition | Recurrent stacking, `LastStep`, existing elementwise layers, dense layers, datasets, losses, metrics, optimizers, training, and serialization | Time-distributed dense layers, richer output adapters, and combined spatial/temporal CNN-RNN layouts |
| Data loading | Caller-prepared flattened fixed-length rows | Tokenization, embeddings, vocabulary management, and sequence-specific loaders |
| Runtime | Clear pure-Go CPU reference kernel | SIMD, Metal, goroutine-parallel, distributed, and third-party recurrent kernels |
| Containers | Existing `matrix.Matrix` and `layer.Layer` contracts | Generic tensors, ragged tensors, sequence containers, and automatic-differentiation graphs |

Existing activation and dropout layers operate elementwise on flattened
sequence values. Existing batch normalization treats each flattened
step-feature column independently; it is not recurrent or temporal batch
normalization.
