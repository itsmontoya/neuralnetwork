# neuralnetwork

`neuralnetwork` is a pure-Go neural network library for dense feed-forward
artificial neural networks (ANNs), an initial convolutional neural network
(CNN) path, and an initial recurrent neural network (RNN) path trained with
backpropagation.

The project is currently an early implementation. The v1 scope and public API
direction are documented in [docs/v1-scope-and-api.md](docs/v1-scope-and-api.md),
and the current stable surface is reviewed in
[docs/v1-api-review.md](docs/v1-api-review.md).

Example import path:

```go
import "github.com/itsmontoya/neuralnetwork/model"
```

## Minimal Usage

```go
package main

import (
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func train() (predictions *matrix.Matrix, err error) {
	var (
		random        *rand.Rand
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		trainingData  *data.Dataset
		hidden        *layer.Dense
		hiddenTanh    *layer.Activation
		output        *layer.Dense
		outputSigmoid *layer.Activation
		network       *model.Sequential
		adam          *optimizer.Adam
	)

	random = rand.New(rand.NewSource(1))

	if inputs, err = matrix.FromSlice(4, 2, []float32{0, 0, 0, 1, 1, 0, 1, 1}); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(4, 1, []float32{0, 1, 1, 0}); err != nil {
		return nil, err
	}

	if trainingData, err = data.NewDataset(inputs, targets); err != nil {
		return nil, err
	}

	if hidden, err = layer.NewDense(2, 4, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if hiddenTanh, err = layer.NewActivation(activation.Tanh{}); err != nil {
		return nil, err
	}

	if output, err = layer.NewDense(4, 1, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if outputSigmoid, err = layer.NewActivation(activation.Sigmoid{}); err != nil {
		return nil, err
	}

	if network, err = model.NewSequential(hidden, hiddenTanh, output, outputSigmoid); err != nil {
		return nil, err
	}

	if adam, err = optimizer.NewAdam(0.05); err != nil {
		return nil, err
	}

	_, err = network.Fit(trainingData, model.FitConfig{
		Epochs:    5000,
		BatchSize: 4,
		Optimizer: adam,
		Loss:      loss.BinaryCrossEntropy{},
	})
	if err != nil {
		return nil, err
	}

	predictions, err = network.Predict(inputs)
	return predictions, err
}
```

## Data

Use `data.NewDataset` for in-memory supervised data and `data.LoadCSV` with
`data.CSVConfig` for simple CSV input. Datasets, batches, and split results own
copies of their matrices so callers can mutate source data without changing
stored samples. Batching and splitting preserve order when the random source is
nil and shuffle deterministically when callers provide a seeded `*rand.Rand`.

For padded sequence rows, `data.NewSequenceLengths` validates one positive
logical length per row and `data.NewSequenceDataset` keeps inputs, targets, and
lengths aligned through batching, shuffling, and splitting.

Large read-only boards can opt into scoped, zero-copy ordered access and
evaluation with `Dataset.WithView`, `ViewBatches`, and
`Sequential.FitWithViews`. The aligned equivalents keep sequence inputs,
targets, and lengths together through `SequenceDatasetView` and
`FitWithLengthViews`. Safe constructors, accessors, `Batches`, `Split`, `Fit`,
and `FitWithLengths` continue to copy by default. Shuffled view fitting
requires the explicit `data.ViewOrCopy` fallback:

```go
history, err := network.FitWithViews(trainingData, model.ViewFitConfig{
	FitConfig: fitConfig,
	Policy:    data.ViewOrCopy,
})
```

See [docs/data.md](docs/data.md) for CSV, batching, train/test split, ownership,
aligned sequence-data contracts, view lifetimes, and performance guidance.

## Convolutional Networks

The initial CNN path represents each image as one flattened matrix row in
channels-first `CHW` order. It composes `Conv2D`, existing activation layers,
`MaxPool2D`, `Flatten`, and `Dense` through the unchanged `model.Sequential`
and `data.Dataset` APIs.

See the [CNN guide](docs/cnn.md) for layout formulas, construction, training,
serialization, ownership, determinism, and current limitations. The runnable
[minimal CNN example](examples/cnn/main.go) trains a deterministic classifier
on synthetic horizontal and vertical line images without external downloads.

## Recurrent Networks

The RNN path represents each physical sequence as one flattened matrix row in
time-major `TF` order. A stateless, fixed-tanh `SimpleRNN` returns every hidden
step. `LastStep` selects the physical final hidden vector for fixed-length
callers. For padded rows, `GatherLastValid` selects each positive logical
length's final valid hidden vector through explicit length-aware model and
dataset APIs.

See the [RNN guide](docs/rnn.md) for layout and recurrence formulas,
fixed-length and padded construction, training, migration, serialization,
ownership, determinism, statelessness, and current limitations. The runnable
[fixed-length RNN example](examples/rnn/main.go) and
[explicit-length RNN example](examples/rnn_lengths/main.go) train deterministic
temporal-order classifiers without external downloads.

## Training Controls

`model.Sequential.Fit` is configured with `model.FitConfig`. In addition to the
optimizer and loss, `FitConfig` supports validation data, an optional
`model.AccuracyFunc`, epoch callbacks through `model.FitCallback`, early
stopping through `model.NewEarlyStopping`, and optimizer learning-rate schedules.

Learning-rate schedules live in `optimizer`: use
`optimizer.NewConstantLearningRate`, `optimizer.NewStepDecay`, or
`optimizer.NewExponentialDecay` and pass the schedule to `FitConfig`.

Regularization wraps an existing optimizer with `optimizer.NewRegularized`.
Built-in regularizers include `optimizer.NewL1` and
`optimizer.NewL2WeightDecay`.

Gradient clipping is also an optimizer wrapper. A zero field disables that
control, while at least one positive finite limit is required:

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

When both controls are enabled, value clipping runs before one global norm is
computed across the complete ordered parameter set. To bound the combined
data and regularization gradient, use `Regularized` as the outer wrapper:

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

l2, err := optimizer.NewL2WeightDecay(0.0001)
if err != nil {
	return err
}

optimizerRule, err := optimizer.NewRegularized(clipping, l2)
if err != nil {
	return err
}
```

Pass `optimizerRule` to `TrainBatch`, `Fit`,
`TrainBatchWithLengths`, or `FitWithLengths`. Learning-rate schedules reach
the base optimizer through the wrapper. `Observation` reports whether the
most recent completed clipping phase clamped a value, its pre-scale global
norm and scale, and whether the base update completed. Non-finite gradients
return an error before clipping mutation or base invocation.

Clipping configuration, scratch, observations, and optimizer state are not
serialized with a model; reconstruct the optimizer stack after loading. A
clipping wrapper around SGD uses the correctness-first CPU fallback in a
Metal execution, while direct unwrapped SGD retains its resident update path.
See the [gradient clipping design](docs/gradient-clipping-design.md) and
[RNN guide](docs/rnn.md) for arithmetic, lifecycle, composition, and recurrent
training details.

## Layers

The `layer` package includes dense layers, activation layers, inverted dropout,
per-feature batch normalization, trainable two-dimensional convolution,
parameter-free two-dimensional max pooling, a spatial-to-dense flatten adapter,
a stateless `SimpleRNN`, a fixed-length sequence-to-dense `LastStep` adapter,
and an explicit-length `GatherLastValid` adapter.
`layer.NewSpatialShape`, `layer.NewConv2DConfig`, and
`layer.NewMaxPool2DConfig` validate explicit channels-first spatial geometry;
`layer.NewSequenceShape` and `layer.NewSimpleRNNConfig` validate explicit
time-major sequence geometry.
`layer.NewDropout` requires a caller-owned random source for deterministic masks
and follows training/evaluation mode. `layer.NewBatchNormalization` and
`layer.NewBatchNormalizationWithConfig` manage trainable gamma and beta
parameters plus running statistics for evaluation; batch normalization remains
per flattened feature rather than per spatial channel.

## Metrics

The `metric` package provides reporting-only metrics for regression, binary
classification, categorical classification, and confusion matrices. Metrics do
not affect optimization. Classification behavior, threshold handling, one-hot
target expectations, and confusion-matrix orientation are documented in
[docs/metrics.md](docs/metrics.md).

## Serialization

Use `Sequential.Save` and `model.LoadSequential` to persist sequential models
with the v1 JSON contract. The format is `neuralnetwork.sequential`, version
`1`, and supports `dense`, `activation`, `dropout`, `batch_normalization`,
`conv2d`, `max_pool2d`, `flatten`, `simple_rnn`, `last_step`, and
`gather_last_valid` layers. CNN and RNN layer names and fields are additive.
The gather record stores its sequence shape but not invocation lengths.
Existing ANN-, CNN-, and fixed-length RNN version `1` documents retain their
encoding and compatibility. Older readers reject documents whose additive
layer types they do not recognize.

Serialization stores model structure and layer parameters. It does not store
optimizer state, accumulated gradients, training history, callbacks,
learning-rate schedules, forward caches, recurrent hidden histories, or
gathered-length snapshots or original random source state. Loaded dropout
layers use deterministic local random sources, and loaded recurrent and
length-aware layers begin with zero gradients and fresh forward state.

## Optional Metal Acceleration

On Darwin with cgo, the optional `metal` build tag enables a transparent
device-resident path when a default Metal device is available. The initial
supported graph is:

```text
matrix rows -> Dense -> ReLU -> Dense -> Softmax
```

That graph can remain resident through `Sequential.Predict`, `Backward`,
`TrainBatch`, and `Fit` when training uses categorical cross entropy and plain
SGD. Public constructors and method signatures are unchanged. Small workloads,
unsupported layers, activations, losses, or optimizers, unavailable devices,
non-Darwin platforms, and cgo-disabled builds use the existing CPU path.
On `amd64` and `arm64`, CPU work in a `metal` build still uses SIMD.

Build and test the optional path with:

```sh
go test -tags=metal ./...
```

Use `-tags=purego` to opt out of Metal and external SIMD wrappers. See the
[Metal design](docs/metal.md) for coherence, synchronization, fallback, and
troubleshooting details; the [SIMD design](docs/simd.md) for hybrid CPU
selection; and [GPU benchmark evidence](Benchmarks_gpu.md) for reproducible
end-to-end measurements.

## Development

The baseline verification command is:

```sh
go test ./...
```

Testing policy, floating-point helpers, and the v1 numeric type decision are
documented in [docs/testing.md](docs/testing.md).

Data loading, batching, and splitting behavior is documented in
[docs/data.md](docs/data.md).

Classification metric semantics are documented in
[docs/metrics.md](docs/metrics.md).

## Examples

Run the deterministic synthetic CNN classifier with:

```sh
go run ./examples/cnn
```

Run the deterministic synthetic RNN temporal-order classifier with:

```sh
go run ./examples/rnn
```

Run the deterministic mixed-length padded RNN classifier with:

```sh
go run ./examples/rnn_lengths
```

Run the XOR smoke test with:

```sh
go run ./examples/xor
```

Run the regression example with:

```sh
go run ./examples/regression
```

Run the multiclass classification example with:

```sh
go run ./examples/multiclass
```

Run the terminal-art classifier example with:

```sh
go run ./examples/heart
```

Run the toy code-generation example with:

```sh
go run ./examples/toycode
```
