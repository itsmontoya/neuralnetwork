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

See [docs/data.md](docs/data.md) for CSV, batching, and train/test split
contracts.

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

The initial RNN path represents each fixed-length sequence as one flattened
matrix row in time-major `TF` order. A stateless, fixed-tanh `SimpleRNN`
returns every hidden step, `LastStep` selects the final hidden vector, and an
existing `Dense` layer produces a many-to-one prediction through the unchanged
`model.Sequential` and `data.Dataset` APIs.

See the [RNN guide](docs/rnn.md) for layout and recurrence formulas,
construction, training, serialization, ownership, determinism, statelessness,
and current limitations. The runnable
[minimal RNN example](examples/rnn/main.go) trains a deterministic classifier
whose label depends on temporal order, without external downloads.

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

## Layers

The `layer` package includes dense layers, activation layers, inverted dropout,
per-feature batch normalization, trainable two-dimensional convolution,
parameter-free two-dimensional max pooling, a spatial-to-dense flatten adapter,
a stateless `SimpleRNN`, and a sequence-to-dense `LastStep` adapter.
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
`conv2d`, `max_pool2d`, `flatten`, `simple_rnn`, and `last_step` layers. CNN and
RNN layer names and fields are additive: existing ANN- and CNN-only version `1`
documents retain their encoding and compatibility. Older readers reject RNN
documents whose additive layer types they do not recognize.

Serialization stores model structure and layer parameters. It does not store
optimizer state, accumulated gradients, training history, callbacks,
learning-rate schedules, forward caches, recurrent hidden histories, or
original random source state. Loaded dropout layers use deterministic local
random sources, and loaded recurrent layers begin with zero gradients and fresh
forward state.

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
