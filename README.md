# neuralnetwork

`neuralnetwork` is a pure-Go neural network library focused on dense feed-forward
models trained with backpropagation.

The project is currently an early implementation. The v1 scope and public API
direction are documented in [docs/v1-scope-and-api.md](docs/v1-scope-and-api.md),
and implementation packages are being added incrementally from that plan.

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

	if inputs, err = matrix.FromSlice(4, 2, []float64{0, 0, 0, 1, 1, 0, 1, 1}); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(4, 1, []float64{0, 1, 1, 0}); err != nil {
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

## Development

The baseline verification command is:

```sh
go test ./...
```

Testing policy, floating-point helpers, and the v1 numeric type decision are
documented in [docs/testing.md](docs/testing.md).

Data loading, batching, and splitting behavior is documented in
[docs/data.md](docs/data.md).

## Examples

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
