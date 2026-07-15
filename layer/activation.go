package layer

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/internal/scratch"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewActivation constructs a layer wrapper around an activation function.
func NewActivation(function activation.Activation) (out *Activation, err error) {
	if function == nil {
		err = errors.New("layer: activation function is nil")
		return nil, err
	}

	var a Activation
	a.function = function
	a.destinationFunction, _ = function.(activation.DestinationActivation)
	return &a, nil
}

// Activation applies a stateless activation function as a trainable-model layer.
type Activation struct {
	function            activation.Activation
	destinationFunction activation.DestinationActivation
	inputCachePool      scratch.MatrixPool
	outputPool          scratch.MatrixPool
	inputGradientPool   scratch.MatrixPool
	inputCache          *matrix.Matrix
}

// Forward applies the wrapped activation function and caches input for Backward.
func (a *Activation) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var (
		rows       int
		cols       int
		inputCache *matrix.Matrix
	)

	if err = a.validate(); err != nil {
		return nil, err
	}

	if input == nil {
		err = errors.New("layer: activation input is nil")
		return nil, err
	}

	if err = input.Validate(); err != nil {
		err = fmt.Errorf("layer: activation input invalid: %w", err)
		return nil, err
	}

	rows, cols = input.Shape()
	if a.destinationFunction == nil {
		if output, err = a.function.Forward(input); err != nil {
			return nil, err
		}
	} else {
		if output, _, err = a.outputPool.Get(rows, cols); err != nil {
			return nil, err
		}

		if err = a.destinationFunction.ForwardInto(input, output); err != nil {
			return nil, err
		}
	}

	if inputCache, _, err = a.inputCachePool.Get(rows, cols); err != nil {
		return nil, err
	}

	if err = inputCache.CopyFrom(input); err != nil {
		return nil, err
	}
	a.inputCache = inputCache

	return output, nil
}

// Backward propagates gradients through the wrapped activation function.
func (a *Activation) Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var (
		rows int
		cols int
	)

	if err = a.validate(); err != nil {
		return nil, err
	}

	if a.inputCache == nil {
		err = errors.New("layer: activation backward called before forward")
		return nil, err
	}

	if outputGradient == nil {
		err = errors.New("layer: activation output gradient is nil")
		return nil, err
	}

	if err = outputGradient.Validate(); err != nil {
		err = fmt.Errorf("layer: activation output gradient invalid: %w", err)
		return nil, err
	}

	if a.destinationFunction == nil {
		inputGradient, err = a.function.Backward(a.inputCache, outputGradient)
		return inputGradient, err
	}

	rows, cols = a.inputCache.Shape()
	if inputGradient, _, err = a.inputGradientPool.Get(rows, cols); err != nil {
		return nil, err
	}

	if err = a.destinationFunction.BackwardInto(a.inputCache, outputGradient, inputGradient); err != nil {
		return nil, err
	}

	return inputGradient, nil
}

// Function returns the wrapped activation function.
func (a *Activation) Function() (function activation.Activation) {
	if a == nil {
		return nil
	}

	function = a.function
	return function
}

func (a *Activation) validate() (err error) {
	if a == nil {
		err = errors.New("layer: activation layer is nil")
		return err
	}

	if a.function == nil {
		err = errors.New("layer: activation function is nil")
		return err
	}

	return nil
}
