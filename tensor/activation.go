package tensor

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// ReLUInto writes the float32 rectified input into a disjoint destination.
// It preserves the existing activation's positive-zero result for NaN and
// negative zero. All returned errors leave destination unchanged.
func ReLUInto(input, destination *Tensor) (err error) {
	err = activationFloat32Into(input, destination, activation.ReLU{})
	return err
}

// SigmoidInto writes the stable float32 logistic activation into destination.
// Destination must be disjoint and match input's shape, including scalars and
// empties. All returned errors leave destination unchanged.
func SigmoidInto(input, destination *Tensor) (err error) {
	err = activationFloat32Into(input, destination, activation.Sigmoid{})
	return err
}

// TanhInto writes the float32 hyperbolic tangent into a disjoint destination.
// Scalars and empties preserve shape. All returned errors leave destination
// unchanged.
func TanhInto(input, destination *Tensor) (err error) {
	err = activationFloat32Into(input, destination, activation.Tanh{})
	return err
}

func activationFloat32Into(input, destination *Tensor, function activation.DestinationActivation) (err error) {
	var (
		inputMatrix, outputMatrix *matrix.Matrix
		values                    []float32
	)

	if err = validateTensorType("input", input, Float32); err != nil {
		return err
	}
	if err = validateDestination(destination, Float32, input.shape, input); err != nil {
		return err
	}
	if len(input.floats) == 0 {
		return nil
	}
	if inputMatrix, err = matrix.FromSlice(1, len(input.floats), input.floats); err != nil {
		return fmt.Errorf("tensor: activation input matrix adapter: %w", err)
	}
	if outputMatrix, err = matrix.New(1, len(input.floats)); err != nil {
		return fmt.Errorf("tensor: activation output matrix adapter: %w", err)
	}
	if err = function.ForwardInto(inputMatrix, outputMatrix); err != nil {
		return fmt.Errorf("tensor: activation matrix adapter: %w", err)
	}
	if values, err = outputMatrix.Values(); err != nil {
		return fmt.Errorf("tensor: activation matrix values: %w", err)
	}
	copy(destination.floats, values)
	return nil
}
