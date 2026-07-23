package activation

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// ReLU applies the rectified linear unit activation.
type ReLU struct{}

// Forward returns max(0, x) for each input value.
func (r ReLU) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var (
		rows int
		cols int
	)

	if rows, cols, err = matrixShape("input", input); err != nil {
		return nil, err
	}
	if output, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}
	if err = r.ForwardInto(input, output); err != nil {
		return nil, err
	}

	return output, nil
}

// ForwardInto writes max(0, x) for each input value into output.
// It follows DestinationActivation's destination and alias contract without
// allocating.
func (r ReLU) ForwardInto(input, output *matrix.Matrix) (err error) {
	var handled bool

	if _, _, err = matrixShape("input", input); err != nil {
		return err
	}
	if handled, err = device.ReLUForward(input, output); err != nil {
		err = fmt.Errorf("activation: output matrix invalid: %w", err)
		return err
	}
	if handled {
		return nil
	}

	err = applyInto(input, output, reLUValue)
	return err
}

// Backward multiplies outputGradient by the ReLU derivative at input.
func (r ReLU) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = applyDerivative(input, outputGradient, reLUDerivative)
	return inputGradient, err
}

// BackwardInto writes the propagated ReLU gradient into inputGradient.
// It follows DestinationActivation's destination and alias contract without
// allocating.
func (r ReLU) BackwardInto(input, outputGradient, inputGradient *matrix.Matrix) (err error) {
	err = applyDerivativeInto(input, outputGradient, inputGradient, reLUDerivative)
	return err
}

func reLUValue(value float32) (result float32) {
	if value > 0 {
		result = value
		return result
	}

	return 0
}

func reLUDerivative(value float32) (result float32) {
	if value > 0 {
		return 1
	}

	return 0
}
