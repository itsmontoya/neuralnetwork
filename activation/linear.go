package activation

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// Linear leaves input values unchanged.
type Linear struct{}

// Forward returns a copy of the input matrix.
func (l Linear) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	if output, err = input.Clone(); err != nil {
		err = fmt.Errorf("activation: input matrix invalid: %w", err)
		return nil, err
	}

	return output, err
}

// Backward returns a copy of outputGradient after validating its shape.
func (l Linear) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var (
		rows           int
		cols           int
		gradientValues []float64
	)

	if rows, cols, _, gradientValues, err = matrixValuePair(input, outputGradient); err != nil {
		return nil, err
	}

	inputGradient, err = matrix.FromSlice(rows, cols, gradientValues)
	return inputGradient, err
}
