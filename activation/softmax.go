package activation

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// Softmax applies row-wise normalized exponentials for batched inputs.
type Softmax struct{}

// Forward returns a row-wise Softmax using the row maximum for numerical stability.
func (s Softmax) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
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

	if err = s.ForwardInto(input, output); err != nil {
		return nil, err
	}

	return output, nil
}

// Backward multiplies outputGradient by the row-wise Softmax Jacobian.
func (s Softmax) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var (
		rows int
		cols int
	)

	if rows, cols, err = matrixPairShape(input, outputGradient); err != nil {
		return nil, err
	}

	if inputGradient, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}

	if err = s.BackwardInto(input, outputGradient, inputGradient); err != nil {
		return nil, err
	}

	return inputGradient, nil
}

// ForwardInto writes a row-wise Softmax into output.
func (s Softmax) ForwardInto(input, output *matrix.Matrix) (err error) {
	if err = input.SoftmaxRowsInto(output); err != nil {
		err = fmt.Errorf("activation: softmax forward failed: %w", err)
		return err
	}

	return nil
}

// BackwardInto writes the product of outputGradient and the row-wise Softmax
// Jacobian into inputGradient.
func (s Softmax) BackwardInto(input, outputGradient, inputGradient *matrix.Matrix) (err error) {
	if err = input.SoftmaxRowsBackwardInto(outputGradient, inputGradient); err != nil {
		err = fmt.Errorf("activation: softmax backward failed: %w", err)
		return err
	}

	return nil
}
