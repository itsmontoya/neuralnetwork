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

// ForwardInto copies input into output.
func (l Linear) ForwardInto(input, output *matrix.Matrix) (err error) {
	if _, _, err = matrixShape("input", input); err != nil {
		return err
	}

	if err = output.CopyFrom(input); err != nil {
		err = fmt.Errorf("activation: output matrix invalid: %w", err)
		return err
	}

	return nil
}

// Backward returns a copy of outputGradient after validating its shape.
func (l Linear) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	if _, _, err = matrixPairShape(input, outputGradient); err != nil {
		return nil, err
	}

	if inputGradient, err = outputGradient.Clone(); err != nil {
		err = fmt.Errorf("activation: output gradient matrix invalid: %w", err)
		return nil, err
	}

	return inputGradient, nil
}

// BackwardInto copies outputGradient into inputGradient after validating input.
func (l Linear) BackwardInto(input, outputGradient, inputGradient *matrix.Matrix) (err error) {
	if _, _, err = matrixPairShape(input, outputGradient); err != nil {
		return err
	}

	if inputGradient == outputGradient {
		err = fmt.Errorf("activation: input gradient must not alias output gradient")
		return err
	}

	if err = inputGradient.CopyFrom(outputGradient); err != nil {
		err = fmt.Errorf("activation: input gradient matrix invalid: %w", err)
		return err
	}

	return nil
}
