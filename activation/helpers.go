package activation

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func apply(input *matrix.Matrix, fn func(float32) float32) (output *matrix.Matrix, err error) {
	if _, _, err = matrixShape("input", input); err != nil {
		return nil, err
	}

	if output, err = input.Apply(fn); err != nil {
		err = fmt.Errorf("activation: input matrix invalid: %w", err)
		return nil, err
	}

	return output, nil
}

func applyInto(input, output *matrix.Matrix, fn func(float32) float32) (err error) {
	if _, _, err = matrixShape("input", input); err != nil {
		return err
	}

	if err = input.ApplyInto(fn, output); err != nil {
		err = fmt.Errorf("activation: output matrix invalid: %w", err)
		return err
	}

	return nil
}

func applyDerivative(input, outputGradient *matrix.Matrix, derivative func(float32) float32) (inputGradient *matrix.Matrix, err error) {
	if _, _, err = matrixPairShape(input, outputGradient); err != nil {
		return nil, err
	}

	if inputGradient, err = input.Apply(derivative); err != nil {
		err = fmt.Errorf("activation: input matrix invalid: %w", err)
		return nil, err
	}

	if err = inputGradient.MultiplyElementsInto(outputGradient, inputGradient); err != nil {
		err = fmt.Errorf("activation: derivative multiply failed: %w", err)
		return nil, err
	}

	return inputGradient, nil
}

func applyDerivativeInto(
	input, outputGradient, inputGradient *matrix.Matrix,
	derivative func(float32) float32,
) (err error) {
	if _, _, err = matrixPairShape(input, outputGradient); err != nil {
		return err
	}

	if inputGradient == outputGradient {
		err = errors.New("activation: input gradient must not alias output gradient")
		return err
	}

	if err = input.ApplyInto(derivative, inputGradient); err != nil {
		err = fmt.Errorf("activation: input gradient matrix invalid: %w", err)
		return err
	}

	if err = inputGradient.MultiplyElementsInto(outputGradient, inputGradient); err != nil {
		err = fmt.Errorf("activation: derivative multiply failed: %w", err)
		return err
	}

	return nil
}

func matrixShape(name string, input *matrix.Matrix) (rows, cols int, err error) {
	if err = input.Validate(); err != nil {
		err = fmt.Errorf("activation: %s matrix invalid: %w", name, err)
		return 0, 0, err
	}

	rows, cols = input.Shape()
	return rows, cols, nil
}

func matrixPairShape(input, outputGradient *matrix.Matrix) (rows, cols int, err error) {
	var (
		gradientRows int
		gradientCols int
	)

	if rows, cols, err = matrixShape("input", input); err != nil {
		return 0, 0, err
	}

	if gradientRows, gradientCols, err = matrixShape("output gradient", outputGradient); err != nil {
		return 0, 0, err
	}

	if rows != gradientRows || cols != gradientCols {
		err = fmt.Errorf(
			"activation: gradient shape mismatch: input %dx%d, gradient %dx%d",
			rows,
			cols,
			gradientRows,
			gradientCols,
		)
		return 0, 0, err
	}

	return rows, cols, nil
}
