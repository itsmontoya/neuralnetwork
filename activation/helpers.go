package activation

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func apply(input *matrix.Matrix, fn func(float64) float64) (output *matrix.Matrix, err error) {
	if _, _, err = matrixShape("input", input); err != nil {
		return nil, err
	}

	if output, err = input.Apply(fn); err != nil {
		err = fmt.Errorf("activation: input matrix invalid: %w", err)
		return nil, err
	}

	return output, nil
}

func applyDerivative(input, outputGradient *matrix.Matrix, derivative func(float64) float64) (inputGradient *matrix.Matrix, err error) {
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

func matrixShape(name string, input *matrix.Matrix) (rows, cols int, err error) {
	if err = input.Validate(); err != nil {
		err = fmt.Errorf("activation: %s matrix invalid: %w", name, err)
		return 0, 0, err
	}

	rows, cols = input.Shape()
	return rows, cols, nil
}

func matrixValues(name string, input *matrix.Matrix) (rows, cols int, values []float64, err error) {
	if rows, cols, err = matrixShape(name, input); err != nil {
		return 0, 0, nil, err
	}

	if values, err = input.Values(); err != nil {
		err = fmt.Errorf("activation: %s matrix invalid: %w", name, err)
		return 0, 0, nil, err
	}

	return rows, cols, values, nil
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

func matrixValuePair(input, outputGradient *matrix.Matrix) (rows, cols int, inputValues, gradientValues []float64, err error) {
	if rows, cols, err = matrixPairShape(input, outputGradient); err != nil {
		return 0, 0, nil, nil, err
	}

	if inputValues, err = input.Values(); err != nil {
		err = fmt.Errorf("activation: input matrix invalid: %w", err)
		return 0, 0, nil, nil, err
	}

	if gradientValues, err = outputGradient.Values(); err != nil {
		err = fmt.Errorf("activation: output gradient matrix invalid: %w", err)
		return 0, 0, nil, nil, err
	}

	return rows, cols, inputValues, gradientValues, nil
}
