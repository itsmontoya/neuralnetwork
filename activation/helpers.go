package activation

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func apply(input *matrix.Matrix, fn func(float64) float64) (output *matrix.Matrix, err error) {
	var (
		rows   int
		cols   int
		values []float64
		result []float64
		index  int
	)

	if rows, cols, values, err = matrixValues("input", input); err != nil {
		return nil, err
	}

	result = make([]float64, len(values))
	for index = range values {
		result[index] = fn(values[index])
	}

	output, err = matrix.FromSlice(rows, cols, result)
	return output, err
}

func applyDerivative(input, outputGradient *matrix.Matrix, derivative func(float64) float64) (inputGradient *matrix.Matrix, err error) {
	var (
		rows           int
		cols           int
		inputValues    []float64
		gradientValues []float64
		result         []float64
		index          int
	)

	if rows, cols, inputValues, gradientValues, err = matrixValuePair(input, outputGradient); err != nil {
		return nil, err
	}

	result = make([]float64, len(inputValues))
	for index = range inputValues {
		result[index] = gradientValues[index] * derivative(inputValues[index])
	}

	inputGradient, err = matrix.FromSlice(rows, cols, result)
	return inputGradient, err
}

func matrixValues(name string, input *matrix.Matrix) (rows, cols int, values []float64, err error) {
	if values, err = input.Values(); err != nil {
		err = fmt.Errorf("activation: %s matrix invalid: %w", name, err)
		return 0, 0, nil, err
	}

	rows, cols = input.Shape()
	return rows, cols, values, nil
}

func matrixValuePair(input, outputGradient *matrix.Matrix) (rows, cols int, inputValues, gradientValues []float64, err error) {
	var (
		gradientRows int
		gradientCols int
	)

	if rows, cols, inputValues, err = matrixValues("input", input); err != nil {
		return 0, 0, nil, nil, err
	}

	if gradientRows, gradientCols, gradientValues, err = matrixValues("output gradient", outputGradient); err != nil {
		return 0, 0, nil, nil, err
	}

	if rows != gradientRows || cols != gradientCols {
		err = fmt.Errorf(
			"activation: gradient shape mismatch: input %dx%d, gradient %dx%d",
			rows,
			cols,
			gradientRows,
			gradientCols,
		)
		return 0, 0, nil, nil, err
	}

	return rows, cols, inputValues, gradientValues, nil
}
