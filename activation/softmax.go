package activation

import (
	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// Softmax applies row-wise normalized exponentials for batched inputs.
type Softmax struct{}

// Forward returns a row-wise Softmax using the row maximum for numerical stability.
func (s Softmax) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var (
		rows   int
		cols   int
		values []float32
		result []float32
	)

	if rows, cols, values, err = matrixValues("input", input); err != nil {
		return nil, err
	}

	result = softmaxRows(rows, cols, values)
	output, err = matrix.FromSlice(rows, cols, result)
	return output, err
}

// Backward multiplies outputGradient by the row-wise Softmax Jacobian.
func (s Softmax) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	var (
		rows           int
		cols           int
		inputValues    []float32
		gradientValues []float32
		softmaxValues  []float32
		result         []float32
		row            int
		col            int
		offset         int
		dot            float32
	)

	if rows, cols, inputValues, gradientValues, err = matrixValuePair(input, outputGradient); err != nil {
		return nil, err
	}

	softmaxValues = softmaxRows(rows, cols, inputValues)
	result = make([]float32, len(inputValues))

	for row = 0; row < rows; row++ {
		offset = row * cols
		dot = 0

		for col = 0; col < cols; col++ {
			dot += gradientValues[offset+col] * softmaxValues[offset+col]
		}

		for col = 0; col < cols; col++ {
			result[offset+col] = softmaxValues[offset+col] * (gradientValues[offset+col] - dot)
		}
	}

	inputGradient, err = matrix.FromSlice(rows, cols, result)
	return inputGradient, err
}

func softmaxRows(rows, cols int, values []float32) (result []float32) {
	var (
		row      int
		col      int
		offset   int
		maxValue float32
		value    float32
		sum      float32
	)

	result = make([]float32, len(values))
	for row = 0; row < rows; row++ {
		offset = row * cols
		maxValue = values[offset]

		for col = 1; col < cols; col++ {
			value = values[offset+col]
			if value > maxValue {
				maxValue = value
			}
		}

		sum = 0
		for col = 0; col < cols; col++ {
			value = f32.Exp(values[offset+col] - maxValue)
			result[offset+col] = value
			sum += value
		}

		for col = 0; col < cols; col++ {
			result[offset+col] /= sum
		}
	}

	return result
}
