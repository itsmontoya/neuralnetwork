package activation

import "github.com/itsmontoya/neuralnetwork/matrix"

// ReLU applies the rectified linear unit activation.
type ReLU struct{}

// Forward returns max(0, x) for each input value.
func (r ReLU) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output, err = apply(input, reLUValue)
	return output, err
}

// Backward multiplies outputGradient by the ReLU derivative at input.
func (r ReLU) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = applyDerivative(input, outputGradient, reLUDerivative)
	return inputGradient, err
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
