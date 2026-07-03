package activation

import (
	"math"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// Sigmoid applies the logistic activation.
type Sigmoid struct{}

// Forward returns 1/(1+exp(-x)) for each input value.
func (s Sigmoid) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output, err = apply(input, sigmoidValue)
	return output, err
}

// Backward multiplies outputGradient by the Sigmoid derivative at input.
func (s Sigmoid) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = applyDerivative(input, outputGradient, sigmoidDerivative)
	return inputGradient, err
}

func sigmoidValue(value float64) (result float64) {
	var exponent float64

	if value >= 0 {
		exponent = math.Exp(-value)
		result = 1 / (1 + exponent)
		return result
	}

	exponent = math.Exp(value)
	result = exponent / (1 + exponent)
	return result
}

func sigmoidDerivative(value float64) (result float64) {
	var activated float64

	activated = sigmoidValue(value)
	result = activated * (1 - activated)
	return result
}
