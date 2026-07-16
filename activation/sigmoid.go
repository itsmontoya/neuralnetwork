package activation

import (
	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// Sigmoid applies the logistic activation.
type Sigmoid struct{}

// Forward returns 1/(1+exp(-x)) for each input value.
func (s Sigmoid) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output, err = apply(input, sigmoidValue)
	return output, err
}

// ForwardInto writes the Sigmoid result into output.
// It follows DestinationActivation's destination and alias contract without
// allocating.
func (s Sigmoid) ForwardInto(input, output *matrix.Matrix) (err error) {
	err = applyInto(input, output, sigmoidValue)
	return err
}

// Backward multiplies outputGradient by the Sigmoid derivative at input.
func (s Sigmoid) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = applyDerivative(input, outputGradient, sigmoidDerivative)
	return inputGradient, err
}

// BackwardInto writes the propagated Sigmoid gradient into inputGradient.
// It follows DestinationActivation's destination and alias contract without
// allocating.
func (s Sigmoid) BackwardInto(input, outputGradient, inputGradient *matrix.Matrix) (err error) {
	err = applyDerivativeInto(input, outputGradient, inputGradient, sigmoidDerivative)
	return err
}

func sigmoidValue(value float32) (result float32) {
	var exponent float32

	if value >= 0 {
		exponent = f32.Exp(-value)
		result = 1 / (1 + exponent)
		return result
	}

	exponent = f32.Exp(value)
	result = exponent / (1 + exponent)
	return result
}

func sigmoidDerivative(value float32) (result float32) {
	var activated float32

	activated = sigmoidValue(value)
	result = activated * (1 - activated)
	return result
}
