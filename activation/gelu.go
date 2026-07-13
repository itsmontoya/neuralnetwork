package activation

import (
	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	geluInverseSqrtTwo   = 0.7071067811865475
	geluInverseSqrtTwoPi = 0.3989422804014327
)

// GELU applies the exact Gaussian error linear unit activation.
type GELU struct{}

// Forward returns x*Phi(x), where Phi is the standard normal CDF.
func (g GELU) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output, err = apply(input, geluValue)
	return output, err
}

// Backward multiplies outputGradient by the GELU derivative at input.
func (g GELU) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = applyDerivative(input, outputGradient, geluDerivative)
	return inputGradient, err
}

func geluValue(value float32) (result float32) {
	result = 0.5 * value * (1 + f32.Erf(value*geluInverseSqrtTwo))
	return result
}

func geluDerivative(value float32) (result float32) {
	var density float32

	density = geluInverseSqrtTwoPi * f32.Exp(-0.5*value*value)
	result = 0.5*(1+f32.Erf(value*geluInverseSqrtTwo)) + value*density
	return result
}
