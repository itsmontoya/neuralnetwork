package activation

import (
	"math"

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

func geluValue(value float64) (result float64) {
	result = 0.5 * value * (1 + math.Erf(value*geluInverseSqrtTwo))
	return result
}

func geluDerivative(value float64) (result float64) {
	var density float64

	density = geluInverseSqrtTwoPi * math.Exp(-0.5*value*value)
	result = 0.5*(1+math.Erf(value*geluInverseSqrtTwo)) + value*density
	return result
}
