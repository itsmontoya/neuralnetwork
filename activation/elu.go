package activation

import (
	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const eluAlpha = 1

// ELU applies the exponential linear unit activation with alpha 1.
type ELU struct{}

// Forward returns x for positive inputs and exp(x)-1 otherwise.
func (e ELU) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output, err = apply(input, eluValue)
	return output, err
}

// ForwardInto writes the ELU result into output.
func (e ELU) ForwardInto(input, output *matrix.Matrix) (err error) {
	err = applyInto(input, output, eluValue)
	return err
}

// Backward multiplies outputGradient by the ELU derivative at input.
func (e ELU) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = applyDerivative(input, outputGradient, eluDerivative)
	return inputGradient, err
}

// BackwardInto writes the propagated ELU gradient into inputGradient.
func (e ELU) BackwardInto(input, outputGradient, inputGradient *matrix.Matrix) (err error) {
	err = applyDerivativeInto(input, outputGradient, inputGradient, eluDerivative)
	return err
}

func eluValue(value float32) (result float32) {
	if value > 0 {
		result = value
		return result
	}

	result = eluAlpha * (f32.Exp(value) - 1)
	return result
}

func eluDerivative(value float32) (result float32) {
	if value > 0 {
		return 1
	}

	result = eluAlpha * f32.Exp(value)
	return result
}
