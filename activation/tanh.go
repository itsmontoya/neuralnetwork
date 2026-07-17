package activation

import (
	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// Tanh applies the hyperbolic tangent activation.
type Tanh struct{}

// Forward returns tanh(x) for each input value.
func (t Tanh) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output, err = apply(input, tanhValue)
	return output, err
}

// ForwardInto writes the hyperbolic tangent result into output.
func (t Tanh) ForwardInto(input, output *matrix.Matrix) (err error) {
	err = applyInto(input, output, tanhValue)
	return err
}

// Backward multiplies outputGradient by the Tanh derivative at input.
func (t Tanh) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = applyDerivative(input, outputGradient, tanhDerivative)
	return inputGradient, err
}

// BackwardInto writes the propagated Tanh gradient into inputGradient.
func (t Tanh) BackwardInto(input, outputGradient, inputGradient *matrix.Matrix) (err error) {
	err = applyDerivativeInto(input, outputGradient, inputGradient, tanhDerivative)
	return err
}

func tanhValue(value float32) (result float32) {
	result = f32.Tanh(value)
	return result
}

func tanhDerivative(value float32) (result float32) {
	var activated float32

	activated = f32.Tanh(value)
	result = 1 - activated*activated
	return result
}
