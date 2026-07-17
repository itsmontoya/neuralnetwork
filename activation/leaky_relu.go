package activation

import "github.com/itsmontoya/neuralnetwork/matrix"

const leakyReLUSlope = 0.01

// LeakyReLU applies rectified linear activation with a small negative slope.
type LeakyReLU struct{}

// Forward returns x for positive inputs and 0.01*x otherwise.
func (l LeakyReLU) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output, err = apply(input, leakyReLUValue)
	return output, err
}

// ForwardInto writes the LeakyReLU result into output.
// It follows DestinationActivation's destination and alias contract without
// allocating.
func (l LeakyReLU) ForwardInto(input, output *matrix.Matrix) (err error) {
	err = applyInto(input, output, leakyReLUValue)
	return err
}

// Backward multiplies outputGradient by the LeakyReLU derivative at input.
func (l LeakyReLU) Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = applyDerivative(input, outputGradient, leakyReLUDerivative)
	return inputGradient, err
}

// BackwardInto writes the propagated LeakyReLU gradient into inputGradient.
// It follows DestinationActivation's destination and alias contract without
// allocating.
func (l LeakyReLU) BackwardInto(input, outputGradient, inputGradient *matrix.Matrix) (err error) {
	err = applyDerivativeInto(input, outputGradient, inputGradient, leakyReLUDerivative)
	return err
}

func leakyReLUValue(value float32) (result float32) {
	if value > 0 {
		result = value
		return result
	}

	result = leakyReLUSlope * value
	return result
}

func leakyReLUDerivative(value float32) (result float32) {
	if value > 0 {
		return 1
	}

	return leakyReLUSlope
}
