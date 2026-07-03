// Package activation provides stateless activation functions for neural network layers.
package activation

import "github.com/itsmontoya/neuralnetwork/matrix"

// Activation transforms matrix values during a forward pass and propagates
// output gradients back to input gradients.
type Activation interface {
	// Forward applies the activation to input values.
	Forward(input *matrix.Matrix) (output *matrix.Matrix, err error)
	// Backward converts output gradients into input gradients.
	Backward(input, outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error)
}
