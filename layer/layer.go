// Package layer defines neural network layer contracts and implementations.
package layer

import "github.com/itsmontoya/neuralnetwork/matrix"

// Layer transforms batched inputs during forward passes and propagates output
// gradients during backward passes.
//
// Implementations should retain only the cached state needed to compute the
// next backward pass.
type Layer interface {
	// Forward transforms a batched input matrix.
	Forward(input *matrix.Matrix) (output *matrix.Matrix, err error)
	// Backward propagates output gradients to input gradients.
	Backward(outputGradient *matrix.Matrix) (inputGradient *matrix.Matrix, err error)
}
