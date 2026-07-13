// Package loss defines training loss contracts and implementations.
package loss

import "github.com/itsmontoya/neuralnetwork/matrix"

// Loss evaluates predictions against targets and computes prediction gradients.
type Loss interface {
	// Value computes the scalar loss for predictions and targets.
	Value(predictions, targets *matrix.Matrix) (value float32, err error)
	// Gradient computes the derivative of the loss with respect to predictions.
	Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error)
}
