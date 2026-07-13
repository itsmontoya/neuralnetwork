// Package metric defines reporting metrics separate from optimization losses.
package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// Metric evaluates predictions against targets for reporting.
type Metric interface {
	// Value computes a reporting metric from predictions and targets.
	Value(predictions, targets *matrix.Matrix) (value float32, err error)
}
