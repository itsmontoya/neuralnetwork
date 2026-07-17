package activation

import "github.com/itsmontoya/neuralnetwork/matrix"

// DestinationActivation writes activation results into caller-owned matrices.
// Destinations must match the input shape and are fully overwritten on success.
// ForwardInto permits output to alias input. BackwardInto permits inputGradient
// to alias input, but not outputGradient.
type DestinationActivation interface {
	// ForwardInto writes the activated input values into output.
	ForwardInto(input, output *matrix.Matrix) (err error)
	// BackwardInto writes the propagated output gradient into inputGradient.
	BackwardInto(input, outputGradient, inputGradient *matrix.Matrix) (err error)
}
