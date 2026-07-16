package loss

import "github.com/itsmontoya/neuralnetwork/matrix"

// DestinationGradient optionally writes a loss gradient into caller-owned storage.
//
// The destination must have the same shape as predictions and targets. On
// success, GradientInto fully overwrites destination and retains none of its
// arguments. Destination may alias predictions or targets because each input
// value is read before the corresponding destination value is written. Valid
// calls do not allocate.
type DestinationGradient interface {
	// GradientInto writes the prediction gradient into destination.
	GradientInto(predictions, targets, destination *matrix.Matrix) (err error)
}
