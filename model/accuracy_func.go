package model

import "github.com/itsmontoya/neuralnetwork/matrix"

// AccuracyFunc computes an accuracy value from predictions and targets.
type AccuracyFunc func(predictions, targets *matrix.Matrix) (accuracy float32, err error)
