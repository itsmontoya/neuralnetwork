package loss

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// BinaryCrossEntropy computes cross entropy for single-output binary classification.
//
// Predictions and targets must be shaped [batchSize, 1]. Targets must contain
// binary labels encoded as 0 or 1. Predictions are clamped to a small epsilon
// before logarithms and divisions to keep boundary probabilities finite.
type BinaryCrossEntropy struct{}

// Value returns the mean binary cross entropy over the batch.
func (b BinaryCrossEntropy) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var (
		rows       int
		prediction float32
		target     float32
	)

	if rows, _, err = b.shape(predictions, targets); err != nil {
		return 0, err
	}

	err = predictions.Pairwise(targets, func(row, col int, left, right float32) (err error) {
		prediction = left
		target = right
		if err = validateBinaryTarget(row, target); err != nil {
			return err
		}

		prediction = clampPrediction(prediction)
		value -= target*f32.Log(prediction) + (1-target)*f32.Log(1-prediction)
		return nil
	})
	if err != nil {
		return 0, err
	}

	value /= float32(rows)
	return value, nil
}

// Gradient returns the prediction gradient of the mean binary cross entropy.
func (b BinaryCrossEntropy) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
	var rows int

	if rows, _, err = b.shape(predictions, targets); err != nil {
		return nil, err
	}

	if err = validateBinaryTargets(targets); err != nil {
		return nil, err
	}

	if gradient, err = matrix.New(rows, 1); err != nil {
		return nil, err
	}

	if err = b.gradientInto(predictions, targets, gradient, rows); err != nil {
		return nil, err
	}

	return gradient, nil
}

// GradientInto writes the prediction gradient into destination.
func (b BinaryCrossEntropy) GradientInto(predictions, targets, destination *matrix.Matrix) (err error) {
	var rows int

	if rows, _, err = b.shape(predictions, targets); err != nil {
		return err
	}

	if err = validateBinaryTargets(targets); err != nil {
		return err
	}

	err = b.gradientInto(predictions, targets, destination, rows)
	return err
}

func (b BinaryCrossEntropy) gradientInto(
	predictions,
	targets,
	destination *matrix.Matrix,
	rows int,
) (err error) {
	var (
		prediction float32
		target     float32
		scale      float32
	)

	scale = 1 / float32(rows)
	err = predictions.PairwiseInto(targets, destination, func(row, col int, left, right float32) (value float32, err error) {
		prediction = left
		target = right
		prediction = clampPrediction(prediction)
		value = (prediction - target) / (prediction * (1 - prediction)) * scale
		return value, nil
	})
	if err != nil {
		return err
	}

	return nil
}

func (b BinaryCrossEntropy) shape(predictions, targets *matrix.Matrix) (rows, cols int, err error) {
	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return 0, 0, err
	}

	if cols != 1 {
		err = fmt.Errorf("loss: binary cross entropy requires one prediction column: cols=%d", cols)
		return 0, 0, err
	}

	return rows, cols, nil
}

func validateBinaryTargets(targets *matrix.Matrix) (err error) {
	err = targets.Pairwise(targets, func(row, col int, value, right float32) (err error) {
		if err = validateBinaryTarget(row, value); err != nil {
			return err
		}

		return nil
	})
	if err != nil {
		return err
	}

	return nil
}

func validateBinaryTarget(index int, value float32) (err error) {
	if value == 0 || value == 1 {
		return nil
	}

	err = fmt.Errorf("loss: binary target at index %d must be 0 or 1: value=%g", index, value)
	return err
}
