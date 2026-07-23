package loss

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

// CategoricalCrossEntropy computes cross entropy for one-hot classification targets.
//
// Predictions and targets must have matching [batchSize, classCount] shape.
// Each target row must be one-hot encoded with exactly one value set to 1 and
// all other values set to 0. Predictions are clamped to a small epsilon before
// logarithms and divisions to keep boundary probabilities finite.
type CategoricalCrossEntropy struct{}

// Value returns the mean categorical cross entropy over the batch.
func (c CategoricalCrossEntropy) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var (
		rows       int
		currentRow int
		ones       int
		prediction float32
		target     float32
		handled    bool
	)

	if rows, _, err = matrixShapePair(predictions, targets); err != nil {
		return 0, err
	}
	if value, handled, err = device.CategoricalCrossEntropyValue(
		predictions,
		targets,
		predictionEpsilon,
	); err != nil {
		return 0, err
	}
	if handled {
		return value, nil
	}

	currentRow = -1
	err = predictions.Pairwise(targets, func(row, col int, left, right float32) (err error) {
		if row != currentRow {
			if currentRow >= 0 && ones != 1 {
				err = fmt.Errorf("loss: categorical target row %d must contain exactly one class: ones=%d", currentRow, ones)
				return err
			}

			currentRow = row
			ones = 0
		}

		prediction = left
		target = right
		if target == 1 {
			ones++
			prediction = clampPrediction(prediction)
			value -= f32.Log(prediction)
			return nil
		}

		if target != 0 {
			err = fmt.Errorf("loss: categorical target at row %d column %d must be 0 or 1: value=%g", row, col, target)
			return err
		}

		return nil
	})
	if err != nil {
		return 0, err
	}

	if ones != 1 {
		err = fmt.Errorf("loss: categorical target row %d must contain exactly one class: ones=%d", currentRow, ones)
		return 0, err
	}

	value /= float32(rows)
	return value, nil
}

// Gradient returns the prediction gradient of the mean categorical cross entropy.
func (c CategoricalCrossEntropy) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
	var (
		rows int
		cols int
	)

	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return nil, err
	}

	if gradient, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}

	if err = c.gradientInto(predictions, targets, gradient, rows); err != nil {
		return nil, err
	}

	return gradient, nil
}

// GradientInto writes the prediction gradient into destination.
// It follows DestinationGradient's destination and alias contract without
// allocating.
func (c CategoricalCrossEntropy) GradientInto(predictions, targets, destination *matrix.Matrix) (err error) {
	var rows int

	if rows, _, err = matrixShapePair(predictions, targets); err != nil {
		return err
	}

	err = c.gradientInto(predictions, targets, destination, rows)
	return err
}

func (c CategoricalCrossEntropy) gradientInto(
	predictions,
	targets,
	destination *matrix.Matrix,
	rows int,
) (err error) {
	var (
		prediction float32
		target     float32
		scale      float32
		handled    bool
	)

	if handled, err = device.CategoricalCrossEntropyGradient(
		predictions,
		targets,
		destination,
		predictionEpsilon,
	); err != nil {
		return err
	}
	if handled {
		return nil
	}
	if err = validateOneHotTargets(targets); err != nil {
		return err
	}

	scale = 1 / float32(rows)
	err = predictions.PairwiseInto(targets, destination, func(row, col int, left, right float32) (value float32, err error) {
		prediction = left
		target = right
		if target == 0 {
			return 0, nil
		}

		prediction = clampPrediction(prediction)
		value = -target / prediction * scale
		return value, nil
	})
	if err != nil {
		return err
	}

	return nil
}
func validateOneHotTargets(targets *matrix.Matrix) (err error) {
	var (
		currentRow int
		ones       int
		value      float32
	)

	currentRow = -1
	err = targets.Pairwise(targets, func(row, col int, left, right float32) (err error) {
		if row != currentRow {
			if currentRow >= 0 && ones != 1 {
				err = fmt.Errorf("loss: categorical target row %d must contain exactly one class: ones=%d", currentRow, ones)
				return err
			}

			currentRow = row
			ones = 0
		}

		value = left
		if value == 1 {
			ones++
			return nil
		}

		if value != 0 {
			err = fmt.Errorf("loss: categorical target at row %d column %d must be 0 or 1: value=%g", row, col, value)
			return err
		}

		return nil
	})
	if err != nil {
		return err
	}

	if currentRow >= 0 && ones != 1 {
		err = fmt.Errorf("loss: categorical target row %d must contain exactly one class: ones=%d", currentRow, ones)
		return err
	}

	return nil
}
