package loss

import (
	"fmt"
	"math"

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
func (c CategoricalCrossEntropy) Value(predictions, targets *matrix.Matrix) (value float64, err error) {
	var (
		rows       int
		currentRow int
		ones       int
		prediction float64
		target     float64
	)

	if rows, _, err = matrixShapePair(predictions, targets); err != nil {
		return 0, err
	}

	currentRow = -1
	err = predictions.Pairwise(targets, func(row, col int, left, right float64) (err error) {
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
			value -= math.Log(prediction)
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

	value /= float64(rows)
	return value, nil
}

// Gradient returns the prediction gradient of the mean categorical cross entropy.
func (c CategoricalCrossEntropy) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
	var (
		rows       int
		cols       int
		prediction float64
		target     float64
		scale      float64
	)

	if rows, cols, err = c.shape(predictions, targets); err != nil {
		return nil, err
	}

	if gradient, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}

	scale = 1 / float64(rows)
	err = predictions.PairwiseInto(targets, gradient, func(row, col int, left, right float64) (value float64, err error) {
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
		return nil, err
	}

	return gradient, nil
}

func (c CategoricalCrossEntropy) shape(predictions, targets *matrix.Matrix) (rows, cols int, err error) {
	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return 0, 0, err
	}

	if err = validateOneHotTargets(targets); err != nil {
		return 0, 0, err
	}

	return rows, cols, nil
}

func validateOneHotTargets(targets *matrix.Matrix) (err error) {
	var (
		currentRow int
		ones       int
		value      float64
	)

	currentRow = -1
	err = targets.Pairwise(targets, func(row, col int, left, right float64) (err error) {
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
