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
		rows             int
		cols             int
		predictionValues []float64
		targetValues     []float64
		row              int
		col              int
		index            int
		prediction       float64
	)

	if rows, cols, predictionValues, targetValues, err = c.values(predictions, targets); err != nil {
		return 0, err
	}

	for row = 0; row < rows; row++ {
		for col = 0; col < cols; col++ {
			index = row*cols + col
			if targetValues[index] == 0 {
				continue
			}

			prediction = clampPrediction(predictionValues[index])
			value -= math.Log(prediction)
		}
	}

	value /= float64(rows)
	return value, nil
}

// Gradient returns the prediction gradient of the mean categorical cross entropy.
func (c CategoricalCrossEntropy) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
	var (
		rows             int
		cols             int
		predictionValues []float64
		targetValues     []float64
		gradientValues   []float64
		index            int
		prediction       float64
		scale            float64
	)

	if rows, cols, predictionValues, targetValues, err = c.values(predictions, targets); err != nil {
		return nil, err
	}

	scale = 1 / float64(rows)
	gradientValues = make([]float64, len(predictionValues))
	for index = range predictionValues {
		if targetValues[index] == 0 {
			continue
		}

		prediction = clampPrediction(predictionValues[index])
		gradientValues[index] = -targetValues[index] / prediction * scale
	}

	gradient, err = matrix.FromSlice(rows, cols, gradientValues)
	return gradient, err
}

func (c CategoricalCrossEntropy) values(predictions, targets *matrix.Matrix) (rows, cols int, predictionValues, targetValues []float64, err error) {
	if rows, cols, predictionValues, targetValues, err = matrixValuePair(predictions, targets); err != nil {
		return 0, 0, nil, nil, err
	}

	if err = validateOneHotTargets(rows, cols, targetValues); err != nil {
		return 0, 0, nil, nil, err
	}

	return rows, cols, predictionValues, targetValues, nil
}

func validateOneHotTargets(rows, cols int, targetValues []float64) (err error) {
	var (
		row   int
		col   int
		index int
		ones  int
		value float64
	)

	for row = 0; row < rows; row++ {
		ones = 0
		for col = 0; col < cols; col++ {
			index = row*cols + col
			value = targetValues[index]
			if value == 1 {
				ones++
				continue
			}

			if value == 0 {
				continue
			}

			err = fmt.Errorf("loss: categorical target at row %d column %d must be 0 or 1: value=%g", row, col, value)
			return err
		}

		if ones != 1 {
			err = fmt.Errorf("loss: categorical target row %d must contain exactly one class: ones=%d", row, ones)
			return err
		}
	}

	return nil
}
