package loss

import (
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// BinaryCrossEntropy computes cross entropy for single-output binary classification.
//
// Predictions and targets must be shaped [batchSize, 1]. Targets must contain
// binary labels encoded as 0 or 1. Predictions are clamped to a small epsilon
// before logarithms and divisions to keep boundary probabilities finite.
type BinaryCrossEntropy struct{}

// Value returns the mean binary cross entropy over the batch.
func (b BinaryCrossEntropy) Value(predictions, targets *matrix.Matrix) (value float64, err error) {
	var (
		rows             int
		predictionValues []float64
		targetValues     []float64
		index            int
		prediction       float64
		target           float64
	)

	if rows, _, predictionValues, targetValues, err = b.values(predictions, targets); err != nil {
		return 0, err
	}

	for index = range predictionValues {
		prediction = clampPrediction(predictionValues[index])
		target = targetValues[index]
		value -= target*math.Log(prediction) + (1-target)*math.Log(1-prediction)
	}

	value /= float64(rows)
	return value, nil
}

// Gradient returns the prediction gradient of the mean binary cross entropy.
func (b BinaryCrossEntropy) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
	var (
		rows             int
		predictionValues []float64
		targetValues     []float64
		gradientValues   []float64
		index            int
		prediction       float64
		target           float64
		scale            float64
	)

	if rows, _, predictionValues, targetValues, err = b.values(predictions, targets); err != nil {
		return nil, err
	}

	scale = 1 / float64(rows)
	gradientValues = make([]float64, len(predictionValues))
	for index = range predictionValues {
		prediction = clampPrediction(predictionValues[index])
		target = targetValues[index]
		gradientValues[index] = (prediction - target) / (prediction * (1 - prediction)) * scale
	}

	gradient, err = matrix.FromSlice(rows, 1, gradientValues)
	return gradient, err
}

func (b BinaryCrossEntropy) values(predictions, targets *matrix.Matrix) (rows, cols int, predictionValues, targetValues []float64, err error) {
	if rows, cols, predictionValues, targetValues, err = matrixValuePair(predictions, targets); err != nil {
		return 0, 0, nil, nil, err
	}

	if cols != 1 {
		err = fmt.Errorf("loss: binary cross entropy requires one prediction column: cols=%d", cols)
		return 0, 0, nil, nil, err
	}

	if err = validateBinaryTargets(targetValues); err != nil {
		return 0, 0, nil, nil, err
	}

	return rows, cols, predictionValues, targetValues, nil
}

func validateBinaryTargets(targetValues []float64) (err error) {
	var (
		index int
		value float64
	)

	for index, value = range targetValues {
		if value == 0 || value == 1 {
			continue
		}

		err = fmt.Errorf("loss: binary target at index %d must be 0 or 1: value=%g", index, value)
		return err
	}

	return nil
}
