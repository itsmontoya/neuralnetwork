package metric

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const defaultBinaryThreshold = 0.5

// NewBinaryAccuracy constructs BinaryAccuracy with the provided finite threshold.
func NewBinaryAccuracy(threshold float32) (b BinaryAccuracy, err error) {
	if f32.IsNaN(threshold) || f32.IsInf(threshold, 0) {
		err = fmt.Errorf("metric: binary accuracy threshold must be finite: threshold=%g", threshold)
		return b, err
	}

	b.threshold = threshold
	b.hasThreshold = true
	return b, nil
}

// BinaryAccuracy reports the fraction of correct single-output binary predictions.
//
// The zero value uses a threshold of 0.5. Predictions greater than or equal to
// the threshold are treated as class 1; lower predictions are treated as class 0.
// Custom thresholds may be any finite float32, including values outside [0, 1].
type BinaryAccuracy struct {
	threshold    float32
	hasThreshold bool
}

// Value returns binary accuracy for [batchSize, 1] predictions and binary targets.
func (b BinaryAccuracy) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var (
		rows             int
		predictionValues []float32
		targetValues     []float32
		index            int
		threshold        float32
		predictedClass   float32
		correct          int
	)

	if rows, _, predictionValues, targetValues, err = b.values(predictions, targets); err != nil {
		return 0, err
	}

	if threshold, err = b.configuredThreshold(); err != nil {
		return 0, err
	}

	for index = range predictionValues {
		predictedClass = 0
		if predictionValues[index] >= threshold {
			predictedClass = 1
		}

		if predictedClass != targetValues[index] {
			continue
		}

		correct++
	}

	value = float32(correct) / float32(rows)
	return value, nil
}

func (b BinaryAccuracy) values(predictions, targets *matrix.Matrix) (rows, cols int, predictionValues, targetValues []float32, err error) {
	if rows, cols, predictionValues, targetValues, err = matrixValuePair(predictions, targets); err != nil {
		return 0, 0, nil, nil, err
	}

	if cols != 1 {
		err = fmt.Errorf("metric: binary accuracy requires one prediction column: cols=%d", cols)
		return 0, 0, nil, nil, err
	}

	if err = validateBinaryTargets(targetValues); err != nil {
		return 0, 0, nil, nil, err
	}

	return rows, cols, predictionValues, targetValues, nil
}

func (b BinaryAccuracy) configuredThreshold() (threshold float32, err error) {
	threshold = defaultBinaryThreshold
	if b.hasThreshold {
		threshold = b.threshold
	}

	if f32.IsNaN(threshold) || f32.IsInf(threshold, 0) {
		err = fmt.Errorf("metric: binary accuracy threshold must be finite: threshold=%g", threshold)
		return 0, err
	}

	return threshold, nil
}
