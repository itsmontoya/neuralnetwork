package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// MeanSquaredError reports the mean squared difference over all prediction values.
type MeanSquaredError struct{}

// Value returns the mean squared error for predictions and targets with equal shape.
func (m MeanSquaredError) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var (
		rows             int
		cols             int
		predictionValues []float32
		targetValues     []float32
		index            int
		difference       float32
	)

	if rows, cols, predictionValues, targetValues, err = matrixValuePair(predictions, targets); err != nil {
		return 0, err
	}

	for index = range predictionValues {
		difference = predictionValues[index] - targetValues[index]
		value += difference * difference
	}

	value /= float32(rows * cols)
	return value, nil
}
