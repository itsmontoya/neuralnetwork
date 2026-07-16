package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// MeanSquaredError reports the mean squared difference over all prediction values.
type MeanSquaredError struct{}

// Value returns the mean squared error for predictions and targets with equal shape.
func (m MeanSquaredError) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var (
		rows       int
		cols       int
		difference float32
	)

	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return 0, err
	}

	err = predictions.Pairwise(targets, func(row, col int, prediction, target float32) (err error) {
		difference = prediction - target
		value += difference * difference
		return nil
	})
	if err != nil {
		return 0, err
	}

	value /= float32(rows * cols)
	return value, nil
}
