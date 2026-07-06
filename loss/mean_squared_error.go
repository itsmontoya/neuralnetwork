package loss

import "github.com/itsmontoya/neuralnetwork/matrix"

// MeanSquaredError computes the mean squared difference over all prediction values.
type MeanSquaredError struct{}

// Value returns the mean squared error for predictions and targets with equal shape.
func (m MeanSquaredError) Value(predictions, targets *matrix.Matrix) (value float64, err error) {
	var (
		rows       int
		cols       int
		difference float64
	)

	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return 0, err
	}

	err = predictions.Pairwise(targets, func(row, col int, prediction, target float64) (err error) {
		difference = prediction - target
		value += difference * difference
		return nil
	})
	if err != nil {
		return 0, err
	}

	value /= float64(rows * cols)
	return value, nil
}

// Gradient returns the prediction gradient of the mean squared error.
func (m MeanSquaredError) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
	var (
		rows  int
		cols  int
		scale float64
	)

	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return nil, err
	}

	if gradient, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}

	if err = predictions.SubtractInto(targets, gradient); err != nil {
		return nil, err
	}

	scale = 2 / float64(rows*cols)
	if err = gradient.MultiplyScalarInPlace(scale); err != nil {
		return nil, err
	}

	return gradient, nil
}
