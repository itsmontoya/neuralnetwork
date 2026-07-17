package loss

import "github.com/itsmontoya/neuralnetwork/matrix"

// MeanSquaredError computes the mean squared difference over all prediction values.
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

// Gradient returns the prediction gradient of the mean squared error.
func (m MeanSquaredError) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
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

	if err = m.gradientInto(predictions, targets, gradient, rows, cols); err != nil {
		return nil, err
	}

	return gradient, nil
}

// GradientInto writes the prediction gradient into destination.
// It follows DestinationGradient's destination and alias contract without
// allocating.
func (m MeanSquaredError) GradientInto(predictions, targets, destination *matrix.Matrix) (err error) {
	var (
		rows int
		cols int
	)

	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return err
	}

	err = m.gradientInto(predictions, targets, destination, rows, cols)
	return err
}

func (m MeanSquaredError) gradientInto(
	predictions,
	targets,
	destination *matrix.Matrix,
	rows,
	cols int,
) (err error) {
	var scale float32

	if err = predictions.SubtractInto(targets, destination); err != nil {
		return err
	}

	scale = 2 / float32(rows*cols)
	if err = destination.MultiplyScalarInPlace(scale); err != nil {
		return err
	}

	return nil
}
