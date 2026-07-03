package loss

import "github.com/itsmontoya/neuralnetwork/matrix"

// MeanSquaredError computes the mean squared difference over all prediction values.
type MeanSquaredError struct{}

// Value returns the mean squared error for predictions and targets with equal shape.
func (m MeanSquaredError) Value(predictions, targets *matrix.Matrix) (value float64, err error) {
	var (
		rows             int
		cols             int
		predictionValues []float64
		targetValues     []float64
		index            int
		difference       float64
	)

	if rows, cols, predictionValues, targetValues, err = matrixValuePair(predictions, targets); err != nil {
		return 0, err
	}

	for index = range predictionValues {
		difference = predictionValues[index] - targetValues[index]
		value += difference * difference
	}

	value /= float64(rows * cols)
	return value, nil
}

// Gradient returns the prediction gradient of the mean squared error.
func (m MeanSquaredError) Gradient(predictions, targets *matrix.Matrix) (gradient *matrix.Matrix, err error) {
	var (
		rows             int
		cols             int
		predictionValues []float64
		targetValues     []float64
		gradientValues   []float64
		index            int
		scale            float64
	)

	if rows, cols, predictionValues, targetValues, err = matrixValuePair(predictions, targets); err != nil {
		return nil, err
	}

	scale = 2 / float64(rows*cols)
	gradientValues = make([]float64, len(predictionValues))
	for index = range predictionValues {
		gradientValues[index] = (predictionValues[index] - targetValues[index]) * scale
	}

	gradient, err = matrix.FromSlice(rows, cols, gradientValues)
	return gradient, err
}
