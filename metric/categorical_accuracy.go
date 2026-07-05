package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// CategoricalAccuracy reports the fraction of correct one-hot categorical predictions.
//
// The predicted class is the first maximum value in each prediction row. Targets
// must be one-hot encoded.
type CategoricalAccuracy struct{}

// Value returns categorical accuracy for one-hot classification targets.
func (c CategoricalAccuracy) Value(predictions, targets *matrix.Matrix) (value float64, err error) {
	var (
		rows             int
		cols             int
		predictionValues []float64
		targetValues     []float64
		row              int
		predictedClass   int
		targetClass      int
		correct          int
	)

	if rows, cols, predictionValues, targetValues, err = c.values(predictions, targets); err != nil {
		return 0, err
	}

	for row = 0; row < rows; row++ {
		predictedClass = rowArgmax(predictionValues, row, cols)
		targetClass = rowArgmax(targetValues, row, cols)
		if predictedClass != targetClass {
			continue
		}

		correct++
	}

	value = float64(correct) / float64(rows)
	return value, nil
}

func (c CategoricalAccuracy) values(predictions, targets *matrix.Matrix) (rows, cols int, predictionValues, targetValues []float64, err error) {
	if rows, cols, predictionValues, targetValues, err = matrixValuePair(predictions, targets); err != nil {
		return 0, 0, nil, nil, err
	}

	if err = validateOneHotTargets(rows, cols, targetValues); err != nil {
		return 0, 0, nil, nil, err
	}

	return rows, cols, predictionValues, targetValues, nil
}
