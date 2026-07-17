package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// CategoricalAccuracy reports the fraction of correct one-hot categorical predictions.
//
// The predicted class is the first maximum value in each prediction row. Targets
// must be one-hot encoded.
type CategoricalAccuracy struct{}

// Value returns categorical accuracy for one-hot classification targets.
func (c CategoricalAccuracy) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var (
		rows    int
		cols    int
		correct int
	)

	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return 0, err
	}

	if correct, err = categoricalClassSummary(predictions, targets, cols, nil); err != nil {
		return 0, err
	}

	value = float32(correct) / float32(rows)
	return value, nil
}
