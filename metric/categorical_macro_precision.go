package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// CategoricalMacroPrecision reports macro-averaged precision for one-hot targets.
type CategoricalMacroPrecision struct{}

// Value returns macro-averaged precision across classes.
func (c CategoricalMacroPrecision) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var confusionMatrix ConfusionMatrix

	if confusionMatrix, err = categoricalConfusionMatrix(predictions, targets); err != nil {
		return 0, err
	}

	value, err = confusionMatrix.MacroPrecision()
	return value, err
}
