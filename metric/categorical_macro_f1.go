package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// CategoricalMacroF1 reports macro-averaged F1 for one-hot targets.
type CategoricalMacroF1 struct{}

// Value returns macro-averaged F1 across classes.
func (c CategoricalMacroF1) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var confusionMatrix ConfusionMatrix

	if confusionMatrix, err = categoricalConfusionMatrix(predictions, targets); err != nil {
		return 0, err
	}

	value, err = confusionMatrix.MacroF1()
	return value, err
}
