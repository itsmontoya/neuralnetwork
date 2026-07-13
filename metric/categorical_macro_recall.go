package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// CategoricalMacroRecall reports macro-averaged recall for one-hot targets.
type CategoricalMacroRecall struct{}

// Value returns macro-averaged recall across classes.
func (c CategoricalMacroRecall) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var confusionMatrix *ConfusionMatrix

	if confusionMatrix, err = NewCategoricalConfusionMatrix(predictions, targets); err != nil {
		return 0, err
	}

	value, err = confusionMatrix.MacroRecall()
	return value, err
}
