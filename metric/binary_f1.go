package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// NewBinaryF1 constructs BinaryF1 with the provided threshold.
func NewBinaryF1(threshold float64) (b BinaryF1, err error) {
	if _, err = configuredBinaryThreshold("binary f1", threshold, true); err != nil {
		return b, err
	}

	b.threshold = threshold
	b.hasThreshold = true
	return b, nil
}

// BinaryF1 reports positive-class F1 for binary predictions.
type BinaryF1 struct {
	threshold    float64
	hasThreshold bool
}

// Value returns positive-class F1 for [batchSize, 1] predictions.
func (b BinaryF1) Value(predictions, targets *matrix.Matrix) (value float64, err error) {
	var confusionMatrix *ConfusionMatrix

	if confusionMatrix, err = b.confusionMatrix(predictions, targets); err != nil {
		return 0, err
	}

	value, err = confusionMatrix.F1(1)
	return value, err
}

func (b BinaryF1) confusionMatrix(predictions, targets *matrix.Matrix) (confusionMatrix *ConfusionMatrix, err error) {
	var threshold float64

	if threshold, err = configuredBinaryThreshold("binary f1", b.threshold, b.hasThreshold); err != nil {
		return nil, err
	}

	confusionMatrix, err = NewBinaryConfusionMatrixWithThreshold(predictions, targets, threshold)
	return confusionMatrix, err
}
