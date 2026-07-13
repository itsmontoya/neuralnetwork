package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// NewBinaryF1 constructs BinaryF1 with the provided finite threshold.
func NewBinaryF1(threshold float32) (b BinaryF1, err error) {
	if _, err = configuredBinaryThreshold("binary f1", threshold, true); err != nil {
		return b, err
	}

	b.threshold = threshold
	b.hasThreshold = true
	return b, nil
}

// BinaryF1 reports positive-class F1 for binary predictions.
//
// The zero value uses a threshold of 0.5. Predictions greater than or equal to
// the threshold are treated as class 1; lower predictions are treated as class 0.
// Custom thresholds may be any finite float32, including values outside [0, 1].
type BinaryF1 struct {
	threshold    float32
	hasThreshold bool
}

// Value returns positive-class F1 for [batchSize, 1] predictions.
func (b BinaryF1) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var confusionMatrix *ConfusionMatrix

	if confusionMatrix, err = b.confusionMatrix(predictions, targets); err != nil {
		return 0, err
	}

	value, err = confusionMatrix.F1(1)
	return value, err
}

func (b BinaryF1) confusionMatrix(predictions, targets *matrix.Matrix) (confusionMatrix *ConfusionMatrix, err error) {
	var threshold float32

	if threshold, err = configuredBinaryThreshold("binary f1", b.threshold, b.hasThreshold); err != nil {
		return nil, err
	}

	confusionMatrix, err = NewBinaryConfusionMatrixWithThreshold(predictions, targets, threshold)
	return confusionMatrix, err
}
