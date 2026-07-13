package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// NewBinaryPrecision constructs BinaryPrecision with the provided finite threshold.
func NewBinaryPrecision(threshold float32) (b BinaryPrecision, err error) {
	if _, err = configuredBinaryThreshold("binary precision", threshold, true); err != nil {
		return b, err
	}

	b.threshold = threshold
	b.hasThreshold = true
	return b, nil
}

// BinaryPrecision reports positive-class precision for binary predictions.
//
// The zero value uses a threshold of 0.5. Predictions greater than or equal to
// the threshold are treated as class 1; lower predictions are treated as class 0.
// Custom thresholds may be any finite float32, including values outside [0, 1].
type BinaryPrecision struct {
	threshold    float32
	hasThreshold bool
}

// Value returns positive-class precision for [batchSize, 1] predictions.
func (b BinaryPrecision) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var confusionMatrix *ConfusionMatrix

	if confusionMatrix, err = b.confusionMatrix(predictions, targets); err != nil {
		return 0, err
	}

	value, err = confusionMatrix.Precision(1)
	return value, err
}

func (b BinaryPrecision) confusionMatrix(predictions, targets *matrix.Matrix) (confusionMatrix *ConfusionMatrix, err error) {
	var threshold float32

	if threshold, err = configuredBinaryThreshold("binary precision", b.threshold, b.hasThreshold); err != nil {
		return nil, err
	}

	confusionMatrix, err = NewBinaryConfusionMatrixWithThreshold(predictions, targets, threshold)
	return confusionMatrix, err
}
