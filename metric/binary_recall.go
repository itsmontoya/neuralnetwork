package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// NewBinaryRecall constructs BinaryRecall with the provided finite threshold.
func NewBinaryRecall(threshold float64) (b BinaryRecall, err error) {
	if _, err = configuredBinaryThreshold("binary recall", threshold, true); err != nil {
		return b, err
	}

	b.threshold = threshold
	b.hasThreshold = true
	return b, nil
}

// BinaryRecall reports positive-class recall for binary predictions.
//
// The zero value uses a threshold of 0.5. Predictions greater than or equal to
// the threshold are treated as class 1; lower predictions are treated as class 0.
// Custom thresholds may be any finite float64, including values outside [0, 1].
type BinaryRecall struct {
	threshold    float64
	hasThreshold bool
}

// Value returns positive-class recall for [batchSize, 1] predictions.
func (b BinaryRecall) Value(predictions, targets *matrix.Matrix) (value float64, err error) {
	var confusionMatrix *ConfusionMatrix

	if confusionMatrix, err = b.confusionMatrix(predictions, targets); err != nil {
		return 0, err
	}

	value, err = confusionMatrix.Recall(1)
	return value, err
}

func (b BinaryRecall) confusionMatrix(predictions, targets *matrix.Matrix) (confusionMatrix *ConfusionMatrix, err error) {
	var threshold float64

	if threshold, err = configuredBinaryThreshold("binary recall", b.threshold, b.hasThreshold); err != nil {
		return nil, err
	}

	confusionMatrix, err = NewBinaryConfusionMatrixWithThreshold(predictions, targets, threshold)
	return confusionMatrix, err
}
