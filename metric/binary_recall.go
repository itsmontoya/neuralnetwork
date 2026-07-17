package metric

import "github.com/itsmontoya/neuralnetwork/matrix"

// NewBinaryRecall constructs BinaryRecall with the provided finite threshold.
func NewBinaryRecall(threshold float32) (b BinaryRecall, err error) {
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
// Custom thresholds may be any finite float32, including values outside [0, 1].
type BinaryRecall struct {
	threshold    float32
	hasThreshold bool
}

// Value returns positive-class recall for [batchSize, 1] predictions.
func (b BinaryRecall) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	var (
		truePositive   int
		targetPositive int
		threshold      float32
	)

	if threshold, err = configuredBinaryThreshold("binary recall", b.threshold, b.hasThreshold); err != nil {
		return 0, err
	}

	if _, truePositive, _, targetPositive, err = binaryPositiveTotals(
		predictions,
		targets,
		threshold,
		"binary classification",
	); err != nil {
		return 0, err
	}

	value = recallValue(truePositive, targetPositive)
	return value, nil
}
