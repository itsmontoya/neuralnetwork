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
	var (
		truePositive      int
		predictedPositive int
		threshold         float32
	)

	if threshold, err = configuredBinaryThreshold("binary precision", b.threshold, b.hasThreshold); err != nil {
		return 0, err
	}

	if _, truePositive, predictedPositive, _, err = binaryPositiveTotals(
		predictions,
		targets,
		threshold,
		"binary classification",
	); err != nil {
		return 0, err
	}

	value = precisionValue(truePositive, predictedPositive)
	return value, nil
}
