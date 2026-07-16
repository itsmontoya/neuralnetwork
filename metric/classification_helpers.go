package metric

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func configuredBinaryThreshold(metricName string, threshold float32, hasThreshold bool) (configured float32, err error) {
	configured = defaultBinaryThreshold
	if hasThreshold {
		configured = threshold
	}

	if f32.IsNaN(configured) || f32.IsInf(configured, 0) {
		err = fmt.Errorf("metric: %s threshold must be finite: threshold=%g", metricName, configured)
		return 0, err
	}

	return configured, nil
}

func binaryPositiveTotals(
	predictions,
	targets *matrix.Matrix,
	threshold float32,
	requirementName string,
) (rows, truePositive, predictedPositive, targetPositive int, err error) {
	var (
		cols           int
		predictedClass int
		targetClass    int
	)

	if rows, cols, err = matrixShapePair(predictions, targets); err != nil {
		return 0, 0, 0, 0, err
	}

	if cols != 1 {
		err = fmt.Errorf("metric: %s requires one prediction column: cols=%d", requirementName, cols)
		return 0, 0, 0, 0, err
	}

	err = predictions.Pairwise(targets, func(row, col int, prediction, target float32) (err error) {
		if err = validateBinaryTarget(row, target); err != nil {
			return err
		}

		predictedClass = 0
		if prediction >= threshold {
			predictedClass = 1
			predictedPositive++
		}

		targetClass = int(target)
		if targetClass == 1 {
			targetPositive++
		}

		if predictedClass == 1 && targetClass == 1 {
			truePositive++
		}

		return nil
	})
	if err != nil {
		return 0, 0, 0, 0, err
	}

	return rows, truePositive, predictedPositive, targetPositive, nil
}

func categoricalClassSummary(
	predictions,
	targets *matrix.Matrix,
	cols int,
	counts []int,
) (correct int, err error) {
	var (
		predictedClass int
		targetClass    int
		ones           int
		maximum        float32
	)

	if counts != nil && len(counts) != cols*cols {
		err = fmt.Errorf("metric: categorical count length mismatch: got %d, want %d", len(counts), cols*cols)
		return 0, err
	}

	err = predictions.Pairwise(targets, func(row, col int, prediction, target float32) (err error) {
		if col == 0 {
			predictedClass = 0
			targetClass = 0
			ones = 0
			maximum = prediction
		} else if !(prediction <= maximum) {
			predictedClass = col
			maximum = prediction
		}

		if target == 1 {
			targetClass = col
			ones++
		} else if target != 0 {
			err = fmt.Errorf("metric: categorical target at row %d column %d must be 0 or 1: value=%g", row, col, target)
			return err
		}

		if col != cols-1 {
			return nil
		}

		if ones != 1 {
			err = fmt.Errorf("metric: categorical target row %d must contain exactly one class: ones=%d", row, ones)
			return err
		}

		if predictedClass == targetClass {
			correct++
		}

		if counts != nil {
			counts[targetClass*cols+predictedClass]++
		}

		return nil
	})
	if err != nil {
		return 0, err
	}

	return correct, nil
}

func precisionValue(truePositive, predictedPositive int) (value float32) {
	if predictedPositive == 0 {
		return 0
	}

	value = float32(truePositive) / float32(predictedPositive)
	return value
}

func recallValue(truePositive, targetPositive int) (value float32) {
	if targetPositive == 0 {
		return 0
	}

	value = float32(truePositive) / float32(targetPositive)
	return value
}

func f1Value(truePositive, predictedPositive, targetPositive int) (value float32) {
	var (
		precision float32
		recall    float32
	)

	precision = precisionValue(truePositive, predictedPositive)
	recall = recallValue(truePositive, targetPositive)
	if precision+recall == 0 {
		return 0
	}

	value = 2 * precision * recall / (precision + recall)
	return value
}
