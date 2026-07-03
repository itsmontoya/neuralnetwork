package metric

import (
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func configuredBinaryThreshold(metricName string, threshold float64, hasThreshold bool) (configured float64, err error) {
	configured = defaultBinaryThreshold
	if hasThreshold {
		configured = threshold
	}

	if math.IsNaN(configured) || math.IsInf(configured, 0) {
		err = fmt.Errorf("metric: %s threshold must be finite: threshold=%g", metricName, configured)
		return 0, err
	}

	return configured, nil
}

func binaryClassValues(predictions, targets *matrix.Matrix, threshold float64) (predictedClasses, targetClasses []int, err error) {
	var (
		rows             int
		cols             int
		predictionValues []float64
		targetValues     []float64
		index            int
	)

	if math.IsNaN(threshold) || math.IsInf(threshold, 0) {
		err = fmt.Errorf("metric: binary classification threshold must be finite: threshold=%g", threshold)
		return nil, nil, err
	}

	if rows, cols, predictionValues, targetValues, err = matrixValuePair(predictions, targets); err != nil {
		return nil, nil, err
	}

	if cols != 1 {
		err = fmt.Errorf("metric: binary classification requires one prediction column: cols=%d", cols)
		return nil, nil, err
	}

	if err = validateBinaryTargets(targetValues); err != nil {
		return nil, nil, err
	}

	predictedClasses = make([]int, rows)
	targetClasses = make([]int, rows)
	for index = range predictionValues {
		if predictionValues[index] >= threshold {
			predictedClasses[index] = 1
		}

		targetClasses[index] = int(targetValues[index])
	}

	return predictedClasses, targetClasses, nil
}

func categoricalClassValues(predictions, targets *matrix.Matrix) (classCount int, predictedClasses, targetClasses []int, err error) {
	var (
		rows             int
		cols             int
		predictionValues []float64
		targetValues     []float64
		row              int
	)

	if rows, cols, predictionValues, targetValues, err = matrixValuePair(predictions, targets); err != nil {
		return 0, nil, nil, err
	}

	if err = validateOneHotTargets(rows, cols, targetValues); err != nil {
		return 0, nil, nil, err
	}

	predictedClasses = make([]int, rows)
	targetClasses = make([]int, rows)
	for row = 0; row < rows; row++ {
		predictedClasses[row] = rowArgmax(predictionValues, row, cols)
		targetClasses[row] = rowArgmax(targetValues, row, cols)
	}

	classCount = cols
	return classCount, predictedClasses, targetClasses, nil
}
