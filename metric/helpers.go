package metric

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func matrixValuePair(predictions, targets *matrix.Matrix) (rows, cols int, predictionValues, targetValues []float64, err error) {
	var (
		targetRows int
		targetCols int
	)

	if predictionValues, err = predictions.Values(); err != nil {
		err = fmt.Errorf("metric: predictions matrix invalid: %w", err)
		return 0, 0, nil, nil, err
	}

	if targetValues, err = targets.Values(); err != nil {
		err = fmt.Errorf("metric: targets matrix invalid: %w", err)
		return 0, 0, nil, nil, err
	}

	rows, cols = predictions.Shape()
	targetRows, targetCols = targets.Shape()
	if rows != targetRows || cols != targetCols {
		err = fmt.Errorf(
			"metric: shape mismatch: predictions %dx%d, targets %dx%d",
			rows,
			cols,
			targetRows,
			targetCols,
		)
		return 0, 0, nil, nil, err
	}

	return rows, cols, predictionValues, targetValues, nil
}

func validateBinaryTargets(targetValues []float64) (err error) {
	var (
		index int
		value float64
	)

	for index, value = range targetValues {
		if value == 0 || value == 1 {
			continue
		}

		err = fmt.Errorf("metric: binary target at index %d must be 0 or 1: value=%g", index, value)
		return err
	}

	return nil
}

func validateOneHotTargets(rows, cols int, targetValues []float64) (err error) {
	var (
		row   int
		col   int
		index int
		ones  int
		value float64
	)

	for row = 0; row < rows; row++ {
		ones = 0
		for col = 0; col < cols; col++ {
			index = row*cols + col
			value = targetValues[index]
			if value == 1 {
				ones++
				continue
			}

			if value == 0 {
				continue
			}

			err = fmt.Errorf("metric: categorical target at row %d column %d must be 0 or 1: value=%g", row, col, value)
			return err
		}

		if ones != 1 {
			err = fmt.Errorf("metric: categorical target row %d must contain exactly one class: ones=%d", row, ones)
			return err
		}
	}

	return nil
}

func rowArgmax(values []float64, row, cols int) (argmax int) {
	var (
		col   int
		index int
		max   float64
		value float64
	)

	index = row * cols
	max = values[index]

	for col = 1; col < cols; col++ {
		index = row*cols + col
		value = values[index]
		if value <= max {
			continue
		}

		max = value
		argmax = col
	}

	return argmax
}
