package metric

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func matrixShapePair(predictions, targets *matrix.Matrix) (rows, cols int, err error) {
	var (
		targetRows int
		targetCols int
	)

	if err = predictions.Validate(); err != nil {
		err = fmt.Errorf("metric: predictions matrix invalid: %w", err)
		return 0, 0, err
	}

	if err = targets.Validate(); err != nil {
		err = fmt.Errorf("metric: targets matrix invalid: %w", err)
		return 0, 0, err
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
		return 0, 0, err
	}

	return rows, cols, nil
}

func validateBinaryTarget(index int, value float32) (err error) {
	if value == 0 || value == 1 {
		return nil
	}

	err = fmt.Errorf("metric: binary target at index %d must be 0 or 1: value=%g", index, value)
	return err
}
