package loss

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

const predictionEpsilon = 1e-15

func matrixValuePair(predictions, targets *matrix.Matrix) (rows, cols int, predictionValues, targetValues []float64, err error) {
	var (
		targetRows int
		targetCols int
	)

	if predictionValues, err = predictions.Values(); err != nil {
		err = fmt.Errorf("loss: predictions matrix invalid: %w", err)
		return 0, 0, nil, nil, err
	}

	if targetValues, err = targets.Values(); err != nil {
		err = fmt.Errorf("loss: targets matrix invalid: %w", err)
		return 0, 0, nil, nil, err
	}

	rows, cols = predictions.Shape()
	targetRows, targetCols = targets.Shape()
	if rows != targetRows || cols != targetCols {
		err = fmt.Errorf(
			"loss: shape mismatch: predictions %dx%d, targets %dx%d",
			rows,
			cols,
			targetRows,
			targetCols,
		)
		return 0, 0, nil, nil, err
	}

	return rows, cols, predictionValues, targetValues, nil
}

func clampPrediction(value float64) (clamped float64) {
	if value < predictionEpsilon {
		return predictionEpsilon
	}

	if value > 1-predictionEpsilon {
		return 1 - predictionEpsilon
	}

	return value
}
