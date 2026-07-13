package loss

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

const predictionEpsilon = 1e-7

func matrixShapePair(predictions, targets *matrix.Matrix) (rows, cols int, err error) {
	var (
		targetRows int
		targetCols int
	)

	if err = predictions.Validate(); err != nil {
		err = fmt.Errorf("loss: predictions matrix invalid: %w", err)
		return 0, 0, err
	}

	if err = targets.Validate(); err != nil {
		err = fmt.Errorf("loss: targets matrix invalid: %w", err)
		return 0, 0, err
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
		return 0, 0, err
	}

	return rows, cols, nil
}

func clampPrediction(value float32) (clamped float32) {
	if value < predictionEpsilon {
		return predictionEpsilon
	}

	if value > 1-predictionEpsilon {
		return 1 - predictionEpsilon
	}

	return value
}
