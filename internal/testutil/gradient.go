package testutil

import (
	"fmt"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// ScalarObjective computes a scalar value from the current test state.
type ScalarObjective func() (value float32, err error)

// FiniteDifferenceGradient approximates the gradient of objective with respect
// to values using central finite differences.
func FiniteDifferenceGradient(values *matrix.Matrix, epsilon float32, objective ScalarObjective) (gradient *matrix.Matrix, err error) {
	var (
		rows          int
		cols          int
		row           int
		col           int
		originalValue float32
		forwardValue  float32
		backwardValue float32
	)

	if values == nil {
		err = fmt.Errorf("testutil: gradient values matrix is nil")
		return nil, err
	}

	if epsilon <= 0 {
		err = fmt.Errorf("testutil: finite difference epsilon must be positive: epsilon=%g", epsilon)
		return nil, err
	}

	if objective == nil {
		err = fmt.Errorf("testutil: scalar objective is nil")
		return nil, err
	}

	rows, cols = values.Shape()
	if gradient, err = matrix.New(rows, cols); err != nil {
		return nil, err
	}

	for row = 0; row < rows; row++ {
		for col = 0; col < cols; col++ {
			if originalValue, err = values.At(row, col); err != nil {
				return nil, err
			}

			if err = values.Set(row, col, originalValue+epsilon); err != nil {
				return nil, err
			}

			if forwardValue, err = objective(); err != nil {
				restoreMatrixValue(values, row, col, originalValue)
				return nil, err
			}

			if err = values.Set(row, col, originalValue-epsilon); err != nil {
				restoreMatrixValue(values, row, col, originalValue)
				return nil, err
			}

			if backwardValue, err = objective(); err != nil {
				restoreMatrixValue(values, row, col, originalValue)
				return nil, err
			}

			if err = values.Set(row, col, originalValue); err != nil {
				return nil, err
			}

			if err = gradient.Set(row, col, (forwardValue-backwardValue)/(2*epsilon)); err != nil {
				return nil, err
			}
		}
	}

	return gradient, nil
}

// WeightedMatrixSum returns sum(values[i] * weights[i]) for two matrices with
// matching shapes.
func WeightedMatrixSum(values, weights *matrix.Matrix) (sum float32, err error) {
	var (
		valueRows    int
		valueCols    int
		weightRows   int
		weightCols   int
		valueValues  []float32
		weightValues []float32
		index        int
	)

	if values == nil {
		err = fmt.Errorf("testutil: weighted sum values matrix is nil")
		return 0, err
	}

	if weights == nil {
		err = fmt.Errorf("testutil: weighted sum weights matrix is nil")
		return 0, err
	}

	valueRows, valueCols = values.Shape()
	weightRows, weightCols = weights.Shape()
	if valueRows != weightRows || valueCols != weightCols {
		err = fmt.Errorf(
			"testutil: weighted sum shape mismatch: values %dx%d, weights %dx%d",
			valueRows,
			valueCols,
			weightRows,
			weightCols,
		)
		return 0, err
	}

	if valueValues, err = values.Values(); err != nil {
		return 0, err
	}

	if weightValues, err = weights.Values(); err != nil {
		return 0, err
	}

	for index = range valueValues {
		sum += valueValues[index] * weightValues[index]
	}

	return sum, nil
}

// RequireMatrixAlmostEqual fails tb when got and want differ in shape or values.
func RequireMatrixAlmostEqual(tb testing.TB, got, want *matrix.Matrix, epsilon float32) {
	var (
		gotRows    int
		gotCols    int
		wantRows   int
		wantCols   int
		gotValues  []float32
		wantValues []float32
		err        error
	)

	tb.Helper()

	if got == nil {
		tb.Fatal("got matrix is nil")
	}

	if want == nil {
		tb.Fatal("want matrix is nil")
	}

	gotRows, gotCols = got.Shape()
	wantRows, wantCols = want.Shape()
	if gotRows != wantRows || gotCols != wantCols {
		tb.Fatalf("matrix shapes differ: got %dx%d, want %dx%d", gotRows, gotCols, wantRows, wantCols)
	}

	gotValues, err = got.Values()
	if err != nil {
		tb.Fatalf("got Values returned error: %v", err)
	}

	wantValues, err = want.Values()
	if err != nil {
		tb.Fatalf("want Values returned error: %v", err)
	}

	RequireSliceAlmostEqual(tb, gotValues, wantValues, epsilon)
}

func restoreMatrixValue(values *matrix.Matrix, row, col int, value float32) {
	_ = values.Set(row, col, value)
}
