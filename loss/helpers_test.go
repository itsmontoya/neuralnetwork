package loss_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const epsilon = 1e-12

func mustMatrix(tb testing.TB, rows, cols int, values []float64) (m *matrix.Matrix) {
	var err error

	tb.Helper()

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func requireMatrixValues(tb testing.TB, got *matrix.Matrix, want []float64) {
	var (
		values []float64
		err    error
	)

	tb.Helper()

	values, err = got.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(tb, values, want, epsilon)
}

func requireFinite(tb testing.TB, value float64) {
	tb.Helper()

	if math.IsInf(value, 0) || math.IsNaN(value) {
		tb.Fatalf("value is not finite: %g", value)
	}
}

func requireFiniteMatrix(tb testing.TB, got *matrix.Matrix) {
	var (
		values []float64
		index  int
		err    error
	)

	tb.Helper()

	values, err = got.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	for index = range values {
		requireFinite(tb, values[index])
	}
}
