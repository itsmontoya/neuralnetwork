package metric_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const epsilon = 1e-5

func mustMatrix(tb testing.TB, rows, cols int, values []float32) (m *matrix.Matrix) {
	var err error

	tb.Helper()

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func requireAlmostEqual(tb testing.TB, got, want float32) {
	tb.Helper()

	testutil.RequireAlmostEqual(tb, got, want, epsilon)
}
