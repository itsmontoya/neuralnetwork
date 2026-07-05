package layer

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_BatchNormalization_BackwardValidatesNormalizedCache(t *testing.T) {
	var (
		batchNorm *BatchNormalization
		err       error
	)

	batchNorm = mustInternalBatchNormalization(t)
	batchNorm.normalizedCache = nil

	_, err = batchNorm.Backward(mustInternalMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	}))
	if err == nil {
		t.Fatal("Backward error = nil, want cache error")
	}

	if !strings.Contains(err.Error(), "normalized cache is nil") {
		t.Fatalf("Backward error = %q, want normalized cache error", err.Error())
	}
}

func Test_BatchNormalization_BackwardValidatesInverseStdCache(t *testing.T) {
	var (
		batchNorm *BatchNormalization
		err       error
	)

	batchNorm = mustInternalBatchNormalization(t)
	batchNorm.inverseStdCache = []float64{1}

	_, err = batchNorm.Backward(mustInternalMatrix(t, 2, 2, []float64{
		1, 2,
		3, 4,
	}))
	if err == nil {
		t.Fatal("Backward error = nil, want cache error")
	}

	if !strings.Contains(err.Error(), "inverse std cache length mismatch") {
		t.Fatalf("Backward error = %q, want inverse std cache error", err.Error())
	}
}

func mustInternalBatchNormalization(tb testing.TB) (batchNorm *BatchNormalization) {
	var err error

	tb.Helper()

	batchNorm, err = NewBatchNormalization(2)
	if err != nil {
		tb.Fatalf("NewBatchNormalization returned error: %v", err)
	}

	_, err = batchNorm.Forward(mustInternalMatrix(tb, 2, 2, []float64{
		1, 2,
		3, 6,
	}))
	if err != nil {
		tb.Fatalf("Forward returned error: %v", err)
	}

	return batchNorm
}

func mustInternalMatrix(tb testing.TB, rows, cols int, values []float64) (m *matrix.Matrix) {
	var err error

	tb.Helper()

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}
