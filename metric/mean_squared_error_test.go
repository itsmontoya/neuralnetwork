package metric_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
)

func Test_MeanSquaredError_Value(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float64
		err         error
	)

	predictions = mustMatrix(t, 2, 2, []float64{1, 2, 3, 4})
	targets = mustMatrix(t, 2, 2, []float64{1.5, 1, 2, 5})

	got, err = metric.MeanSquaredError{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("Value returned error: %v", err)
	}

	requireAlmostEqual(t, got, 0.8125)
}

func Test_MeanSquaredError_ValidatesShape(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float64
		err         error
	)

	predictions = mustMatrix(t, 1, 2, []float64{1, 2})
	targets = mustMatrix(t, 2, 1, []float64{1, 2})

	got, err = metric.MeanSquaredError{}.Value(predictions, targets)
	if err == nil {
		t.Fatalf("Value returned %g and nil error, want error", got)
	}
}
