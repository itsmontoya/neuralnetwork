package loss_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
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

	got, err = loss.MeanSquaredError{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("Value returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, got, 0.8125, epsilon)
}

func Test_MeanSquaredError_Gradient(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		gradient    *matrix.Matrix
		err         error
	)

	predictions = mustMatrix(t, 2, 2, []float64{1, 2, 3, 4})
	targets = mustMatrix(t, 2, 2, []float64{1.5, 1, 2, 5})

	gradient, err = loss.MeanSquaredError{}.Gradient(predictions, targets)
	if err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}

	if gradient.Rows() != 2 {
		t.Fatalf("Gradient rows = %d, want 2", gradient.Rows())
	}

	if gradient.Cols() != 2 {
		t.Fatalf("Gradient cols = %d, want 2", gradient.Cols())
	}

	requireMatrixValues(t, gradient, []float64{-0.25, 0.5, 0.5, -0.5})
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

	got, err = loss.MeanSquaredError{}.Value(predictions, targets)
	if err == nil {
		t.Fatalf("Value returned %g and nil error, want error", got)
	}
}

func Test_MeanSquaredError_GradientValidatesShape(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		gradient    *matrix.Matrix
		err         error
	)

	predictions = mustMatrix(t, 1, 2, []float64{1, 2})
	targets = mustMatrix(t, 2, 1, []float64{1, 2})

	gradient, err = loss.MeanSquaredError{}.Gradient(predictions, targets)
	if err == nil {
		t.Fatalf("Gradient returned %#v and nil error, want error", gradient)
	}
}
