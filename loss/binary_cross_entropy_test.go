package loss_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_BinaryCrossEntropy_Value(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float32
		err         error
	)

	predictions = mustMatrix(t, 2, 1, []float32{0.8, 0.25})
	targets = mustMatrix(t, 2, 1, []float32{1, 0})

	got, err = loss.BinaryCrossEntropy{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("Value returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, got, 0.25541281188299536, epsilon)
}

func Test_BinaryCrossEntropy_Gradient(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		gradient    *matrix.Matrix
		err         error
	)

	predictions = mustMatrix(t, 2, 1, []float32{0.8, 0.25})
	targets = mustMatrix(t, 2, 1, []float32{1, 0})

	gradient, err = loss.BinaryCrossEntropy{}.Gradient(predictions, targets)
	if err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}

	if gradient.Rows() != 2 {
		t.Fatalf("Gradient rows = %d, want 2", gradient.Rows())
	}

	if gradient.Cols() != 1 {
		t.Fatalf("Gradient cols = %d, want 1", gradient.Cols())
	}

	requireMatrixValues(t, gradient, []float32{-0.625, 0.6666666666666666})
}

func Test_BinaryCrossEntropy_StableAroundZeroAndOne(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		gradient    *matrix.Matrix
		value       float32
		lower       float32
		upper       float32
		wantValue   float32
		err         error
	)

	predictions = mustMatrix(t, 2, 1, []float32{0, 1})
	targets = mustMatrix(t, 2, 1, []float32{1, 0})

	value, err = loss.BinaryCrossEntropy{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("Value returned error: %v", err)
	}

	requireFinite(t, value)
	lower = clampEpsilon
	upper = 1 - clampEpsilon
	wantValue = -(f32.Log(lower) + f32.Log(1-upper)) / 2
	testutil.RequireAlmostEqual(t, value, wantValue, epsilon)

	gradient, err = loss.BinaryCrossEntropy{}.Gradient(predictions, targets)
	if err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}

	requireFiniteMatrix(t, gradient)
	requireMatrixValues(t, gradient, []float32{
		(lower - 1) / (lower * (1 - lower)) / 2,
		upper / (upper * (1 - upper)) / 2,
	})
}

func Test_BinaryCrossEntropy_ValidatesTargetFormat(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float32
		err         error
	)

	predictions = mustMatrix(t, 1, 1, []float32{0.5})
	targets = mustMatrix(t, 1, 1, []float32{0.5})

	got, err = loss.BinaryCrossEntropy{}.Value(predictions, targets)
	if err == nil {
		t.Fatalf("Value returned %g and nil error, want error", got)
	}
}

func Test_BinaryCrossEntropy_GradientValidatesTargetFormat(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		gradient    *matrix.Matrix
		err         error
	)

	predictions = mustMatrix(t, 1, 1, []float32{0.5})
	targets = mustMatrix(t, 1, 1, []float32{0.5})

	gradient, err = loss.BinaryCrossEntropy{}.Gradient(predictions, targets)
	if err == nil {
		t.Fatalf("Gradient returned %#v and nil error, want error", gradient)
	}
}

func Test_BinaryCrossEntropy_ValidatesSingleOutput(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float32
		err         error
	)

	predictions = mustMatrix(t, 1, 2, []float32{0.5, 0.5})
	targets = mustMatrix(t, 1, 2, []float32{1, 0})

	got, err = loss.BinaryCrossEntropy{}.Value(predictions, targets)
	if err == nil {
		t.Fatalf("Value returned %g and nil error, want error", got)
	}
}
