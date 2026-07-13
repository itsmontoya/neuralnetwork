package loss_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_CategoricalCrossEntropy_Value(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float32
		err         error
	)

	predictions = mustMatrix(t, 2, 3, []float32{
		0.7, 0.2, 0.1,
		0.1, 0.6, 0.3,
	})
	targets = mustMatrix(t, 2, 3, []float32{
		1, 0, 0,
		0, 1, 0,
	})

	got, err = loss.CategoricalCrossEntropy{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("Value returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, got, 0.4337502838523616, epsilon)
}

func Test_CategoricalCrossEntropy_Gradient(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		gradient    *matrix.Matrix
		err         error
	)

	predictions = mustMatrix(t, 2, 3, []float32{
		0.7, 0.2, 0.1,
		0.1, 0.6, 0.3,
	})
	targets = mustMatrix(t, 2, 3, []float32{
		1, 0, 0,
		0, 1, 0,
	})

	gradient, err = loss.CategoricalCrossEntropy{}.Gradient(predictions, targets)
	if err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}

	if gradient.Rows() != 2 {
		t.Fatalf("Gradient rows = %d, want 2", gradient.Rows())
	}

	if gradient.Cols() != 3 {
		t.Fatalf("Gradient cols = %d, want 3", gradient.Cols())
	}

	requireMatrixValues(t, gradient, []float32{
		-0.7142857142857143, 0, 0,
		0, -0.8333333333333334, 0,
	})
}

func Test_CategoricalCrossEntropy_StableAroundZeroAndOne(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		gradient    *matrix.Matrix
		value       float32
		lower       float32
		err         error
	)

	predictions = mustMatrix(t, 2, 3, []float32{
		1, 0, 0,
		0, 1, 0,
	})
	targets = mustMatrix(t, 2, 3, []float32{
		0, 1, 0,
		0, 0, 1,
	})

	value, err = loss.CategoricalCrossEntropy{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("Value returned error: %v", err)
	}

	requireFinite(t, value)
	lower = clampEpsilon
	testutil.RequireAlmostEqual(t, value, -f32.Log(lower), epsilon)

	gradient, err = loss.CategoricalCrossEntropy{}.Gradient(predictions, targets)
	if err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}

	requireFiniteMatrix(t, gradient)
	requireMatrixValues(t, gradient, []float32{
		0, -1 / lower / 2, 0,
		0, 0, -1 / lower / 2,
	})
}

func Test_CategoricalCrossEntropy_ValidatesOneHotTargets(t *testing.T) {
	type testcase struct {
		name   string
		values []float32
	}

	var tests []testcase
	tests = []testcase{
		{
			name:   "no class",
			values: []float32{0, 0, 0},
		},
		{
			name:   "multiple classes",
			values: []float32{1, 1, 0},
		},
		{
			name:   "fractional class",
			values: []float32{0.5, 0.5, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				predictions *matrix.Matrix
				targets     *matrix.Matrix
				got         float32
				err         error
			)

			predictions = mustMatrix(t, 1, 3, []float32{0.4, 0.4, 0.2})
			targets = mustMatrix(t, 1, 3, tt.values)

			got, err = loss.CategoricalCrossEntropy{}.Value(predictions, targets)
			if err == nil {
				t.Fatalf("Value returned %g and nil error, want error", got)
			}
		})
	}
}

func Test_CategoricalCrossEntropy_GradientValidatesOneHotTargets(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		gradient    *matrix.Matrix
		err         error
	)

	predictions = mustMatrix(t, 1, 3, []float32{0.4, 0.4, 0.2})
	targets = mustMatrix(t, 1, 3, []float32{0.5, 0.5, 0})

	gradient, err = loss.CategoricalCrossEntropy{}.Gradient(predictions, targets)
	if err == nil {
		t.Fatalf("Gradient returned %#v and nil error, want error", gradient)
	}
}
