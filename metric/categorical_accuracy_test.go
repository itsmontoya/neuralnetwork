package metric_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
)

func Test_CategoricalAccuracy_ValueUsesArgmax(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float32
		err         error
	)

	predictions = mustMatrix(t, 3, 3, []float32{
		0.1, 0.8, 0.1,
		0.4, 0.4, 0.2,
		0.2, 0.3, 0.5,
	})
	targets = mustMatrix(t, 3, 3, []float32{
		0, 1, 0,
		1, 0, 0,
		0, 1, 0,
	})

	got, err = metric.CategoricalAccuracy{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("Value returned error: %v", err)
	}

	requireAlmostEqual(t, got, 2.0/3.0)
}

func Test_CategoricalAccuracy_ValidatesShape(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float32
		err         error
	)

	predictions = mustMatrix(t, 1, 3, []float32{0.2, 0.6, 0.2})
	targets = mustMatrix(t, 1, 2, []float32{0, 1})

	got, err = metric.CategoricalAccuracy{}.Value(predictions, targets)
	if err == nil {
		t.Fatalf("Value returned %g and nil error, want error", got)
	}
}

func Test_CategoricalAccuracy_ValidatesOneHotTargets(t *testing.T) {
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

			got, err = metric.CategoricalAccuracy{}.Value(predictions, targets)
			if err == nil {
				t.Fatalf("Value returned %g and nil error, want error", got)
			}
		})
	}
}
