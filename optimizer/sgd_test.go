package optimizer_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_NewSGD_ValidatesLearningRate(t *testing.T) {
	type testcase struct {
		name         string
		learningRate float32
	}

	tests := []testcase{
		{
			name:         "zero",
			learningRate: 0,
		},
		{
			name:         "negative",
			learningRate: -0.1,
		},
		{
			name:         "nan",
			learningRate: float32(math.NaN()),
		},
		{
			name:         "infinite",
			learningRate: float32(math.Inf(1)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				sgd *optimizer.SGD
				err error
			)

			sgd, err = optimizer.NewSGD(tt.learningRate)
			if err == nil {
				t.Fatal("NewSGD error = nil, want error")
			}

			if sgd != nil {
				t.Fatal("NewSGD returned optimizer on error")
			}
		})
	}
}

func Test_SGD_Update(t *testing.T) {
	var (
		parameter *optimizer.Parameter
		gradient  *matrix.Matrix
		sgd       *optimizer.SGD
		err       error
	)

	parameter = mustParameter(t, 1, 3, []float32{1, -2, 3})
	gradient = mustMatrix(t, 1, 3, []float32{0.5, -1, 2})

	err = parameter.AccumulateGradient(gradient)
	if err != nil {
		t.Fatalf("AccumulateGradient returned error: %v", err)
	}

	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	err = sgd.Update([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	requireMatrixValues(t, parameter.Values(), []float32{0.95, -1.9, 2.8})
	requireMatrixValues(t, parameter.Gradient(), []float32{0, 0, 0})
}

func Test_SGD_Update_Repeated(t *testing.T) {
	var (
		parameter *optimizer.Parameter
		sgd       *optimizer.SGD
		err       error
	)

	parameter = mustParameter(t, 1, 2, []float32{1, 2})
	sgd, err = optimizer.NewSGD(0.25)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	accumulateGradient(t, parameter, []float32{0.4, -0.8})
	err = sgd.Update([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	accumulateGradient(t, parameter, []float32{0.2, 0.4})
	err = sgd.Update([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	requireMatrixValues(t, parameter.Values(), []float32{0.85, 2.1})
	requireMatrixValues(t, parameter.Gradient(), []float32{0, 0})
}

func Test_SGD_SetLearningRate(t *testing.T) {
	var (
		parameter *optimizer.Parameter
		sgd       *optimizer.SGD
		err       error
	)

	parameter = mustParameter(t, 1, 1, []float32{1})
	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	if err = sgd.SetLearningRate(0.5); err != nil {
		t.Fatalf("SetLearningRate returned error: %v", err)
	}

	accumulateGradient(t, parameter, []float32{0.2})
	if err = sgd.Update([]*optimizer.Parameter{parameter}); err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, sgd.LearningRate(), 0.5, epsilon)
	requireMatrixValues(t, parameter.Values(), []float32{0.9})
}

func Test_SGD_Update_ValidatesParameters(t *testing.T) {
	var (
		sgd *optimizer.SGD
		err error
	)

	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	err = sgd.Update([]*optimizer.Parameter{nil})
	if err == nil {
		t.Fatal("Update error = nil, want error")
	}
}

func accumulateGradient(tb testing.TB, parameter *optimizer.Parameter, values []float32) {
	var (
		rows     int
		cols     int
		gradient *matrix.Matrix
		err      error
	)

	tb.Helper()

	rows, cols = parameter.Values().Shape()
	gradient = mustMatrix(tb, rows, cols, values)
	err = parameter.AccumulateGradient(gradient)
	if err != nil {
		tb.Fatalf("AccumulateGradient returned error: %v", err)
	}
}

func requireMatrixValues(tb testing.TB, actual *matrix.Matrix, want []float32) {
	var (
		values []float32
		err    error
	)

	tb.Helper()

	values, err = actual.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(tb, values, want, epsilon)
}
