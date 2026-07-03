package optimizer_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_NewMomentum_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name         string
		learningRate float64
		coefficient  float64
	}

	tests := []testcase{
		{
			name:         "learning rate",
			learningRate: 0,
			coefficient:  0.9,
		},
		{
			name:         "negative coefficient",
			learningRate: 0.1,
			coefficient:  -0.1,
		},
		{
			name:         "coefficient one",
			learningRate: 0.1,
			coefficient:  1,
		},
		{
			name:         "coefficient nan",
			learningRate: 0.1,
			coefficient:  math.NaN(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				momentum *optimizer.Momentum
				err      error
			)

			momentum, err = optimizer.NewMomentumWithCoefficient(tt.learningRate, tt.coefficient)
			if err == nil {
				t.Fatal("NewMomentumWithCoefficient error = nil, want error")
			}

			if momentum != nil {
				t.Fatal("NewMomentumWithCoefficient returned optimizer on error")
			}
		})
	}
}

func Test_Momentum_Update_Repeated(t *testing.T) {
	var (
		parameter *optimizer.Parameter
		momentum  *optimizer.Momentum
		err       error
	)

	parameter = mustParameter(t, 1, 2, []float64{1, 2})
	momentum, err = optimizer.NewMomentumWithCoefficient(0.1, 0.9)
	if err != nil {
		t.Fatalf("NewMomentumWithCoefficient returned error: %v", err)
	}

	accumulateGradient(t, parameter, []float64{0.5, -0.25})
	err = momentum.Update([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	requireMatrixValues(t, parameter.Values(), []float64{0.95, 2.025})
	requireMatrixValues(t, parameter.Gradient(), []float64{0, 0})

	accumulateGradient(t, parameter, []float64{0.5, -0.25})
	err = momentum.Update([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	requireMatrixValues(t, parameter.Values(), []float64{0.855, 2.0725})
	requireMatrixValues(t, parameter.Gradient(), []float64{0, 0})
}

func Test_Momentum_StateIsolation(t *testing.T) {
	var (
		first    *optimizer.Parameter
		second   *optimizer.Parameter
		momentum *optimizer.Momentum
		err      error
	)

	first = mustParameter(t, 1, 2, []float64{1, 2})
	second = mustParameter(t, 1, 2, []float64{1, 2})
	momentum, err = optimizer.NewMomentumWithCoefficient(0.1, 0.9)
	if err != nil {
		t.Fatalf("NewMomentumWithCoefficient returned error: %v", err)
	}

	accumulateGradient(t, first, []float64{0.5, -0.25})
	err = momentum.Update([]*optimizer.Parameter{first})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	accumulateGradient(t, first, []float64{0.5, -0.25})
	accumulateGradient(t, second, []float64{0.5, -0.25})
	err = momentum.Update([]*optimizer.Parameter{first, second})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	requireMatrixValues(t, first.Values(), []float64{0.855, 2.0725})
	requireMatrixValues(t, second.Values(), []float64{0.95, 2.025})
}

func Test_Momentum_Setters(t *testing.T) {
	var (
		momentum *optimizer.Momentum
		err      error
	)

	momentum, err = optimizer.NewMomentum(0.1)
	if err != nil {
		t.Fatalf("NewMomentum returned error: %v", err)
	}

	if err = momentum.SetLearningRate(0.2); err != nil {
		t.Fatalf("SetLearningRate returned error: %v", err)
	}

	if err = momentum.SetCoefficient(0.8); err != nil {
		t.Fatalf("SetCoefficient returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, momentum.LearningRate(), 0.2, epsilon)
	testutil.RequireAlmostEqual(t, momentum.Coefficient(), 0.8, epsilon)
}
