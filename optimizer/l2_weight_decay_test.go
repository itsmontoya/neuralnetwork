package optimizer_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_L2WeightDecay_Apply(t *testing.T) {
	var (
		parameter *optimizer.Parameter
		decay     *optimizer.L2WeightDecay
		err       error
	)

	parameter = mustParameter(t, 1, 3, []float64{2, -4, 0})
	accumulateGradient(t, parameter, []float64{0.5, 0.5, 0.5})

	decay, err = optimizer.NewL2WeightDecay(0.25)
	if err != nil {
		t.Fatalf("NewL2WeightDecay returned error: %v", err)
	}

	err = decay.Apply([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Apply returned error: %v", err)
	}

	requireMatrixValues(t, parameter.Gradient(), []float64{1, -0.5, 0.5})
}

func Test_L2WeightDecay_SetCoefficient(t *testing.T) {
	var (
		decay *optimizer.L2WeightDecay
		err   error
	)

	decay, err = optimizer.NewL2WeightDecay(0.2)
	if err != nil {
		t.Fatalf("NewL2WeightDecay returned error: %v", err)
	}

	if err = decay.SetCoefficient(0.4); err != nil {
		t.Fatalf("SetCoefficient returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, decay.Coefficient(), 0.4, epsilon)
}

func Test_NewL2WeightDecay_ValidatesCoefficient(t *testing.T) {
	type testcase struct {
		name        string
		coefficient float64
	}

	tests := []testcase{
		{
			name:        "negative",
			coefficient: -0.1,
		},
		{
			name:        "infinite",
			coefficient: math.Inf(1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				decay *optimizer.L2WeightDecay
				err   error
			)

			decay, err = optimizer.NewL2WeightDecay(tt.coefficient)
			if err == nil {
				t.Fatal("NewL2WeightDecay error = nil, want error")
			}

			if decay != nil {
				t.Fatal("NewL2WeightDecay returned regularizer on error")
			}
		})
	}
}
