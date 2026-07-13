package optimizer_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_L1_Apply(t *testing.T) {
	var (
		parameter *optimizer.Parameter
		l1        *optimizer.L1
		err       error
	)

	parameter = mustParameter(t, 1, 3, []float32{2, -4, 0})
	accumulateGradient(t, parameter, []float32{0.5, 0.5, 0.5})

	l1, err = optimizer.NewL1(0.25)
	if err != nil {
		t.Fatalf("NewL1 returned error: %v", err)
	}

	err = l1.Apply([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Apply returned error: %v", err)
	}

	requireMatrixValues(t, parameter.Gradient(), []float32{0.75, 0.25, 0.5})
}

func Test_L1_SetCoefficient(t *testing.T) {
	var (
		l1  *optimizer.L1
		err error
	)

	l1, err = optimizer.NewL1(0.1)
	if err != nil {
		t.Fatalf("NewL1 returned error: %v", err)
	}

	if err = l1.SetCoefficient(0.3); err != nil {
		t.Fatalf("SetCoefficient returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, l1.Coefficient(), 0.3, epsilon)
}

func Test_NewL1_ValidatesCoefficient(t *testing.T) {
	type testcase struct {
		name        string
		coefficient float32
	}

	tests := []testcase{
		{
			name:        "negative",
			coefficient: -0.1,
		},
		{
			name:        "nan",
			coefficient: float32(math.NaN()),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				l1  *optimizer.L1
				err error
			)

			l1, err = optimizer.NewL1(tt.coefficient)
			if err == nil {
				t.Fatal("NewL1 error = nil, want error")
			}

			if l1 != nil {
				t.Fatal("NewL1 returned regularizer on error")
			}
		})
	}
}
