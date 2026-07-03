package optimizer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Regularized_ImplementsOptimizer(t *testing.T) {
	var _ optimizer.Optimizer = (*optimizer.Regularized)(nil)
}

func Test_NewRegularized_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name         string
		base         optimizer.Optimizer
		regularizers []optimizer.Regularizer
	}

	var l1 *optimizer.L1
	var err error

	l1, err = optimizer.NewL1(0.1)
	if err != nil {
		t.Fatalf("NewL1 returned error: %v", err)
	}

	tests := []testcase{
		{
			name:         "nil base",
			base:         nil,
			regularizers: []optimizer.Regularizer{l1},
		},
		{
			name:         "no regularizers",
			base:         &mockOptimizer{},
			regularizers: nil,
		},
		{
			name:         "nil regularizer",
			base:         &mockOptimizer{},
			regularizers: []optimizer.Regularizer{nil},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				regularized *optimizer.Regularized
				err         error
			)

			regularized, err = optimizer.NewRegularized(tt.base, tt.regularizers...)
			if err == nil {
				t.Fatal("NewRegularized error = nil, want error")
			}

			if regularized != nil {
				t.Fatal("NewRegularized returned optimizer on error")
			}
		})
	}
}

func Test_Regularized_UpdateAppliesRegularizersBeforeBaseOptimizer(t *testing.T) {
	var (
		parameter   *optimizer.Parameter
		sgd         *optimizer.SGD
		l1          *optimizer.L1
		decay       *optimizer.L2WeightDecay
		regularized *optimizer.Regularized
		err         error
	)

	parameter = mustParameter(t, 1, 3, []float64{2, -4, 0})
	accumulateGradient(t, parameter, []float64{0.5, 0.5, 0.5})

	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	l1, err = optimizer.NewL1(0.25)
	if err != nil {
		t.Fatalf("NewL1 returned error: %v", err)
	}

	decay, err = optimizer.NewL2WeightDecay(0.25)
	if err != nil {
		t.Fatalf("NewL2WeightDecay returned error: %v", err)
	}

	regularized, err = optimizer.NewRegularized(sgd, l1, decay)
	if err != nil {
		t.Fatalf("NewRegularized returned error: %v", err)
	}

	err = regularized.Update([]*optimizer.Parameter{parameter})
	if err != nil {
		t.Fatalf("Update returned error: %v", err)
	}

	requireMatrixValues(t, parameter.Values(), []float64{1.875, -3.925, -0.05})
	requireMatrixValues(t, parameter.Gradient(), []float64{0, 0, 0})
}

func Test_Regularized_DelegatesLearningRate(t *testing.T) {
	var (
		base        *mockOptimizer
		l1          *optimizer.L1
		regularized *optimizer.Regularized
		err         error
	)

	base = &mockOptimizer{learningRate: 0.1}
	l1, err = optimizer.NewL1(0.1)
	if err != nil {
		t.Fatalf("NewL1 returned error: %v", err)
	}

	regularized, err = optimizer.NewRegularized(base, l1)
	if err != nil {
		t.Fatalf("NewRegularized returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, regularized.LearningRate(), 0.1, epsilon)
	err = regularized.SetLearningRate(0.25)
	if err != nil {
		t.Fatalf("SetLearningRate returned error: %v", err)
	}

	testutil.RequireAlmostEqual(t, base.learningRate, 0.25, epsilon)
	testutil.RequireAlmostEqual(t, regularized.LearningRate(), 0.25, epsilon)
}

func Test_Regularized_ReturnsRegularizerCopy(t *testing.T) {
	var (
		l1           *optimizer.L1
		sgd          *optimizer.SGD
		regularized  *optimizer.Regularized
		regularizers []optimizer.Regularizer
		err          error
	)

	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	l1, err = optimizer.NewL1(0.1)
	if err != nil {
		t.Fatalf("NewL1 returned error: %v", err)
	}

	regularized, err = optimizer.NewRegularized(sgd, l1)
	if err != nil {
		t.Fatalf("NewRegularized returned error: %v", err)
	}

	regularizers = regularized.Regularizers()
	if len(regularizers) != 1 {
		t.Fatalf("Regularizers length = %d, want 1", len(regularizers))
	}

	regularizers[0] = nil
	if regularized.Regularizers()[0] == nil {
		t.Fatal("Regularizers exposed internal slice")
	}
}
