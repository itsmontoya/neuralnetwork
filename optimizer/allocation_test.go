package optimizer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

var allocationOptimizerParameters []*optimizer.Parameter

func Test_OptimizerSteadyStateUpdateAllocations(t *testing.T) {
	tests := []struct {
		name string
		new  func(testing.TB) optimizer.Optimizer
	}{
		{
			name: "SGD",
			new: func(tb testing.TB) (optimizerRule optimizer.Optimizer) {
				var err error

				optimizerRule, err = optimizer.NewSGD(0.01)
				if err != nil {
					tb.Fatalf("NewSGD returned error: %v", err)
				}

				return optimizerRule
			},
		},
		{
			name: "Momentum",
			new: func(tb testing.TB) (optimizerRule optimizer.Optimizer) {
				var err error

				optimizerRule, err = optimizer.NewMomentum(0.01)
				if err != nil {
					tb.Fatalf("NewMomentum returned error: %v", err)
				}

				return optimizerRule
			},
		},
		{
			name: "Adam",
			new: func(tb testing.TB) (optimizerRule optimizer.Optimizer) {
				var err error

				optimizerRule, err = optimizer.NewAdam(0.001)
				if err != nil {
					tb.Fatalf("NewAdam returned error: %v", err)
				}

				return optimizerRule
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				optimizerRule optimizer.Optimizer
				parameters    []*optimizer.Parameter
				gradients     []*matrix.Matrix
				err           error
			)

			optimizerRule = tt.new(t)
			parameters, gradients = allocationOptimizerParametersAndGradients(t)
			if err = allocationAccumulateGradients(parameters, gradients); err != nil {
				t.Fatalf("AccumulateGradient returned error: %v", err)
			}

			if err = optimizerRule.Update(parameters); err != nil {
				t.Fatalf("Update returned error: %v", err)
			}

			requireMaxAllocs(t, tt.name, 0, func() {
				if err = allocationAccumulateGradients(parameters, gradients); err != nil {
					panic(err)
				}

				if err = optimizerRule.Update(parameters); err != nil {
					panic(err)
				}
			})

			allocationOptimizerParameters = parameters
		})
	}
}

func allocationOptimizerParametersAndGradients(tb testing.TB) (parameters []*optimizer.Parameter, gradients []*matrix.Matrix) {
	var (
		parameter *optimizer.Parameter
		gradient  *matrix.Matrix
	)

	tb.Helper()

	parameter = mustParameter(tb, 2, 3, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})
	gradient = mustMatrix(tb, 2, 3, []float32{0.01, -0.02, 0.03, -0.04, 0.05, -0.06})
	parameters = append(parameters, parameter)
	gradients = append(gradients, gradient)

	parameter = mustParameter(tb, 1, 3, []float32{0.7, 0.8, 0.9})
	gradient = mustMatrix(tb, 1, 3, []float32{-0.03, 0.02, -0.01})
	parameters = append(parameters, parameter)
	gradients = append(gradients, gradient)

	return parameters, gradients
}

func allocationAccumulateGradients(parameters []*optimizer.Parameter, gradients []*matrix.Matrix) (err error) {
	var (
		index     int
		parameter *optimizer.Parameter
	)

	for index, parameter = range parameters {
		if err = parameter.AccumulateGradient(gradients[index]); err != nil {
			return err
		}
	}

	return nil
}

func requireMaxAllocs(tb testing.TB, name string, max float64, run func()) {
	var got float64

	tb.Helper()

	got = testing.AllocsPerRun(100, run)
	if got > max {
		tb.Fatalf("%s allocations = %.0f, want <= %.0f", name, got, max)
	}
}
