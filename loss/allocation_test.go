package loss_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var allocationLossValue float32
var allocationLossGradient *matrix.Matrix

func Test_LossValueAllocations(t *testing.T) {
	tests := []struct {
		name        string
		lossFunc    loss.Loss
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}{
		{
			name:        "MeanSquaredError",
			lossFunc:    loss.MeanSquaredError{},
			predictions: mustMatrix(t, 2, 2, []float32{0.1, 0.2, 0.3, 0.4}),
			targets:     mustMatrix(t, 2, 2, []float32{0, 0.25, 0.5, 0.75}),
		},
		{
			name:        "BinaryCrossEntropy",
			lossFunc:    loss.BinaryCrossEntropy{},
			predictions: mustMatrix(t, 4, 1, []float32{0.1, 0.8, 0.25, 0.75}),
			targets:     mustMatrix(t, 4, 1, []float32{0, 1, 0, 1}),
		},
		{
			name:        "CategoricalCrossEntropy",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: mustMatrix(t, 2, 3, []float32{0.7, 0.2, 0.1, 0.1, 0.8, 0.1}),
			targets:     mustMatrix(t, 2, 3, []float32{1, 0, 0, 0, 1, 0}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			requireMaxAllocs(t, tt.name, 0, func() {
				allocationLossValue, err = tt.lossFunc.Value(tt.predictions, tt.targets)
				if err != nil {
					panic(err)
				}
			})
		})
	}
}

func Test_LossGradientAllocationCeilings(t *testing.T) {
	tests := []struct {
		name        string
		lossFunc    loss.Loss
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}{
		{
			name:        "MeanSquaredError",
			lossFunc:    loss.MeanSquaredError{},
			predictions: mustMatrix(t, 2, 2, []float32{0.1, 0.2, 0.3, 0.4}),
			targets:     mustMatrix(t, 2, 2, []float32{0, 0.25, 0.5, 0.75}),
		},
		{
			name:        "BinaryCrossEntropy",
			lossFunc:    loss.BinaryCrossEntropy{},
			predictions: mustMatrix(t, 4, 1, []float32{0.1, 0.8, 0.25, 0.75}),
			targets:     mustMatrix(t, 4, 1, []float32{0, 1, 0, 1}),
		},
		{
			name:        "CategoricalCrossEntropy",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: mustMatrix(t, 2, 3, []float32{0.7, 0.2, 0.1, 0.1, 0.8, 0.1}),
			targets:     mustMatrix(t, 2, 3, []float32{1, 0, 0, 0, 1, 0}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			requireMaxAllocs(t, tt.name, 2, func() {
				allocationLossGradient, err = tt.lossFunc.Gradient(tt.predictions, tt.targets)
				if err != nil {
					panic(err)
				}
			})
		})
	}
}

func Test_DestinationGradientAllocations(t *testing.T) {
	var tests []struct {
		name        string
		lossFunc    loss.DestinationGradient
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
	}

	tests = []struct {
		name        string
		lossFunc    loss.DestinationGradient
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
	}{
		{
			name:        "MeanSquaredError",
			lossFunc:    loss.MeanSquaredError{},
			predictions: mustMatrix(t, 2, 2, []float32{0.1, 0.2, 0.3, 0.4}),
			targets:     mustMatrix(t, 2, 2, []float32{0, 0.25, 0.5, 0.75}),
			destination: mustMatrix(t, 2, 2, []float32{0, 0, 0, 0}),
		},
		{
			name:        "BinaryCrossEntropy",
			lossFunc:    loss.BinaryCrossEntropy{},
			predictions: mustMatrix(t, 4, 1, []float32{0.1, 0.8, 0.25, 0.75}),
			targets:     mustMatrix(t, 4, 1, []float32{0, 1, 0, 1}),
			destination: mustMatrix(t, 4, 1, []float32{0, 0, 0, 0}),
		},
		{
			name:        "CategoricalCrossEntropy",
			lossFunc:    loss.CategoricalCrossEntropy{},
			predictions: mustMatrix(t, 2, 3, []float32{0.7, 0.2, 0.1, 0.1, 0.8, 0.1}),
			targets:     mustMatrix(t, 2, 3, []float32{1, 0, 0, 0, 1, 0}),
			destination: mustMatrix(t, 2, 3, []float32{0, 0, 0, 0, 0, 0}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			requireMaxAllocs(t, tt.name, 0, func() {
				if err = tt.lossFunc.GradientInto(tt.predictions, tt.targets, tt.destination); err != nil {
					panic(err)
				}
			})
		})
	}
}

func requireMaxAllocs(tb testing.TB, name string, max float64, run func()) {
	var got float64

	tb.Helper()

	got = testing.AllocsPerRun(100, run)
	if got > max {
		tb.Fatalf("%s allocations = %.0f, want <= %.0f", name, got, max)
	}
}
