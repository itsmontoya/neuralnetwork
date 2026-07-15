package activation_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var allocationActivationResult *matrix.Matrix

func Test_BuiltInActivationAllocationCeilings(t *testing.T) {
	var tests []struct {
		name          string
		function      activation.Activation
		forwardLimit  float64
		backwardLimit float64
	}

	tests = []struct {
		name          string
		function      activation.Activation
		forwardLimit  float64
		backwardLimit float64
	}{
		{name: "ELU", function: activation.ELU{}, forwardLimit: 2, backwardLimit: 2},
		{name: "GELU", function: activation.GELU{}, forwardLimit: 2, backwardLimit: 2},
		{name: "LeakyReLU", function: activation.LeakyReLU{}, forwardLimit: 2, backwardLimit: 2},
		{name: "Linear", function: activation.Linear{}, forwardLimit: 2, backwardLimit: 2},
		{name: "ReLU", function: activation.ReLU{}, forwardLimit: 2, backwardLimit: 2},
		{name: "Sigmoid", function: activation.Sigmoid{}, forwardLimit: 2, backwardLimit: 2},
		{name: "Softmax", function: activation.Softmax{}, forwardLimit: 4, backwardLimit: 6},
		{name: "Tanh", function: activation.Tanh{}, forwardLimit: 2, backwardLimit: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input          *matrix.Matrix
				outputGradient *matrix.Matrix
				err            error
			)

			input = mustMatrix(t, 2, 3, []float32{-1, 0, 2, -2, 1, 3})
			outputGradient = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})

			requireActivationMaxAllocs(t, "Forward", tt.forwardLimit, func() {
				allocationActivationResult, err = tt.function.Forward(input)
				if err != nil {
					panic(err)
				}
			})

			requireActivationMaxAllocs(t, "Backward", tt.backwardLimit, func() {
				allocationActivationResult, err = tt.function.Backward(input, outputGradient)
				if err != nil {
					panic(err)
				}
			})
		})
	}
}

func Test_BuiltInDestinationActivationAllocations(t *testing.T) {
	var tests []struct {
		name     string
		function activation.DestinationActivation
	}

	tests = []struct {
		name     string
		function activation.DestinationActivation
	}{
		{name: "ELU", function: activation.ELU{}},
		{name: "GELU", function: activation.GELU{}},
		{name: "LeakyReLU", function: activation.LeakyReLU{}},
		{name: "Linear", function: activation.Linear{}},
		{name: "ReLU", function: activation.ReLU{}},
		{name: "Sigmoid", function: activation.Sigmoid{}},
		{name: "Tanh", function: activation.Tanh{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input          *matrix.Matrix
				output         *matrix.Matrix
				outputGradient *matrix.Matrix
				inputGradient  *matrix.Matrix
				err            error
			)

			input = mustMatrix(t, 2, 3, []float32{-1, 0, 2, -2, 1, 3})
			output = mustMatrix(t, 2, 3, []float32{0, 0, 0, 0, 0, 0})
			outputGradient = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
			inputGradient = mustMatrix(t, 2, 3, []float32{0, 0, 0, 0, 0, 0})

			requireActivationMaxAllocs(t, "ForwardInto", 0, func() {
				if err = tt.function.ForwardInto(input, output); err != nil {
					panic(err)
				}
			})

			requireActivationMaxAllocs(t, "BackwardInto", 0, func() {
				if err = tt.function.BackwardInto(input, outputGradient, inputGradient); err != nil {
					panic(err)
				}
			})
		})
	}
}

func requireActivationMaxAllocs(tb testing.TB, name string, max float64, run func()) {
	var got float64

	tb.Helper()

	got = testing.AllocsPerRun(100, run)
	if got > max {
		tb.Fatalf("%s allocations = %.0f, want <= %.0f", name, got, max)
	}
}
