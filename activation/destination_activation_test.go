package activation_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_DestinationActivation_Interface(t *testing.T) {
	var _ activation.DestinationActivation = activation.ELU{}
	var _ activation.DestinationActivation = activation.GELU{}
	var _ activation.DestinationActivation = activation.LeakyReLU{}
	var _ activation.DestinationActivation = activation.Linear{}
	var _ activation.DestinationActivation = activation.ReLU{}
	var _ activation.DestinationActivation = activation.Sigmoid{}
	var _ activation.DestinationActivation = activation.Tanh{}
}

func Test_DestinationActivation_EquivalentToAllocatingMethods(t *testing.T) {
	var tests []struct {
		name     string
		function activation.Activation
	}

	tests = []struct {
		name     string
		function activation.Activation
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
				destinationFunction activation.DestinationActivation
				input               *matrix.Matrix
				outputGradient      *matrix.Matrix
				wantOutput          *matrix.Matrix
				wantInputGradient   *matrix.Matrix
				output              *matrix.Matrix
				inputGradient       *matrix.Matrix
				forwardAlias        *matrix.Matrix
				backwardAlias       *matrix.Matrix
				wantOutputValues    []float32
				wantGradientValues  []float32
				err                 error
			)

			destinationFunction = tt.function.(activation.DestinationActivation)
			input = mustMatrix(t, 2, 3, []float32{-1, 0, 2, -2, 1, 3})
			outputGradient = mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6})
			output = mustMatrix(t, 2, 3, []float32{9, 9, 9, 9, 9, 9})
			inputGradient = mustMatrix(t, 2, 3, []float32{9, 9, 9, 9, 9, 9})

			wantOutput, err = tt.function.Forward(input)
			if err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			wantInputGradient, err = tt.function.Backward(input, outputGradient)
			if err != nil {
				t.Fatalf("Backward returned error: %v", err)
			}

			wantOutputValues, err = wantOutput.Values()
			if err != nil {
				t.Fatalf("output Values returned error: %v", err)
			}

			wantGradientValues, err = wantInputGradient.Values()
			if err != nil {
				t.Fatalf("input gradient Values returned error: %v", err)
			}

			if err = destinationFunction.ForwardInto(input, output); err != nil {
				t.Fatalf("ForwardInto returned error: %v", err)
			}
			requireMatrixValues(t, output, wantOutputValues)

			if err = destinationFunction.BackwardInto(input, outputGradient, inputGradient); err != nil {
				t.Fatalf("BackwardInto returned error: %v", err)
			}
			requireMatrixValues(t, inputGradient, wantGradientValues)

			forwardAlias, err = input.Clone()
			if err != nil {
				t.Fatalf("input Clone returned error: %v", err)
			}

			if err = destinationFunction.ForwardInto(forwardAlias, forwardAlias); err != nil {
				t.Fatalf("aliased ForwardInto returned error: %v", err)
			}
			requireMatrixValues(t, forwardAlias, wantOutputValues)

			backwardAlias, err = input.Clone()
			if err != nil {
				t.Fatalf("input Clone returned error: %v", err)
			}

			if err = destinationFunction.BackwardInto(backwardAlias, outputGradient, backwardAlias); err != nil {
				t.Fatalf("aliased BackwardInto returned error: %v", err)
			}
			requireMatrixValues(t, backwardAlias, wantGradientValues)
		})
	}
}

func Test_DestinationActivation_ValidatesDestinations(t *testing.T) {
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
				outputGradient *matrix.Matrix
				wrongShape     *matrix.Matrix
				err            error
			)

			input = mustMatrix(t, 1, 2, []float32{-1, 2})
			outputGradient = mustMatrix(t, 1, 2, []float32{3, 4})
			wrongShape = mustMatrix(t, 2, 1, []float32{0, 0})

			if err = tt.function.ForwardInto(input, wrongShape); err == nil {
				t.Fatal("ForwardInto error = nil for wrong destination shape")
			}

			if err = tt.function.BackwardInto(input, outputGradient, wrongShape); err == nil {
				t.Fatal("BackwardInto error = nil for wrong destination shape")
			}

			if err = tt.function.BackwardInto(input, outputGradient, outputGradient); err == nil {
				t.Fatal("BackwardInto error = nil for output-gradient alias")
			}
		})
	}
}
