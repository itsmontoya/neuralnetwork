package activation_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var benchmarkActivationResult *matrix.Matrix

func Benchmark_ActivationForward(b *testing.B) {
	var tests []struct {
		name     string
		function activation.Activation
	}

	tests = benchmarkActivations()
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkActivationForwardShapes(b, tt.function)
		})
	}
}

func Benchmark_ActivationBackward(b *testing.B) {
	var tests []struct {
		name     string
		function activation.Activation
	}

	tests = benchmarkActivations()
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkActivationBackwardShapes(b, tt.function)
		})
	}
}

func benchmarkActivations() (tests []struct {
	name     string
	function activation.Activation
}) {
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
		{name: "Softmax", function: activation.Softmax{}},
		{name: "Tanh", function: activation.Tanh{}},
	}
	return tests
}

func benchmarkActivationForwardShapes(b *testing.B, function activation.Activation) {
	var shapes []struct {
		name string
		rows int
		cols int
	}

	shapes = []struct {
		name string
		rows int
		cols int
	}{
		{name: "Small", rows: 4, cols: 4},
		{name: "Medium", rows: 128, cols: 64},
	}

	for _, shape := range shapes {
		b.Run(shape.name, func(b *testing.B) {
			var (
				input  *matrix.Matrix
				output *matrix.Matrix
				err    error
				index  int
			)

			input = benchmarkActivationMatrix(b, shape.rows, shape.cols)
			if output, err = function.Forward(input); err != nil {
				b.Fatalf("Forward returned error: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()

			for index = 0; index < b.N; index++ {
				if output, err = function.Forward(input); err != nil {
					b.Fatalf("Forward returned error: %v", err)
				}
			}

			benchmarkActivationResult = output
		})
	}
}

func benchmarkActivationBackwardShapes(b *testing.B, function activation.Activation) {
	var shapes []struct {
		name string
		rows int
		cols int
	}

	shapes = []struct {
		name string
		rows int
		cols int
	}{
		{name: "Small", rows: 4, cols: 4},
		{name: "Medium", rows: 128, cols: 64},
	}

	for _, shape := range shapes {
		b.Run(shape.name, func(b *testing.B) {
			var (
				input          *matrix.Matrix
				outputGradient *matrix.Matrix
				inputGradient  *matrix.Matrix
				err            error
				index          int
			)

			input = benchmarkActivationMatrix(b, shape.rows, shape.cols)
			outputGradient = benchmarkActivationMatrix(b, shape.rows, shape.cols)
			if inputGradient, err = function.Backward(input, outputGradient); err != nil {
				b.Fatalf("Backward returned error: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()

			for index = 0; index < b.N; index++ {
				if inputGradient, err = function.Backward(input, outputGradient); err != nil {
					b.Fatalf("Backward returned error: %v", err)
				}
			}

			benchmarkActivationResult = inputGradient
		})
	}
}

func benchmarkActivationMatrix(tb testing.TB, rows, cols int) (m *matrix.Matrix) {
	var (
		values []float32
		err    error
		index  int
	)

	tb.Helper()

	values = make([]float32, rows*cols)
	for index = range values {
		values[index] = float32(index%31)/15 - 1
	}

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}
