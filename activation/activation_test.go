package activation_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const epsilon = 1e-12

func Test_Activation_Interface(t *testing.T) {
	var _ activation.Activation = activation.ELU{}
	var _ activation.Activation = activation.GELU{}
	var _ activation.Activation = activation.LeakyReLU{}
	var _ activation.Activation = activation.ReLU{}
	var _ activation.Activation = activation.Sigmoid{}
	var _ activation.Activation = activation.Tanh{}
	var _ activation.Activation = activation.Linear{}
	var _ activation.Activation = activation.Softmax{}
}

func Test_Activation_Forward(t *testing.T) {
	type testcase struct {
		name       string
		activation activation.Activation
		input      *matrix.Matrix
		want       []float64
	}

	var (
		baseInput    *matrix.Matrix
		softmaxInput *matrix.Matrix
		tests        []testcase
	)

	baseInput = mustMatrix(t, 2, 3, []float64{-1, 0, 2, -2, 1, 3})
	softmaxInput = mustMatrix(t, 2, 3, []float64{1, 2, 3, -1, 0, 1})
	tests = []testcase{
		{
			name:       "elu",
			activation: activation.ELU{},
			input:      baseInput,
			want: []float64{
				math.Exp(-1) - 1,
				0,
				2,
				math.Exp(-2) - 1,
				1,
				3,
			},
		},
		{
			name:       "gelu",
			activation: activation.GELU{},
			input:      baseInput,
			want: []float64{
				geluTestValue(-1),
				geluTestValue(0),
				geluTestValue(2),
				geluTestValue(-2),
				geluTestValue(1),
				geluTestValue(3),
			},
		},
		{
			name:       "leaky relu",
			activation: activation.LeakyReLU{},
			input:      baseInput,
			want:       []float64{-0.01, 0, 2, -0.02, 1, 3},
		},
		{
			name:       "relu",
			activation: activation.ReLU{},
			input:      baseInput,
			want:       []float64{0, 0, 2, 0, 1, 3},
		},
		{
			name:       "sigmoid",
			activation: activation.Sigmoid{},
			input:      baseInput,
			want: []float64{
				0.2689414213699951,
				0.5,
				0.8807970779778823,
				0.11920292202211755,
				0.7310585786300049,
				0.9525741268224334,
			},
		},
		{
			name:       "tanh",
			activation: activation.Tanh{},
			input:      baseInput,
			want: []float64{
				-0.7615941559557649,
				0,
				0.9640275800758169,
				-0.9640275800758169,
				0.7615941559557649,
				0.9950547536867305,
			},
		},
		{
			name:       "linear",
			activation: activation.Linear{},
			input:      baseInput,
			want:       []float64{-1, 0, 2, -2, 1, 3},
		},
		{
			name:       "softmax",
			activation: activation.Softmax{},
			input:      softmaxInput,
			want: []float64{
				0.09003057317038046,
				0.24472847105479764,
				0.6652409557748218,
				0.09003057317038046,
				0.24472847105479764,
				0.6652409557748218,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				before []float64
				got    *matrix.Matrix
				err    error
			)

			before, err = tt.input.Values()
			if err != nil {
				t.Fatalf("Values returned error: %v", err)
			}

			got, err = tt.activation.Forward(tt.input)
			if err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			requireMatrixValues(t, got, tt.want)
			requireMatrixValues(t, tt.input, before)
		})
	}
}

func Test_Activation_Backward(t *testing.T) {
	type testcase struct {
		name       string
		activation activation.Activation
		input      *matrix.Matrix
		gradient   *matrix.Matrix
		want       []float64
	}

	var (
		input    *matrix.Matrix
		gradient *matrix.Matrix
		tests    []testcase
	)

	input = mustMatrix(t, 2, 3, []float64{-1, 0, 2, -2, 1, 3})
	gradient = mustMatrix(t, 2, 3, []float64{1, 2, 3, 4, 5, 6})
	tests = []testcase{
		{
			name:       "elu",
			activation: activation.ELU{},
			input:      input,
			gradient:   gradient,
			want: []float64{
				1 * math.Exp(-1),
				2,
				3,
				4 * math.Exp(-2),
				5,
				6,
			},
		},
		{
			name:       "gelu",
			activation: activation.GELU{},
			input:      input,
			gradient:   gradient,
			want: []float64{
				1 * geluTestDerivative(-1),
				2 * geluTestDerivative(0),
				3 * geluTestDerivative(2),
				4 * geluTestDerivative(-2),
				5 * geluTestDerivative(1),
				6 * geluTestDerivative(3),
			},
		},
		{
			name:       "leaky relu",
			activation: activation.LeakyReLU{},
			input:      input,
			gradient:   gradient,
			want:       []float64{0.01, 0.02, 3, 0.04, 5, 6},
		},
		{
			name:       "relu",
			activation: activation.ReLU{},
			input:      input,
			gradient:   gradient,
			want:       []float64{0, 0, 3, 0, 5, 6},
		},
		{
			name:       "sigmoid",
			activation: activation.Sigmoid{},
			input:      input,
			gradient:   gradient,
			want: []float64{
				1 * 0.19661193324148185,
				2 * 0.25,
				3 * 0.10499358540350662,
				4 * 0.1049935854035065,
				5 * 0.19661193324148185,
				6 * 0.045176659730912,
			},
		},
		{
			name:       "tanh",
			activation: activation.Tanh{},
			input:      input,
			gradient:   gradient,
			want: []float64{
				1 * 0.41997434161402614,
				2,
				3 * 0.07065082485316443,
				4 * 0.07065082485316443,
				5 * 0.41997434161402614,
				6 * 0.009866037165440211,
			},
		},
		{
			name:       "linear",
			activation: activation.Linear{},
			input:      input,
			gradient:   gradient,
			want:       []float64{1, 2, 3, 4, 5, 6},
		},
		{
			name:       "softmax",
			activation: activation.Softmax{},
			input:      mustMatrix(t, 1, 2, []float64{0, 0}),
			gradient:   mustMatrix(t, 1, 2, []float64{1, 3}),
			want:       []float64{-0.5, 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				got *matrix.Matrix
				err error
			)

			got, err = tt.activation.Backward(tt.input, tt.gradient)
			if err != nil {
				t.Fatalf("Backward returned error: %v", err)
			}

			requireMatrixValues(t, got, tt.want)
		})
	}
}

func Test_Activation_Backward_ValidatesShape(t *testing.T) {
	var (
		input         *matrix.Matrix
		gradient      *matrix.Matrix
		inputGradient *matrix.Matrix
		err           error
	)

	input = mustMatrix(t, 1, 3, []float64{1, 2, 3})
	gradient = mustMatrix(t, 3, 1, []float64{1, 2, 3})

	inputGradient, err = activation.ReLU{}.Backward(input, gradient)
	if err == nil {
		t.Fatalf("Backward returned gradient %v and nil error, want error", inputGradient)
	}
}

func Test_Softmax_RowsSumToOne(t *testing.T) {
	var (
		input   *matrix.Matrix
		output  *matrix.Matrix
		rowSums []float64
		err     error
	)

	input = mustMatrix(t, 3, 4, []float64{
		1, 2, 3, 4,
		-4, -3, -2, -1,
		0.5, -0.25, 0.75, 1.25,
	})

	output, err = activation.Softmax{}.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	rowSums, err = output.RowSums()
	if err != nil {
		t.Fatalf("RowSums returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, rowSums, []float64{1, 1, 1}, epsilon)
}

func Test_Softmax_StableForLargeValues(t *testing.T) {
	var (
		input  *matrix.Matrix
		output *matrix.Matrix
		values []float64
		index  int
		err    error
	)

	input = mustMatrix(t, 2, 3, []float64{
		1000, 1001, 1002,
		-1000, -1001, -1002,
	})

	output, err = activation.Softmax{}.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	values, err = output.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	for index = range values {
		if math.IsInf(values[index], 0) || math.IsNaN(values[index]) {
			t.Fatalf("Softmax value at index %d is unstable: %g", index, values[index])
		}
	}

	testutil.RequireSliceAlmostEqual(t, values, []float64{
		0.09003057317038046,
		0.24472847105479764,
		0.6652409557748218,
		0.6652409557748218,
		0.24472847105479764,
		0.09003057317038046,
	}, epsilon)
}

func mustMatrix(tb testing.TB, rows, cols int, values []float64) (m *matrix.Matrix) {
	var err error

	tb.Helper()

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func requireMatrixValues(tb testing.TB, got *matrix.Matrix, want []float64) {
	var (
		values []float64
		err    error
	)

	tb.Helper()

	values, err = got.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(tb, values, want, epsilon)
}

func geluTestValue(value float64) (result float64) {
	result = 0.5 * value * (1 + math.Erf(value/math.Sqrt2))
	return result
}

func geluTestDerivative(value float64) (result float64) {
	var density float64

	density = math.Exp(-0.5*value*value) / math.Sqrt(2*math.Pi)
	result = 0.5*(1+math.Erf(value/math.Sqrt2)) + value*density
	return result
}
