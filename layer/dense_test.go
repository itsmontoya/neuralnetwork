package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const epsilon = 1e-5

func Test_Dense_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.Dense)(nil)
}

func Test_NewDense_UsesWeightInitializer(t *testing.T) {
	var (
		gotInputSize  int
		gotOutputSize int
		dense         *layer.Dense
		err           error
	)

	dense, err = layer.NewDense(2, 3, func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
		gotInputSize = inputSize
		gotOutputSize = outputSize
		weights, err = matrix.FromSlice(inputSize, outputSize, []float32{1, 2, 3, 4, 5, 6})
		return weights, err
	})
	if err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}

	if gotInputSize != 2 {
		t.Fatalf("initializer inputSize = %d, want 2", gotInputSize)
	}

	if gotOutputSize != 3 {
		t.Fatalf("initializer outputSize = %d, want 3", gotOutputSize)
	}

	requireMatrixValues(t, dense.Weights().Values(), []float32{1, 2, 3, 4, 5, 6})
	requireMatrixValues(t, dense.Biases().Values(), []float32{0, 0, 0})
}

func Test_Dense_Accessors(t *testing.T) {
	var (
		dense      *layer.Dense
		parameters []*optimizer.Parameter
	)

	dense = mustDense(
		t,
		2,
		3,
		[]float32{
			1, 2, 3,
			4, 5, 6,
		},
		[]float32{0.1, 0.2, 0.3},
	)

	if dense.InputSize() != 2 {
		t.Fatalf("InputSize = %d, want 2", dense.InputSize())
	}

	if dense.OutputSize() != 3 {
		t.Fatalf("OutputSize = %d, want 3", dense.OutputSize())
	}

	parameters = dense.Parameters()
	if len(parameters) != 2 {
		t.Fatalf("Parameters length = %d, want 2", len(parameters))
	}

	if parameters[0] != dense.Weights() {
		t.Fatal("Parameters[0] did not match weights")
	}

	if parameters[1] != dense.Biases() {
		t.Fatal("Parameters[1] did not match biases")
	}
}

func Test_Dense_NilReceiverAccessors(t *testing.T) {
	var dense *layer.Dense

	if dense.InputSize() != 0 {
		t.Fatalf("InputSize = %d, want 0", dense.InputSize())
	}

	if dense.OutputSize() != 0 {
		t.Fatalf("OutputSize = %d, want 0", dense.OutputSize())
	}

	if dense.Weights() != nil {
		t.Fatal("Weights returned value for nil receiver")
	}

	if dense.Biases() != nil {
		t.Fatal("Biases returned value for nil receiver")
	}

	if dense.Parameters() != nil {
		t.Fatal("Parameters returned value for nil receiver")
	}
}

func Test_NewDense_ValidatesConfig(t *testing.T) {
	type testcase struct {
		name        string
		inputSize   int
		outputSize  int
		initializer layer.WeightInitializer
	}

	tests := []testcase{
		{
			name:        "input size",
			inputSize:   0,
			outputSize:  1,
			initializer: layer.ZeroWeights,
		},
		{
			name:        "output size",
			inputSize:   1,
			outputSize:  0,
			initializer: layer.ZeroWeights,
		},
		{
			name:        "initializer",
			inputSize:   1,
			outputSize:  1,
			initializer: nil,
		},
		{
			name:       "initializer shape",
			inputSize:  2,
			outputSize: 3,
			initializer: func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
				weights, err = matrix.New(1, outputSize)
				return weights, err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				dense *layer.Dense
				err   error
			)

			dense, err = layer.NewDense(tt.inputSize, tt.outputSize, tt.initializer)
			if err == nil {
				t.Fatal("NewDense error = nil, want error")
			}

			if dense != nil {
				t.Fatal("NewDense returned dense layer on error")
			}
		})
	}
}

func Test_Dense_Forward(t *testing.T) {
	var (
		dense  *layer.Dense
		input  *matrix.Matrix
		output *matrix.Matrix
		err    error
	)

	dense = mustDense(
		t,
		2,
		3,
		[]float32{
			0.5, -1, 2,
			1.5, 0, -0.5,
		},
		[]float32{0.1, 0.2, 0.3},
	)
	input = mustMatrix(t, 2, 2, []float32{
		1, 2,
		3, 4,
	})

	output, err = dense.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	if output.Rows() != 2 {
		t.Fatalf("Forward output rows = %d, want 2", output.Rows())
	}

	if output.Cols() != 3 {
		t.Fatalf("Forward output cols = %d, want 3", output.Cols())
	}

	requireMatrixValues(t, output, []float32{
		3.6, -0.8, 1.3,
		7.6, -2.8, 4.3,
	})
}

func Test_Dense_ForwardReportsInputShapeMismatch(t *testing.T) {
	var (
		dense *layer.Dense
		input *matrix.Matrix
		err   error
	)

	dense = mustDense(
		t,
		3,
		2,
		[]float32{
			1, 2,
			3, 4,
			5, 6,
		},
		[]float32{0, 0},
	)
	input = mustMatrix(t, 4, 2, []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})

	_, err = dense.Forward(input)
	if err == nil {
		t.Fatal("Forward error = nil, want shape error")
	}

	if !strings.Contains(err.Error(), "got 4x2, want batch rows x 3") {
		t.Fatalf("Forward error = %q, want received and expected shape", err.Error())
	}
}

func Test_Dense_Backward(t *testing.T) {
	var (
		dense          *layer.Dense
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	dense = mustDense(
		t,
		2,
		3,
		[]float32{
			0.5, -1, 2,
			1.5, 0, -0.5,
		},
		[]float32{0.1, 0.2, 0.3},
	)
	input = mustMatrix(t, 2, 2, []float32{
		1, 2,
		3, 4,
	})
	outputGradient = mustMatrix(t, 2, 3, []float32{
		1, -2, 0.5,
		0.25, 1.5, -1,
	})

	_, err = dense.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = dense.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{
		3.5, 1.25,
		-3.375, 0.875,
	})
	requireMatrixValues(t, dense.Weights().Gradient(), []float32{
		1.75, 2.5, -2.5,
		3, 2, -3,
	})
	requireMatrixValues(t, dense.Biases().Gradient(), []float32{1.25, -0.5, -0.5})
}

func Test_Dense_ResetGradients(t *testing.T) {
	var (
		dense          *layer.Dense
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		err            error
	)

	dense = mustDense(
		t,
		2,
		2,
		[]float32{
			1, 2,
			3, 4,
		},
		[]float32{0.5, -0.5},
	)
	input = mustMatrix(t, 1, 2, []float32{2, 3})
	outputGradient = mustMatrix(t, 1, 2, []float32{0.25, -0.75})

	_, err = dense.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	_, err = dense.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	err = dense.ResetGradients()
	if err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}

	requireMatrixValues(t, dense.Weights().Gradient(), []float32{0, 0, 0, 0})
	requireMatrixValues(t, dense.Biases().Gradient(), []float32{0, 0})
}

func mustDense(tb testing.TB, inputSize, outputSize int, weightValues, biasValues []float32) (dense *layer.Dense) {
	var (
		biases *matrix.Matrix
		err    error
	)

	tb.Helper()

	dense, err = layer.NewDense(inputSize, outputSize, func(layerInputSize, layerOutputSize int) (weights *matrix.Matrix, err error) {
		weights, err = matrix.FromSlice(layerInputSize, layerOutputSize, weightValues)
		return weights, err
	})
	if err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	biases = mustMatrix(tb, 1, outputSize, biasValues)
	err = dense.Biases().Values().CopyFrom(biases)
	if err != nil {
		tb.Fatalf("CopyFrom returned error: %v", err)
	}

	return dense
}

func mustMatrix(tb testing.TB, rows, cols int, values []float32) (m *matrix.Matrix) {
	var err error

	tb.Helper()

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func requireMatrixValues(tb testing.TB, got *matrix.Matrix, want []float32) {
	var (
		values []float32
		err    error
	)

	tb.Helper()

	values, err = got.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(tb, values, want, epsilon)
}
