package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_Flatten_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.Flatten)(nil)
}

func Test_NewFlatten_ValidatesShapeAndExposesAccessors(t *testing.T) {
	var (
		shape      layer.SpatialShape
		flatten    *layer.Flatten
		nilFlatten *layer.Flatten
		err        error
	)

	shape = mustSpatialShape(t, 2, 3, 4)
	flatten, err = layer.NewFlatten(shape)
	if err != nil {
		t.Fatalf("NewFlatten returned error: %v", err)
	}

	if flatten.InputShape() != shape {
		t.Fatalf("InputShape = %#v, want %#v", flatten.InputShape(), shape)
	}

	if flatten.OutputSize() != 24 {
		t.Fatalf("OutputSize = %d, want 24", flatten.OutputSize())
	}

	if nilFlatten.InputShape() != (layer.SpatialShape{}) {
		t.Fatal("nil InputShape returned a nonzero shape")
	}

	if nilFlatten.OutputSize() != 0 {
		t.Fatalf("nil OutputSize = %d, want 0", nilFlatten.OutputSize())
	}

	if _, err = nilFlatten.Backward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("nil Backward error = nil, want error")
	}

	flatten, err = layer.NewFlatten(layer.SpatialShape{})
	if err == nil {
		t.Fatal("NewFlatten error = nil, want invalid shape error")
	}

	if flatten != nil {
		t.Fatal("NewFlatten returned layer on error")
	}

	if !strings.HasPrefix(err.Error(), "layer: flatten input shape invalid:") {
		t.Fatalf("NewFlatten error = %q, want flatten context", err)
	}
}

func Test_Flatten_ForwardPreservesBatchAndCHWOrder(t *testing.T) {
	var (
		flatten *layer.Flatten
		input   *matrix.Matrix
		output  *matrix.Matrix
		err     error
	)

	flatten = mustFlatten(t, 2, 2, 3)
	input = mustMatrix(t, 2, 12, []float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	})

	output, err = flatten.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	if output == input {
		t.Fatal("Forward output aliases input")
	}

	if output.Rows() != 2 || output.Cols() != 12 {
		t.Fatalf("Forward output shape = %dx%d, want 2x12", output.Rows(), output.Cols())
	}

	if err = input.Set(0, 0, 99); err != nil {
		t.Fatalf("input Set returned error: %v", err)
	}

	requireMatrixValues(t, output, []float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	})
}

func Test_Flatten_BackwardPreservesGradientOrder(t *testing.T) {
	var (
		flatten        *layer.Flatten
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	flatten = mustFlatten(t, 2, 2, 2)
	if _, err = flatten.Forward(mustMatrix(t, 2, 8, make([]float32, 16))); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 2, 8, []float32{
		-1, -2, -3, -4, 5, 6, 7, 8,
		9, 10, 11, 12, -13, -14, -15, -16,
	})
	inputGradient, err = flatten.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	if inputGradient == outputGradient {
		t.Fatal("Backward input gradient aliases output gradient")
	}

	if err = outputGradient.Set(0, 0, 99); err != nil {
		t.Fatalf("output gradient Set returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{
		-1, -2, -3, -4, 5, 6, 7, 8,
		9, 10, 11, 12, -13, -14, -15, -16,
	})
}

func Test_Flatten_ResultsDoNotAliasLiveArguments(t *testing.T) {
	var (
		flatten             *layer.Flatten
		input               *matrix.Matrix
		firstOutput         *matrix.Matrix
		secondOutput        *matrix.Matrix
		firstInputGradient  *matrix.Matrix
		secondInputGradient *matrix.Matrix
		err                 error
	)

	flatten = mustFlatten(t, 1, 2, 2)
	input = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
	firstOutput, err = flatten.Forward(input)
	if err != nil {
		t.Fatalf("first Forward returned error: %v", err)
	}

	secondOutput, err = flatten.Forward(firstOutput)
	if err != nil {
		t.Fatalf("second Forward returned error: %v", err)
	}

	if secondOutput == firstOutput {
		t.Fatal("second Forward output aliases its input")
	}

	requireMatrixValues(t, secondOutput, []float32{1, 2, 3, 4})

	firstInputGradient, err = flatten.Backward(mustMatrix(t, 1, 4, []float32{5, 6, 7, 8}))
	if err != nil {
		t.Fatalf("first Backward returned error: %v", err)
	}

	if _, err = flatten.Forward(input); err != nil {
		t.Fatalf("Forward before second Backward returned error: %v", err)
	}

	secondInputGradient, err = flatten.Backward(firstInputGradient)
	if err != nil {
		t.Fatalf("second Backward returned error: %v", err)
	}

	if secondInputGradient == firstInputGradient {
		t.Fatal("second Backward input gradient aliases its output gradient")
	}

	requireMatrixValues(t, secondInputGradient, []float32{5, 6, 7, 8})
}

func Test_Flatten_ForwardValidatesReceiverAndInput(t *testing.T) {
	type testcase struct {
		name      string
		flatten   *layer.Flatten
		input     *matrix.Matrix
		wantError string
	}

	validFlatten := mustFlatten(t, 1, 2, 2)
	tests := []testcase{
		{
			name:      "nil receiver",
			flatten:   nil,
			input:     mustMatrix(t, 1, 4, []float32{1, 2, 3, 4}),
			wantError: "flatten layer is nil",
		},
		{
			name:      "zero value",
			flatten:   &layer.Flatten{},
			input:     mustMatrix(t, 1, 4, []float32{1, 2, 3, 4}),
			wantError: "flatten input shape invalid",
		},
		{
			name:      "nil input",
			flatten:   validFlatten,
			input:     nil,
			wantError: "flatten input is nil",
		},
		{
			name:      "invalid input",
			flatten:   validFlatten,
			input:     &matrix.Matrix{},
			wantError: "flatten input invalid",
		},
		{
			name:      "column mismatch",
			flatten:   validFlatten,
			input:     mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6}),
			wantError: "got 2x3, want batch rows x 4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			_, err = tt.flatten.Forward(tt.input)
			if err == nil {
				t.Fatal("Forward error = nil, want error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Forward error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func Test_Flatten_BackwardValidatesStateAndGradient(t *testing.T) {
	var (
		flatten *layer.Flatten
		err     error
	)

	flatten = mustFlatten(t, 1, 2, 2)
	_, err = flatten.Backward(mustMatrix(t, 1, 4, []float32{1, 2, 3, 4}))
	if err == nil || !strings.Contains(err.Error(), "backward called before forward") {
		t.Fatalf("Backward error = %v, want before-forward error", err)
	}

	if _, err = flatten.Forward(mustMatrix(t, 2, 4, make([]float32, 8))); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	type testcase struct {
		name           string
		outputGradient *matrix.Matrix
		wantError      string
	}

	tests := []testcase{
		{
			name:           "nil gradient",
			outputGradient: nil,
			wantError:      "flatten output gradient is nil",
		},
		{
			name:           "invalid gradient",
			outputGradient: &matrix.Matrix{},
			wantError:      "flatten output gradient invalid",
		},
		{
			name:           "batch mismatch",
			outputGradient: mustMatrix(t, 1, 4, make([]float32, 4)),
			wantError:      "got 1x4, want 2x4",
		},
		{
			name:           "column mismatch",
			outputGradient: mustMatrix(t, 2, 3, make([]float32, 6)),
			wantError:      "got 2x3, want 2x4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			_, err = flatten.Backward(tt.outputGradient)
			if err == nil {
				t.Fatal("Backward error = nil, want error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Backward error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func Test_Flatten_ComposesWithDenseInSequential(t *testing.T) {
	var (
		flatten        *layer.Flatten
		dense          *layer.Dense
		network        *model.Sequential
		input          *matrix.Matrix
		output         *matrix.Matrix
		inputGradient  *matrix.Matrix
		parameterCount int
		err            error
	)

	flatten = mustFlatten(t, 1, 2, 2)
	dense = mustDense(
		t,
		4,
		2,
		[]float32{
			1, 0,
			0, 1,
			1, 0,
			0, 1,
		},
		[]float32{0, 0},
	)
	network, err = model.NewSequential(flatten, dense)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	parameterCount = len(network.Parameters())
	if parameterCount != 2 {
		t.Fatalf("Sequential parameter count = %d, want only Dense's 2 parameters", parameterCount)
	}

	input = mustMatrix(t, 2, 4, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	})
	output, err = network.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	requireMatrixValues(t, output, []float32{4, 6, 12, 14})

	inputGradient, err = network.Backward(mustMatrix(t, 2, 2, []float32{1, 2, 3, 4}))
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{
		1, 2, 1, 2,
		3, 4, 3, 4,
	})
}

func mustFlatten(tb testing.TB, channels, height, width int) (flatten *layer.Flatten) {
	var (
		shape layer.SpatialShape
		err   error
	)

	tb.Helper()
	shape = mustSpatialShape(tb, channels, height, width)
	flatten, err = layer.NewFlatten(shape)
	if err != nil {
		tb.Fatalf("NewFlatten returned error: %v", err)
	}

	return flatten
}
