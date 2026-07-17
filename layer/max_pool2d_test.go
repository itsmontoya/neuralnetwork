package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_MaxPool2D_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.MaxPool2D)(nil)
}

func Test_NewMaxPool2D_ValidatesConfiguration(t *testing.T) {
	var (
		pool *layer.MaxPool2D
		err  error
	)

	pool, err = layer.NewMaxPool2D(layer.MaxPool2DConfig{})
	if err == nil {
		t.Fatal("NewMaxPool2D error = nil, want error")
	}

	if pool != nil {
		t.Fatal("NewMaxPool2D returned layer on error")
	}

	if !strings.HasPrefix(err.Error(), "layer: max pool2d configuration invalid:") {
		t.Fatalf("NewMaxPool2D error = %q, want configuration context", err)
	}
}

func Test_MaxPool2D_AccessorsAndParameterFreeModel(t *testing.T) {
	var (
		config  layer.MaxPool2DConfig
		pool    *layer.MaxPool2D
		network *model.Sequential
		err     error
	)

	config = mustMaxPool2DConfig(t, 2, 4, 5, 2, 3, 1, 2)
	pool, err = layer.NewMaxPool2D(config)
	if err != nil {
		t.Fatalf("NewMaxPool2D returned error: %v", err)
	}

	if pool.Config() != config {
		t.Fatalf("Config = %#v, want %#v", pool.Config(), config)
	}

	if pool.InputShape() != config.InputShape() {
		t.Fatalf("InputShape = %#v, want %#v", pool.InputShape(), config.InputShape())
	}

	if pool.OutputShape() != config.OutputShape() {
		t.Fatalf("OutputShape = %#v, want %#v", pool.OutputShape(), config.OutputShape())
	}

	network, err = model.NewSequential(pool)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	if len(network.Parameters()) != 0 {
		t.Fatalf("Sequential parameters length = %d, want 0", len(network.Parameters()))
	}
}

func Test_MaxPool2D_NilReceiverAndZeroValue(t *testing.T) {
	var (
		nilPool  *layer.MaxPool2D
		zeroPool layer.MaxPool2D
		err      error
	)

	if nilPool.Config() != (layer.MaxPool2DConfig{}) {
		t.Fatal("Config returned value for nil receiver")
	}

	if nilPool.InputShape() != (layer.SpatialShape{}) {
		t.Fatal("InputShape returned value for nil receiver")
	}

	if nilPool.OutputShape() != (layer.SpatialShape{}) {
		t.Fatal("OutputShape returned value for nil receiver")
	}

	if _, err = nilPool.Forward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("Forward error = nil, want nil receiver error")
	} else if !strings.Contains(err.Error(), "layer is nil") {
		t.Fatalf("Forward error = %q, want nil receiver context", err)
	}

	if _, err = nilPool.Backward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("Backward error = nil, want nil receiver error")
	} else if !strings.Contains(err.Error(), "layer is nil") {
		t.Fatalf("Backward error = %q, want nil receiver context", err)
	}

	if _, err = zeroPool.Forward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("zero-value Forward error = nil, want invalid state error")
	} else if !strings.Contains(err.Error(), "configuration invalid") {
		t.Fatalf("zero-value Forward error = %q, want configuration context", err)
	}
}

func Test_MaxPool2D_Forward(t *testing.T) {
	type testcase struct {
		name       string
		config     layer.MaxPool2DConfig
		rows       int
		input      []float32
		wantOutput []float32
	}

	tests := []testcase{
		{
			name:   "batches channels and negative values",
			config: mustMaxPool2DConfig(t, 2, 2, 3, 2, 2, 1, 1),
			rows:   2,
			input: []float32{
				1, 5, 2,
				4, 3, 6,
				-1, -2, -3,
				-4, -5, -6,
				9, 8, 7,
				6, 5, 4,
				0, 3, 2,
				1, 4, -1,
			},
			wantOutput: []float32{
				5, 6,
				-1, -2,
				9, 8,
				4, 4,
			},
		},
		{
			name:   "rectangular window and stride",
			config: mustMaxPool2DConfig(t, 1, 3, 5, 2, 3, 1, 2),
			rows:   1,
			input: []float32{
				1, 9, 3, 4, 5,
				6, 2, 8, 7, 0,
				-1, 10, 11, 12, 13,
			},
			wantOutput: []float32{9, 8, 11, 13},
		},
		{
			name:   "first maximum wins ties",
			config: mustMaxPool2DConfig(t, 1, 2, 3, 2, 2, 1, 1),
			rows:   1,
			input: []float32{
				5, 5, 1,
				5, 0, 5,
			},
			wantOutput: []float32{5, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				pool   *layer.MaxPool2D
				input  *matrix.Matrix
				output *matrix.Matrix
				err    error
			)

			pool, err = layer.NewMaxPool2D(tt.config)
			if err != nil {
				t.Fatalf("NewMaxPool2D returned error: %v", err)
			}

			input = mustMatrix(t, tt.rows, tt.config.InputShape().Size(), tt.input)
			output, err = pool.Forward(input)
			if err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			if output.Rows() != tt.rows || output.Cols() != tt.config.OutputShape().Size() {
				t.Fatalf(
					"Forward output shape = %dx%d, want %dx%d",
					output.Rows(),
					output.Cols(),
					tt.rows,
					tt.config.OutputShape().Size(),
				)
			}

			requireMatrixValues(t, output, tt.wantOutput)
		})
	}
}

func Test_MaxPool2D_Backward(t *testing.T) {
	type testcase struct {
		name              string
		config            layer.MaxPool2DConfig
		rows              int
		input             []float32
		outputGradient    []float32
		wantInputGradient []float32
	}

	tests := []testcase{
		{
			name:   "batches and channels",
			config: mustMaxPool2DConfig(t, 2, 2, 2, 2, 2, 2, 2),
			rows:   2,
			input: []float32{
				1, 4, 2, 3,
				-1, -2, -4, -3,
				9, 7, 8, 6,
				0, 1, 3, 2,
			},
			outputGradient: []float32{2, -3, 4, 5},
			wantInputGradient: []float32{
				0, 2, 0, 0,
				-3, 0, 0, 0,
				4, 0, 0, 0,
				0, 0, 5, 0,
			},
		},
		{
			name:   "ties route to first maximum",
			config: mustMaxPool2DConfig(t, 1, 2, 3, 2, 2, 1, 1),
			rows:   1,
			input: []float32{
				5, 5, 1,
				5, 0, 5,
			},
			outputGradient:    []float32{2, 3},
			wantInputGradient: []float32{2, 3, 0, 0, 0, 0},
		},
		{
			name:   "overlapping windows accumulate",
			config: mustMaxPool2DConfig(t, 1, 3, 3, 2, 2, 1, 1),
			rows:   1,
			input: []float32{
				1, 2, 3,
				4, 9, 5,
				6, 7, 8,
			},
			outputGradient:    []float32{1, 2, 3, 4},
			wantInputGradient: []float32{0, 0, 0, 0, 10, 0, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				pool           *layer.MaxPool2D
				input          *matrix.Matrix
				outputGradient *matrix.Matrix
				inputGradient  *matrix.Matrix
				err            error
			)

			pool, err = layer.NewMaxPool2D(tt.config)
			if err != nil {
				t.Fatalf("NewMaxPool2D returned error: %v", err)
			}

			input = mustMatrix(t, tt.rows, tt.config.InputShape().Size(), tt.input)
			if _, err = pool.Forward(input); err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			outputGradient = mustMatrix(t, tt.rows, tt.config.OutputShape().Size(), tt.outputGradient)
			inputGradient, err = pool.Backward(outputGradient)
			if err != nil {
				t.Fatalf("Backward returned error: %v", err)
			}

			if inputGradient.Rows() != tt.rows || inputGradient.Cols() != tt.config.InputShape().Size() {
				t.Fatalf(
					"Backward output shape = %dx%d, want %dx%d",
					inputGradient.Rows(),
					inputGradient.Cols(),
					tt.rows,
					tt.config.InputShape().Size(),
				)
			}

			requireMatrixValues(t, inputGradient, tt.wantInputGradient)
		})
	}
}

func Test_MaxPool2D_ValidatesForwardAndBackwardShapes(t *testing.T) {
	var (
		config layer.MaxPool2DConfig
		pool   *layer.MaxPool2D
		err    error
	)

	config = mustMaxPool2DConfig(t, 1, 2, 3, 2, 2, 1, 1)
	pool, err = layer.NewMaxPool2D(config)
	if err != nil {
		t.Fatalf("NewMaxPool2D returned error: %v", err)
	}

	if _, err = pool.Forward(nil); err == nil || !strings.Contains(err.Error(), "input is nil") {
		t.Fatalf("nil Forward error = %v, want input error", err)
	}

	if _, err = pool.Forward(mustMatrix(t, 2, config.InputShape().Size()-1, make([]float32, 10))); err == nil {
		t.Fatal("Forward error = nil, want input shape error")
	} else if !strings.Contains(err.Error(), "got 2x5, want batch rows x 6") {
		t.Fatalf("Forward error = %q, want diagnostic shape", err)
	}

	if _, err = pool.Backward(mustMatrix(t, 1, config.OutputShape().Size(), make([]float32, config.OutputShape().Size()))); err == nil {
		t.Fatal("Backward error = nil, want backward-before-forward error")
	} else if !strings.Contains(err.Error(), "backward called before forward") {
		t.Fatalf("Backward error = %q, want state context", err)
	}

	if _, err = pool.Forward(mustMatrix(t, 2, config.InputShape().Size(), make([]float32, 12))); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	if _, err = pool.Backward(nil); err == nil || !strings.Contains(err.Error(), "output gradient is nil") {
		t.Fatalf("nil Backward error = %v, want output gradient error", err)
	}

	if _, err = pool.Backward(mustMatrix(t, 1, config.OutputShape().Size(), make([]float32, config.OutputShape().Size()))); err == nil {
		t.Fatal("Backward error = nil, want batch shape error")
	} else if !strings.Contains(err.Error(), "got 1x2, want 2x2") {
		t.Fatalf("Backward error = %q, want diagnostic batch shape", err)
	}

	if _, err = pool.Backward(mustMatrix(t, 2, config.OutputShape().Size()-1, make([]float32, 2))); err == nil {
		t.Fatal("Backward error = nil, want column shape error")
	} else if !strings.Contains(err.Error(), "got 2x1, want 2x2") {
		t.Fatalf("Backward error = %q, want diagnostic column shape", err)
	}
}

func Test_MaxPool2D_BackwardUsesForwardArgmaxAfterInputMutation(t *testing.T) {
	var (
		config         layer.MaxPool2DConfig
		pool           *layer.MaxPool2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	config = mustMaxPool2DConfig(t, 1, 1, 3, 1, 2, 1, 1)
	pool, err = layer.NewMaxPool2D(config)
	if err != nil {
		t.Fatalf("NewMaxPool2D returned error: %v", err)
	}

	input = mustMatrix(t, 1, 3, []float32{3, 1, 2})
	if _, err = pool.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	if err = input.CopyValuesFrom([]float32{0, 100, 0}); err != nil {
		t.Fatalf("input CopyValuesFrom returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 1, 2, []float32{4, 5})
	inputGradient, err = pool.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{4, 0, 5})
}

func Test_MaxPool2D_DoesNotAliasOrRetainCallerStorage(t *testing.T) {
	var (
		config              layer.MaxPool2DConfig
		pool                *layer.MaxPool2D
		input               *matrix.Matrix
		outputGradient      *matrix.Matrix
		firstOutput         *matrix.Matrix
		secondOutput        *matrix.Matrix
		firstInputGradient  *matrix.Matrix
		secondInputGradient *matrix.Matrix
		err                 error
	)

	config = mustMaxPool2DConfig(t, 1, 1, 2, 1, 1, 1, 1)
	pool, err = layer.NewMaxPool2D(config)
	if err != nil {
		t.Fatalf("NewMaxPool2D returned error: %v", err)
	}

	input = mustMatrix(t, 1, 2, []float32{3, 4})
	outputGradient = mustMatrix(t, 1, 2, []float32{1, 2})
	firstOutput, err = pool.Forward(input)
	if err != nil {
		t.Fatalf("first Forward returned error: %v", err)
	}

	if firstOutput == input {
		t.Fatal("Forward output aliases input")
	}

	requireMatrixValues(t, input, []float32{3, 4})
	if err = input.Fill(100); err != nil {
		t.Fatalf("input Fill returned error: %v", err)
	}

	firstInputGradient, err = pool.Backward(outputGradient)
	if err != nil {
		t.Fatalf("first Backward returned error: %v", err)
	}

	requireMatrixValues(t, firstOutput, []float32{3, 4})
	requireMatrixValues(t, outputGradient, []float32{1, 2})
	requireMatrixValues(t, firstInputGradient, []float32{1, 2})
	if firstInputGradient == outputGradient {
		t.Fatal("Backward input gradient aliases output gradient")
	}

	secondOutput, err = pool.Forward(firstOutput)
	if err != nil {
		t.Fatalf("aliased Forward returned error: %v", err)
	}

	if secondOutput == firstOutput {
		t.Fatal("Forward scratch output aliases its live input")
	}

	if _, err = pool.Forward(mustMatrix(t, 1, 2, []float32{3, 4})); err != nil {
		t.Fatalf("Forward before aliased Backward returned error: %v", err)
	}

	secondInputGradient, err = pool.Backward(firstInputGradient)
	if err != nil {
		t.Fatalf("aliased Backward returned error: %v", err)
	}

	if secondInputGradient == firstInputGradient {
		t.Fatal("Backward scratch output aliases its live output gradient")
	}
}

func mustMaxPool2DConfig(
	tb testing.TB,
	inputChannels, inputHeight, inputWidth int,
	windowHeight, windowWidth int,
	strideHeight, strideWidth int,
) (config layer.MaxPool2DConfig) {
	var (
		inputShape layer.SpatialShape
		err        error
	)

	tb.Helper()
	inputShape, err = layer.NewSpatialShape(inputChannels, inputHeight, inputWidth)
	if err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}

	config, err = layer.NewMaxPool2DConfig(
		inputShape,
		windowHeight,
		windowWidth,
		strideHeight,
		strideWidth,
	)
	if err != nil {
		tb.Fatalf("NewMaxPool2DConfig returned error: %v", err)
	}

	return config
}
