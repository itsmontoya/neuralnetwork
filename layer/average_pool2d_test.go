package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_AveragePool2D_ImplementsLayerAndExposesShapes(t *testing.T) {
	var (
		config  layer.AveragePool2DConfig
		pooling *layer.AveragePool2D
		network *model.Sequential
		err     error
	)

	var _ layer.Layer = (*layer.AveragePool2D)(nil)
	config = mustAveragePool2DConfig(t, 2, 3, 5, 2, 2, 2, 2)
	if pooling, err = layer.NewAveragePool2D(config); err != nil {
		t.Fatalf("NewAveragePool2D returned error: %v", err)
	}
	if pooling.Config() != config || pooling.InputShape() != config.InputShape() || pooling.OutputShape() != config.OutputShape() {
		t.Fatal("AveragePool2D accessors do not preserve configuration")
	}
	if config.OutputShape().Channels() != 2 || config.OutputShape().Height() != 1 || config.OutputShape().Width() != 2 {
		t.Fatalf("OutputShape = %#v, want 2x1x2", config.OutputShape())
	}
	if network, err = model.NewSequential(pooling); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if len(network.Parameters()) != 0 {
		t.Fatalf("parameter count = %d, want 0", len(network.Parameters()))
	}
}

func Test_NewAveragePool2DConfig_RejectsInvalidGeometry(t *testing.T) {
	type testcase struct {
		name          string
		inputShape    layer.SpatialShape
		windowHeight  int
		windowWidth   int
		strideHeight  int
		strideWidth   int
		wantErrorPart string
	}

	var (
		inputShape layer.SpatialShape
		tests      []testcase
		err        error
	)

	if inputShape, err = layer.NewSpatialShape(1, 3, 5); err != nil {
		t.Fatalf("NewSpatialShape returned error: %v", err)
	}
	tests = []testcase{
		{
			name:          "invalid input shape",
			inputShape:    layer.SpatialShape{},
			windowHeight:  1,
			windowWidth:   1,
			strideHeight:  1,
			strideWidth:   1,
			wantErrorPart: "input shape invalid",
		},
		{
			name:          "zero window height",
			inputShape:    inputShape,
			windowHeight:  0,
			windowWidth:   1,
			strideHeight:  1,
			strideWidth:   1,
			wantErrorPart: "window height must be positive",
		},
		{
			name:          "negative window width",
			inputShape:    inputShape,
			windowHeight:  1,
			windowWidth:   -1,
			strideHeight:  1,
			strideWidth:   1,
			wantErrorPart: "window width must be positive",
		},
		{
			name:          "zero stride height",
			inputShape:    inputShape,
			windowHeight:  1,
			windowWidth:   1,
			strideHeight:  0,
			strideWidth:   1,
			wantErrorPart: "stride height must be positive",
		},
		{
			name:          "negative stride width",
			inputShape:    inputShape,
			windowHeight:  1,
			windowWidth:   1,
			strideHeight:  1,
			strideWidth:   -1,
			wantErrorPart: "stride width must be positive",
		},
		{
			name:          "window taller than input",
			inputShape:    inputShape,
			windowHeight:  4,
			windowWidth:   1,
			strideHeight:  1,
			strideWidth:   1,
			wantErrorPart: "window height",
		},
		{
			name:          "window wider than input",
			inputShape:    inputShape,
			windowHeight:  1,
			windowWidth:   6,
			strideHeight:  1,
			strideWidth:   1,
			wantErrorPart: "window width",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			if _, err = layer.NewAveragePool2DConfig(
				tt.inputShape,
				tt.windowHeight,
				tt.windowWidth,
				tt.strideHeight,
				tt.strideWidth,
			); err == nil || !strings.Contains(err.Error(), tt.wantErrorPart) {
				t.Fatalf("NewAveragePool2DConfig error = %v, want containing %q", err, tt.wantErrorPart)
			}
		})
	}
}

func Test_AveragePool2D_ForwardUsesCompleteRectangularWindows(t *testing.T) {
	var (
		config  layer.AveragePool2DConfig
		pooling *layer.AveragePool2D
		input   *matrix.Matrix
		output  *matrix.Matrix
		err     error
	)

	config = mustAveragePool2DConfig(t, 1, 3, 5, 2, 2, 2, 2)
	if pooling, err = layer.NewAveragePool2D(config); err != nil {
		t.Fatalf("NewAveragePool2D returned error: %v", err)
	}
	input = mustMatrix(t, 2, 15, []float32{
		1, 2, 3, 4, 100,
		6, 7, 8, 9, 100,
		100, 100, 100, 100, 100,
		-1, -2, -3, -4, -100,
		-6, -7, -8, -9, -100,
		-100, -100, -100, -100, -100,
	})
	if output, err = pooling.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	if output.Rows() != 2 || output.Cols() != 2 {
		t.Fatalf("output shape = %dx%d, want 2x2", output.Rows(), output.Cols())
	}
	requireMatrixValues(t, output, []float32{4, 6, -4, -6})
}

func Test_AveragePool2D_BackwardDistributesAndAccumulates(t *testing.T) {
	var (
		config         layer.AveragePool2DConfig
		pooling        *layer.AveragePool2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	config = mustAveragePool2DConfig(t, 1, 3, 3, 2, 2, 1, 1)
	if pooling, err = layer.NewAveragePool2D(config); err != nil {
		t.Fatalf("NewAveragePool2D returned error: %v", err)
	}
	input = mustMatrix(t, 1, 9, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	if _, err = pooling.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	outputGradient = mustMatrix(t, 1, 4, []float32{4, 8, 12, 16})
	if inputGradient, err = pooling.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{
		1, 3, 2,
		4, 10, 6,
		3, 7, 4,
	})
}

func Test_AveragePool2D_TrailingEdgesReceiveZeroGradient(t *testing.T) {
	var (
		pooling       *layer.AveragePool2D
		inputGradient *matrix.Matrix
		err           error
	)

	if pooling, err = layer.NewAveragePool2D(mustAveragePool2DConfig(t, 1, 3, 5, 2, 2, 2, 2)); err != nil {
		t.Fatalf("NewAveragePool2D returned error: %v", err)
	}
	if _, err = pooling.Forward(mustMatrix(t, 1, 15, make([]float32, 15))); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if inputGradient, err = pooling.Backward(mustMatrix(t, 1, 2, []float32{4, 8})); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{
		1, 1, 2, 2, 0,
		1, 1, 2, 2, 0,
		0, 0, 0, 0, 0,
	})
}

func Test_AveragePool2D_ValidatesConfigurationAndCallOrder(t *testing.T) {
	var (
		pooling *layer.AveragePool2D
		err     error
	)

	if pooling, err = layer.NewAveragePool2D(layer.AveragePool2DConfig{}); err == nil {
		t.Fatal("NewAveragePool2D error = nil, want error")
	}
	if pooling != nil || !strings.Contains(err.Error(), "configuration invalid") {
		t.Fatalf("NewAveragePool2D result = %v, %v", pooling, err)
	}
	if pooling, err = layer.NewAveragePool2D(mustAveragePool2DConfig(t, 1, 2, 2, 2, 2, 2, 2)); err != nil {
		t.Fatalf("NewAveragePool2D returned error: %v", err)
	}
	if _, err = pooling.Backward(mustMatrix(t, 1, 1, []float32{1})); err == nil || !strings.Contains(err.Error(), "before forward") {
		t.Fatalf("Backward error = %v, want call-order error", err)
	}
	if _, err = pooling.Forward(mustMatrix(t, 1, 3, []float32{1, 2, 3})); err == nil || !strings.Contains(err.Error(), "input shape mismatch") {
		t.Fatalf("Forward error = %v, want shape error", err)
	}
}

func mustAveragePool2DConfig(
	tb testing.TB,
	inputChannels, inputHeight, inputWidth int,
	windowHeight, windowWidth int,
	strideHeight, strideWidth int,
) (config layer.AveragePool2DConfig) {
	var (
		inputShape layer.SpatialShape
		err        error
	)

	tb.Helper()
	if inputShape, err = layer.NewSpatialShape(inputChannels, inputHeight, inputWidth); err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}
	if config, err = layer.NewAveragePool2DConfig(
		inputShape,
		windowHeight,
		windowWidth,
		strideHeight,
		strideWidth,
	); err != nil {
		tb.Fatalf("NewAveragePool2DConfig returned error: %v", err)
	}
	return config
}
