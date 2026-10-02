package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_AdaptiveAveragePool2D_ImplementsLayerAndExposesShapes(t *testing.T) {
	var (
		config  layer.AdaptiveAveragePool2DConfig
		pooling *layer.AdaptiveAveragePool2D
		network *model.Sequential
		err     error
	)

	var _ layer.Layer = (*layer.AdaptiveAveragePool2D)(nil)
	config = mustAdaptiveAveragePool2DConfig(t, 2, 3, 5, 2, 4)
	if pooling, err = layer.NewAdaptiveAveragePool2D(config); err != nil {
		t.Fatalf("NewAdaptiveAveragePool2D returned error: %v", err)
	}
	if pooling.Config() != config || pooling.InputShape() != config.InputShape() || pooling.OutputShape() != config.OutputShape() {
		t.Fatal("AdaptiveAveragePool2D accessors do not preserve configuration")
	}
	if config.OutputHeight() != 2 || config.OutputWidth() != 4 || config.OutputShape().Channels() != 2 {
		t.Fatalf("OutputShape = %#v, want 2x2x4", config.OutputShape())
	}
	if network, err = model.NewSequential(pooling); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if len(network.Parameters()) != 0 {
		t.Fatalf("parameter count = %d, want 0", len(network.Parameters()))
	}
}

func Test_NewAdaptiveAveragePool2DConfig_RejectsInvalidGeometry(t *testing.T) {
	type testcase struct {
		name          string
		inputShape    layer.SpatialShape
		outputHeight  int
		outputWidth   int
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
			outputHeight:  1,
			outputWidth:   1,
			wantErrorPart: "input shape invalid",
		},
		{
			name:          "zero output height",
			inputShape:    inputShape,
			outputHeight:  0,
			outputWidth:   1,
			wantErrorPart: "output height must be positive",
		},
		{
			name:          "negative output height",
			inputShape:    inputShape,
			outputHeight:  -1,
			outputWidth:   1,
			wantErrorPart: "output height must be positive",
		},
		{
			name:          "zero output width",
			inputShape:    inputShape,
			outputHeight:  1,
			outputWidth:   0,
			wantErrorPart: "output width must be positive",
		},
		{
			name:          "negative output width",
			inputShape:    inputShape,
			outputHeight:  1,
			outputWidth:   -1,
			wantErrorPart: "output width must be positive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			if _, err = layer.NewAdaptiveAveragePool2DConfig(
				tt.inputShape,
				tt.outputHeight,
				tt.outputWidth,
			); err == nil || !strings.Contains(err.Error(), tt.wantErrorPart) {
				t.Fatalf("NewAdaptiveAveragePool2DConfig error = %v, want containing %q", err, tt.wantErrorPart)
			}
		})
	}
}

func Test_AdaptiveAveragePool2D_ForwardUsesFloorCeilBins(t *testing.T) {
	var (
		pooling *layer.AdaptiveAveragePool2D
		output  *matrix.Matrix
		err     error
	)

	if pooling, err = layer.NewAdaptiveAveragePool2D(mustAdaptiveAveragePool2DConfig(t, 1, 3, 5, 2, 2)); err != nil {
		t.Fatalf("NewAdaptiveAveragePool2D returned error: %v", err)
	}
	if output, err = pooling.Forward(mustMatrix(t, 1, 15, []float32{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
	})); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{4.5, 6.5, 9.5, 11.5})
}

func Test_AdaptiveAveragePool2D_GlobalAveragePooling(t *testing.T) {
	var (
		pooling       *layer.AdaptiveAveragePool2D
		output        *matrix.Matrix
		inputGradient *matrix.Matrix
		err           error
	)

	if pooling, err = layer.NewAdaptiveAveragePool2D(mustAdaptiveAveragePool2DConfig(t, 2, 2, 4, 1, 1)); err != nil {
		t.Fatalf("NewAdaptiveAveragePool2D returned error: %v", err)
	}
	if output, err = pooling.Forward(mustMatrix(t, 1, 16, []float32{
		1, 2, 3, 4, 5, 6, 7, 8,
		-1, -2, -3, -4, -5, -6, -7, -8,
	})); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{4.5, -4.5})
	if inputGradient, err = pooling.Backward(mustMatrix(t, 1, 2, []float32{8, -16})); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{
		1, 1, 1, 1, 1, 1, 1, 1,
		-2, -2, -2, -2, -2, -2, -2, -2,
	})
}

func Test_AdaptiveAveragePool2D_SupportsLargerOutputWithOverlappingBins(t *testing.T) {
	var (
		pooling *layer.AdaptiveAveragePool2D
		output  *matrix.Matrix
		err     error
	)

	if pooling, err = layer.NewAdaptiveAveragePool2D(mustAdaptiveAveragePool2DConfig(t, 1, 2, 2, 3, 3)); err != nil {
		t.Fatalf("NewAdaptiveAveragePool2D returned error: %v", err)
	}
	if output, err = pooling.Forward(mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{
		1, 1.5, 2,
		2, 2.5, 3,
		3, 3.5, 4,
	})
}

func Test_AdaptiveAveragePool2D_BackwardAccumulatesOverlappingBins(t *testing.T) {
	var (
		pooling       *layer.AdaptiveAveragePool2D
		inputGradient *matrix.Matrix
		err           error
	)

	if pooling, err = layer.NewAdaptiveAveragePool2D(mustAdaptiveAveragePool2DConfig(t, 1, 3, 5, 2, 2)); err != nil {
		t.Fatalf("NewAdaptiveAveragePool2D returned error: %v", err)
	}
	if _, err = pooling.Forward(mustMatrix(t, 1, 15, make([]float32, 15))); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if inputGradient, err = pooling.Backward(mustMatrix(t, 1, 4, []float32{1, 1, 1, 1})); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{
		1.0 / 6, 1.0 / 6, 2.0 / 6, 1.0 / 6, 1.0 / 6,
		2.0 / 6, 2.0 / 6, 4.0 / 6, 2.0 / 6, 2.0 / 6,
		1.0 / 6, 1.0 / 6, 2.0 / 6, 1.0 / 6, 1.0 / 6,
	})
}

func Test_AdaptiveAveragePool2D_ValidatesConfigurationAndCallOrder(t *testing.T) {
	var (
		pooling *layer.AdaptiveAveragePool2D
		err     error
	)

	if pooling, err = layer.NewAdaptiveAveragePool2D(layer.AdaptiveAveragePool2DConfig{}); err == nil {
		t.Fatal("NewAdaptiveAveragePool2D error = nil, want error")
	}
	if pooling != nil || !strings.Contains(err.Error(), "configuration invalid") {
		t.Fatalf("NewAdaptiveAveragePool2D result = %v, %v", pooling, err)
	}
	if pooling, err = layer.NewAdaptiveAveragePool2D(mustAdaptiveAveragePool2DConfig(t, 1, 2, 2, 1, 1)); err != nil {
		t.Fatalf("NewAdaptiveAveragePool2D returned error: %v", err)
	}
	if _, err = pooling.Backward(mustMatrix(t, 1, 1, []float32{1})); err == nil || !strings.Contains(err.Error(), "before forward") {
		t.Fatalf("Backward error = %v, want call-order error", err)
	}
	if _, err = pooling.Forward(mustMatrix(t, 1, 3, []float32{1, 2, 3})); err == nil || !strings.Contains(err.Error(), "input shape mismatch") {
		t.Fatalf("Forward error = %v, want shape error", err)
	}
}

func mustAdaptiveAveragePool2DConfig(
	tb testing.TB,
	inputChannels, inputHeight, inputWidth int,
	outputHeight, outputWidth int,
) (config layer.AdaptiveAveragePool2DConfig) {
	var (
		inputShape layer.SpatialShape
		err        error
	)

	tb.Helper()
	if inputShape, err = layer.NewSpatialShape(inputChannels, inputHeight, inputWidth); err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}
	if config, err = layer.NewAdaptiveAveragePool2DConfig(inputShape, outputHeight, outputWidth); err != nil {
		tb.Fatalf("NewAdaptiveAveragePool2DConfig returned error: %v", err)
	}
	return config
}
