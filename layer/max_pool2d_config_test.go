package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
)

func Test_NewMaxPool2DConfig(t *testing.T) {
	type testcase struct {
		name            string
		inputShape      layer.SpatialShape
		windowHeight    int
		windowWidth     int
		strideHeight    int
		strideWidth     int
		wantOutputShape layer.SpatialShape
	}

	tests := []testcase{
		{
			name:            "rectangular with floor behavior",
			inputShape:      mustSpatialShape(t, 2, 5, 8),
			windowHeight:    2,
			windowWidth:     3,
			strideHeight:    2,
			strideWidth:     3,
			wantOutputShape: mustSpatialShape(t, 2, 2, 2),
		},
		{
			name:            "overlapping windows",
			inputShape:      mustSpatialShape(t, 3, 4, 5),
			windowHeight:    2,
			windowWidth:     3,
			strideHeight:    1,
			strideWidth:     2,
			wantOutputShape: mustSpatialShape(t, 3, 3, 2),
		},
		{
			name:            "exact window",
			inputShape:      mustSpatialShape(t, 1, 2, 4),
			windowHeight:    2,
			windowWidth:     4,
			strideHeight:    2,
			strideWidth:     4,
			wantOutputShape: mustSpatialShape(t, 1, 1, 1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				config layer.MaxPool2DConfig
				err    error
			)

			config, err = layer.NewMaxPool2DConfig(
				tt.inputShape,
				tt.windowHeight,
				tt.windowWidth,
				tt.strideHeight,
				tt.strideWidth,
			)
			if err != nil {
				t.Fatalf("NewMaxPool2DConfig returned error: %v", err)
			}

			if config.InputShape() != tt.inputShape {
				t.Fatalf("InputShape = %#v, want %#v", config.InputShape(), tt.inputShape)
			}

			if config.OutputShape() != tt.wantOutputShape {
				t.Fatalf("OutputShape = %#v, want %#v", config.OutputShape(), tt.wantOutputShape)
			}

			if config.WindowHeight() != tt.windowHeight || config.WindowWidth() != tt.windowWidth {
				t.Fatalf(
					"window = %dx%d, want %dx%d",
					config.WindowHeight(),
					config.WindowWidth(),
					tt.windowHeight,
					tt.windowWidth,
				)
			}

			if config.StrideHeight() != tt.strideHeight || config.StrideWidth() != tt.strideWidth {
				t.Fatalf(
					"stride = %dx%d, want %dx%d",
					config.StrideHeight(),
					config.StrideWidth(),
					tt.strideHeight,
					tt.strideWidth,
				)
			}
		})
	}
}

func Test_NewMaxPool2DConfig_ValidatesDimensions(t *testing.T) {
	type testcase struct {
		name          string
		inputShape    layer.SpatialShape
		windowHeight  int
		windowWidth   int
		strideHeight  int
		strideWidth   int
		wantErrorPart string
	}

	validShape := mustSpatialShape(t, 1, 3, 3)
	tests := []testcase{
		{name: "zero input shape", inputShape: layer.SpatialShape{}, windowHeight: 1, windowWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "input shape"},
		{name: "zero window height", inputShape: validShape, windowHeight: 0, windowWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "window height"},
		{name: "negative window height", inputShape: validShape, windowHeight: -1, windowWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "window height"},
		{name: "zero window width", inputShape: validShape, windowHeight: 1, windowWidth: 0, strideHeight: 1, strideWidth: 1, wantErrorPart: "window width"},
		{name: "negative window width", inputShape: validShape, windowHeight: 1, windowWidth: -1, strideHeight: 1, strideWidth: 1, wantErrorPart: "window width"},
		{name: "zero stride height", inputShape: validShape, windowHeight: 1, windowWidth: 1, strideHeight: 0, strideWidth: 1, wantErrorPart: "stride height"},
		{name: "negative stride height", inputShape: validShape, windowHeight: 1, windowWidth: 1, strideHeight: -1, strideWidth: 1, wantErrorPart: "stride height"},
		{name: "zero stride width", inputShape: validShape, windowHeight: 1, windowWidth: 1, strideHeight: 1, strideWidth: 0, wantErrorPart: "stride width"},
		{name: "negative stride width", inputShape: validShape, windowHeight: 1, windowWidth: 1, strideHeight: 1, strideWidth: -1, wantErrorPart: "stride width"},
		{name: "window height exceeds input", inputShape: validShape, windowHeight: 4, windowWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "got=4 want<=3"},
		{name: "window width exceeds input", inputShape: validShape, windowHeight: 1, windowWidth: 4, strideHeight: 1, strideWidth: 1, wantErrorPart: "got=4 want<=3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				config layer.MaxPool2DConfig
				err    error
			)

			config, err = layer.NewMaxPool2DConfig(
				tt.inputShape,
				tt.windowHeight,
				tt.windowWidth,
				tt.strideHeight,
				tt.strideWidth,
			)
			if err == nil {
				t.Fatal("NewMaxPool2DConfig error = nil, want error")
			}

			if !strings.HasPrefix(err.Error(), "layer: ") {
				t.Fatalf("NewMaxPool2DConfig error = %q, want layer context", err)
			}

			if !strings.Contains(err.Error(), tt.wantErrorPart) {
				t.Fatalf("NewMaxPool2DConfig error = %q, want substring %q", err, tt.wantErrorPart)
			}

			if config != (layer.MaxPool2DConfig{}) {
				t.Fatalf("NewMaxPool2DConfig config = %#v, want zero value on error", config)
			}
		})
	}
}

func Test_NewMaxPool2DConfig_AcceptsMaximumDimension(t *testing.T) {
	var (
		config layer.MaxPool2DConfig
		err    error
	)

	maxInt := int(^uint(0) >> 1)
	config, err = layer.NewMaxPool2DConfig(
		mustSpatialShape(t, 1, maxInt, 1),
		maxInt,
		1,
		maxInt,
		1,
	)
	if err != nil {
		t.Fatalf("NewMaxPool2DConfig returned error at maximum dimension: %v", err)
	}

	if config.OutputShape() != mustSpatialShape(t, 1, 1, 1) {
		t.Fatalf("OutputShape = %#v, want 1x1x1", config.OutputShape())
	}
}
