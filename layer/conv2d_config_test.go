package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
)

func Test_NewConv2DConfig(t *testing.T) {
	type testcase struct {
		name            string
		inputShape      layer.SpatialShape
		outputChannels  int
		kernelHeight    int
		kernelWidth     int
		strideHeight    int
		strideWidth     int
		paddingHeight   int
		paddingWidth    int
		wantOutputShape layer.SpatialShape
	}

	tests := []testcase{
		{
			name:            "rectangular with floor behavior",
			inputShape:      mustSpatialShape(t, 2, 5, 8),
			outputChannels:  3,
			kernelHeight:    2,
			kernelWidth:     3,
			strideHeight:    2,
			strideWidth:     3,
			paddingHeight:   0,
			paddingWidth:    0,
			wantOutputShape: mustSpatialShape(t, 3, 2, 2),
		},
		{
			name:            "rectangular padding",
			inputShape:      mustSpatialShape(t, 3, 5, 7),
			outputChannels:  4,
			kernelHeight:    2,
			kernelWidth:     3,
			strideHeight:    2,
			strideWidth:     2,
			paddingHeight:   1,
			paddingWidth:    0,
			wantOutputShape: mustSpatialShape(t, 4, 3, 3),
		},
		{
			name:            "kernel fits padded input",
			inputShape:      mustSpatialShape(t, 1, 2, 3),
			outputChannels:  2,
			kernelHeight:    4,
			kernelWidth:     5,
			strideHeight:    1,
			strideWidth:     1,
			paddingHeight:   1,
			paddingWidth:    1,
			wantOutputShape: mustSpatialShape(t, 2, 1, 1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				config layer.Conv2DConfig
				err    error
			)

			config, err = layer.NewConv2DConfig(
				tt.inputShape,
				tt.outputChannels,
				tt.kernelHeight,
				tt.kernelWidth,
				tt.strideHeight,
				tt.strideWidth,
				tt.paddingHeight,
				tt.paddingWidth,
			)
			if err != nil {
				t.Fatalf("NewConv2DConfig returned error: %v", err)
			}

			if config.InputShape() != tt.inputShape {
				t.Fatalf("InputShape = %#v, want %#v", config.InputShape(), tt.inputShape)
			}

			if config.OutputShape() != tt.wantOutputShape {
				t.Fatalf("OutputShape = %#v, want %#v", config.OutputShape(), tt.wantOutputShape)
			}

			if config.OutputChannels() != tt.outputChannels {
				t.Fatalf("OutputChannels = %d, want %d", config.OutputChannels(), tt.outputChannels)
			}

			if config.KernelHeight() != tt.kernelHeight || config.KernelWidth() != tt.kernelWidth {
				t.Fatalf(
					"kernel = %dx%d, want %dx%d",
					config.KernelHeight(),
					config.KernelWidth(),
					tt.kernelHeight,
					tt.kernelWidth,
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

			if config.PaddingHeight() != tt.paddingHeight || config.PaddingWidth() != tt.paddingWidth {
				t.Fatalf(
					"padding = %dx%d, want %dx%d",
					config.PaddingHeight(),
					config.PaddingWidth(),
					tt.paddingHeight,
					tt.paddingWidth,
				)
			}
		})
	}
}

func Test_NewConv2DConfig_ValidatesDimensions(t *testing.T) {
	type testcase struct {
		name           string
		inputShape     layer.SpatialShape
		outputChannels int
		kernelHeight   int
		kernelWidth    int
		strideHeight   int
		strideWidth    int
		paddingHeight  int
		paddingWidth   int
		wantErrorPart  string
	}

	maxInt := int(^uint(0) >> 1)
	validShape := mustSpatialShape(t, 1, 3, 3)
	tests := []testcase{
		{name: "zero input shape", inputShape: layer.SpatialShape{}, outputChannels: 1, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "input shape"},
		{name: "zero output channels", inputShape: validShape, outputChannels: 0, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "output channels"},
		{name: "negative output channels", inputShape: validShape, outputChannels: -1, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "output channels"},
		{name: "zero kernel height", inputShape: validShape, outputChannels: 1, kernelHeight: 0, kernelWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "kernel height"},
		{name: "negative kernel height", inputShape: validShape, outputChannels: 1, kernelHeight: -1, kernelWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "kernel height"},
		{name: "zero kernel width", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 0, strideHeight: 1, strideWidth: 1, wantErrorPart: "kernel width"},
		{name: "negative kernel width", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: -1, strideHeight: 1, strideWidth: 1, wantErrorPart: "kernel width"},
		{name: "zero stride height", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 1, strideHeight: 0, strideWidth: 1, wantErrorPart: "stride height"},
		{name: "negative stride height", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 1, strideHeight: -1, strideWidth: 1, wantErrorPart: "stride height"},
		{name: "zero stride width", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: 0, wantErrorPart: "stride width"},
		{name: "negative stride width", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: -1, wantErrorPart: "stride width"},
		{name: "negative padding height", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: 1, paddingHeight: -1, wantErrorPart: "padding height"},
		{name: "negative padding width", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: 1, paddingWidth: -1, wantErrorPart: "padding width"},
		{name: "kernel height exceeds input", inputShape: validShape, outputChannels: 1, kernelHeight: 4, kernelWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "got=4 want<=3"},
		{name: "kernel width exceeds input", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 4, strideHeight: 1, strideWidth: 1, wantErrorPart: "got=4 want<=3"},
		{name: "padded height overflow", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: 1, paddingHeight: maxInt, wantErrorPart: "padded input dimension overflows"},
		{name: "padded width overflow", inputShape: validShape, outputChannels: 1, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: 1, paddingWidth: maxInt, wantErrorPart: "padded input dimension overflows"},
		{name: "kernel fan-in overflow", inputShape: mustSpatialShape(t, maxInt, 1, 1), outputChannels: 1, kernelHeight: 1, kernelWidth: 2, strideHeight: 1, strideWidth: 1, paddingWidth: 1, wantErrorPart: "kernel size overflows"},
		{name: "output shape overflow", inputShape: mustSpatialShape(t, 1, 1, 2), outputChannels: maxInt, kernelHeight: 1, kernelWidth: 1, strideHeight: 1, strideWidth: 1, wantErrorPart: "output shape invalid"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				config layer.Conv2DConfig
				err    error
			)

			config, err = layer.NewConv2DConfig(
				tt.inputShape,
				tt.outputChannels,
				tt.kernelHeight,
				tt.kernelWidth,
				tt.strideHeight,
				tt.strideWidth,
				tt.paddingHeight,
				tt.paddingWidth,
			)
			if err == nil {
				t.Fatal("NewConv2DConfig error = nil, want error")
			}

			if !strings.HasPrefix(err.Error(), "layer: ") {
				t.Fatalf("NewConv2DConfig error = %q, want layer context", err)
			}

			if !strings.Contains(err.Error(), tt.wantErrorPart) {
				t.Fatalf("NewConv2DConfig error = %q, want substring %q", err, tt.wantErrorPart)
			}

			if config != (layer.Conv2DConfig{}) {
				t.Fatalf("NewConv2DConfig config = %#v, want zero value on error", config)
			}
		})
	}
}

func Test_NewConv2DConfig_AcceptsMaximumPaddedDimension(t *testing.T) {
	var (
		config layer.Conv2DConfig
		err    error
	)

	maxInt := int(^uint(0) >> 1)
	config, err = layer.NewConv2DConfig(
		mustSpatialShape(t, 1, 1, 1),
		1,
		maxInt,
		1,
		1,
		1,
		(maxInt-1)/2,
		0,
	)
	if err != nil {
		t.Fatalf("NewConv2DConfig returned error at maximum padded dimension: %v", err)
	}

	if config.OutputShape() != mustSpatialShape(t, 1, 1, 1) {
		t.Fatalf("OutputShape = %#v, want 1x1x1", config.OutputShape())
	}
}
