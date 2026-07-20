package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
)

func Test_NewSimpleRNNConfig(t *testing.T) {
	type testcase struct {
		name            string
		inputShape      layer.SequenceShape
		hiddenSize      int
		wantOutputShape layer.SequenceShape
	}

	tests := []testcase{
		{
			name:            "one step",
			inputShape:      mustSequenceShape(t, 1, 2),
			hiddenSize:      3,
			wantOutputShape: mustSequenceShape(t, 1, 3),
		},
		{
			name:            "multiple steps",
			inputShape:      mustSequenceShape(t, 4, 3),
			hiddenSize:      2,
			wantOutputShape: mustSequenceShape(t, 4, 2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				config layer.SimpleRNNConfig
				err    error
			)

			config, err = layer.NewSimpleRNNConfig(tt.inputShape, tt.hiddenSize)
			if err != nil {
				t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
			}

			if config.InputShape() != tt.inputShape {
				t.Fatalf("InputShape = %#v, want %#v", config.InputShape(), tt.inputShape)
			}

			if config.OutputShape() != tt.wantOutputShape {
				t.Fatalf("OutputShape = %#v, want %#v", config.OutputShape(), tt.wantOutputShape)
			}

			if config.HiddenSize() != tt.hiddenSize {
				t.Fatalf("HiddenSize = %d, want %d", config.HiddenSize(), tt.hiddenSize)
			}

			if config.OutputShape().Steps() != config.InputShape().Steps() {
				t.Fatalf(
					"output steps = %d, want input steps %d",
					config.OutputShape().Steps(),
					config.InputShape().Steps(),
				)
			}

			if config.OutputShape().FeatureSize() != config.HiddenSize() {
				t.Fatalf(
					"output feature size = %d, want hidden size %d",
					config.OutputShape().FeatureSize(),
					config.HiddenSize(),
				)
			}
		})
	}
}

func Test_NewSimpleRNNConfig_ValidatesDimensions(t *testing.T) {
	type testcase struct {
		name          string
		inputShape    layer.SequenceShape
		hiddenSize    int
		wantErrorPart string
	}

	maxInt := int(^uint(0) >> 1)
	tests := []testcase{
		{name: "zero input shape", inputShape: layer.SequenceShape{}, hiddenSize: 1, wantErrorPart: "input shape invalid"},
		{name: "zero hidden size", inputShape: mustSequenceShape(t, 1, 1), hiddenSize: 0, wantErrorPart: "hidden size must be positive: got=0 want>0"},
		{name: "negative hidden size", inputShape: mustSequenceShape(t, 1, 1), hiddenSize: -1, wantErrorPart: "hidden size must be positive: got=-1 want>0"},
		{name: "output shape overflow", inputShape: mustSequenceShape(t, maxInt, 1), hiddenSize: 2, wantErrorPart: "output shape invalid"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				config layer.SimpleRNNConfig
				err    error
			)

			config, err = layer.NewSimpleRNNConfig(tt.inputShape, tt.hiddenSize)
			if err == nil {
				t.Fatal("NewSimpleRNNConfig error = nil, want error")
			}

			if !strings.HasPrefix(err.Error(), "layer: ") {
				t.Fatalf("NewSimpleRNNConfig error = %q, want layer context", err)
			}

			if !strings.Contains(err.Error(), tt.wantErrorPart) {
				t.Fatalf("NewSimpleRNNConfig error = %q, want substring %q", err, tt.wantErrorPart)
			}

			if config != (layer.SimpleRNNConfig{}) {
				t.Fatalf("NewSimpleRNNConfig config = %#v, want zero value on error", config)
			}
		})
	}
}

func Test_NewSimpleRNNConfig_AcceptsMaximumOutputSize(t *testing.T) {
	var (
		config layer.SimpleRNNConfig
		err    error
	)

	maxInt := int(^uint(0) >> 1)
	config, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, maxInt, 1), 1)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error at maximum output size: %v", err)
	}

	if config.OutputShape().Size() != maxInt {
		t.Fatalf("OutputShape Size = %d, want %d", config.OutputShape().Size(), maxInt)
	}
}

func Test_SimpleRNNConfig_ZeroValue(t *testing.T) {
	var config layer.SimpleRNNConfig

	if config.InputShape() != (layer.SequenceShape{}) {
		t.Fatalf("InputShape = %#v, want zero sequence shape", config.InputShape())
	}

	if config.OutputShape() != (layer.SequenceShape{}) {
		t.Fatalf("OutputShape = %#v, want zero sequence shape", config.OutputShape())
	}

	if config.HiddenSize() != 0 {
		t.Fatalf("HiddenSize = %d, want 0", config.HiddenSize())
	}
}
