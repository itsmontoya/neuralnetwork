package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
)

func Test_NewSequenceShape(t *testing.T) {
	type testcase struct {
		name        string
		steps       int
		featureSize int
		wantSize    int
	}

	tests := []testcase{
		{name: "one step and one feature", steps: 1, featureSize: 1, wantSize: 1},
		{name: "one step and multiple features", steps: 1, featureSize: 4, wantSize: 4},
		{name: "multiple steps and one feature", steps: 3, featureSize: 1, wantSize: 3},
		{name: "multiple steps and features", steps: 3, featureSize: 4, wantSize: 12},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				shape layer.SequenceShape
				err   error
			)

			shape, err = layer.NewSequenceShape(tt.steps, tt.featureSize)
			if err != nil {
				t.Fatalf("NewSequenceShape returned error: %v", err)
			}

			if shape.Steps() != tt.steps {
				t.Fatalf("Steps = %d, want %d", shape.Steps(), tt.steps)
			}

			if shape.FeatureSize() != tt.featureSize {
				t.Fatalf("FeatureSize = %d, want %d", shape.FeatureSize(), tt.featureSize)
			}

			if shape.Size() != tt.wantSize {
				t.Fatalf("Size = %d, want %d", shape.Size(), tt.wantSize)
			}
		})
	}
}

func Test_NewSequenceShape_ValidatesDimensions(t *testing.T) {
	type testcase struct {
		name          string
		steps         int
		featureSize   int
		wantErrorPart string
	}

	maxInt := int(^uint(0) >> 1)
	tests := []testcase{
		{name: "zero steps", steps: 0, featureSize: 1, wantErrorPart: "steps must be positive: got=0 want>0"},
		{name: "negative steps", steps: -1, featureSize: 1, wantErrorPart: "steps must be positive: got=-1 want>0"},
		{name: "zero feature size", steps: 1, featureSize: 0, wantErrorPart: "feature size must be positive: got=0 want>0"},
		{name: "negative feature size", steps: 1, featureSize: -1, wantErrorPart: "feature size must be positive: got=-1 want>0"},
		{name: "minimum overflow", steps: maxInt/2 + 1, featureSize: 2, wantErrorPart: "size overflows int"},
		{name: "maximum factor overflow", steps: maxInt, featureSize: 2, wantErrorPart: "size overflows int"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				shape layer.SequenceShape
				err   error
			)

			shape, err = layer.NewSequenceShape(tt.steps, tt.featureSize)
			if err == nil {
				t.Fatal("NewSequenceShape error = nil, want error")
			}

			if !strings.HasPrefix(err.Error(), "layer: ") {
				t.Fatalf("NewSequenceShape error = %q, want layer context", err)
			}

			if !strings.Contains(err.Error(), tt.wantErrorPart) {
				t.Fatalf("NewSequenceShape error = %q, want substring %q", err, tt.wantErrorPart)
			}

			if shape != (layer.SequenceShape{}) {
				t.Fatalf("NewSequenceShape shape = %#v, want zero value on error", shape)
			}
		})
	}
}

func Test_NewSequenceShape_AcceptsOverflowBoundaries(t *testing.T) {
	type testcase struct {
		name        string
		steps       int
		featureSize int
		wantSize    int
	}

	maxInt := int(^uint(0) >> 1)
	tests := []testcase{
		{name: "maximum size", steps: maxInt, featureSize: 1, wantSize: maxInt},
		{name: "largest even product", steps: maxInt / 2, featureSize: 2, wantSize: maxInt - 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				shape layer.SequenceShape
				err   error
			)

			shape, err = layer.NewSequenceShape(tt.steps, tt.featureSize)
			if err != nil {
				t.Fatalf("NewSequenceShape returned error at overflow boundary: %v", err)
			}

			if shape.Size() != tt.wantSize {
				t.Fatalf("Size = %d, want %d", shape.Size(), tt.wantSize)
			}
		})
	}
}

func Test_SequenceShape_Equality(t *testing.T) {
	var (
		first      layer.SequenceShape
		equivalent layer.SequenceShape
		different  layer.SequenceShape
	)

	first = mustSequenceShape(t, 2, 3)
	equivalent = mustSequenceShape(t, 2, 3)
	different = mustSequenceShape(t, 3, 2)

	if first != equivalent {
		t.Fatalf("equivalent shapes compare unequal: %#v != %#v", first, equivalent)
	}

	if first == different {
		t.Fatalf("different shapes compare equal: %#v == %#v", first, different)
	}
}

func Test_SequenceShape_TimeMajorIndexingContract(t *testing.T) {
	type testcase struct {
		name       string
		step       int
		feature    int
		wantColumn int
	}

	shape := mustSequenceShape(t, 3, 4)
	tests := []testcase{
		{name: "first value", step: 0, feature: 0, wantColumn: 0},
		{name: "last first-step value", step: 0, feature: 3, wantColumn: 3},
		{name: "first second-step value", step: 1, feature: 0, wantColumn: 4},
		{name: "interior value", step: 1, feature: 2, wantColumn: 6},
		{name: "last value", step: 2, feature: 3, wantColumn: 11},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var column int

			column = tt.step*shape.FeatureSize() + tt.feature
			if column != tt.wantColumn {
				t.Fatalf("time-major column = %d, want %d", column, tt.wantColumn)
			}

			if column >= shape.Size() {
				t.Fatalf("time-major column = %d, want less than shape size %d", column, shape.Size())
			}
		})
	}
}

func Test_SequenceShape_ZeroValueIsInvalidConfigurationInput(t *testing.T) {
	var (
		config layer.SimpleRNNConfig
		err    error
		shape  layer.SequenceShape
	)

	config, err = layer.NewSimpleRNNConfig(shape, 1)
	if err == nil {
		t.Fatal("NewSimpleRNNConfig error = nil, want error for zero-value input shape")
	}

	if !strings.Contains(err.Error(), "input shape invalid") {
		t.Fatalf("NewSimpleRNNConfig error = %q, want input shape context", err)
	}

	if config != (layer.SimpleRNNConfig{}) {
		t.Fatalf("NewSimpleRNNConfig config = %#v, want zero value on error", config)
	}
}

func mustSequenceShape(tb testing.TB, steps, featureSize int) (shape layer.SequenceShape) {
	var err error

	tb.Helper()

	shape, err = layer.NewSequenceShape(steps, featureSize)
	if err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}

	return shape
}
