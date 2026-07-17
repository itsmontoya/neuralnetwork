package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
)

func Test_NewSpatialShape(t *testing.T) {
	type testcase struct {
		name     string
		channels int
		height   int
		width    int
		wantSize int
	}

	tests := []testcase{
		{name: "unit", channels: 1, height: 1, width: 1, wantSize: 1},
		{name: "rectangular", channels: 3, height: 2, width: 4, wantSize: 24},
		{name: "single channel", channels: 1, height: 5, width: 7, wantSize: 35},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				shape layer.SpatialShape
				err   error
			)

			shape, err = layer.NewSpatialShape(tt.channels, tt.height, tt.width)
			if err != nil {
				t.Fatalf("NewSpatialShape returned error: %v", err)
			}

			if shape.Channels() != tt.channels {
				t.Fatalf("Channels = %d, want %d", shape.Channels(), tt.channels)
			}

			if shape.Height() != tt.height {
				t.Fatalf("Height = %d, want %d", shape.Height(), tt.height)
			}

			if shape.Width() != tt.width {
				t.Fatalf("Width = %d, want %d", shape.Width(), tt.width)
			}

			if shape.Size() != tt.wantSize {
				t.Fatalf("Size = %d, want %d", shape.Size(), tt.wantSize)
			}
		})
	}
}

func Test_NewSpatialShape_ValidatesDimensions(t *testing.T) {
	type testcase struct {
		name     string
		channels int
		height   int
		width    int
	}

	maxInt := int(^uint(0) >> 1)
	tests := []testcase{
		{name: "zero channels", channels: 0, height: 1, width: 1},
		{name: "negative channels", channels: -1, height: 1, width: 1},
		{name: "zero height", channels: 1, height: 0, width: 1},
		{name: "negative height", channels: 1, height: -1, width: 1},
		{name: "zero width", channels: 1, height: 1, width: 0},
		{name: "negative width", channels: 1, height: 1, width: -1},
		{name: "first product overflow", channels: maxInt, height: 2, width: 1},
		{name: "second product overflow", channels: maxInt/2 + 1, height: 1, width: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				shape layer.SpatialShape
				err   error
			)

			shape, err = layer.NewSpatialShape(tt.channels, tt.height, tt.width)
			if err == nil {
				t.Fatal("NewSpatialShape error = nil, want error")
			}

			if !strings.HasPrefix(err.Error(), "layer: ") {
				t.Fatalf("NewSpatialShape error = %q, want layer context", err)
			}

			if shape != (layer.SpatialShape{}) {
				t.Fatalf("NewSpatialShape shape = %#v, want zero value on error", shape)
			}
		})
	}
}

func Test_SpatialShape_Equality(t *testing.T) {
	var (
		first      layer.SpatialShape
		equivalent layer.SpatialShape
		different  layer.SpatialShape
	)

	first = mustSpatialShape(t, 2, 3, 4)
	equivalent = mustSpatialShape(t, 2, 3, 4)
	different = mustSpatialShape(t, 2, 4, 3)

	if first != equivalent {
		t.Fatalf("equivalent shapes compare unequal: %#v != %#v", first, equivalent)
	}

	if first == different {
		t.Fatalf("different shapes compare equal: %#v == %#v", first, different)
	}
}

func Test_SpatialShape_CHWIndexingContract(t *testing.T) {
	type testcase struct {
		name       string
		channel    int
		height     int
		width      int
		wantColumn int
	}

	shape := mustSpatialShape(t, 2, 3, 4)
	tests := []testcase{
		{name: "first value", channel: 0, height: 0, width: 0, wantColumn: 0},
		{name: "last first-channel value", channel: 0, height: 2, width: 3, wantColumn: 11},
		{name: "first second-channel value", channel: 1, height: 0, width: 0, wantColumn: 12},
		{name: "interior second-channel value", channel: 1, height: 1, width: 2, wantColumn: 18},
		{name: "last value", channel: 1, height: 2, width: 3, wantColumn: 23},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var column int

			column = tt.channel*shape.Height()*shape.Width() + tt.height*shape.Width() + tt.width
			if column != tt.wantColumn {
				t.Fatalf("CHW column = %d, want %d", column, tt.wantColumn)
			}

			if column >= shape.Size() {
				t.Fatalf("CHW column = %d, want less than shape size %d", column, shape.Size())
			}
		})
	}
}

func mustSpatialShape(tb testing.TB, channels, height, width int) (shape layer.SpatialShape) {
	var err error

	tb.Helper()

	shape, err = layer.NewSpatialShape(channels, height, width)
	if err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}

	return shape
}
