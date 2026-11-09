package tensor

import (
	"math"
	"testing"
)

func Test_Conv2D_ClippedColumns(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		weights  []float32
		stride   int64
		dilation int64
		left     int64
		right    int64
		want     []float32
	}{
		{
			name:  "dilated strided asymmetric padding",
			input: []float32{1, 2, 3, 4}, weights: []float32{1, 10, 100},
			stride: 2, dilation: 2, left: 3, right: 4,
			want: []float32{200, 420, 42, 4},
		},
		{
			name:  "windows entirely left of input",
			input: []float32{1, 2, 3, 4}, weights: []float32{1, 10, 100},
			stride: 1, dilation: 2, left: 7, right: 1,
			want: []float32{0, 0, 0, 100, 200, 310, 420, 31},
		},
		{
			name:  "dilation skips narrow input",
			input: []float32{2}, weights: []float32{3, 4},
			stride: 1, dilation: 3, left: 1, right: 2,
			want: []float32{0},
		},
		{
			name:  "padding skips nonfinite products",
			input: []float32{2}, weights: []float32{float32(math.Inf(1))},
			stride: 1, dilation: 1, left: 2, right: 2,
			want: []float32{0, 0, float32(math.Inf(1)), 0, 0},
		},
		{
			name:  "float32 accumulation order",
			input: []float32{16777216, 1, -16777216}, weights: []float32{1, 1, 1},
			stride: 1, dilation: 1,
			want: []float32{0},
		},
		{
			name:    "empty width with padded output",
			weights: []float32{float32(math.NaN())},
			stride:  1, dilation: 1, left: 1, right: 1,
			want: []float32{0, 0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, weights, output *Tensor
				shape                  []int64
				values                 []float32
				config                 = Conv2DConfig{
					Strides: [2]int64{1, tt.stride}, Dilations: [2]int64{1, tt.dilation},
					Pads: [4]int64{0, tt.left, 0, tt.right}, Groups: 1,
				}
				err error
			)

			if input, err = FromFloat32([]int64{1, 1, 1, int64(len(tt.input))}, tt.input); err != nil {
				t.Fatal(err)
			}

			if weights, err = FromFloat32([]int64{1, 1, 1, int64(len(tt.weights))}, tt.weights); err != nil {
				t.Fatal(err)
			}

			if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
				t.Fatal(err)
			}

			if output, err = New(Float32, shape); err != nil {
				t.Fatal(err)
			}

			if err = Conv2DInto(input, weights, nil, output, config); err != nil {
				t.Fatal(err)
			}

			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}

			if len(values) != len(tt.want) {
				t.Fatalf("output length = %d, want %d", len(values), len(tt.want))
			}

			for index, want := range tt.want {
				if math.Float32bits(values[index]) != math.Float32bits(want) {
					t.Fatalf("output[%d] = %v, want %v", index, values[index], want)
				}
			}
		})
	}
}
