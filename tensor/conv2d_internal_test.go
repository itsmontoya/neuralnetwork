package tensor

import (
	"math"
	"slices"
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

func Test_Conv2D_FourColumns(t *testing.T) {
	var (
		unit     = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		strided  = unit
		dilated  = unit
		fallback = unit
		padded   = unit
		grouped  = unit
	)

	strided.Strides = [2]int64{2, 2}
	dilated.Dilations[0] = 2
	fallback.Dilations[1] = 2
	padded.Pads = [4]int64{2, 3, 1, 4}
	grouped.Groups = 2
	var tests = []struct {
		name      string
		input     []int64
		weights   []int64
		config    Conv2DConfig
		bias      bool
		nonfinite bool
		rounded   bool
	}{
		{name: "exact block", input: []int64{1, 1, 3, 6}, weights: []int64{1, 1, 3, 3}, config: unit},
		{name: "multiple blocks and tail", input: []int64{2, 3, 5, 13}, weights: []int64{4, 3, 2, 3}, config: unit, bias: true},
		{name: "one column tail", input: []int64{1, 2, 3, 11}, weights: []int64{2, 2, 2, 3}, config: unit},
		{name: "two column tail", input: []int64{1, 2, 3, 12}, weights: []int64{2, 2, 2, 3}, config: unit},
		{name: "short row fallback", input: []int64{1, 1, 3, 5}, weights: []int64{1, 1, 2, 3}, config: unit},
		{name: "strided blocks", input: []int64{2, 2, 7, 22}, weights: []int64{3, 2, 2, 3}, config: strided, bias: true},
		{name: "height dilation", input: []int64{1, 2, 8, 14}, weights: []int64{3, 2, 3, 2}, config: dilated},
		{name: "width dilation fallback", input: []int64{1, 2, 4, 18}, weights: []int64{2, 2, 2, 3}, config: fallback},
		{name: "padding and blocks", input: []int64{1, 2, 4, 14}, weights: []int64{2, 2, 3, 3}, config: padded, bias: true},
		{name: "groups and batches", input: []int64{2, 4, 4, 14}, weights: []int64{6, 2, 2, 3}, config: grouped, bias: true},
		{name: "depthwise groups", input: []int64{2, 2, 4, 14}, weights: []int64{4, 1, 2, 3}, config: grouped, bias: true},
		{name: "pointwise", input: []int64{2, 3, 4, 12}, weights: []int64{2, 3, 1, 1}, config: unit},
		{name: "rounded products", input: []int64{1, 2, 1, 4}, weights: []int64{1, 2, 1, 1}, config: unit, rounded: true},
		{name: "nonfinite interior", input: []int64{1, 2, 3, 14}, weights: []int64{2, 2, 2, 3}, config: unit, nonfinite: true},
		{name: "nonfinite padding", input: []int64{1, 2, 3, 14}, weights: []int64{2, 2, 2, 3}, config: padded, nonfinite: true},
		{name: "empty height", input: []int64{1, 1, 0, 14}, weights: []int64{1, 1, 1, 2}, config: padded},
		{name: "empty width", input: []int64{1, 1, 3, 0}, weights: []int64{1, 1, 1, 2}, config: padded},
		{name: "empty batch", input: []int64{0, 2, 4, 14}, weights: []int64{2, 2, 2, 3}, config: unit, bias: true},
		{name: "empty output height", input: []int64{1, 1, 1, 14}, weights: []int64{1, 1, 2, 2}, config: unit},
		{name: "empty output width", input: []int64{1, 1, 3, 1}, weights: []int64{1, 1, 2, 2}, config: unit},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, weights, bias, output *Tensor
				shape                        []int64
				originals                    [3][]float32
				offset                       int
				err                          error
			)

			if input, err = New(Float32, tt.input); err != nil {
				t.Fatal(err)
			}

			if weights, err = New(Float32, tt.weights); err != nil {
				t.Fatal(err)
			}

			for index := range input.floats {
				input.floats[index] = []float32{16777216, 1, -16777216, -3, 0.25, 2, -0.5}[index%7]
			}
			for index := range weights.floats {
				weights.floats[index] = float32(index%5-2) * 0.5
			}
			if tt.rounded {
				var aboveOne = math.Float32frombits(0x3f800001)

				// Go permits fused products; compare with the original scalar kernel.
				copy(input.floats, []float32{-1, -1, -1, -1, aboveOne, aboveOne, aboveOne, aboveOne})
				copy(weights.floats, []float32{1, math.Float32frombits(0x3f7ffffe)})
			}
			if tt.nonfinite {
				input.floats[0] = float32(math.Inf(1))
				input.floats[4] = math.Float32frombits(0x80000000)
				weights.floats[0] = float32(math.Inf(-1))
				weights.floats[3] = float32(math.NaN())
			}

			if tt.bias {
				if bias, err = New(Float32, []int64{tt.weights[0]}); err != nil {
					t.Fatal(err)
				}

				for index := range bias.floats {
					bias.floats[index] = float32(index-2) * 0.25
				}
			}

			if shape, err = Conv2DShape(tt.input, tt.weights, tt.config); err != nil {
				t.Fatal(err)
			}

			if output, err = New(Float32, shape); err != nil {
				t.Fatal(err)
			}

			for index, source := range []*Tensor{input, weights, bias} {
				if source != nil {
					originals[index] = slices.Clone(source.floats)
				}
			}

			if err = Conv2DInto(input, weights, bias, output, tt.config); err != nil {
				t.Fatal(err)
			}

			for batch := 0; batch < int(shape[0]); batch++ {
				for channel := 0; channel < int(shape[1]); channel++ {
					for row := 0; row < int(shape[2]); row++ {
						for col := 0; col < int(shape[3]); col++ {
							want := conv2DValue(input, weights, batch, channel, row, col, tt.config)
							if bias != nil {
								want += bias.floats[channel]
							}

							got := output.floats[offset]
							if math.Float32bits(got) != math.Float32bits(want) && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
								t.Fatalf("output[%d] = %v (%08x), want %v (%08x)", offset, got, math.Float32bits(got), want, math.Float32bits(want))
							}

							offset++
						}
					}
				}
			}
			for sourceIndex, source := range []*Tensor{input, weights, bias} {
				if source == nil {
					continue
				}
				for index, original := range originals[sourceIndex] {
					if math.Float32bits(source.floats[index]) != math.Float32bits(original) {
						t.Fatalf("source %d changed at index %d", sourceIndex, index)
					}
				}
			}
		})
	}
}

func Test_Conv2D_FourColumnsNumerics(t *testing.T) {
	var (
		unit = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		top  = unit
	)

	top.Pads[0] = 1
	var tests = []struct {
		name         string
		inputShape   []int64
		input        []float32
		weightsShape []int64
		weights      []float32
		bias         float32
		config       Conv2DConfig
		want         []float32
	}{
		{
			name: "bias after cancellation", inputShape: []int64{1, 3, 1, 4},
			input:        []float32{16777216, 16777216, 16777216, 16777216, 1, 1, 1, 1, -16777216, -16777216, -16777216, -16777216},
			weightsShape: []int64{1, 3, 1, 1}, weights: []float32{1, 1, 1}, bias: 1, config: unit,
			want: []float32{1, 1, 1, 1},
		},
		{
			name: "skip padded nonfinite weights", inputShape: []int64{1, 1, 1, 4}, input: []float32{1, 2, 3, 4},
			weightsShape: []int64{1, 1, 2, 1}, weights: []float32{float32(math.Inf(1)), 1}, config: top,
			want: []float32{1, 2, 3, 4},
		},
		{
			name: "signed zero products", inputShape: []int64{1, 1, 1, 4}, input: []float32{0, 0, 0, 0},
			weightsShape: []int64{1, 1, 1, 1}, weights: []float32{-1}, bias: math.Float32frombits(0x80000000), config: unit,
			want: []float32{0, 0, 0, 0},
		},
		{
			name: "nonfinite bias", inputShape: []int64{1, 1, 1, 4}, input: []float32{1, -1, 0, float32(math.Inf(-1))},
			weightsShape: []int64{1, 1, 1, 1}, weights: []float32{1}, bias: float32(math.Inf(1)), config: unit,
			want: []float32{float32(math.Inf(1)), float32(math.Inf(1)), float32(math.Inf(1)), float32(math.NaN())},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, weights, bias, output *Tensor
				shape                        []int64
				err                          error
			)

			if input, err = FromFloat32(tt.inputShape, tt.input); err != nil {
				t.Fatal(err)
			}
			if weights, err = FromFloat32(tt.weightsShape, tt.weights); err != nil {
				t.Fatal(err)
			}
			if bias, err = FromFloat32([]int64{1}, []float32{tt.bias}); err != nil {
				t.Fatal(err)
			}
			if shape, err = Conv2DShape(tt.inputShape, tt.weightsShape, tt.config); err != nil {
				t.Fatal(err)
			}
			if output, err = New(Float32, shape); err != nil {
				t.Fatal(err)
			}
			if err = Conv2DInto(input, weights, bias, output, tt.config); err != nil {
				t.Fatal(err)
			}
			if len(output.floats) != len(tt.want) {
				t.Fatalf("output length = %d, want %d", len(output.floats), len(tt.want))
			}
			for index, want := range tt.want {
				var got = output.floats[index]

				if math.Float32bits(got) != math.Float32bits(want) && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
					t.Fatalf("output[%d] = %v (%08x), want %v (%08x)", index, got, math.Float32bits(got), want, math.Float32bits(want))
				}
			}
		})
	}
}
