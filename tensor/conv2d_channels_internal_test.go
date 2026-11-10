package tensor

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func Test_Conv2D_ChannelBlocks(t *testing.T) {
	var tests = []struct {
		name                           string
		batch, channels, height, width int64
		kernelH, kernelW               int64
		strides, dilations             [2]int64
		pads                           [4]int64
		bias                           bool
	}{
		{name: "tile ends at payload boundary", batch: 1, channels: 1, height: 1, width: 4, kernelH: 1, kernelW: 1, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}, bias: true},
		{name: "two tiles end at row boundary", batch: 2, channels: 3, height: 2, width: 10, kernelH: 2, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}},
		{name: "height stride with unit width stride", batch: 2, channels: 2, height: 7, width: 11, kernelH: 3, kernelW: 4, strides: [2]int64{2, 1}, dilations: [2]int64{1, 1}, pads: [4]int64{2, 0, 1, 0}, bias: true},
		{name: "tiles with entirely padded rows", batch: 1, channels: 2, height: 1, width: 7, kernelH: 2, kernelW: 1, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}, pads: [4]int64{4, 1, 3, 1}},
		{name: "padding and spatial blocks", batch: 2, channels: 3, height: 5, width: 14, kernelH: 3, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}, pads: [4]int64{2, 3, 1, 4}, bias: true},
		{name: "one column", batch: 1, channels: 3, height: 3, width: 3, kernelH: 3, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}},
		{name: "three columns", batch: 1, channels: 3, height: 3, width: 5, kernelH: 3, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}, bias: true},
		{name: "exact spatial block", batch: 1, channels: 2, height: 4, width: 6, kernelH: 2, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}},
		{name: "one remainder", batch: 1, channels: 2, height: 4, width: 7, kernelH: 2, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}},
		{name: "two remainders", batch: 1, channels: 2, height: 4, width: 8, kernelH: 2, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}, bias: true},
		{name: "three remainders", batch: 1, channels: 2, height: 4, width: 9, kernelH: 2, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}},
		{name: "strided asymmetric padding", batch: 2, channels: 2, height: 8, width: 22, kernelH: 3, kernelW: 3, strides: [2]int64{2, 2}, dilations: [2]int64{1, 1}, pads: [4]int64{1, 3, 2, 1}, bias: true},
		{name: "height dilation", batch: 1, channels: 3, height: 7, width: 14, kernelH: 3, kernelW: 2, strides: [2]int64{1, 1}, dilations: [2]int64{2, 1}, pads: [4]int64{2, 1, 1, 2}},
		{name: "width dilation", batch: 1, channels: 3, height: 4, width: 18, kernelH: 2, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 2}, pads: [4]int64{1, 3, 2, 1}, bias: true},
		{name: "dilated strided", batch: 2, channels: 2, height: 9, width: 17, kernelH: 2, kernelW: 3, strides: [2]int64{2, 3}, dilations: [2]int64{2, 2}, pads: [4]int64{3, 4, 1, 2}},
		{name: "depthwise multiplier", batch: 2, channels: 1, height: 4, width: 14, kernelH: 2, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}, pads: [4]int64{1, 1, 0, 1}, bias: true},
		{name: "fully padded windows", batch: 1, channels: 1, height: 1, width: 1, kernelH: 3, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{2, 2}, pads: [4]int64{6, 7, 4, 5}, bias: true},
		{name: "pointwise", batch: 1, channels: 3, height: 2, width: 7, kernelH: 1, kernelW: 1, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}},
		{name: "empty height", batch: 1, channels: 2, height: 0, width: 14, kernelH: 1, kernelW: 2, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}, pads: [4]int64{2, 3, 1, 4}, bias: true},
		{name: "empty width", batch: 1, channels: 2, height: 3, width: 0, kernelH: 1, kernelW: 2, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}, pads: [4]int64{2, 3, 1, 4}},
		{name: "empty batch", batch: 0, channels: 2, height: 4, width: 14, kernelH: 2, kernelW: 3, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}, bias: true},
		{name: "empty output", batch: 1, channels: 2, height: 1, width: 1, kernelH: 2, kernelW: 2, strides: [2]int64{1, 1}, dilations: [2]int64{1, 1}},
	}
	for _, tt := range tests {
		for _, filters := range []int64{4, 5, 7, 8} {
			for _, groups := range []int64{1, 2} {
				for _, numbers := range []string{"finite", "rounded", "nonfinite"} {
					t.Run(fmt.Sprintf("%s/filters%d/groups%d/%s", tt.name, filters, groups, numbers), func(t *testing.T) {
						var (
							input, weights, bias, output *Tensor
							shape                        []int64
							originals                    [3][]float32
							offset                       int
							config                       = Conv2DConfig{Strides: tt.strides, Dilations: tt.dilations, Pads: tt.pads, Groups: groups}
							err                          error
						)

						if input, err = New(Float32, []int64{tt.batch, tt.channels * groups, tt.height, tt.width}); err != nil {
							t.Fatal(err)
						}
						if weights, err = New(Float32, []int64{filters * groups, tt.channels, tt.kernelH, tt.kernelW}); err != nil {
							t.Fatal(err)
						}
						for index := range input.floats {
							input.floats[index] = []float32{16777216, 1, -16777216, -3, 0.25, 2, -0.5}[index%7]
							if numbers == "rounded" {
								input.floats[index] = []float32{-1, math.Float32frombits(0x3f800001), 1, -1}[index%4]
							}
						}
						for index := range weights.floats {
							weights.floats[index] = float32(index%5-2) * 0.5
							if numbers == "rounded" {
								weights.floats[index] = []float32{1, math.Float32frombits(0x3f7ffffe), -1}[index%3]
							}
						}
						if numbers == "nonfinite" {
							for index := range input.floats {
								input.floats[index] = []float32{0, math.Float32frombits(0x80000000), 1, -1, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}[index%7]
							}
							for index := range weights.floats {
								weights.floats[index] = []float32{float32(math.Inf(1)), 1, float32(math.NaN()), -1, float32(math.Inf(-1))}[index%5]
							}
						}
						if tt.bias {
							if bias, err = New(Float32, []int64{filters * groups}); err != nil {
								t.Fatal(err)
							}
							for index := range bias.floats {
								bias.floats[index] = float32(index-2) * 0.25
							}
						}
						if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
							t.Fatal(err)
						}
						if output, err = New(Float32, shape); err != nil {
							t.Fatal(err)
						}
						for index := range output.floats {
							output.floats[index] = 99
						}
						for index, source := range []*Tensor{input, weights, bias} {
							if source != nil {
								originals[index] = slices.Clone(source.floats)
							}
						}
						if err = Conv2DInto(input, weights, bias, output, config); err != nil {
							t.Fatal(err)
						}
						for batch := 0; batch < int(shape[0]); batch++ {
							for channel := 0; channel < int(shape[1]); channel++ {
								for row := 0; row < int(shape[2]); row++ {
									for col := 0; col < int(shape[3]); col++ {
										var (
											want = conv2DValue(input, weights, batch, channel, row, col, config)
											got  = output.floats[offset]
										)

										if bias != nil {
											want += bias.floats[channel]
										}
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
		}
	}
}

func Test_Conv2D_ChannelBlocksNumerics(t *testing.T) {
	var tests = []struct {
		name     string
		channels int64
		input    []float32
		kernelH  int64
		weights  []float32
		bias     float32
		want     []float32
	}{
		{
			name: "bias after cancellation", channels: 3,
			input: []float32{16777216, 1, -16777216}, kernelH: 1, weights: []float32{1, 1, 1}, bias: 1,
			want: []float32{1, 1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name: "skip nonfinite padded row", channels: 1,
			input: []float32{2}, kernelH: 2, weights: []float32{float32(math.Inf(1)), 1},
			want: []float32{0, 2, 2, 2, 2, 2, 2, 2, 0},
		},
		{
			name: "signed zero", channels: 1,
			input: []float32{math.Float32frombits(0x80000000)}, kernelH: 1, weights: []float32{1}, bias: math.Float32frombits(0x80000000),
			want: []float32{0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name: "infinite bias", channels: 1,
			input: []float32{2}, kernelH: 1, weights: []float32{1}, bias: float32(math.Inf(-1)),
			want: slices.Repeat([]float32{float32(math.Inf(-1))}, 9),
		},
		{
			name: "NaN bias", channels: 1,
			input: []float32{2}, kernelH: 1, weights: []float32{1}, bias: float32(math.NaN()),
			want: slices.Repeat([]float32{float32(math.NaN())}, 9),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, weights, bias, output *Tensor
				config                       = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{tt.kernelH - 1, 1, 0, 1}, Groups: 2}
				shape                        []int64
				err                          error
			)

			if input, err = New(Float32, []int64{1, tt.channels * 2, 1, 7}); err != nil {
				t.Fatal(err)
			}
			if weights, err = New(Float32, []int64{14, tt.channels, tt.kernelH, 1}); err != nil {
				t.Fatal(err)
			}
			if bias, err = New(Float32, []int64{14}); err != nil {
				t.Fatal(err)
			}
			for index := range input.floats {
				input.floats[index] = tt.input[(index/7)%len(tt.input)]
			}
			for index := range weights.floats {
				weights.floats[index] = tt.weights[index%len(tt.weights)]
			}
			for index := range bias.floats {
				bias.floats[index] = tt.bias
			}
			if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
				t.Fatal(err)
			}
			if output, err = New(Float32, shape); err != nil {
				t.Fatal(err)
			}
			if err = Conv2DInto(input, weights, bias, output, config); err != nil {
				t.Fatal(err)
			}
			for index, got := range output.floats {
				var want = tt.want[index%len(tt.want)]

				if math.Float32bits(got) != math.Float32bits(want) && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
					t.Fatalf("output[%d] = %v (%08x), want %v (%08x)", index, got, math.Float32bits(got), want, math.Float32bits(want))
				}
			}
		})
	}
}
