//go:build arm64 && !purego

package tensor

import (
	"math"
	"testing"
)

func Test_Conv2D_ResidualBits(t *testing.T) {
	var tests = []struct {
		name                  string
		height, width, kernel int64
		pad                   int64
	}{
		{name: "tail one", height: 4, width: 7, kernel: 3},
		{name: "tail two", height: 4, width: 8, kernel: 3},
		{name: "tail three", height: 4, width: 9, kernel: 3},
		{name: "MNIST edges", height: 14, width: 14, kernel: 5, pad: 2},
		{name: "smallest span", height: 1, width: 1, kernel: 1},
		{name: "fully padded", height: 1, width: 1, kernel: 3, pad: 5},
		{name: "empty width", height: 2, width: 0, kernel: 1, pad: 2},
		{name: "empty height", height: 0, width: 2, kernel: 1, pad: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, weights, bias, output *Tensor
				shape                        []int64
				config                       = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{tt.pad, tt.pad, tt.pad, tt.pad}, Groups: 2}
				patterns                     = [][]uint32{
					{0x3f800001, 0xbf7ffffe, 0x4b800000, 0x3f800000, 0xcb800000},
					{0, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x00800000},
					{0x7f7fffff, 0xff7fffff, 0x40000000, 0x3f800000},
					{0x7fc00123, 0xffc00456, 0x3f800000, 0x7f800000, 0xff800000},
					{0x7f800123, 0xff800456, 0x3f800000, 0, 0x80000000},
				}
				err error
			)
			if input, err = New(Float32, []int64{2, 6, tt.height, tt.width}); err != nil {
				t.Fatal(err)
			}
			if weights, err = New(Float32, []int64{8, 3, tt.kernel, tt.kernel}); err != nil {
				t.Fatal(err)
			}
			if bias, err = New(Float32, []int64{8}); err != nil {
				t.Fatal(err)
			}
			if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
				t.Fatal(err)
			}
			if output, err = New(Float32, shape); err != nil {
				t.Fatal(err)
			}
			for _, numbers := range patterns {
				for index := range input.floats {
					input.floats[index] = math.Float32frombits(numbers[index%len(numbers)])
				}
				for index := range weights.floats {
					weights.floats[index] = math.Float32frombits(numbers[(index+1)%len(numbers)])
				}
				for index := range bias.floats {
					bias.floats[index] = math.Float32frombits(numbers[(index+2)%len(numbers)])
				}
				for _, optionalBias := range []*Tensor{nil, bias} {
					var plane = int(shape[2] * shape[3])
					for batch := 0; batch < int(shape[0]); batch++ {
						for channel := 0; channel < int(shape[1]); channel += 4 {
							for row := 0; row < int(shape[2]); row++ {
								for col := 0; col < int(shape[3]); col++ {
									var (
										offset                       = (batch*int(shape[1])+channel)*plane + row*int(shape[3]) + col
										first, second, third, fourth = conv2DFourChannels(input, weights, batch, channel, row, col, config)
										want                         = [4]float32{first, second, third, fourth}
									)
									if !conv2DResidualChannels(input, weights, optionalBias, output, batch, channel, row, col-int(tt.pad), offset, config) {
										t.Fatal("residual unavailable")
									}

									for lane, expected := range want {
										if optionalBias != nil {
											expected += optionalBias.floats[channel+lane]
											// Reduction NaNs are already quiet. A NaN bias takes
											// priority in the released caller and is quieted.
											if math.IsNaN(float64(optionalBias.floats[channel+lane])) {
												expected = math.Float32frombits(math.Float32bits(optionalBias.floats[channel+lane]) | 0x00400000)
											}
										}
										if math.Float32bits(output.floats[offset+lane*plane]) != math.Float32bits(expected) {
											t.Fatalf("batch=%d channel=%d row=%d col=%d lane=%d got %08x want %08x", batch, channel, row, col, lane, math.Float32bits(output.floats[offset+lane*plane]), math.Float32bits(expected))
										}
									}
								}
							}
						}
					}
				}
			}
		})
	}
}

func Test_Conv2D_ResidualBiasBits(t *testing.T) {
	var (
		input, weights, bias, output *Tensor
		config                       = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		numbers                      = []uint32{0x7fc00123, 0xffc00456, 0x3f800000, 0x7f800000, 0xff800000}
		// Released public Into bits for the final column of every output row.
		want = []uint32{
			0xffc00456, 0xffc00456, 0x7fc00123, 0xffc00456,
			0x7fc00123, 0xffc00456, 0x7fc00123, 0x7fc00123,
			0xffc00456, 0x7fc00123, 0x7fc00123, 0x7fc00123,
			0x7fc00123, 0x7fc00123, 0x7fc00123, 0x7fc00123,
		}
		err error
	)

	if input, err = New(Float32, []int64{2, 3, 4, 7}); err != nil {
		t.Fatal(err)
	}
	if weights, err = New(Float32, []int64{4, 3, 3, 3}); err != nil {
		t.Fatal(err)
	}
	if bias, err = New(Float32, []int64{4}); err != nil {
		t.Fatal(err)
	}
	if output, err = New(Float32, []int64{2, 4, 2, 5}); err != nil {
		t.Fatal(err)
	}
	for index := range input.floats {
		input.floats[index] = math.Float32frombits(numbers[index%len(numbers)])
	}
	for index := range weights.floats {
		weights.floats[index] = math.Float32frombits(numbers[(index+1)%len(numbers)])
	}
	for index := range bias.floats {
		bias.floats[index] = math.Float32frombits(numbers[(index+2)%len(numbers)])
	}
	if err = Conv2DInto(input, weights, bias, output, config); err != nil {
		t.Fatal(err)
	}
	for row, expected := range want {
		var got = math.Float32bits(output.floats[row*5+4])

		if got != expected {
			t.Fatalf("row %d final column bits=%08x, want %08x", row, got, expected)
		}
	}
}
