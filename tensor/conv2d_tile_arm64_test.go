package tensor

import (
	"math"
	"testing"
)

func Test_Conv2D_TileOutputBits(t *testing.T) {
	var tests = []struct {
		name    string
		input   []uint32
		weights []uint32
		bias    uint32
	}{
		{name: "subnormals", input: []uint32{0x00000001, 0x80000001, 0x007fffff, 0x00800000}, weights: []uint32{0x3f000000, 0xbf000000, 0x3f800001}},
		{name: "overflow and cancellation", input: []uint32{0x7f7fffff, 0xff7fffff, 0x3f800001, 0xbf800000}, weights: []uint32{0x40000000, 0x3f800001, 0x3f7ffffe}, bias: 0x3f800000},
		{name: "signed zeros", input: []uint32{0x80000000, 0, 0x80000000}, weights: []uint32{0x3f800000, 0xbf800000}, bias: 0x80000000},
		{name: "quiet NaN payloads", input: []uint32{0x7fc00123, 0xffc00456, 0x3f800000}, weights: []uint32{0xffc00789, 0x7fc00abc, 0x3f800000}},
		{name: "signaling NaN payloads", input: []uint32{0x7f800123, 0xff800456, 0x3f800000}, weights: []uint32{0xff800789, 0x7f800abc, 0x3f800000}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				input, weights, bias, output *Tensor
				config                       = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 2}
				offset                       int
				err                          error
			)

			if input, err = New(Float32, []int64{1, 6, 1, 4}); err != nil {
				t.Fatal(err)
			}
			if weights, err = New(Float32, []int64{10, 3, 1, 1}); err != nil {
				t.Fatal(err)
			}
			if bias, err = New(Float32, []int64{10}); err != nil {
				t.Fatal(err)
			}
			if output, err = New(Float32, []int64{1, 10, 1, 4}); err != nil {
				t.Fatal(err)
			}
			for index := range input.floats {
				input.floats[index] = math.Float32frombits(tt.input[index%len(tt.input)])
			}
			for index := range weights.floats {
				weights.floats[index] = math.Float32frombits(tt.weights[index%len(tt.weights)])
			}
			for index := range bias.floats {
				bias.floats[index] = math.Float32frombits(tt.bias)
			}
			if err = Conv2DInto(input, weights, bias, output, config); err != nil {
				t.Fatal(err)
			}
			for channel := 0; channel < 10; channel++ {
				for column := 0; column < 4; column++ {
					var want = conv2DValue(input, weights, 0, channel, 0, column, config) + bias.floats[channel]

					if math.Float32bits(output.floats[offset]) != math.Float32bits(want) {
						t.Fatalf("output[%d] bits = %08x, want %08x", offset, math.Float32bits(output.floats[offset]), math.Float32bits(want))
					}
					offset++
				}
			}
		})
	}
}
