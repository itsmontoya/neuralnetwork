//go:build arm64 && !purego

package tensor

import (
	"fmt"
	"math"
	"testing"
)

func Test_Conv2D_EightFilterBits(t *testing.T) {
	var tests = []struct {
		name                     string
		channels, height, kernel int64
		pads                     [4]int64
		stride                   int64
	}{
		{name: "interior", channels: 3, height: 4, kernel: 3, stride: 1},
		{name: "padded", channels: 2, height: 3, kernel: 5, pads: [4]int64{2, 2, 2, 2}, stride: 1},
		{name: "asymmetric strided rows", channels: 1, height: 2, kernel: 3, pads: [4]int64{4, 3, 2, 5}, stride: 2},
	}
	var patterns = [][]uint32{
		{0x3f800001, 0xbf7ffffe, 0x4b800000, 0x3f800000, 0xcb800000, 0x3eaaaaab},
		{0, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x00800000},
		{0x7f7fffff, 0xff7fffff, 0x40000000, 0x3f800000, 0xbf800000},
		{0x7fc00123, 0xffc00456, 0x3f800000, 0x7f800000, 0xff800000},
		{0x7f800123, 0xff800456, 0x3f800000, 0, 0x80000000},
	}
	for _, tt := range tests {
		for _, filters := range []int64{8, 9, 10, 11, 12, 15, 16, 17} {
			for _, groups := range []int64{1, 2} {
				for _, width := range []int64{1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 27, 28, 29, 32, 33} {
					t.Run(fmt.Sprintf("%s/filters%d/groups%d/width%d", tt.name, filters, groups, width), func(t *testing.T) {
						var (
							input, weights, bias, output, reference *Tensor
							config                                  = Conv2DConfig{Strides: [2]int64{tt.stride, 1}, Dilations: [2]int64{1, 1}, Pads: tt.pads, Groups: groups}
							shape                                   []int64
							err                                     error
						)
						if input, err = New(Float32, []int64{2, tt.channels * groups, tt.height, width}); err != nil {
							t.Fatal(err)
						}
						if weights, err = New(Float32, []int64{filters * groups, tt.channels, tt.kernel, tt.kernel}); err != nil {
							t.Fatal(err)
						}
						if bias, err = New(Float32, []int64{filters * groups}); err != nil {
							t.Fatal(err)
						}
						if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
							t.Fatal(err)
						}
						if output, err = New(Float32, shape); err != nil {
							t.Fatal(err)
						}
						if reference, err = New(Float32, shape); err != nil {
							t.Fatal(err)
						}
						for pattern, numbers := range patterns {
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
								conv2DReleasedChannelBlocks(input, weights, optionalBias, reference, config)
								if err = Conv2DInto(input, weights, optionalBias, output, config); err != nil {
									t.Fatal(err)
								}
								for index, expected := range reference.floats {
									if math.Float32bits(output.floats[index]) != math.Float32bits(expected) {
										t.Fatalf("pattern=%d bias=%v output[%d]=%08x, released=%08x", pattern, optionalBias != nil, index, math.Float32bits(output.floats[index]), math.Float32bits(expected))
									}
								}
							}
							for operand, values := range [][]float32{input.floats, weights.floats, bias.floats} {
								for index, value := range values {
									if math.Float32bits(value) != numbers[(index+operand)%len(numbers)] {
										t.Fatalf("operand %d changed at %d", operand, index)
									}
								}
							}
						}
					})
				}
			}
		}
	}
}
