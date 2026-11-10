//go:build arm64 && !purego

package tensor

import (
	"fmt"
	"math"
	"testing"
)

func Test_Conv2D_ResidualTileBits(t *testing.T) {
	var tests = []struct {
		name                             string
		batch, channels, filters, groups int64
		height, kernelH, kernelW         int64
		pads                             [4]int64
		stride                           int64
	}{
		{name: "unpadded", batch: 1, channels: 3, filters: 8, groups: 1, height: 7, kernelH: 3, kernelW: 3, stride: 1},
		{name: "padded", batch: 1, channels: 8, filters: 16, groups: 1, height: 14, kernelH: 5, kernelW: 5, pads: [4]int64{2, 2, 2, 2}, stride: 1},
		{name: "groups and remainder", batch: 2, channels: 6, filters: 14, groups: 2, height: 4, kernelH: 2, kernelW: 3, pads: [4]int64{3, 1, 0, 4}, stride: 2},
		{name: "depthwise multiplier", batch: 1, channels: 3, filters: 12, groups: 3, height: 3, kernelH: 3, kernelW: 5, pads: [4]int64{1, 2, 4, 3}, stride: 1},
		{name: "single row", batch: 1, channels: 1, filters: 5, groups: 1, height: 1, kernelH: 1, kernelW: 1, pads: [4]int64{1, 3, 1, 3}, stride: 1},
		{name: "large padding", batch: 1, channels: 2, filters: 4, groups: 1, height: 1, kernelH: 2, kernelW: 2, pads: [4]int64{5, 5, 4, 6}, stride: 1},
	}
	var patterns = [][]uint32{
		{0x3f800001, 0xbf7ffffe, 0x4b800000, 0x3f800000, 0xcb800000, 0x3eaaaaab},
		{0, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x00800000},
		{0x7f7fffff, 0xff7fffff, 0x40000000, 0x3f800000, 0xbf800000},
		{0x7fc00123, 0xffc00456, 0x3f800000, 0x7f800000, 0xff800000},
		{0x7f800123, 0xff800456, 0x3f800000, 0, 0x80000000},
	}

	for _, tt := range tests {
		for _, width := range []int64{1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 27, 28, 29} {
			t.Run(fmt.Sprintf("%s/width%d", tt.name, width), func(t *testing.T) {
				var (
					input, weights, bias, output, reference *Tensor
					shape                                   []int64
					config                                  = Conv2DConfig{Strides: [2]int64{tt.stride, 1}, Dilations: [2]int64{1, 1}, Pads: tt.pads, Groups: tt.groups}
					err                                     error
				)

				if input, err = New(Float32, []int64{tt.batch, tt.channels, tt.height, width}); err != nil {
					t.Fatal(err)
				}
				if weights, err = New(Float32, []int64{tt.filters, tt.channels / tt.groups, tt.kernelH, tt.kernelW}); err != nil {
					t.Fatal(err)
				}
				if bias, err = New(Float32, []int64{tt.filters}); err != nil {
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
						for index := range output.floats {
							output.floats[index] = math.Float32frombits(0x7fc0ffff)
						}
						if err = Conv2DInto(input, weights, optionalBias, output, config); err != nil {
							t.Fatal(err)
						}
						for index, expected := range reference.floats {
							if math.Float32bits(output.floats[index]) != math.Float32bits(expected) {
								t.Fatalf("pattern=%d bias=%v output[%d]=%08x, released=%08x", pattern, optionalBias != nil, index, math.Float32bits(output.floats[index]), math.Float32bits(expected))
							}
						}
					}
				}
			})
		}
	}
}
