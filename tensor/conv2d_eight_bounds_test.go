//go:build arm64 && !purego && (darwin || linux)

package tensor

import (
	"math"
	"testing"
)

func Test_Conv2D_EightFilterPayloadBounds(t *testing.T) {
	var tests = []struct {
		name                                          string
		channels, height, width, filters, kernel, pad int64
	}{
		{name: "two lanes", channels: 2, height: 1, width: 2, filters: 8, kernel: 3, pad: 2},
		{name: "three lanes", channels: 1, height: 2, width: 3, filters: 16, kernel: 5, pad: 3},
		{name: "interior and residual", channels: 2, height: 3, width: 14, filters: 17, kernel: 5, pad: 2},
		{name: "interior and filter tails", channels: 2, height: 3, width: 29, filters: 9, kernel: 3, pad: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, atEnd := range []bool{false, true} {
				var (
					input, weights, bias, output, reference *Tensor
					shape                                   []int64
					config                                  = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{tt.pad, tt.pad, tt.pad, tt.pad}, Groups: 1}
					numbers                                 = []uint32{0x3f800001, 0xbf7ffffe, 0x80000000, 0x00000001, 0x7fc00123, 0x7f800000, 0xff800456}
					err                                     error
				)
				if input, err = New(Float32, []int64{1, tt.channels, tt.height, tt.width}); err != nil {
					t.Fatal(err)
				}
				if weights, err = New(Float32, []int64{tt.filters, tt.channels, tt.kernel, tt.kernel}); err != nil {
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
				input.floats = conv2DGuardedFloats(t, len(input.floats), atEnd)
				weights.floats = conv2DGuardedFloats(t, len(weights.floats), atEnd)
				bias.floats = conv2DGuardedFloats(t, len(bias.floats), atEnd)
				output.floats = conv2DGuardedFloats(t, len(output.floats), atEnd)
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
							t.Fatalf("atEnd=%v bias=%v output[%d]=%08x, released=%08x", atEnd, optionalBias != nil, index, math.Float32bits(output.floats[index]), math.Float32bits(expected))
						}
					}
				}
			}
		})
	}
}
