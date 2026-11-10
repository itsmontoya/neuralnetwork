//go:build arm64 && !purego && (darwin || linux)

package tensor

import (
	"math"
	"testing"
)

func Test_Conv2D_ResidualTilePayloadBounds(t *testing.T) {
	var tests = []struct {
		name                    string
		channels, height, width int64
		kernelH, kernelW, pad   int64
	}{
		{name: "single sample", channels: 1, height: 1, width: 1, kernelH: 1, kernelW: 1, pad: 3},
		{name: "two samples", channels: 2, height: 1, width: 2, kernelH: 1, kernelW: 3, pad: 3},
		{name: "three samples", channels: 1, height: 2, width: 3, kernelH: 2, kernelW: 5, pad: 3},
		{name: "last channel and row", channels: 3, height: 3, width: 7, kernelH: 3, kernelW: 3, pad: 2},
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
				if weights, err = New(Float32, []int64{4, tt.channels, tt.kernelH, tt.kernelW}); err != nil {
					t.Fatal(err)
				}
				if bias, err = New(Float32, []int64{4}); err != nil {
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
					for row := 0; row < int(shape[2]); row++ {
						for column := 0; column < int(shape[3]); column++ {
							var offset = row*int(shape[3]) + column

							conv2DResidualChannels(input, weights, optionalBias, reference, 0, 0, row, column-int(tt.pad), offset, config)
						}
					}
					for tileWidth := 2; tileWidth <= 4; tileWidth++ {
						for row := 0; row < int(shape[2]); row++ {
							for column := 0; column < int(shape[3]); {
								var (
									columns = min(tileWidth, int(shape[3])-column)
									offset  = row*int(shape[3]) + column
								)

								if columns == 1 {
									conv2DResidualChannels(input, weights, optionalBias, output, 0, 0, row, column-int(tt.pad), offset, config)
									column++
									continue
								}
								conv2DResidualTile(input, weights, optionalBias, output, 0, 0, row, column-int(tt.pad), offset, columns, config)
								column += columns
							}
						}
						for index, expected := range reference.floats {
							if math.Float32bits(output.floats[index]) != math.Float32bits(expected) {
								t.Fatalf("atEnd=%v tile=%d bias=%v output[%d]=%08x want %08x", atEnd, tileWidth, optionalBias != nil, index, math.Float32bits(output.floats[index]), math.Float32bits(expected))
							}
						}
					}
				}
			}
		})
	}
}

func Test_Conv2D_ResidualTileSumsBounds(t *testing.T) {
	var tests = []struct {
		name              string
		width, first, end int
		input             []float32
		want              [4]float32
	}{
		{name: "two lanes", width: 2, end: 2, input: []float32{2, 3}, want: [4]float32{2, 3}},
		{name: "three lanes", width: 3, end: 3, input: []float32{2, 3, 4}, want: [4]float32{2, 3, 4}},
		{name: "four lanes", width: 4, end: 4, input: []float32{2, 3, 4, 5}, want: [4]float32{2, 3, 4, 5}},
		{name: "left clipping", width: 4, first: 3, end: 4, input: []float32{2}, want: [4]float32{0, 0, 0, 2}},
		{name: "right clipping", width: 4, end: 1, input: []float32{2}, want: [4]float32{2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, atEnd := range []bool{false, true} {
				var (
					input   = conv2DGuardedFloats(t, len(tt.input), atEnd)
					weights = conv2DGuardedFloats(t, 4, atEnd)
					sums    = conv2DGuardedFloats(t, 16, atEnd)
				)

				copy(input, tt.input)
				copy(weights, []float32{3, 5, 7, 11})
				conv2DResidualTileSums(input, weights, (*[16]float32)(sums), 1, 1, 1, len(input), len(input), 1, 1, tt.first, tt.end, tt.width)
				for lane, weight := range weights {
					for column := 0; column < tt.width; column++ {
						var expected = tt.want[column] * weight

						if sums[lane*4+column] != expected {
							t.Fatalf("atEnd=%v lane=%d column=%d got %v want %v", atEnd, lane, column, sums[lane*4+column], expected)
						}
					}
				}
			}
		})
	}
}
