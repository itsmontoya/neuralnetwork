package tensor

import (
	"math"
	"testing"
)

// BenchmarkConv2DInto includes validation and reuses a disjoint destination.
// Tensor construction and payload initialization are outside timing.
func BenchmarkConv2DInto(b *testing.B) {
	var tests = []struct {
		name                           string
		batch, channels, height, width int64
		filters, groups, kernel        int64
		stride, dilation, pad          int64
		nonfinite                      bool
	}{
		{name: "MNISTFirst", batch: 1, channels: 1, height: 28, width: 28, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "MNISTSecond", batch: 1, channels: 8, height: 14, width: 14, filters: 16, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width07", batch: 1, channels: 3, height: 5, width: 7, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width08", batch: 1, channels: 3, height: 5, width: 8, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width09", batch: 1, channels: 3, height: 5, width: 9, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width10", batch: 1, channels: 3, height: 5, width: 10, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width13", batch: 1, channels: 3, height: 5, width: 13, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width14", batch: 1, channels: 3, height: 5, width: 14, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width15", batch: 1, channels: 3, height: 5, width: 15, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width16", batch: 1, channels: 3, height: 5, width: 16, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width27", batch: 1, channels: 3, height: 5, width: 27, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width28", batch: 1, channels: 3, height: 5, width: 28, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Width29", batch: 1, channels: 3, height: 5, width: 29, filters: 8, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "ShortHeight", batch: 1, channels: 1, height: 1, width: 14, filters: 4, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2},
		{name: "Nonfinite", batch: 1, channels: 8, height: 14, width: 14, filters: 16, groups: 1, kernel: 5, stride: 1, dilation: 1, pad: 2, nonfinite: true},
		{name: "Narrow1", batch: 1, channels: 3, height: 8, width: 3, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1},
		{name: "Narrow3", batch: 1, channels: 3, height: 8, width: 5, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1},
		{name: "Exact4", batch: 1, channels: 3, height: 8, width: 6, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1},
		{name: "Tail1", batch: 1, channels: 3, height: 8, width: 7, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1},
		{name: "Tail2", batch: 1, channels: 3, height: 8, width: 8, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1},
		{name: "Tail3", batch: 1, channels: 3, height: 8, width: 9, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1},
		{name: "WideExact", batch: 1, channels: 8, height: 8, width: 130, filters: 16, groups: 1, kernel: 3, stride: 1, dilation: 1},
		{name: "WideTail", batch: 1, channels: 8, height: 8, width: 131, filters: 16, groups: 1, kernel: 3, stride: 1, dilation: 1},
		{name: "WidePadded", batch: 1, channels: 8, height: 8, width: 128, filters: 16, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Filters4", batch: 1, channels: 3, height: 8, width: 14, filters: 4, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Filters5", batch: 1, channels: 3, height: 8, width: 14, filters: 5, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Filters7", batch: 1, channels: 3, height: 8, width: 14, filters: 7, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Filters8", batch: 1, channels: 3, height: 8, width: 14, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Filters9", batch: 1, channels: 3, height: 8, width: 14, filters: 9, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Filters12", batch: 1, channels: 3, height: 8, width: 14, filters: 12, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Filters15", batch: 1, channels: 3, height: 8, width: 14, filters: 15, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Filters17", batch: 1, channels: 3, height: 8, width: 14, filters: 17, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Groups4", batch: 2, channels: 6, height: 8, width: 14, filters: 8, groups: 2, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Groups5", batch: 2, channels: 6, height: 8, width: 14, filters: 10, groups: 2, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Groups7", batch: 2, channels: 6, height: 8, width: 14, filters: 14, groups: 2, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Groups8", batch: 2, channels: 6, height: 8, width: 14, filters: 16, groups: 2, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Groups9", batch: 2, channels: 6, height: 8, width: 14, filters: 18, groups: 2, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Depthwise", batch: 1, channels: 8, height: 14, width: 14, filters: 8, groups: 8, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "DepthwiseMultiplier4", batch: 1, channels: 8, height: 14, width: 14, filters: 32, groups: 8, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Fallback3", batch: 1, channels: 3, height: 8, width: 14, filters: 3, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Strided", batch: 1, channels: 3, height: 17, width: 29, filters: 8, groups: 1, kernel: 3, stride: 2, dilation: 1, pad: 2},
		{name: "Dilated", batch: 1, channels: 3, height: 9, width: 17, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 2, pad: 3},
		{name: "DilatedStrided", batch: 1, channels: 3, height: 17, width: 29, filters: 8, groups: 1, kernel: 3, stride: 2, dilation: 2, pad: 3},
		{name: "HeavyPadding", batch: 1, channels: 3, height: 2, width: 3, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 5},
		{name: "Batches", batch: 4, channels: 3, height: 8, width: 14, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "Pointwise", batch: 1, channels: 8, height: 8, width: 16, filters: 8, groups: 1, kernel: 1, stride: 1, dilation: 1},
		{name: "TinyPointwise", batch: 1, channels: 1, height: 1, width: 1, filters: 4, groups: 1, kernel: 1, stride: 1, dilation: 1},
		{name: "EmptyHeight", batch: 1, channels: 3, height: 0, width: 14, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 2},
		{name: "EmptyWidth", batch: 1, channels: 3, height: 8, width: 0, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 2},
		{name: "EmptyBatch", batch: 0, channels: 3, height: 8, width: 14, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1, pad: 1},
		{name: "EmptyOutput", batch: 1, channels: 3, height: 1, width: 1, filters: 8, groups: 1, kernel: 3, stride: 1, dilation: 1},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			var (
				input, weights, bias, output *Tensor
				shape                        []int64
				config                       = Conv2DConfig{
					Strides:   [2]int64{tt.stride, tt.stride},
					Dilations: [2]int64{tt.dilation, tt.dilation},
					Pads:      [4]int64{tt.pad, tt.pad, tt.pad, tt.pad}, Groups: tt.groups,
				}
				err error
			)

			if input, err = New(Float32, []int64{tt.batch, tt.channels, tt.height, tt.width}); err != nil {
				b.Fatal(err)
			}
			if weights, err = New(Float32, []int64{tt.filters, tt.channels / tt.groups, tt.kernel, tt.kernel}); err != nil {
				b.Fatal(err)
			}
			if bias, err = New(Float32, []int64{tt.filters}); err != nil {
				b.Fatal(err)
			}
			for index := range input.floats {
				input.floats[index] = float32(index%17-8) * 0.125
			}
			for index := range weights.floats {
				weights.floats[index] = float32(index%13-6) * 0.125
			}
			for index := range bias.floats {
				bias.floats[index] = float32(index%7-3) * 0.125
			}
			if tt.nonfinite {
				input.floats[0] = math.Float32frombits(0x7fc00123)
				weights.floats[0] = float32(math.Inf(1))
				bias.floats[0] = math.Float32frombits(0xff800456)
			}
			if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
				b.Fatal(err)
			}
			if output, err = New(Float32, shape); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err = Conv2DInto(input, weights, bias, output, config); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkConv2DIntoNoBias matches the two existing MNIST cases with nil bias,
// as used by ONNX graphs that add channel bias in a separate node.
func BenchmarkConv2DIntoNoBias(b *testing.B) {
	var tests = []struct {
		name                     string
		channels, width, filters int64
	}{
		{name: "MNISTFirst", channels: 1, width: 28, filters: 8},
		{name: "MNISTSecond", channels: 8, width: 14, filters: 16},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			var (
				input, weights, output *Tensor
				shape                  []int64
				config                 = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{2, 2, 2, 2}, Groups: 1}
				err                    error
			)

			if input, err = New(Float32, []int64{1, tt.channels, tt.width, tt.width}); err != nil {
				b.Fatal(err)
			}
			if weights, err = New(Float32, []int64{tt.filters, tt.channels, 5, 5}); err != nil {
				b.Fatal(err)
			}
			for index := range input.floats {
				input.floats[index] = float32(index%17-8) * 0.125
			}
			for index := range weights.floats {
				weights.floats[index] = float32(index%13-6) * 0.125
			}
			if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
				b.Fatal(err)
			}
			if output, err = New(Float32, shape); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err = Conv2DInto(input, weights, nil, output, config); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
