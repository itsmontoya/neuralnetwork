package tensor

import (
	"math"
	"testing"
)

// BenchmarkMaxPool2DInto includes validation and reuses a disjoint destination.
// Construction and initialization are outside timing.
func BenchmarkMaxPool2DInto(b *testing.B) {
	type testcase struct {
		name                string
		shape               []int64
		kernel, strides     [2]int64
		dilation            int64
		pads                [4]int64
		ceilMode, nonfinite bool
	}
	var tests = []testcase{
		{name: "MNISTFirst", shape: []int64{1, 8, 28, 28}, kernel: [2]int64{2, 2}, strides: [2]int64{2, 2}},
		{name: "MNISTSecond", shape: []int64{1, 16, 14, 14}, kernel: [2]int64{3, 3}, strides: [2]int64{3, 3}},
		{name: "Single2", shape: []int64{1, 1, 2, 2}, kernel: [2]int64{2, 2}, strides: [2]int64{1, 1}},
		{name: "Single3", shape: []int64{1, 1, 3, 3}, kernel: [2]int64{3, 3}, strides: [2]int64{1, 1}},
		{name: "Narrow2", shape: []int64{2, 3, 17, 2}, kernel: [2]int64{2, 2}, strides: [2]int64{1, 4}},
		{name: "Narrow3", shape: []int64{2, 3, 17, 3}, kernel: [2]int64{3, 3}, strides: [2]int64{2, 4}},
		{name: "ShortWide", shape: []int64{1, 3, 3, 131}, kernel: [2]int64{3, 3}, strides: [2]int64{4, 2}},
		{name: "Uneven2", shape: []int64{2, 3, 9, 11}, kernel: [2]int64{2, 2}, strides: [2]int64{3, 4}},
		{name: "Uneven3", shape: []int64{2, 3, 9, 11}, kernel: [2]int64{3, 3}, strides: [2]int64{4, 3}},
		{name: "Overlap2", shape: []int64{2, 3, 17, 19}, kernel: [2]int64{2, 2}, strides: [2]int64{1, 1}},
		{name: "Overlap3", shape: []int64{2, 3, 17, 19}, kernel: [2]int64{3, 3}, strides: [2]int64{1, 1}},
		{name: "LargeStride", shape: []int64{2, 3, 7, 9}, kernel: [2]int64{3, 3}, strides: [2]int64{100, 200}},
		{name: "Nonfinite", shape: []int64{2, 3, 17, 19}, kernel: [2]int64{3, 3}, strides: [2]int64{2, 3}, nonfinite: true},
		{name: "Padded", shape: []int64{2, 3, 9, 11}, kernel: [2]int64{2, 2}, strides: [2]int64{2, 2}, pads: [4]int64{1, 0, 0, 2}},
		{name: "Dilated", shape: []int64{2, 3, 9, 11}, kernel: [2]int64{3, 3}, strides: [2]int64{2, 2}, dilation: 2},
		{name: "Ceil", shape: []int64{2, 3, 9, 11}, kernel: [2]int64{3, 3}, strides: [2]int64{3, 3}, ceilMode: true},
		{name: "Rectangular", shape: []int64{2, 3, 9, 11}, kernel: [2]int64{2, 3}, strides: [2]int64{2, 3}},
		{name: "Empty", shape: []int64{2, 3, 1, 1}, kernel: [2]int64{2, 2}, strides: [2]int64{2, 2}},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			var (
				input, output *Tensor
				shape         []int64
				config        = Pool2DConfig{
					Kernel: tt.kernel, Strides: tt.strides, Dilations: [2]int64{1, 1},
					Pads: tt.pads, CeilMode: tt.ceilMode,
				}
				err error
			)
			if tt.dilation != 0 {
				config.Dilations = [2]int64{tt.dilation, tt.dilation}
			}
			if input, err = New(Float32, tt.shape); err != nil {
				b.Fatal(err)
			}
			for index := range input.floats {
				input.floats[index] = float32(index%31-15) * 0.125
				if tt.nonfinite && index%7 == 0 {
					input.floats[index] = math.Float32frombits(0xff800123)
				}
			}
			if shape, err = Pool2DShape(input.shape, config); err != nil {
				b.Fatal(err)
			}
			if output, err = New(Float32, shape); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err = MaxPool2DInto(input, output, config); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
