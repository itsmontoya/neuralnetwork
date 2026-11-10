package tensor

import "testing"

// BenchmarkAddInto includes validation and dispatch with disjoint, reused output.
// Both input orders use varied finite values initialized outside timing.
func BenchmarkAddInto(b *testing.B) {
	var tests = []struct {
		name        string
		left, right []int64
	}{
		{name: "MNISTFirst", left: []int64{1, 8, 28, 28}, right: []int64{8, 1, 1}},
		{name: "MNISTSecond", left: []int64{1, 16, 14, 14}, right: []int64{16, 1, 1}},
		{name: "Scalar", left: []int64{2, 3, 31}},
		{name: "Batch", left: []int64{3, 8, 7, 11}, right: []int64{8, 1, 1}},
		{name: "RankFive", left: []int64{2, 1, 3, 5, 7}, right: []int64{1, 4, 3, 1, 1}},
		{name: "SingletonSuffix", left: []int64{2, 3, 1, 7, 1}, right: []int64{3, 1, 1, 1}},
		{name: "TailOne", left: []int64{3, 17}, right: []int64{3, 1}},
		{name: "TailTwo", left: []int64{3, 18}, right: []int64{3, 1}},
		{name: "TailThree", left: []int64{3, 19}, right: []int64{3, 1}},
		{name: "SpanTwo", left: []int64{3, 2}, right: []int64{3, 1}},
		{name: "Tiny", left: []int64{1}},
		{name: "TinyTwo", left: []int64{2}},
		{name: "Scalars"},
		{name: "Equal", left: []int64{1, 10}, right: []int64{1, 10}},
		{name: "Fallback", left: []int64{2, 1, 3}, right: []int64{1, 4, 3}},
		{name: "FallbackVector", left: []int64{7, 31}, right: []int64{31}},
		{name: "Empty", left: []int64{2, 0, 3}, right: []int64{1, 1, 3}},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			for _, reverse := range []bool{false, true} {
				var name = "Forward"
				if reverse {
					name = "Reverse"
				}
				b.Run(name, func(b *testing.B) {
					var (
						left, right, output *Tensor
						shape               []int64
						err                 error
					)
					if left, err = New(Float32, tt.left); err != nil {
						b.Fatal(err)
					}
					if right, err = New(Float32, tt.right); err != nil {
						b.Fatal(err)
					}
					for index := range left.floats {
						left.floats[index] = float32(index%29-14) * 0.125
					}
					for index := range right.floats {
						right.floats[index] = float32(index%13-6) * 0.25
					}
					if reverse {
						left, right = right, left
					}
					if shape, err = BroadcastShape(left.shape, right.shape); err != nil {
						b.Fatal(err)
					}
					if output, err = New(Float32, shape); err != nil {
						b.Fatal(err)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						if err = AddInto(left, right, output); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		})
	}
}

// BenchmarkBroadcastAdd separates generic traversal from stride planning.
func BenchmarkBroadcastAdd(b *testing.B) {
	var tests = []struct {
		name           string
		channels, side int64
	}{
		{name: "MNISTFirst", channels: 8, side: 28},
		{name: "MNISTSecond", channels: 16, side: 14},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			var (
				left, right, output *Tensor
				err                 error
			)
			if left, err = New(Float32, []int64{1, tt.channels, tt.side, tt.side}); err != nil {
				b.Fatal(err)
			}
			if right, err = New(Float32, []int64{tt.channels, 1, 1}); err != nil {
				b.Fatal(err)
			}
			if output, err = New(Float32, left.shape); err != nil {
				b.Fatal(err)
			}
			for index := range left.floats {
				left.floats[index] = float32(index%29-14) * 0.125
			}
			for index := range right.floats {
				right.floats[index] = float32(index%13-6) * 0.25
			}
			b.Run("Generic", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err = broadcastFloat32Into(left, right, output, addValue); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("Strides", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err = broadcastStrides(left.shape, output.shape); err != nil {
						b.Fatal(err)
					}
					if _, err = broadcastStrides(right.shape, output.shape); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}
