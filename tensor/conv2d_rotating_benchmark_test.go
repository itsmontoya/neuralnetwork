package tensor

import "testing"

// BenchmarkConv2DIntoRotating varies reuse while keeping geometry and payloads
// fixed. Each independent set owns disjoint input, weights, and output tensors.
func BenchmarkConv2DIntoRotating(b *testing.B) {
	var tests = []struct {
		name string
		sets int
	}{
		{name: "Sets1", sets: 1},
		{name: "Sets64", sets: 64},
		{name: "Sets1024", sets: 1024},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			var (
				inputs   = make([]*Tensor, tt.sets)
				weights  = make([]*Tensor, tt.sets)
				outputs  = make([]*Tensor, tt.sets)
				config   = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{2, 2, 2, 2}, Groups: 1}
				position int
				err      error
			)
			for set := 0; set < tt.sets; set++ {
				if inputs[set], err = New(Float32, []int64{1, 8, 14, 14}); err != nil {
					b.Fatal(err)
				}
				if weights[set], err = New(Float32, []int64{16, 8, 5, 5}); err != nil {
					b.Fatal(err)
				}
				if outputs[set], err = New(Float32, []int64{1, 16, 14, 14}); err != nil {
					b.Fatal(err)
				}
				for index := range inputs[set].floats {
					inputs[set].floats[index] = float32(index%17-8) * 0.125
				}
				for index := range weights[set].floats {
					weights[set].floats[index] = float32(index%13-6) * 0.125
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if err = Conv2DInto(inputs[position], weights[position], nil, outputs[position], config); err != nil {
					b.Fatal(err)
				}
				position++
				if position == tt.sets {
					position = 0
				}
			}
		})
	}
}
