package tensor

import (
	"math"
	"slices"
	"sync"
	"testing"
)

func Test_AddInto_GenericBroadcast(t *testing.T) {
	var (
		tests = []struct {
			name        string
			left, right []int64
		}{
			{name: "first bias", left: []int64{1, 8, 28, 28}, right: []int64{8, 1, 1}},
			{name: "second bias", left: []int64{1, 16, 14, 14}, right: []int64{16, 1, 1}},
			{name: "scalar", left: []int64{2, 3, 17}},
			{name: "batch", left: []int64{3, 4, 5, 7}, right: []int64{4, 1, 1}},
			{name: "multidirectional prefix", left: []int64{2, 1, 3, 5, 7}, right: []int64{1, 4, 3, 1, 1}},
			{name: "singleton axes", left: []int64{2, 3, 1, 7, 1}, right: []int64{3, 1, 1, 1}},
			{name: "broadcast values prefix", left: []int64{1, 3, 7}, right: []int64{2, 3, 1}},
			{name: "promoted values", left: []int64{5}, right: []int64{2, 1}},
			{name: "span two", left: []int64{3, 2}, right: []int64{3, 1}},
			{name: "span three", left: []int64{3, 3}, right: []int64{3, 1}},
			{name: "span four", left: []int64{3, 4}, right: []int64{3, 1}},
			{name: "span eighteen", left: []int64{3, 18}, right: []int64{3, 1}},
			{name: "span nineteen", left: []int64{3, 19}, right: []int64{3, 1}},
			{name: "both vary fallback", left: []int64{7, 31}, right: []int64{31}},
			{name: "singleton suffix fallback", left: []int64{2, 1, 3, 1}, right: []int64{1, 4, 1, 1}},
			{name: "tiny", left: []int64{1}},
			{name: "tiny two", left: []int64{2}},
			{name: "scalars"},
			{name: "all singleton", left: []int64{1, 1, 1, 1}, right: []int64{1}},
			{name: "equal", left: []int64{1, 10}, right: []int64{1, 10}},
			{name: "empty", left: []int64{2, 0, 1}, right: []int64{1, 1, 3}},
		}
		patterns = [][]uint32{
			{0x3f800001, 0xbf7ffffe, 0x4b800000, 0x3f800000, 0xcb800000, 0x7f7fffff, 0xff7fffff},
			{0, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x00800000},
			{0x7fc00123, 0xffc00456, 0x7f800123, 0xff800456, 0x7f800000, 0xff800000, 0, 0x80000000},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				left, right, output, reference *Tensor
				shape                          []int64
				err                            error
			)
			if left, err = New(Float32, tt.left); err != nil {
				t.Fatal(err)
			}
			if right, err = New(Float32, tt.right); err != nil {
				t.Fatal(err)
			}
			if shape, err = BroadcastShape(tt.left, tt.right); err != nil {
				t.Fatal(err)
			}
			if output, err = New(Float32, shape); err != nil {
				t.Fatal(err)
			}
			if reference, err = New(Float32, shape); err != nil {
				t.Fatal(err)
			}
			for _, pattern := range patterns {
				for index := range left.floats {
					left.floats[index] = math.Float32frombits(pattern[index%len(pattern)])
				}
				for index := range right.floats {
					right.floats[index] = math.Float32frombits(pattern[(index+1)%len(pattern)])
				}
				for _, reverse := range []bool{false, true} {
					var (
						first, second = left, right
						beforeLeft    = slices.Clone(left.floats)
						beforeRight   = slices.Clone(right.floats)
					)
					if reverse {
						first, second = right, left
					}
					if err = AddInto(first, second, output); err != nil {
						t.Fatal(err)
					}
					if len(reference.floats) != 0 {
						if err = broadcastFloat32Into(first, second, reference, addValue); err != nil {
							t.Fatal(err)
						}
					}
					for index, want := range reference.floats {
						var got = output.floats[index]
						// Existing arithmetic fixtures leave NaN payloads unspecified.
						if math.Float32bits(got) != math.Float32bits(want) && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
							t.Fatalf("reverse=%v output[%d]=%08x, want %08x", reverse, index, math.Float32bits(got), math.Float32bits(want))
						}
					}
					for sourceIndex, source := range []*Tensor{left, right} {
						var before = [][]float32{beforeLeft, beforeRight}[sourceIndex]
						for index, value := range source.floats {
							if math.Float32bits(value) != math.Float32bits(before[index]) {
								t.Fatalf("source %d changed at %d", sourceIndex, index)
							}
						}
					}
				}
			}
		})
	}
}

func Test_AddInto_SpanArgumentErrors(t *testing.T) {
	var tests = []string{
		"left payload", "right payload", "destination payload", "left dimension",
		"right dimension", "destination dimension", "left overflow", "right overflow",
		"destination overflow", "incompatible prefix", "destination shape", "left alias", "right alias",
	}
	for _, name := range tests {
		t.Run(name, func(t *testing.T) {
			var (
				left, right, output *Tensor
				snapshot            []float32
				limit               = int64(int(^uint(0) >> 1))
				err                 error
			)
			if left, err = New(Float32, []int64{2, 3, 17}); err != nil {
				t.Fatal(err)
			}
			if right, err = New(Float32, []int64{3, 1}); err != nil {
				t.Fatal(err)
			}
			if output, err = New(Float32, left.shape); err != nil {
				t.Fatal(err)
			}
			switch name {
			case "left payload":
				left.floats = left.floats[:len(left.floats)-1]
			case "right payload":
				right.floats = nil
			case "destination payload":
				output.floats = output.floats[:len(output.floats)-1]
			case "left dimension":
				left.shape[0] = -1
			case "right dimension":
				right.shape[0] = -1
			case "destination dimension":
				output.shape[0] = -1
			case "left overflow":
				left.shape[0] = limit
			case "right overflow":
				right.shape = []int64{limit, 3, 1}
			case "destination overflow":
				output.shape[0] = limit
			case "incompatible prefix":
				right.shape = []int64{3, 1, 1}
			case "destination shape":
				output.shape = []int64{3, 2, 17}
			case "left alias":
				var shallow = *left
				output = &shallow
			case "right alias":
				left, right = right, left
				var shallow = *right
				output = &shallow
			}
			for index := range output.floats {
				output.floats[index] = float32(index + 1)
			}
			snapshot = slices.Clone(output.floats)
			if err = AddInto(left, right, output); err == nil {
				t.Fatal("invalid arguments succeeded")
			}
			if !slices.Equal(output.floats, snapshot) {
				t.Fatal("destination changed on error")
			}
		})
	}
}

func Test_AddInto_ConcurrentSpans(t *testing.T) {
	var (
		left, right, reference *Tensor
		workers                sync.WaitGroup
		err                    error
	)
	if left, err = New(Float32, []int64{2, 3, 17}); err != nil {
		t.Fatal(err)
	}
	if right, err = FromFloat32([]int64{3, 1}, []float32{1.25, -2.5, 3.75}); err != nil {
		t.Fatal(err)
	}
	if reference, err = New(Float32, left.shape); err != nil {
		t.Fatal(err)
	}
	for index := range left.floats {
		left.floats[index] = float32(index%29-14) * 0.125
	}
	if err = broadcastFloat32Into(left, right, reference, addValue); err != nil {
		t.Fatal(err)
	}
	for worker := range 8 {
		workers.Add(1)
		go func() {
			var (
				output        *Tensor
				first, second = left, right
				workerErr     error
			)
			defer workers.Done()
			if worker%2 == 1 {
				first, second = right, left
			}
			if output, workerErr = New(Float32, reference.shape); workerErr != nil {
				t.Error(workerErr)
				return
			}
			for range 10 {
				if workerErr = AddInto(first, second, output); workerErr != nil {
					t.Error(workerErr)
					return
				}
				if !slices.Equal(output.floats, reference.floats) {
					t.Error("concurrent output differs")
					return
				}
			}
		}()
	}
	workers.Wait()
}
