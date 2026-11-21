package float32product

import (
	"fmt"
	"math"
	"runtime"
	"testing"
)

func TestRightTransposePackedRange(t *testing.T) {
	const inner, cols = 68, 21
	var (
		dense  = make([]float32, inner*cols)
		packed = make([]float32, len(dense))
	)
	for index := range dense {
		dense[index] = float32(index%79-39) / 79
	}
	dense[0] = math.Float32frombits(0x7fc12345)
	dense[inner+3] = float32(math.Inf(-1))
	clear(dense[7*inner : 8*inner])
	dense[7*inner+3] = math.Float32frombits(0x80000000)
	PackRightTranspose(dense, packed, inner, cols)

	for rows := 1; rows <= 5; rows++ {
		var (
			left = make([]float32, rows*inner)
			want = make([]float32, rows*cols)
			got  = make([]float32, len(want))
		)
		for index := range left {
			left[index] = float32(index%39-19) / 39
		}
		RightTranspose(left, dense, want, rows, inner, cols)
		for start := 0; start <= cols; start++ {
			for end := start; end <= cols; end++ {
				for index := range got {
					got[index] = 123
				}
				rightTransposePackedRange(left, packed, got, rows, inner, cols, start, end)
				for index, value := range got {
					expected := float32(123)
					if column := index % cols; start <= column && column < end {
						expected = want[index]
					}
					if math.Float32bits(value) != math.Float32bits(expected) {
						t.Fatalf("rows=%d range=[%d,%d) index=%d: got %08x, want %08x", rows, start, end, index, math.Float32bits(value), math.Float32bits(expected))
					}
				}
			}
		}
		if allocs := testing.AllocsPerRun(3, func() {
			rightTransposePackedRange(left, packed, got, rows, inner, cols, 3, 15)
		}); allocs != 0 {
			t.Fatalf("rows=%d: partial packed range allocated %v times", rows, allocs)
		}
	}
}

func TestRightTransposePackedWorkerCaps(t *testing.T) {
	var previous = runtime.GOMAXPROCS(0)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })

	for _, shape := range []struct {
		inner, cols int
		arm64Only   bool
	}{
		{inner: 2048, cols: 781},
		{inner: 2048, cols: 8192, arm64Only: true},
		{inner: 8192, cols: 2048, arm64Only: true},
	} {
		t.Run(fmt.Sprintf("K%dN%d", shape.inner, shape.cols), func(t *testing.T) {
			if shape.arm64Only && !packedRightARM64 {
				t.Skip("full FFN sizes exercise the ARM64 packed dispatch")
			}
			var (
				inner  = shape.inner
				cols   = shape.cols
				dense  = make([]float32, inner*cols)
				packed = make([]float32, len(dense))
				left   = make([]float32, 3*inner)
				want   = make([]float32, 3*cols)
				got    = make([]float32, len(want))
			)
			for index := range dense {
				dense[index] = float32(index%131-65) / 131
			}
			for index := range left {
				left[index] = float32(index%83-41) / 83
			}
			PackRightTranspose(dense, packed, inner, cols)
			for rows := 1; rows <= 3; rows++ {
				runtime.GOMAXPROCS(1)
				RightTranspose(left, dense, want[:rows*cols], rows, inner, cols)
				for workerCap := 1; workerCap <= 8; workerCap++ {
					runtime.GOMAXPROCS(workerCap)
					for index := range got {
						got[index] = 123
					}
					RightTransposePacked(left, packed, got[:rows*cols], rows, inner, cols)
					for index, value := range got[:rows*cols] {
						if math.Float32bits(value) != math.Float32bits(want[index]) {
							t.Fatalf("rows=%d P=%d index=%d: got %08x, want %08x", rows, workerCap, index, math.Float32bits(value), math.Float32bits(want[index]))
						}
					}
					for index, value := range got[rows*cols:] {
						if value != 123 {
							t.Fatalf("rows=%d P=%d: output changed past row %d at index %d", rows, workerCap, rows, rows*cols+index)
						}
					}
				}
			}
		})
	}
}
