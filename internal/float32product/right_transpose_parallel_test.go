package float32product

import (
	"math"
	"runtime"
	"slices"
	"testing"
)

func TestRightTransposeOneRowWorkerCount(t *testing.T) {
	var (
		previous = runtime.GOMAXPROCS(8)
		tests    = []struct {
			name                       string
			inner, cols, threads, want int
		}{
			{name: "incomplete tile", inner: 1 << 22, cols: 7, threads: 8, want: 1},
			{name: "below threshold", inner: 2048, cols: 511, threads: 8, want: 1},
			{name: "exact threshold", inner: 2048, cols: 512, threads: 8, want: 2},
			{name: "one scheduler thread", inner: 2048, cols: 8192, threads: 1, want: 1},
			{name: "limited scheduler threads", inner: 2048, cols: 8192, threads: 4, want: 4},
			{name: "worker cap", inner: 2048, cols: 128256, threads: 16, want: 8},
		}
	)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runtime.GOMAXPROCS(tt.threads)
			if got := rightTransposeOneRowWorkerCount(tt.inner, tt.cols); got != tt.want {
				t.Fatalf("worker count = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestRightTransposeOneRowAgainstOrderedReference(t *testing.T) {
	var (
		previous = runtime.GOMAXPROCS(8)
		tests    = []struct {
			name                 string
			inner, cols, workers int
		}{
			{name: "below threshold", inner: 2048, cols: 511, workers: 1},
			{name: "exact threshold", inner: 2048, cols: 512, workers: 2},
			{name: "odd contraction and partition tail", inner: 2049, cols: 775, workers: 3},
			{name: "worker cap and column tail", inner: 2048, cols: 2055, workers: 8},
			{name: "FFN down contraction", inner: 8192, cols: 137, workers: 2},
		}
	)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				left         = make([]float32, tt.inner)
				right        = make([]float32, tt.inner*tt.cols)
				serial       = make([]float32, tt.cols)
				dispatched   = make([]float32, tt.cols)
				want         = make([]float32, tt.cols)
				negativeZero = math.Float32frombits(0x80000000)
				infinity     = float32(math.Inf(1))
				nan          = math.Float32frombits(0x7fc12345)
			)
			for index := range left {
				left[index] = float32(index%31-15) / 17
			}
			left[0], left[1], left[2], left[3] = 1, 1, 1, math.SmallestNonzeroFloat32
			for column := range want {
				weight := right[column*tt.inner:][:tt.inner]
				switch column % 23 {
				case 0:
					// Adding the unit after cancellation would change this result.
					weight[0], weight[1], weight[2] = 1<<25, 1, -(1 << 25)
				case 1:
					weight[0], weight[3] = negativeZero, -0.25
				case 2:
					weight[0] = infinity
				case 3:
					weight[0] = -infinity
				case 4:
					weight[0], weight[1] = infinity, -infinity
				case 5:
					weight[0] = nan
				default:
					for index := range weight {
						weight[index] = float32((column+index)%53-26) / 29
					}
				}
				var sum float32
				for index, input := range left {
					sum += input * weight[index]
				}
				want[column] = sum
				serial[column], dispatched[column] = 99, 99
			}
			leftBefore, rightBefore := slices.Clone(left), slices.Clone(right)
			runtime.GOMAXPROCS(1)
			RightTranspose(left, right, serial, 1, tt.inner, tt.cols)
			runtime.GOMAXPROCS(8)
			if got := rightTransposeOneRowWorkerCount(tt.inner, tt.cols); got != tt.workers {
				t.Fatalf("worker count = %d, want %d", got, tt.workers)
			}
			RightTranspose(left, right, dispatched, 1, tt.inner, tt.cols)
			for column, expected := range want {
				for _, got := range []float32{serial[column], dispatched[column]} {
					if math.IsNaN(float64(expected)) && math.IsNaN(float64(got)) {
						continue
					}
					if math.Float32bits(got) != math.Float32bits(expected) {
						t.Fatalf("column %d: got %08x, want %08x", column, math.Float32bits(got), math.Float32bits(expected))
					}
				}
			}
			var equalBits = func(left, right float32) (equal bool) {
				return math.Float32bits(left) == math.Float32bits(right)
			}
			if !slices.EqualFunc(left, leftBefore, equalBits) || !slices.EqualFunc(right, rightBefore, equalBits) {
				t.Fatal("projection modified read-only inputs")
			}
		})
	}
}
