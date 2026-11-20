package float32product

import (
	"math"
	"runtime"
	"slices"
	"testing"
)

func TestRightTransposeGroupWorkerCount(t *testing.T) {
	var (
		previous = runtime.GOMAXPROCS(8)
		tests    = []struct {
			name                               string
			groups, inner, cols, threads, want int
		}{
			{name: "incomplete tile", groups: 2, inner: 1 << 20, cols: 3, threads: 8, want: 1},
			{name: "one complete tile", groups: 2, inner: 1 << 20, cols: 7, threads: 8, want: 1},
			{name: "below threshold", groups: 1, inner: 2048, cols: 255, threads: 8, want: 1},
			{name: "exact threshold", groups: 1, inner: 2048, cols: 256, threads: 8, want: 2},
			{name: "groups share dispatch", groups: 2, inner: 2048, cols: 129, threads: 8, want: 2},
			{name: "one scheduler thread", groups: 9, inner: 2048, cols: 8192, threads: 1, want: 1},
			{name: "limited scheduler threads", groups: 2, inner: 2048, cols: 8192, threads: 4, want: 4},
			{name: "worker cap", groups: 2, inner: 2048, cols: 8192, threads: 16, want: 8},
		}
	)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runtime.GOMAXPROCS(tt.threads)
			if workers := rightTransposeGroupWorkerCount(tt.groups, tt.inner, tt.cols); workers != tt.want {
				t.Fatalf("worker count = %d, want %d", workers, tt.want)
			}
		})
	}
}

func TestRightTransposeGroupsAgainstOrderedReference(t *testing.T) {
	var (
		previous = runtime.GOMAXPROCS(8)
		tests    = []struct {
			name              string
			rows, inner, cols int
		}{
			{name: "irregular split and scalar column tail", rows: 3, inner: 2048, cols: 519},
			{name: "odd contraction and one scalar row", rows: 7, inner: 2049, cols: 519},
			{name: "wide contraction and two scalar rows", rows: 8, inner: 8192, cols: 139},
		}
	)
	t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				left    = make([]float32, tt.rows*tt.inner)
				right   = make([]float32, tt.cols*tt.inner)
				serial  = make([]float32, tt.rows*tt.cols)
				output  = make([]float32, len(serial))
				want    = make([]float32, len(serial))
				inf     = float32(math.Inf(1))
				nan     = float32(math.NaN())
				negZero = math.Float32frombits(0x80000000)
			)
			for row := range tt.rows {
				var input = left[row*tt.inner:][:tt.inner]
				for index := range input {
					input[index] = float32((row+index)%31-15) / 17
				}
				input[0], input[1], input[2], input[tt.inner-1] = 1, 1, 1, math.SmallestNonzeroFloat32
			}
			for column := range tt.cols {
				var weight = right[column*tt.inner:][:tt.inner]
				switch column % 19 {
				case 0:
					// The unit term is lost before the large terms cancel.
					weight[0], weight[1], weight[2] = 1<<25, 1, -(1 << 25)
				case 1:
					weight[0], weight[tt.inner-1] = negZero, -0.25
				case 2:
					weight[0] = inf
				case 3:
					weight[0] = -inf
				case 4:
					weight[0], weight[1] = inf, -inf
				case 5:
					weight[0] = nan
				default:
					for index := range weight {
						weight[index] = float32((column+index)%53-26) / 29
					}
				}
				for row := range tt.rows {
					var (
						input = left[row*tt.inner:][:tt.inner]
						sum   float32
					)
					for index, value := range input {
						sum += value * weight[index]
					}
					want[row*tt.cols+column] = sum
				}
			}
			var (
				leftBefore  = slices.Clone(left)
				rightBefore = slices.Clone(right)
			)
			for index := range want {
				serial[index], output[index] = 99, 99
			}
			runtime.GOMAXPROCS(1)
			if workers := rightTransposeGroupWorkerCount(tt.rows/3, tt.inner, tt.cols); workers != 1 {
				t.Fatalf("one-thread dispatch selected %d workers", workers)
			}
			RightTranspose(left, right, serial, tt.rows, tt.inner, tt.cols)
			runtime.GOMAXPROCS(8)
			if workers := rightTransposeGroupWorkerCount(tt.rows/3, tt.inner, tt.cols); workers < 2 {
				t.Fatalf("parallel dispatch selected %d workers", workers)
			}
			RightTranspose(left, right, output, tt.rows, tt.inner, tt.cols)
			for index, expected := range want {
				for _, got := range []float32{serial[index], output[index]} {
					if math.IsNaN(float64(expected)) && math.IsNaN(float64(got)) {
						continue
					}
					if math.Float32bits(got) != math.Float32bits(expected) {
						t.Fatalf("element [%d,%d]: got %08x, want %08x", index/tt.cols, index%tt.cols, math.Float32bits(got), math.Float32bits(expected))
					}
				}
			}
			var equalBits = func(a, b float32) (equal bool) {
				return math.Float32bits(a) == math.Float32bits(b)
			}
			if !slices.EqualFunc(left, leftBefore, equalBits) || !slices.EqualFunc(right, rightBefore, equalBits) {
				t.Fatal("projection modified read-only inputs")
			}
		})
	}
}
