package float32product

import (
	"math"
	"slices"
	"testing"
)

func TestProducts(t *testing.T) {
	tests := []struct {
		name                          string
		leftTranspose, rightTranspose bool
		operation                     func([]float32, []float32, []float32, int, int, int)
	}{
		{name: "standard", operation: Standard},
		{name: "left transpose", leftTranspose: true, operation: LeftTranspose},
		{name: "right transpose", rightTranspose: true, operation: RightTranspose},
		{name: "both transpose", leftTranspose: true, rightTranspose: true, operation: BothTranspose},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			const rows, inner, cols = 3, 5, 4
			left, right := make([]float32, rows*inner), make([]float32, inner*cols)
			for index := range left {
				left[index] = float32(index%7-3) / 3
			}
			for index := range right {
				right[index] = float32(index%11-5) / 7
			}
			originalLeft, originalRight := slices.Clone(left), slices.Clone(right)
			output := make([]float32, rows*cols)
			for index := range output {
				output[index] = float32(math.NaN())
			}
			test.operation(left, right, output, rows, inner, cols)
			for row := 0; row < rows; row++ {
				for col := 0; col < cols; col++ {
					var expected float32
					for contraction := 0; contraction < inner; contraction++ {
						leftIndex, rightIndex := row*inner+contraction, contraction*cols+col
						if test.leftTranspose {
							leftIndex = contraction*rows + row
						}

						if test.rightTranspose {
							rightIndex = col*inner + contraction
						}

						expected += left[leftIndex] * right[rightIndex]
					}
					if math.Float32bits(output[row*cols+col]) != math.Float32bits(expected) {
						t.Fatalf("element [%d,%d]: got %v, want %v", row, col, output[row*cols+col], expected)
					}
				}
			}
			if !slices.Equal(left, originalLeft) || !slices.Equal(right, originalRight) {
				t.Fatal("product modified read-only inputs")
			}
		})
	}
}

func TestEmptyContraction(t *testing.T) {
	operations := []func([]float32, []float32, []float32, int, int, int){Standard, LeftTranspose, RightTranspose, BothTranspose}
	for _, operation := range operations {
		output := []float32{float32(math.NaN()), 1, -2, 3}
		operation(nil, nil, output, 2, 0, 2)
		for _, value := range output {
			if math.Float32bits(value) != 0 {
				t.Fatalf("empty contraction did not clear destination: %v", output)
			}
		}
	}
}

func TestRightTranspose(t *testing.T) {
	var tests = []struct {
		name              string
		rows, inner, cols int
	}{
		{name: "single row", rows: 1, inner: 33, cols: 5},
		{name: "two rows", rows: 2, inner: 31, cols: 3},
		{name: "five rows", rows: 5, inner: 63, cols: 9},
		{name: "six rows", rows: 6, inner: 64, cols: 19},
		{name: "irregular", rows: 7, inner: 257, cols: 17},
		{name: "no input rows", rows: 0, inner: 3, cols: 5},
		{name: "no weight rows", rows: 3, inner: 5, cols: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				left    = make([]float32, tt.rows*tt.inner)
				right   = make([]float32, tt.cols*tt.inner)
				changed = make([]float32, len(right))
				output  = make([]float32, tt.rows*tt.cols)
			)
			for index := range left {
				left[index] = float32(index%13-6) / 7
			}
			for index := range right {
				right[index] = float32(index%29-14) / 11
				changed[index] = -right[index]
			}
			for index := range output {
				output[index] = float32(math.NaN())
			}
			for _, weights := range [][]float32{right, changed} {
				var (
					originalLeft  = slices.Clone(left)
					originalRight = slices.Clone(weights)
				)
				RightTranspose(left, weights, output, tt.rows, tt.inner, tt.cols)
				for row := range tt.rows {
					for col := range tt.cols {
						var want float32
						for contraction := range tt.inner {
							want += originalLeft[row*tt.inner+contraction] * originalRight[col*tt.inner+contraction]
						}
						got := output[row*tt.cols+col]
						if math.Float32bits(got) != math.Float32bits(want) {
							t.Fatalf("element [%d,%d]: got %g, want %g", row, col, got, want)
						}
					}
				}
				if !slices.Equal(left, originalLeft) || !slices.Equal(weights, originalRight) {
					t.Fatal("product modified read-only inputs")
				}
			}
		})
	}
}

func TestRightTransposeNonfiniteAndSignedZero(t *testing.T) {
	var (
		negativeZero = math.Float32frombits(0x80000000)
		inf          = float32(math.Inf(1))
		nan          = float32(math.NaN())
		left         = []float32{1, 1, 1, 1, 1, 1}
		right        = []float32{negativeZero, negativeZero, inf, 1, -inf, 1, inf, -inf, nan, 1}
		want         = []float32{0, inf, -inf, nan, nan}
		output       = make([]float32, 3*len(want))
	)
	RightTranspose(left, right, output, 3, 2, len(want))
	for index, got := range output {
		expected := want[index%len(want)]
		if math.IsNaN(float64(expected)) && math.IsNaN(float64(got)) {
			continue
		}
		if math.Float32bits(got) != math.Float32bits(expected) {
			t.Fatalf("element %d: got %g, want %g", index, got, expected)
		}
	}
}
