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
