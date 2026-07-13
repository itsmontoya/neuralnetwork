package matrix

import "testing"

var benchmarkDotProductMatrixResult *Matrix
var benchmarkDotProductResult float32

func Benchmark_DotProduct(b *testing.B) {
	type testcase struct {
		name   string
		length int
	}

	tests := []testcase{
		{name: "Length1", length: 1},
		{name: "Length2", length: 2},
		{name: "Length3", length: 3},
		{name: "Length4", length: 4},
		{name: "Length5", length: 5},
		{name: "Length31", length: 31},
		{name: "Length33", length: 33},
		{name: "Length64", length: 64},
		{name: "Length257", length: 257},
		{name: "Length4096", length: 4096},
		{name: "Length4099", length: 4099},
		{name: "Length65537", length: 65537},
	}

	var tt testcase
	for _, tt = range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkDotProductLength(b, tt.length)
		})
	}
}

func Benchmark_MatMulRightTransposeDotCandidate(b *testing.B) {
	type testcase struct {
		name      string
		leftRows  int
		leftCols  int
		rightRows int
	}

	tests := []testcase{
		{name: "Small2x2", leftRows: 2, leftCols: 2, rightRows: 2},
		{name: "Small4x4", leftRows: 4, leftCols: 4, rightRows: 4},
		{name: "Medium64x64", leftRows: 64, leftCols: 64, rightRows: 64},
		{name: "Large128x256x128", leftRows: 128, leftCols: 256, rightRows: 128},
		{name: "Uneven17x33x19", leftRows: 17, leftCols: 33, rightRows: 19},
		{name: "Uneven63x65x31", leftRows: 63, leftCols: 65, rightRows: 31},
	}

	var tt testcase
	for _, tt = range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkMatMulRightTransposeDotCandidate(b, tt.leftRows, tt.leftCols, tt.rightRows)
		})
	}
}

func benchmarkDotProductLength(b *testing.B, length int) {
	var (
		left   []float32
		right  []float32
		result float32
		index  int
	)

	left = benchmarkDotProductValues(length, 0.25)
	right = benchmarkDotProductValues(length, -0.75)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result = dotProduct(left, right)
	}

	benchmarkDotProductResult = result
}

func benchmarkMatMulRightTransposeDotCandidate(b *testing.B, leftRows, leftCols, rightRows int) {
	var (
		left   *Matrix
		right  *Matrix
		result *Matrix
		err    error
		index  int
	)

	left = benchmarkDotProductMatrix(b, leftRows, leftCols)
	right = benchmarkDotProductMatrix(b, rightRows, leftCols)
	result, err = New(leftRows, rightRows)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		matMulRightTransposeDotCandidate(left, right, result)
	}

	benchmarkDotProductMatrixResult = result
}

func matMulRightTransposeDotCandidate(left, right, result *Matrix) {
	var (
		row          int
		col          int
		leftOffset   int
		rightOffset  int
		resultOffset int
	)

	for row = 0; row < left.rows; row++ {
		resultOffset = row * result.cols
		leftOffset = row * left.cols
		for col = 0; col < right.rows; col++ {
			rightOffset = col * right.cols
			result.data[resultOffset+col] = dotProduct(
				left.data[leftOffset:leftOffset+left.cols],
				right.data[rightOffset:rightOffset+right.cols],
			)
		}
	}
}

func benchmarkDotProductMatrix(b *testing.B, rows, cols int) (m *Matrix) {
	var (
		values []float32
		err    error
		index  int
	)

	values = make([]float32, rows*cols)
	for index = range values {
		values[index] = float32(index%31) / 31
	}

	m, err = FromSlice(rows, cols, values)
	if err != nil {
		b.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func benchmarkDotProductValues(length int, offset float32) (values []float32) {
	var index int

	values = make([]float32, length)
	for index = range values {
		values[index] = offset + float32(index%31)/31
	}

	return values
}
