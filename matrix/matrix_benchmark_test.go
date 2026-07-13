package matrix_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

var benchmarkResult *matrix.Matrix
var benchmarkValues []float32

func Benchmark_MatMul(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 64, 64)
	right = benchmarkMatrix(b, 64, 64)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = left.MatMul(right)
		if err != nil {
			b.Fatalf("MatMul returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_MatMulInto(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 64, 64)
	right = benchmarkMatrix(b, 64, 64)
	result, err = matrix.New(64, 64)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.MatMulInto(right, result)
		if err != nil {
			b.Fatalf("MatMulInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_MatMulLeftTransposeInto(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 128, 32)
	right = benchmarkMatrix(b, 128, 64)
	result, err = matrix.New(32, 64)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.MatMulLeftTransposeInto(right, result)
		if err != nil {
			b.Fatalf("MatMulLeftTransposeInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_MatMulRightTransposeInto(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 128, 64)
	right = benchmarkMatrix(b, 32, 64)
	result, err = matrix.New(128, 32)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.MatMulRightTransposeInto(right, result)
		if err != nil {
			b.Fatalf("MatMulRightTransposeInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_Clone(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = source.Clone()
		if err != nil {
			b.Fatalf("Clone returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_Values(b *testing.B) {
	var (
		source *matrix.Matrix
		values []float32
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		values, err = source.Values()
		if err != nil {
			b.Fatalf("Values returned error: %v", err)
		}
	}

	benchmarkValues = values
}

func Benchmark_Add(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 256, 256)
	right = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = left.Add(right)
		if err != nil {
			b.Fatalf("Add returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_AddInto(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 256, 256)
	right = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(256, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.AddInto(right, result)
		if err != nil {
			b.Fatalf("AddInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_AddInPlace(b *testing.B) {
	var (
		left  *matrix.Matrix
		right *matrix.Matrix
		err   error
		index int
	)

	left = benchmarkMatrix(b, 256, 256)
	right = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.AddInPlace(right)
		if err != nil {
			b.Fatalf("AddInPlace returned error: %v", err)
		}
	}

	benchmarkResult = left
}

func Benchmark_AddScaledInPlace(b *testing.B) {
	var (
		left  *matrix.Matrix
		right *matrix.Matrix
		err   error
		index int
	)

	left = benchmarkMatrix(b, 256, 256)
	right = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.AddScaledInPlace(right, 0.125)
		if err != nil {
			b.Fatalf("AddScaledInPlace returned error: %v", err)
		}
	}

	benchmarkResult = left
}

func Benchmark_Subtract(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 256, 256)
	right = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = left.Subtract(right)
		if err != nil {
			b.Fatalf("Subtract returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_SubtractInto(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 256, 256)
	right = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(256, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.SubtractInto(right, result)
		if err != nil {
			b.Fatalf("SubtractInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_MultiplyElements(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 256, 256)
	right = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = left.MultiplyElements(right)
		if err != nil {
			b.Fatalf("MultiplyElements returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_MultiplyElementsInto(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, 256, 256)
	right = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(256, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.MultiplyElementsInto(right, result)
		if err != nil {
			b.Fatalf("MultiplyElementsInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_DivideElements(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkPositiveMatrix(b, 256, 256)
	right = benchmarkPositiveMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = left.DivideElements(right)
		if err != nil {
			b.Fatalf("DivideElements returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_DivideElementsInto(b *testing.B) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkPositiveMatrix(b, 256, 256)
	right = benchmarkPositiveMatrix(b, 256, 256)
	result, err = matrix.New(256, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.DivideElementsInto(right, result)
		if err != nil {
			b.Fatalf("DivideElementsInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_AddScalar(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = source.AddScalar(0.125)
		if err != nil {
			b.Fatalf("AddScalar returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_AddScalarInto(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(256, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.AddScalarInto(0.125, result)
		if err != nil {
			b.Fatalf("AddScalarInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_MultiplyScalar(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = source.MultiplyScalar(1.125)
		if err != nil {
			b.Fatalf("MultiplyScalar returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_MultiplyScalarInto(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(256, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.MultiplyScalarInto(1.125, result)
		if err != nil {
			b.Fatalf("MultiplyScalarInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_MultiplyScalarInPlace(b *testing.B) {
	var (
		source *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.MultiplyScalarInPlace(0.999999)
		if err != nil {
			b.Fatalf("MultiplyScalarInPlace returned error: %v", err)
		}
	}

	benchmarkResult = source
}

func Benchmark_DivideScalar(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = source.DivideScalar(1.125)
		if err != nil {
			b.Fatalf("DivideScalar returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_DivideScalarInto(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(256, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.DivideScalarInto(1.125, result)
		if err != nil {
			b.Fatalf("DivideScalarInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_Transpose(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 128, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = source.Transpose()
		if err != nil {
			b.Fatalf("Transpose returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_TransposeInto(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 128, 256)
	result, err = matrix.New(256, 128)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.TransposeInto(result)
		if err != nil {
			b.Fatalf("TransposeInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_RowSums(b *testing.B) {
	var (
		source *matrix.Matrix
		values []float32
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		values, err = source.RowSums()
		if err != nil {
			b.Fatalf("RowSums returned error: %v", err)
		}
	}

	benchmarkValues = values
}

func Benchmark_RowSumsInto(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(256, 1)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.RowSumsInto(result)
		if err != nil {
			b.Fatalf("RowSumsInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_ColumnSums(b *testing.B) {
	var (
		source *matrix.Matrix
		values []float32
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		values, err = source.ColumnSums()
		if err != nil {
			b.Fatalf("ColumnSums returned error: %v", err)
		}
	}

	benchmarkValues = values
}

func Benchmark_ColumnSumsInto(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(1, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.ColumnSumsInto(result)
		if err != nil {
			b.Fatalf("ColumnSumsInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_AccumulateColumnSumsInto(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(1, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.AccumulateColumnSumsInto(result)
		if err != nil {
			b.Fatalf("AccumulateColumnSumsInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_AddRowVectorInPlace(b *testing.B) {
	var (
		source    *matrix.Matrix
		rowVector *matrix.Matrix
		err       error
		index     int
	)

	source = benchmarkMatrix(b, 256, 256)
	rowVector = benchmarkMatrix(b, 1, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.AddRowVectorInPlace(rowVector)
		if err != nil {
			b.Fatalf("AddRowVectorInPlace returned error: %v", err)
		}
	}

	benchmarkResult = source
}

func Benchmark_Apply(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = source.Apply(benchmarkApply)
		if err != nil {
			b.Fatalf("Apply returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_ApplyInto(b *testing.B) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, 256, 256)
	result, err = matrix.New(256, 256)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = source.ApplyInto(benchmarkApply, result)
		if err != nil {
			b.Fatalf("ApplyInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func Benchmark_MatMulShapes(b *testing.B) {
	type testcase struct {
		name      string
		leftRows  int
		leftCols  int
		rightCols int
	}

	tests := []testcase{
		{name: "Small2x2", leftRows: 2, leftCols: 2, rightCols: 2},
		{name: "Small4x4", leftRows: 4, leftCols: 4, rightCols: 4},
		{name: "Medium64x64", leftRows: 64, leftCols: 64, rightCols: 64},
		{name: "Large128x256x128", leftRows: 128, leftCols: 256, rightCols: 128},
		{name: "Uneven17x33x19", leftRows: 17, leftCols: 33, rightCols: 19},
		{name: "Uneven63x65x31", leftRows: 63, leftCols: 65, rightCols: 31},
	}

	var tt testcase
	for _, tt = range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkMatMulShape(b, tt.leftRows, tt.leftCols, tt.rightCols)
		})
	}
}

func Benchmark_MatMulRightTransposeIntoShapes(b *testing.B) {
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
			benchmarkMatMulRightTransposeShape(b, tt.leftRows, tt.leftCols, tt.rightRows)
		})
	}
}

func benchmarkMatMulShape(b *testing.B, leftRows, leftCols, rightCols int) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, leftRows, leftCols)
	right = benchmarkMatrix(b, leftCols, rightCols)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		result, err = left.MatMul(right)
		if err != nil {
			b.Fatalf("MatMul returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func benchmarkMatMulRightTransposeShape(b *testing.B, leftRows, leftCols, rightRows int) {
	var (
		left   *matrix.Matrix
		right  *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	left = benchmarkMatrix(b, leftRows, leftCols)
	right = benchmarkMatrix(b, rightRows, leftCols)
	result, err = matrix.New(leftRows, rightRows)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		err = left.MatMulRightTransposeInto(right, result)
		if err != nil {
			b.Fatalf("MatMulRightTransposeInto returned error: %v", err)
		}
	}

	benchmarkResult = result
}

func benchmarkMatrix(tb testing.TB, rows, cols int) (m *matrix.Matrix) {
	tb.Helper()

	var (
		values []float32
		err    error
		index  int
	)

	values = make([]float32, rows*cols)
	for index = range values {
		values[index] = float32(index%31) / 31
	}

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func benchmarkPositiveMatrix(tb testing.TB, rows, cols int) (m *matrix.Matrix) {
	tb.Helper()

	var (
		values []float32
		err    error
		index  int
	)

	values = make([]float32, rows*cols)
	for index = range values {
		values[index] = 1 + float32(index%31)/31
	}

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func benchmarkApply(value float32) (out float32) {
	out = value * 1.125
	return out
}
