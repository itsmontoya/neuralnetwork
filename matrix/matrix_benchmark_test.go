package matrix_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

var benchmarkResult *matrix.Matrix

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

func benchmarkMatrix(tb testing.TB, rows, cols int) (m *matrix.Matrix) {
	tb.Helper()

	var (
		values []float64
		err    error
		index  int
	)

	values = make([]float64, rows*cols)
	for index = range values {
		values[index] = float64(index%31) / 31
	}

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}
