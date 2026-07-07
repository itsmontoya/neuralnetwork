package matrix_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Benchmark_ReductionCandidates(b *testing.B) {
	type testcase struct {
		name string
		rows int
		cols int
	}

	tests := []testcase{
		{name: "Small1x1", rows: 1, cols: 1},
		{name: "Small1x3", rows: 1, cols: 3},
		{name: "Small3x1", rows: 3, cols: 1},
		{name: "Medium64x64", rows: 64, cols: 64},
		{name: "Medium128x256", rows: 128, cols: 256},
		{name: "DenseBias128x64", rows: 128, cols: 64},
		{name: "Large512x512", rows: 512, cols: 512},
		{name: "Uneven17x257", rows: 17, cols: 257},
		{name: "Uneven257x17", rows: 257, cols: 17},
	}

	var tt testcase
	for _, tt = range tests {
		b.Run(tt.name, func(b *testing.B) {
			b.Run("RowSumsInto", func(b *testing.B) {
				benchmarkReductionRowSumsInto(b, tt.rows, tt.cols)
			})
			b.Run("ColumnSumsInto", func(b *testing.B) {
				benchmarkReductionColumnSumsInto(b, tt.rows, tt.cols)
			})
			b.Run("AccumulateColumnSumsInto", func(b *testing.B) {
				benchmarkReductionAccumulateColumnSumsInto(b, tt.rows, tt.cols)
			})
		})
	}
}

func benchmarkReductionRowSumsInto(b *testing.B, rows, cols int) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, rows, cols)
	result, err = matrix.New(rows, 1)
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

func benchmarkReductionColumnSumsInto(b *testing.B, rows, cols int) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, rows, cols)
	result, err = matrix.New(1, cols)
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

func benchmarkReductionAccumulateColumnSumsInto(b *testing.B, rows, cols int) {
	var (
		source *matrix.Matrix
		result *matrix.Matrix
		err    error
		index  int
	)

	source = benchmarkMatrix(b, rows, cols)
	result, err = matrix.New(1, cols)
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
