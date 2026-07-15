package matrix_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Benchmark_SoftmaxRowsInto_MediumBatch(b *testing.B) {
	var (
		input  *matrix.Matrix
		output *matrix.Matrix
		err    error
		index  int
	)

	input = benchmarkMatrix(b, 128, 64)
	output, err = matrix.New(128, 64)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if err = input.SoftmaxRowsInto(output); err != nil {
			b.Fatalf("SoftmaxRowsInto returned error: %v", err)
		}
	}

	benchmarkResult = output
}

func Benchmark_SoftmaxRowsBackwardInto_MediumBatch(b *testing.B) {
	var (
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
		index          int
	)

	input = benchmarkMatrix(b, 128, 64)
	outputGradient = benchmarkMatrix(b, 128, 64)
	inputGradient, err = matrix.New(128, 64)
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if err = input.SoftmaxRowsBackwardInto(outputGradient, inputGradient); err != nil {
			b.Fatalf("SoftmaxRowsBackwardInto returned error: %v", err)
		}
	}

	benchmarkResult = inputGradient
}
