package layer_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var benchmarkDenseResult *matrix.Matrix

func Benchmark_DenseForward_XOR(b *testing.B) {
	var (
		dense  *layer.Dense
		input  *matrix.Matrix
		output *matrix.Matrix
		err    error
		index  int
	)

	dense = benchmarkDense(b, 2, 4)
	input = benchmarkLayerMatrix(b, 4, 2)
	output, err = dense.Forward(input)
	if err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		output, err = dense.Forward(input)
		if err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkDenseResult = output
}

func Benchmark_DenseForward_MediumBatch(b *testing.B) {
	var (
		dense  *layer.Dense
		input  *matrix.Matrix
		output *matrix.Matrix
		err    error
		index  int
	)

	dense = benchmarkDense(b, 32, 64)
	input = benchmarkLayerMatrix(b, 128, 32)
	output, err = dense.Forward(input)
	if err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		output, err = dense.Forward(input)
		if err != nil {
			b.Fatalf("Forward returned error: %v", err)
		}
	}

	benchmarkDenseResult = output
}

func Benchmark_DenseBackward_XOR(b *testing.B) {
	var (
		dense          *layer.Dense
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
		index          int
	)

	dense = benchmarkDense(b, 2, 4)
	input = benchmarkLayerMatrix(b, 4, 2)
	outputGradient = benchmarkLayerMatrix(b, 4, 4)
	if _, err = dense.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}
	inputGradient, err = dense.Backward(outputGradient)
	if err != nil {
		b.Fatalf("Backward returned error: %v", err)
	}
	err = dense.ResetGradients()
	if err != nil {
		b.Fatalf("ResetGradients returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		inputGradient, err = dense.Backward(outputGradient)
		if err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkDenseResult = inputGradient
}

func Benchmark_DenseBackward_MediumBatch(b *testing.B) {
	var (
		dense          *layer.Dense
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
		index          int
	)

	dense = benchmarkDense(b, 32, 64)
	input = benchmarkLayerMatrix(b, 128, 32)
	outputGradient = benchmarkLayerMatrix(b, 128, 64)
	if _, err = dense.Forward(input); err != nil {
		b.Fatalf("Forward returned error: %v", err)
	}
	inputGradient, err = dense.Backward(outputGradient)
	if err != nil {
		b.Fatalf("Backward returned error: %v", err)
	}
	err = dense.ResetGradients()
	if err != nil {
		b.Fatalf("ResetGradients returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		inputGradient, err = dense.Backward(outputGradient)
		if err != nil {
			b.Fatalf("Backward returned error: %v", err)
		}
	}

	benchmarkDenseResult = inputGradient
}

func benchmarkDense(tb testing.TB, inputSize, outputSize int) (dense *layer.Dense) {
	var err error

	tb.Helper()

	dense, err = layer.NewDense(inputSize, outputSize, func(layerInputSize, layerOutputSize int) (weights *matrix.Matrix, err error) {
		weights = benchmarkLayerMatrix(tb, layerInputSize, layerOutputSize)
		return weights, nil
	})
	if err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	return dense
}

func benchmarkLayerMatrix(tb testing.TB, rows, cols int) (m *matrix.Matrix) {
	var (
		values []float32
		err    error
		index  int
	)

	tb.Helper()

	values = make([]float32, rows*cols)
	for index = range values {
		values[index] = float32(index%29)/29 - 0.5
	}

	m, err = matrix.FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}
