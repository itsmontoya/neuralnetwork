package loss_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var benchmarkLossValue float32
var benchmarkLossGradient *matrix.Matrix

func Benchmark_MeanSquaredErrorValue_Small(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkRegressionMatrices(b, 2, 2)
	benchmarkLossValueMethod(b, loss.MeanSquaredError{}, predictions, targets)
}

func Benchmark_MeanSquaredErrorValue_MediumBatch(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkRegressionMatrices(b, 128, 16)
	benchmarkLossValueMethod(b, loss.MeanSquaredError{}, predictions, targets)
}

func Benchmark_MeanSquaredErrorGradient_Small(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkRegressionMatrices(b, 2, 2)
	benchmarkLossGradientMethod(b, loss.MeanSquaredError{}, predictions, targets)
}

func Benchmark_MeanSquaredErrorGradient_MediumBatch(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkRegressionMatrices(b, 128, 16)
	benchmarkLossGradientMethod(b, loss.MeanSquaredError{}, predictions, targets)
}

func Benchmark_MeanSquaredErrorGradientInto_Small(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	predictions, targets = benchmarkRegressionMatrices(b, 2, 2)
	if destination, err = matrix.New(2, 2); err != nil {
		b.Fatalf("New destination returned error: %v", err)
	}
	benchmarkLossGradientIntoMethod(b, loss.MeanSquaredError{}, predictions, targets, destination)
}

func Benchmark_MeanSquaredErrorGradientInto_MediumBatch(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	predictions, targets = benchmarkRegressionMatrices(b, 128, 16)
	if destination, err = matrix.New(128, 16); err != nil {
		b.Fatalf("New destination returned error: %v", err)
	}
	benchmarkLossGradientIntoMethod(b, loss.MeanSquaredError{}, predictions, targets, destination)
}

func Benchmark_BinaryCrossEntropyValue_Small(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkBinaryMatrices(b, 4)
	benchmarkLossValueMethod(b, loss.BinaryCrossEntropy{}, predictions, targets)
}

func Benchmark_BinaryCrossEntropyValue_MediumBatch(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkBinaryMatrices(b, 128)
	benchmarkLossValueMethod(b, loss.BinaryCrossEntropy{}, predictions, targets)
}

func Benchmark_BinaryCrossEntropyGradient_Small(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkBinaryMatrices(b, 4)
	benchmarkLossGradientMethod(b, loss.BinaryCrossEntropy{}, predictions, targets)
}

func Benchmark_BinaryCrossEntropyGradient_MediumBatch(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkBinaryMatrices(b, 128)
	benchmarkLossGradientMethod(b, loss.BinaryCrossEntropy{}, predictions, targets)
}

func Benchmark_BinaryCrossEntropyGradientInto_Small(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	predictions, targets = benchmarkBinaryMatrices(b, 4)
	if destination, err = matrix.New(4, 1); err != nil {
		b.Fatalf("New destination returned error: %v", err)
	}
	benchmarkLossGradientIntoMethod(b, loss.BinaryCrossEntropy{}, predictions, targets, destination)
}

func Benchmark_BinaryCrossEntropyGradientInto_MediumBatch(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	predictions, targets = benchmarkBinaryMatrices(b, 128)
	if destination, err = matrix.New(128, 1); err != nil {
		b.Fatalf("New destination returned error: %v", err)
	}
	benchmarkLossGradientIntoMethod(b, loss.BinaryCrossEntropy{}, predictions, targets, destination)
}

func Benchmark_CategoricalCrossEntropyValue_Small(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkCategoricalMatrices(b, 4, 3)
	benchmarkLossValueMethod(b, loss.CategoricalCrossEntropy{}, predictions, targets)
}

func Benchmark_CategoricalCrossEntropyValue_MediumBatch(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkCategoricalMatrices(b, 128, 16)
	benchmarkLossValueMethod(b, loss.CategoricalCrossEntropy{}, predictions, targets)
}

func Benchmark_CategoricalCrossEntropyGradient_Small(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkCategoricalMatrices(b, 4, 3)
	benchmarkLossGradientMethod(b, loss.CategoricalCrossEntropy{}, predictions, targets)
}

func Benchmark_CategoricalCrossEntropyGradient_MediumBatch(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	)

	predictions, targets = benchmarkCategoricalMatrices(b, 128, 16)
	benchmarkLossGradientMethod(b, loss.CategoricalCrossEntropy{}, predictions, targets)
}

func Benchmark_CategoricalCrossEntropyGradientInto_Small(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	predictions, targets = benchmarkCategoricalMatrices(b, 4, 3)
	if destination, err = matrix.New(4, 3); err != nil {
		b.Fatalf("New destination returned error: %v", err)
	}
	benchmarkLossGradientIntoMethod(b, loss.CategoricalCrossEntropy{}, predictions, targets, destination)
}

func Benchmark_CategoricalCrossEntropyGradientInto_MediumBatch(b *testing.B) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		destination *matrix.Matrix
		err         error
	)

	predictions, targets = benchmarkCategoricalMatrices(b, 128, 16)
	if destination, err = matrix.New(128, 16); err != nil {
		b.Fatalf("New destination returned error: %v", err)
	}
	benchmarkLossGradientIntoMethod(b, loss.CategoricalCrossEntropy{}, predictions, targets, destination)
}

func benchmarkLossValueMethod(b *testing.B, lossFunc loss.Loss, predictions, targets *matrix.Matrix) {
	var (
		value float32
		err   error
		index int
	)

	b.Helper()
	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		value, err = lossFunc.Value(predictions, targets)
		if err != nil {
			b.Fatalf("Value returned error: %v", err)
		}
	}

	benchmarkLossValue = value
}

func benchmarkLossGradientMethod(b *testing.B, lossFunc loss.Loss, predictions, targets *matrix.Matrix) {
	var (
		gradient *matrix.Matrix
		err      error
		index    int
	)

	b.Helper()
	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		gradient, err = lossFunc.Gradient(predictions, targets)
		if err != nil {
			b.Fatalf("Gradient returned error: %v", err)
		}
	}

	benchmarkLossGradient = gradient
}

func benchmarkLossGradientIntoMethod(
	b *testing.B,
	lossFunc loss.DestinationGradient,
	predictions,
	targets,
	destination *matrix.Matrix,
) {
	var (
		err   error
		index int
	)

	b.Helper()
	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if err = lossFunc.GradientInto(predictions, targets, destination); err != nil {
			b.Fatalf("GradientInto returned error: %v", err)
		}
	}

	benchmarkLossGradient = destination
}

func benchmarkRegressionMatrices(tb testing.TB, rows, cols int) (predictions, targets *matrix.Matrix) {
	var (
		predictionValues []float32
		targetValues     []float32
		size             int
		index            int
		err              error
	)

	tb.Helper()

	size = rows * cols
	predictionValues = make([]float32, size)
	targetValues = make([]float32, size)
	for index = 0; index < size; index++ {
		predictionValues[index] = float32((index%17)+1) / 20
		targetValues[index] = float32((index%13)+2) / 18
	}

	predictions, err = matrix.FromSlice(rows, cols, predictionValues)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	targets, err = matrix.FromSlice(rows, cols, targetValues)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return predictions, targets
}

func benchmarkBinaryMatrices(tb testing.TB, rows int) (predictions, targets *matrix.Matrix) {
	var (
		predictionValues []float32
		targetValues     []float32
		index            int
		err              error
	)

	tb.Helper()

	predictionValues = make([]float32, rows)
	targetValues = make([]float32, rows)
	for index = 0; index < rows; index++ {
		predictionValues[index] = 0.1 + float32(index%9)/10
		if index%2 == 0 {
			targetValues[index] = 1
			continue
		}

		targetValues[index] = 0
	}

	predictions, err = matrix.FromSlice(rows, 1, predictionValues)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	targets, err = matrix.FromSlice(rows, 1, targetValues)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return predictions, targets
}

func benchmarkCategoricalMatrices(tb testing.TB, rows, cols int) (predictions, targets *matrix.Matrix) {
	var (
		predictionValues []float32
		targetValues     []float32
		row              int
		col              int
		index            int
		selected         int
		err              error
	)

	tb.Helper()

	predictionValues = make([]float32, rows*cols)
	targetValues = make([]float32, rows*cols)
	for row = 0; row < rows; row++ {
		selected = row % cols
		for col = 0; col < cols; col++ {
			index = row*cols + col
			predictionValues[index] = 0.02
			if col == selected {
				predictionValues[index] = 0.7
				targetValues[index] = 1
			}
		}
	}

	predictions, err = matrix.FromSlice(rows, cols, predictionValues)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	targets, err = matrix.FromSlice(rows, cols, targetValues)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return predictions, targets
}
