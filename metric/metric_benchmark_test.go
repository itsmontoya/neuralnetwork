package metric_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
)

var benchmarkMetricValue float32
var benchmarkConfusionMatrix *metric.ConfusionMatrix
var benchmarkConfusionCounts [][]int

func Benchmark_MetricValue(b *testing.B) {
	var tests []struct {
		name       string
		metric     metric.Metric
		matrixPair func(testing.TB, int) (*matrix.Matrix, *matrix.Matrix)
	}

	tests = []struct {
		name       string
		metric     metric.Metric
		matrixPair func(testing.TB, int) (*matrix.Matrix, *matrix.Matrix)
	}{
		{name: "MeanSquaredError", metric: metric.MeanSquaredError{}, matrixPair: benchmarkRegressionMatrices},
		{name: "BinaryAccuracy", metric: metric.BinaryAccuracy{}, matrixPair: benchmarkBinaryMatrices},
		{name: "BinaryPrecision", metric: metric.BinaryPrecision{}, matrixPair: benchmarkBinaryMatrices},
		{name: "BinaryRecall", metric: metric.BinaryRecall{}, matrixPair: benchmarkBinaryMatrices},
		{name: "BinaryF1", metric: metric.BinaryF1{}, matrixPair: benchmarkBinaryMatrices},
		{name: "CategoricalAccuracy", metric: metric.CategoricalAccuracy{}, matrixPair: benchmarkCategoricalMatrices},
		{name: "CategoricalMacroPrecision", metric: metric.CategoricalMacroPrecision{}, matrixPair: benchmarkCategoricalMatrices},
		{name: "CategoricalMacroRecall", metric: metric.CategoricalMacroRecall{}, matrixPair: benchmarkCategoricalMatrices},
		{name: "CategoricalMacroF1", metric: metric.CategoricalMacroF1{}, matrixPair: benchmarkCategoricalMatrices},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			benchmarkMetricShapes(b, tt.metric, tt.matrixPair)
		})
	}
}

func Benchmark_ConfusionMatrixConstruction(b *testing.B) {
	var tests []struct {
		name       string
		matrixPair func(testing.TB, int) (*matrix.Matrix, *matrix.Matrix)
		construct  func(*matrix.Matrix, *matrix.Matrix) (*metric.ConfusionMatrix, error)
	}

	tests = []struct {
		name       string
		matrixPair func(testing.TB, int) (*matrix.Matrix, *matrix.Matrix)
		construct  func(*matrix.Matrix, *matrix.Matrix) (*metric.ConfusionMatrix, error)
	}{
		{name: "Binary", matrixPair: benchmarkBinaryMatrices, construct: metric.NewBinaryConfusionMatrix},
		{name: "Categorical", matrixPair: benchmarkCategoricalMatrices, construct: metric.NewCategoricalConfusionMatrix},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			var shapes []struct {
				name string
				rows int
			}

			shapes = benchmarkMetricShapesTable()
			for _, shape := range shapes {
				b.Run(shape.name, func(b *testing.B) {
					var (
						predictions     *matrix.Matrix
						targets         *matrix.Matrix
						confusionMatrix *metric.ConfusionMatrix
						err             error
						index           int
					)

					predictions, targets = tt.matrixPair(b, shape.rows)
					if confusionMatrix, err = tt.construct(predictions, targets); err != nil {
						b.Fatalf("confusion matrix constructor returned error: %v", err)
					}

					b.ReportAllocs()
					b.ResetTimer()

					for index = 0; index < b.N; index++ {
						if confusionMatrix, err = tt.construct(predictions, targets); err != nil {
							b.Fatalf("confusion matrix constructor returned error: %v", err)
						}
					}

					benchmarkConfusionMatrix = confusionMatrix
				})
			}
		})
	}
}

func Benchmark_ConfusionMatrixCounts_ColdPath(b *testing.B) {
	var (
		predictions     *matrix.Matrix
		targets         *matrix.Matrix
		confusionMatrix *metric.ConfusionMatrix
		counts          [][]int
		err             error
		index           int
	)

	predictions, targets = benchmarkCategoricalMatrices(b, 128)
	if confusionMatrix, err = metric.NewCategoricalConfusionMatrix(predictions, targets); err != nil {
		b.Fatalf("NewCategoricalConfusionMatrix returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		counts, err = confusionMatrix.Counts()
		if err != nil {
			b.Fatalf("Counts returned error: %v", err)
		}
	}

	benchmarkConfusionCounts = counts
}

func benchmarkMetricShapes(
	b *testing.B,
	metricRule metric.Metric,
	matrixPair func(testing.TB, int) (*matrix.Matrix, *matrix.Matrix),
) {
	var shapes []struct {
		name string
		rows int
	}

	shapes = benchmarkMetricShapesTable()
	for _, shape := range shapes {
		b.Run(shape.name, func(b *testing.B) {
			var (
				predictions *matrix.Matrix
				targets     *matrix.Matrix
				value       float32
				err         error
				index       int
			)

			predictions, targets = matrixPair(b, shape.rows)
			if value, err = metricRule.Value(predictions, targets); err != nil {
				b.Fatalf("Value returned error: %v", err)
			}

			b.ReportAllocs()
			b.ResetTimer()

			for index = 0; index < b.N; index++ {
				if value, err = metricRule.Value(predictions, targets); err != nil {
					b.Fatalf("Value returned error: %v", err)
				}
			}

			benchmarkMetricValue = value
		})
	}
}

func benchmarkMetricShapesTable() (shapes []struct {
	name string
	rows int
}) {
	shapes = []struct {
		name string
		rows int
	}{
		{name: "Small", rows: 4},
		{name: "Medium", rows: 128},
	}
	return shapes
}

func benchmarkRegressionMatrices(tb testing.TB, rows int) (predictions, targets *matrix.Matrix) {
	var (
		predictionValues []float32
		targetValues     []float32
		cols             int
		index            int
	)

	tb.Helper()

	cols = 16
	predictionValues = make([]float32, rows*cols)
	targetValues = make([]float32, rows*cols)
	for index = range predictionValues {
		predictionValues[index] = float32(index%29) / 29
		targetValues[index] = float32((index+7)%31) / 31
	}

	predictions = mustMatrix(tb, rows, cols, predictionValues)
	targets = mustMatrix(tb, rows, cols, targetValues)
	return predictions, targets
}

func benchmarkBinaryMatrices(tb testing.TB, rows int) (predictions, targets *matrix.Matrix) {
	var (
		predictionValues []float32
		targetValues     []float32
		row              int
	)

	tb.Helper()

	predictionValues = make([]float32, rows)
	targetValues = make([]float32, rows)
	for row = 0; row < rows; row++ {
		predictionValues[row] = float32((row*7)%101) / 100
		targetValues[row] = float32(row % 2)
	}

	predictions = mustMatrix(tb, rows, 1, predictionValues)
	targets = mustMatrix(tb, rows, 1, targetValues)
	return predictions, targets
}

func benchmarkCategoricalMatrices(tb testing.TB, rows int) (predictions, targets *matrix.Matrix) {
	var (
		predictionValues []float32
		targetValues     []float32
		cols             int
		row              int
		col              int
	)

	tb.Helper()

	cols = 16
	predictionValues = make([]float32, rows*cols)
	targetValues = make([]float32, rows*cols)
	for row = 0; row < rows; row++ {
		for col = 0; col < cols; col++ {
			predictionValues[row*cols+col] = float32((row*3+col*7)%37) / 37
		}

		targetValues[row*cols+(row%cols)] = 1
	}

	predictions = mustMatrix(tb, rows, cols, predictionValues)
	targets = mustMatrix(tb, rows, cols, targetValues)
	return predictions, targets
}
