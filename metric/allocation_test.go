package metric_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
)

var allocationMetricValue float32
var allocationConfusionMatrix *metric.ConfusionMatrix
var allocationConfusionCounts [][]int

func Test_ScalarMetricAllocationCeilings(t *testing.T) {
	var tests []struct {
		name        string
		metricRule  metric.Metric
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}

	tests = []struct {
		name        string
		metricRule  metric.Metric
		predictions *matrix.Matrix
		targets     *matrix.Matrix
	}{
		{
			name:        "MeanSquaredError",
			metricRule:  metric.MeanSquaredError{},
			predictions: mustMatrix(t, 2, 2, []float32{0.1, 0.2, 0.3, 0.4}),
			targets:     mustMatrix(t, 2, 2, []float32{0, 0.25, 0.5, 0.75}),
		},
		{
			name:        "BinaryAccuracy",
			metricRule:  metric.BinaryAccuracy{},
			predictions: mustMatrix(t, 4, 1, []float32{0.1, 0.8, 0.25, 0.75}),
			targets:     mustMatrix(t, 4, 1, []float32{0, 1, 0, 1}),
		},
		{
			name:        "BinaryPrecision",
			metricRule:  metric.BinaryPrecision{},
			predictions: mustMatrix(t, 4, 1, []float32{0.1, 0.8, 0.25, 0.75}),
			targets:     mustMatrix(t, 4, 1, []float32{0, 1, 0, 1}),
		},
		{
			name:        "BinaryRecall",
			metricRule:  metric.BinaryRecall{},
			predictions: mustMatrix(t, 4, 1, []float32{0.1, 0.8, 0.25, 0.75}),
			targets:     mustMatrix(t, 4, 1, []float32{0, 1, 0, 1}),
		},
		{
			name:        "BinaryF1",
			metricRule:  metric.BinaryF1{},
			predictions: mustMatrix(t, 4, 1, []float32{0.1, 0.8, 0.25, 0.75}),
			targets:     mustMatrix(t, 4, 1, []float32{0, 1, 0, 1}),
		},
		{
			name:       "CategoricalAccuracy",
			metricRule: metric.CategoricalAccuracy{},
			predictions: mustMatrix(t, 2, 3, []float32{
				0.7, 0.2, 0.1,
				0.1, 0.8, 0.1,
			}),
			targets: mustMatrix(t, 2, 3, []float32{
				1, 0, 0,
				0, 1, 0,
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			requireMaxAllocs(t, tt.name, 0, func() {
				allocationMetricValue, err = tt.metricRule.Value(tt.predictions, tt.targets)
				if err != nil {
					panic(err)
				}
			})
		})
	}
}

func Test_CategoricalMacroMetricAllocationCeilings(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		tests       []struct {
			name       string
			metricRule metric.Metric
		}
	)

	predictions = mustMatrix(t, 2, 3, []float32{
		0.7, 0.2, 0.1,
		0.1, 0.8, 0.1,
	})
	targets = mustMatrix(t, 2, 3, []float32{
		1, 0, 0,
		0, 1, 0,
	})
	tests = []struct {
		name       string
		metricRule metric.Metric
	}{
		{name: "CategoricalMacroPrecision", metricRule: metric.CategoricalMacroPrecision{}},
		{name: "CategoricalMacroRecall", metricRule: metric.CategoricalMacroRecall{}},
		{name: "CategoricalMacroF1", metricRule: metric.CategoricalMacroF1{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			requireMaxAllocs(t, tt.name, 1, func() {
				allocationMetricValue, err = tt.metricRule.Value(predictions, targets)
				if err != nil {
					panic(err)
				}
			})
		})
	}
}

func Test_ConfusionMatrixAllocationCeilings(t *testing.T) {
	var tests []struct {
		name        string
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		construct   func(*matrix.Matrix, *matrix.Matrix) (*metric.ConfusionMatrix, error)
		countAllocs float64
	}

	tests = []struct {
		name        string
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		construct   func(*matrix.Matrix, *matrix.Matrix) (*metric.ConfusionMatrix, error)
		countAllocs float64
	}{
		{
			name:        "Binary",
			predictions: mustMatrix(t, 4, 1, []float32{0.1, 0.8, 0.25, 0.75}),
			targets:     mustMatrix(t, 4, 1, []float32{0, 1, 0, 1}),
			construct:   metric.NewBinaryConfusionMatrix,
			countAllocs: 3,
		},
		{
			name: "Categorical",
			predictions: mustMatrix(t, 2, 3, []float32{
				0.7, 0.2, 0.1,
				0.1, 0.8, 0.1,
			}),
			targets: mustMatrix(t, 2, 3, []float32{
				1, 0, 0,
				0, 1, 0,
			}),
			construct:   metric.NewCategoricalConfusionMatrix,
			countAllocs: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error

			requireMaxAllocs(t, tt.name+" constructor", 2, func() {
				allocationConfusionMatrix, err = tt.construct(tt.predictions, tt.targets)
				if err != nil {
					panic(err)
				}
			})

			allocationConfusionMatrix, err = tt.construct(tt.predictions, tt.targets)
			if err != nil {
				t.Fatalf("confusion matrix constructor returned error: %v", err)
			}

			requireMaxAllocs(t, tt.name+" Counts", tt.countAllocs, func() {
				allocationConfusionCounts, err = allocationConfusionMatrix.Counts()
				if err != nil {
					panic(err)
				}
			})
		})
	}
}

func requireMaxAllocs(tb testing.TB, name string, max float64, run func()) {
	var got float64

	tb.Helper()

	got = testing.AllocsPerRun(100, run)
	if got > max {
		tb.Fatalf("%s allocations = %.0f, want <= %.0f", name, got, max)
	}
}
