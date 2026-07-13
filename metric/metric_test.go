package metric_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
)

func Test_Metric_Interface(t *testing.T) {
	var _ metric.Metric = metric.MeanSquaredError{}
	var _ metric.Metric = metric.BinaryAccuracy{}
	var _ metric.Metric = metric.BinaryPrecision{}
	var _ metric.Metric = metric.BinaryRecall{}
	var _ metric.Metric = metric.BinaryF1{}
	var _ metric.Metric = metric.CategoricalAccuracy{}
	var _ metric.Metric = metric.CategoricalMacroPrecision{}
	var _ metric.Metric = metric.CategoricalMacroRecall{}
	var _ metric.Metric = metric.CategoricalMacroF1{}
	var _ metric.Metric = mockMetric{}
}

type mockMetric struct{}

func (m mockMetric) Value(predictions, targets *matrix.Matrix) (value float32, err error) {
	value = 0
	return value, nil
}
