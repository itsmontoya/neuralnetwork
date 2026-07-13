package metric_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
)

func Test_BinaryClassificationMetrics(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		value       float32
		err         error
	)

	predictions = mustMatrix(t, 6, 1, []float32{0.9, 0.8, 0.7, 0.1, 0.2, 0.3})
	targets = mustMatrix(t, 6, 1, []float32{1, 0, 1, 1, 0, 0})

	value, err = metric.BinaryPrecision{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("BinaryPrecision returned error: %v", err)
	}
	requireAlmostEqual(t, value, 2.0/3.0)

	value, err = metric.BinaryRecall{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("BinaryRecall returned error: %v", err)
	}
	requireAlmostEqual(t, value, 2.0/3.0)

	value, err = metric.BinaryF1{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("BinaryF1 returned error: %v", err)
	}
	requireAlmostEqual(t, value, 2.0/3.0)
}

func Test_NewBinaryClassificationMetrics_UseCustomThreshold(t *testing.T) {
	var (
		precision   metric.BinaryPrecision
		recall      metric.BinaryRecall
		f1          metric.BinaryF1
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		value       float32
		err         error
	)

	precision, err = metric.NewBinaryPrecision(0.75)
	if err != nil {
		t.Fatalf("NewBinaryPrecision returned error: %v", err)
	}

	recall, err = metric.NewBinaryRecall(0.75)
	if err != nil {
		t.Fatalf("NewBinaryRecall returned error: %v", err)
	}

	f1, err = metric.NewBinaryF1(0.75)
	if err != nil {
		t.Fatalf("NewBinaryF1 returned error: %v", err)
	}

	predictions = mustMatrix(t, 3, 1, []float32{0.74, 0.75, 0.76})
	targets = mustMatrix(t, 3, 1, []float32{0, 1, 1})

	value, err = precision.Value(predictions, targets)
	if err != nil {
		t.Fatalf("BinaryPrecision returned error: %v", err)
	}
	requireAlmostEqual(t, value, 1)

	value, err = recall.Value(predictions, targets)
	if err != nil {
		t.Fatalf("BinaryRecall returned error: %v", err)
	}
	requireAlmostEqual(t, value, 1)

	value, err = f1.Value(predictions, targets)
	if err != nil {
		t.Fatalf("BinaryF1 returned error: %v", err)
	}
	requireAlmostEqual(t, value, 1)
}

func Test_NewBinaryClassificationMetrics_RejectNonFiniteThreshold(t *testing.T) {
	var err error

	_, err = metric.NewBinaryPrecision(float32(math.NaN()))
	if err == nil {
		t.Fatal("NewBinaryPrecision error = nil, want error")
	}

	_, err = metric.NewBinaryRecall(float32(math.NaN()))
	if err == nil {
		t.Fatal("NewBinaryRecall error = nil, want error")
	}

	_, err = metric.NewBinaryF1(float32(math.NaN()))
	if err == nil {
		t.Fatal("NewBinaryF1 error = nil, want error")
	}
}

func Test_CategoricalMacroClassificationMetrics(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		value       float32
		err         error
	)

	predictions = mustMatrix(t, 4, 3, []float32{
		0.1, 0.8, 0.1,
		0.6, 0.2, 0.2,
		0.1, 0.2, 0.7,
		0.1, 0.3, 0.6,
	})
	targets = mustMatrix(t, 4, 3, []float32{
		0, 1, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})

	value, err = metric.CategoricalMacroPrecision{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("CategoricalMacroPrecision returned error: %v", err)
	}
	requireAlmostEqual(t, value, 5.0/6.0)

	value, err = metric.CategoricalMacroRecall{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("CategoricalMacroRecall returned error: %v", err)
	}
	requireAlmostEqual(t, value, 5.0/6.0)

	value, err = metric.CategoricalMacroF1{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("CategoricalMacroF1 returned error: %v", err)
	}
	requireAlmostEqual(t, value, 7.0/9.0)
}
