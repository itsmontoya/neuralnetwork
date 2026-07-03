package metric_test

import (
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
)

func Test_BinaryAccuracy_ValueUsesDefaultThreshold(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float64
		err         error
	)

	predictions = mustMatrix(t, 4, 1, []float64{0.49, 0.5, 0.51, 0.1})
	targets = mustMatrix(t, 4, 1, []float64{0, 1, 1, 1})

	got, err = metric.BinaryAccuracy{}.Value(predictions, targets)
	if err != nil {
		t.Fatalf("Value returned error: %v", err)
	}

	requireAlmostEqual(t, got, 0.75)
}

func Test_NewBinaryAccuracy_UsesCustomThreshold(t *testing.T) {
	var (
		binaryAccuracy metric.BinaryAccuracy
		predictions    *matrix.Matrix
		targets        *matrix.Matrix
		got            float64
		err            error
	)

	binaryAccuracy, err = metric.NewBinaryAccuracy(0.75)
	if err != nil {
		t.Fatalf("NewBinaryAccuracy returned error: %v", err)
	}

	predictions = mustMatrix(t, 3, 1, []float64{0.74, 0.75, 0.76})
	targets = mustMatrix(t, 3, 1, []float64{0, 1, 1})

	got, err = binaryAccuracy.Value(predictions, targets)
	if err != nil {
		t.Fatalf("Value returned error: %v", err)
	}

	requireAlmostEqual(t, got, 1)
}

func Test_NewBinaryAccuracy_RejectsNonFiniteThreshold(t *testing.T) {
	var err error

	_, err = metric.NewBinaryAccuracy(math.NaN())
	if err == nil {
		t.Fatal("NewBinaryAccuracy error = nil, want error")
	}
}

func Test_BinaryAccuracy_ValidatesShape(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float64
		err         error
	)

	predictions = mustMatrix(t, 1, 1, []float64{0.5})
	targets = mustMatrix(t, 2, 1, []float64{1, 0})

	got, err = metric.BinaryAccuracy{}.Value(predictions, targets)
	if err == nil {
		t.Fatalf("Value returned %g and nil error, want error", got)
	}
}

func Test_BinaryAccuracy_ValidatesTargetFormat(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float64
		err         error
	)

	predictions = mustMatrix(t, 1, 1, []float64{0.5})
	targets = mustMatrix(t, 1, 1, []float64{0.5})

	got, err = metric.BinaryAccuracy{}.Value(predictions, targets)
	if err == nil {
		t.Fatalf("Value returned %g and nil error, want error", got)
	}
}

func Test_BinaryAccuracy_ValidatesSingleOutput(t *testing.T) {
	var (
		predictions *matrix.Matrix
		targets     *matrix.Matrix
		got         float64
		err         error
	)

	predictions = mustMatrix(t, 1, 2, []float64{0.4, 0.6})
	targets = mustMatrix(t, 1, 2, []float64{0, 1})

	got, err = metric.BinaryAccuracy{}.Value(predictions, targets)
	if err == nil {
		t.Fatalf("Value returned %g and nil error, want error", got)
	}
}
