package loss_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_SegmentedCategoricalCrossEntropy_Interfaces(t *testing.T) {
	var lossFunction *loss.SegmentedCategoricalCrossEntropy
	var err error

	if lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(2, 3); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}

	var _ loss.Loss = lossFunction
	var _ loss.DestinationGradient = lossFunction
}

func Test_NewSegmentedCategoricalCrossEntropy_ValidatesAndOwnsWidths(t *testing.T) {
	type testcase struct {
		name      string
		widths    []int
		wantError string
	}

	var (
		maxInt       int
		tests        []testcase
		tt           testcase
		lossFunction *loss.SegmentedCategoricalCrossEntropy
		widths       []int
		err          error
	)

	maxInt = int(^uint(0) >> 1)
	tests = []testcase{
		{name: "empty", wantError: "at least one width"},
		{name: "zero", widths: []int{2, 0}, wantError: "width 1 must be positive"},
		{name: "negative", widths: []int{-1}, wantError: "width 0 must be positive"},
		{name: "overflow", widths: []int{maxInt, 1}, wantError: "overflows int"},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(tt.widths...)
			if err == nil {
				t.Fatal("NewSegmentedCategoricalCrossEntropy error = nil, want error")
			}
			if lossFunction != nil {
				t.Fatal("NewSegmentedCategoricalCrossEntropy returned loss on error")
			}
			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("NewSegmentedCategoricalCrossEntropy error = %q, want %q", err, tt.wantError)
			}
		})
	}

	widths = []int{2, 3}
	if lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(widths...); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}
	widths[0] = 9
	widths = lossFunction.Widths()
	if widths[0] != 2 {
		t.Fatalf("Widths()[0] = %d, want 2", widths[0])
	}
	widths[0] = 8
	if lossFunction.Widths()[0] != 2 {
		t.Fatal("Widths returned mutable configuration")
	}
}

func Test_SegmentedCategoricalCrossEntropy_OneSegmentMatchesCategorical(t *testing.T) {
	var (
		segmented    *loss.SegmentedCategoricalCrossEntropy
		predictions  *matrix.Matrix
		targets      *matrix.Matrix
		gotValue     float32
		wantValue    float32
		gotGradient  *matrix.Matrix
		wantGradient *matrix.Matrix
		err          error
	)

	if segmented, err = loss.NewSegmentedCategoricalCrossEntropy(3); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}
	predictions = mustMatrix(t, 2, 3, []float32{0.7, 0.2, 0.1, 0.1, 0.6, 0.3})
	targets = mustMatrix(t, 2, 3, []float32{1, 0, 0, 0, 1, 0})
	if gotValue, err = segmented.Value(predictions, targets); err != nil {
		t.Fatalf("segmented Value returned error: %v", err)
	}
	if wantValue, err = (loss.CategoricalCrossEntropy{}).Value(predictions, targets); err != nil {
		t.Fatalf("categorical Value returned error: %v", err)
	}
	testutil.RequireAlmostEqual(t, gotValue, wantValue, epsilon)

	if gotGradient, err = segmented.Gradient(predictions, targets); err != nil {
		t.Fatalf("segmented Gradient returned error: %v", err)
	}
	if wantGradient, err = (loss.CategoricalCrossEntropy{}).Gradient(predictions, targets); err != nil {
		t.Fatalf("categorical Gradient returned error: %v", err)
	}
	testutil.RequireMatrixAlmostEqual(t, gotGradient, wantGradient, epsilon)
}

func Test_SegmentedCategoricalCrossEntropy_ValueAndGradient(t *testing.T) {
	var (
		lossFunction *loss.SegmentedCategoricalCrossEntropy
		predictions  *matrix.Matrix
		targets      *matrix.Matrix
		gradient     *matrix.Matrix
		value        float32
		wantValue    float32
		err          error
	)

	if lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(2, 3); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}
	predictions = mustMatrix(t, 2, 5, []float32{
		0.8, 0.2, 0.1, 0.7, 0.2,
		0.25, 0.75, 0.6, 0.3, 0.1,
	})
	targets = mustMatrix(t, 2, 5, []float32{
		1, 0, 0, 1, 0,
		0, 1, 1, 0, 0,
	})
	if value, err = lossFunction.Value(predictions, targets); err != nil {
		t.Fatalf("Value returned error: %v", err)
	}
	wantValue = (-f32.Log(0.8) - f32.Log(0.7) - f32.Log(0.75) - f32.Log(0.6)) / 4
	testutil.RequireAlmostEqual(t, value, wantValue, epsilon)

	if gradient, err = lossFunction.Gradient(predictions, targets); err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}
	requireMatrixValues(t, gradient, []float32{
		-1 / 0.8 / 4, 0, 0, -1 / 0.7 / 4, 0,
		0, -1 / 0.75 / 4, -1 / 0.6 / 4, 0, 0,
	})
}

func Test_SegmentedCategoricalCrossEntropy_MeanScaling(t *testing.T) {
	var (
		lossFunction *loss.SegmentedCategoricalCrossEntropy
		predictions  *matrix.Matrix
		targets      *matrix.Matrix
		gradient     *matrix.Matrix
		value        float32
		err          error
	)

	if lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(2, 2); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}
	predictions = mustMatrix(t, 2, 4, []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5})
	targets = mustMatrix(t, 2, 4, []float32{1, 0, 0, 1, 0, 1, 1, 0})
	if value, err = lossFunction.Value(predictions, targets); err != nil {
		t.Fatalf("Value returned error: %v", err)
	}
	testutil.RequireAlmostEqual(t, value, -f32.Log(0.5), epsilon)

	if gradient, err = lossFunction.Gradient(predictions, targets); err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}
	requireMatrixValues(t, gradient, []float32{-0.5, 0, 0, -0.5, 0, -0.5, -0.5, 0})
}

func Test_SegmentedCategoricalCrossEntropy_GradientMatchesFiniteDifference(t *testing.T) {
	var (
		lossFunction      *loss.SegmentedCategoricalCrossEntropy
		predictions       *matrix.Matrix
		targets           *matrix.Matrix
		gradient          *matrix.Matrix
		numericalGradient *matrix.Matrix
		err               error
	)

	if lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(2, 3); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}
	predictions = mustMatrix(t, 1, 5, []float32{0.7, 0.3, 0.2, 0.5, 0.3})
	targets = mustMatrix(t, 1, 5, []float32{1, 0, 0, 1, 0})
	if gradient, err = lossFunction.Gradient(predictions, targets); err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}
	numericalGradient, err = testutil.FiniteDifferenceGradient(
		predictions,
		1e-3,
		func() (value float32, err error) {
			value, err = lossFunction.Value(predictions, targets)
			return value, err
		},
	)
	if err != nil {
		t.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(t, gradient, numericalGradient, 5e-3)
}

func Test_SegmentedCategoricalCrossEntropy_ComposesWithSegmentedSoftmax(t *testing.T) {
	var (
		softmax            *activation.SegmentedSoftmax
		lossFunction       *loss.SegmentedCategoricalCrossEntropy
		logits             *matrix.Matrix
		targets            *matrix.Matrix
		predictions        *matrix.Matrix
		predictionGradient *matrix.Matrix
		logitGradient      *matrix.Matrix
		predictionValues   []float32
		targetValues       []float32
		wantValues         []float32
		index              int
		err                error
	)

	if softmax, err = activation.NewSegmentedSoftmax(2, 3); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	if lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(2, 3); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}
	logits = mustMatrix(t, 2, 5, []float32{1, 2, 0, 2, -1, -1, 1, 3, 0, 2})
	targets = mustMatrix(t, 2, 5, []float32{0, 1, 1, 0, 0, 1, 0, 0, 0, 1})
	if predictions, err = softmax.Forward(logits); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if predictionGradient, err = lossFunction.Gradient(predictions, targets); err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}
	if logitGradient, err = softmax.Backward(logits, predictionGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	if predictionValues, err = predictions.Values(); err != nil {
		t.Fatalf("prediction Values returned error: %v", err)
	}
	if targetValues, err = targets.Values(); err != nil {
		t.Fatalf("target Values returned error: %v", err)
	}
	wantValues = make([]float32, len(predictionValues))
	for index = range wantValues {
		wantValues[index] = (predictionValues[index] - targetValues[index]) / 4
	}

	requireMatrixValues(t, logitGradient, wantValues)
}

func Test_SegmentedCategoricalCrossEntropy_IdentifiesInvalidTargetSegment(t *testing.T) {
	type testcase struct {
		name      string
		targets   []float32
		wantError string
	}

	var (
		lossFunction *loss.SegmentedCategoricalCrossEntropy
		predictions  *matrix.Matrix
		tests        []testcase
		tt           testcase
		targets      *matrix.Matrix
		value        float32
		err          error
	)

	if lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(2, 3); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}
	predictions = mustMatrix(t, 2, 5, []float32{
		0.5, 0.5, 0.2, 0.5, 0.3,
		0.5, 0.5, 0.2, 0.5, 0.3,
	})
	tests = []testcase{
		{
			name:      "no selected class",
			targets:   []float32{1, 0, 0, 1, 0, 0, 1, 0, 0, 0},
			wantError: "row 1 segment 1 must contain exactly one class",
		},
		{
			name:      "multiple selected classes",
			targets:   []float32{1, 0, 0, 1, 0, 1, 1, 1, 0, 0},
			wantError: "row 1 segment 0 contains multiple selected classes",
		},
		{
			name:      "non-binary value",
			targets:   []float32{1, 0, 0, 0.5, 0.5, 0, 1, 1, 0, 0},
			wantError: "row 0 segment 1 column 3 must be 0 or 1",
		},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			targets = mustMatrix(t, 2, 5, tt.targets)
			value, err = lossFunction.Value(predictions, targets)
			if err == nil {
				t.Fatalf("Value = %g and nil error, want error", value)
			}
			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Value error = %q, want %q", err, tt.wantError)
			}
			if _, err = lossFunction.Gradient(predictions, targets); err == nil {
				t.Fatal("Gradient error = nil, want target error")
			}
		})
	}
}

func Test_SegmentedCategoricalCrossEntropy_ShapeAndDestinationContracts(t *testing.T) {
	var (
		lossFunction      *loss.SegmentedCategoricalCrossEntropy
		predictions       *matrix.Matrix
		targets           *matrix.Matrix
		wantGradient      *matrix.Matrix
		predictionAlias   *matrix.Matrix
		predictionTargets *matrix.Matrix
		targetAlias       *matrix.Matrix
		targetPredictions *matrix.Matrix
		wrongColumns      *matrix.Matrix
		err               error
	)

	if lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(2, 3); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}
	predictions = mustMatrix(t, 1, 5, []float32{0.7, 0.3, 0.2, 0.5, 0.3})
	targets = mustMatrix(t, 1, 5, []float32{1, 0, 0, 1, 0})
	wrongColumns = mustMatrix(t, 1, 4, []float32{0.5, 0.5, 0.5, 0.5})
	if wantGradient, err = lossFunction.Gradient(predictions, targets); err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}

	if predictionAlias, err = predictions.Clone(); err != nil {
		t.Fatalf("predictions Clone returned error: %v", err)
	}
	if predictionTargets, err = targets.Clone(); err != nil {
		t.Fatalf("targets Clone returned error: %v", err)
	}
	if err = lossFunction.GradientInto(predictionAlias, predictionTargets, predictionAlias); err != nil {
		t.Fatalf("prediction-aliased GradientInto returned error: %v", err)
	}
	testutil.RequireMatrixAlmostEqual(t, predictionAlias, wantGradient, epsilon)

	if targetPredictions, err = predictions.Clone(); err != nil {
		t.Fatalf("predictions Clone returned error: %v", err)
	}
	if targetAlias, err = targets.Clone(); err != nil {
		t.Fatalf("targets Clone returned error: %v", err)
	}
	if err = lossFunction.GradientInto(targetPredictions, targetAlias, targetAlias); err != nil {
		t.Fatalf("target-aliased GradientInto returned error: %v", err)
	}
	testutil.RequireMatrixAlmostEqual(t, targetAlias, wantGradient, epsilon)

	if _, err = lossFunction.Value(nil, targets); err == nil {
		t.Fatal("Value error = nil for nil predictions")
	}
	if _, err = lossFunction.Value(predictions, nil); err == nil {
		t.Fatal("Value error = nil for nil targets")
	}
	if _, err = lossFunction.Value(wrongColumns, wrongColumns); err == nil {
		t.Fatal("Value error = nil for wrong column count")
	}
	if _, err = lossFunction.Value(predictions, wrongColumns); err == nil {
		t.Fatal("Value error = nil for shape mismatch")
	}
}

func Test_SegmentedCategoricalCrossEntropy_DestinationAllocations(t *testing.T) {
	var (
		lossFunction *loss.SegmentedCategoricalCrossEntropy
		predictions  *matrix.Matrix
		targets      *matrix.Matrix
		destination  *matrix.Matrix
		err          error
	)

	if lossFunction, err = loss.NewSegmentedCategoricalCrossEntropy(2, 3); err != nil {
		t.Fatalf("NewSegmentedCategoricalCrossEntropy returned error: %v", err)
	}
	predictions = mustMatrix(t, 2, 5, []float32{
		0.7, 0.3, 0.2, 0.5, 0.3,
		0.4, 0.6, 0.6, 0.1, 0.3,
	})
	targets = mustMatrix(t, 2, 5, []float32{
		1, 0, 0, 1, 0,
		0, 1, 1, 0, 0,
	})
	destination = mustMatrix(t, 2, 5, make([]float32, 10))

	requireMaxAllocs(t, "SegmentedCategoricalCrossEntropy Value", 0, func() {
		if allocationLossValue, err = lossFunction.Value(predictions, targets); err != nil {
			panic(err)
		}
	})
	requireMaxAllocs(t, "SegmentedCategoricalCrossEntropy GradientInto", 0, func() {
		if err = lossFunction.GradientInto(predictions, targets, destination); err != nil {
			panic(err)
		}
	})
}
