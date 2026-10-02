package activation_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_SegmentedSoftmax_Interfaces(t *testing.T) {
	var function *activation.SegmentedSoftmax
	var err error

	if function, err = activation.NewSegmentedSoftmax(2, 3); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}

	var _ activation.Activation = function
	var _ activation.DestinationActivation = function
}

func Test_NewSegmentedSoftmax_ValidatesAndOwnsWidths(t *testing.T) {
	type testcase struct {
		name      string
		widths    []int
		wantError string
	}

	var (
		maxInt   int
		tests    []testcase
		tt       testcase
		function *activation.SegmentedSoftmax
		widths   []int
		err      error
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
			function, err = activation.NewSegmentedSoftmax(tt.widths...)
			if err == nil {
				t.Fatal("NewSegmentedSoftmax error = nil, want error")
			}
			if function != nil {
				t.Fatal("NewSegmentedSoftmax returned function on error")
			}
			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("NewSegmentedSoftmax error = %q, want %q", err, tt.wantError)
			}
		})
	}

	widths = []int{2, 3}
	if function, err = activation.NewSegmentedSoftmax(widths...); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	widths[0] = 9
	widths = function.Widths()
	if widths[0] != 2 {
		t.Fatalf("Widths()[0] = %d, want 2", widths[0])
	}
	widths[0] = 8
	if function.Widths()[0] != 2 {
		t.Fatal("Widths returned mutable configuration")
	}
}

func Test_SegmentedSoftmax_OneSegmentMatchesSoftmax(t *testing.T) {
	var (
		segmented *activation.SegmentedSoftmax
		input     *matrix.Matrix
		got       *matrix.Matrix
		want      *matrix.Matrix
		err       error
	)

	if segmented, err = activation.NewSegmentedSoftmax(4); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	input = mustMatrix(t, 2, 4, []float32{
		1, -2, 3, 0,
		-4, -1, -3, -2,
	})
	if got, err = segmented.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if want, err = (activation.Softmax{}).Forward(input); err != nil {
		t.Fatalf("Softmax Forward returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(t, got, want, epsilon)
}

func Test_SegmentedSoftmax_NormalizesIrregularSegments(t *testing.T) {
	var (
		segmented *activation.SegmentedSoftmax
		input     *matrix.Matrix
		output    *matrix.Matrix
		values    []float32
		row       int
		index     int
		start     int
		width     int
		sum       float32
		err       error
	)

	if segmented, err = activation.NewSegmentedSoftmax(2, 1, 3); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	input = mustMatrix(t, 2, 6, []float32{
		1000, 1001, -1000, 2, 0, -2,
		-1001, -1000, 7, -3, -1, -2,
	})
	if output, err = segmented.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if values, err = output.Values(); err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	for index = range values {
		if f32.IsInf(values[index], 0) || f32.IsNaN(values[index]) {
			t.Fatalf("output[%d] is unstable: %g", index, values[index])
		}
	}
	for row = 0; row < 2; row++ {
		start = row * 6
		for _, width = range []int{2, 1, 3} {
			sum = 0
			for index = start; index < start+width; index++ {
				sum += values[index]
			}
			if !testutil.AlmostEqual(sum, 1, epsilon) {
				t.Fatalf("row %d segment at %d sum = %g, want 1", row, start-row*6, sum)
			}
			start += width
		}
	}
}

func Test_SegmentedSoftmax_BackwardMatchesFiniteDifference(t *testing.T) {
	var (
		segmented         *activation.SegmentedSoftmax
		input             *matrix.Matrix
		outputGradient    *matrix.Matrix
		inputGradient     *matrix.Matrix
		numericalGradient *matrix.Matrix
		err               error
	)

	if segmented, err = activation.NewSegmentedSoftmax(2, 3); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	input = mustMatrix(t, 2, 5, []float32{
		0.3, -0.7, 1.1, 0.2, -0.4,
		-1.2, 0.4, 0.8, -0.9, 0.5,
	})
	outputGradient = mustMatrix(t, 2, 5, []float32{
		0.5, -0.25, 1.2, 0.3, -0.8,
		-0.4, 0.9, -0.1, 0.7, 0.2,
	})
	if inputGradient, err = segmented.Backward(input, outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	numericalGradient, err = testutil.FiniteDifferenceGradient(
		input,
		activationGradientCheckStep,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = segmented.Forward(input); err != nil {
				return 0, err
			}
			value, err = testutil.WeightedMatrixSum(output, outputGradient)
			return value, err
		},
	)
	if err != nil {
		t.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(t, inputGradient, numericalGradient, activationGradientCheckTolerance)
}

func Test_SegmentedSoftmax_BackwardDoesNotLeakAcrossSegments(t *testing.T) {
	var (
		segmented      *activation.SegmentedSoftmax
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		values         []float32
		err            error
	)

	if segmented, err = activation.NewSegmentedSoftmax(2, 2); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	input = mustMatrix(t, 1, 4, []float32{1, -1, 0.5, 2})
	outputGradient = mustMatrix(t, 1, 4, []float32{0, 0, 1, -2})
	if inputGradient, err = segmented.Backward(input, outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	if values, err = inputGradient.Values(); err != nil {
		t.Fatalf("Values returned error: %v", err)
	}
	if values[0] != 0 || values[1] != 0 {
		t.Fatalf("first-segment gradient = %v, want [0 0]", values[:2])
	}
}

func Test_SegmentedSoftmax_DestinationAndShapeContracts(t *testing.T) {
	var (
		segmented      *activation.SegmentedSoftmax
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		wantOutput     *matrix.Matrix
		wantGradient   *matrix.Matrix
		forwardAlias   *matrix.Matrix
		backwardAlias  *matrix.Matrix
		wrongColumns   *matrix.Matrix
		err            error
	)

	if segmented, err = activation.NewSegmentedSoftmax(2, 3); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	input = mustMatrix(t, 1, 5, []float32{1, 2, 3, 1, -1})
	outputGradient = mustMatrix(t, 1, 5, []float32{1, -2, 0.5, 2, -1})
	wrongColumns = mustMatrix(t, 1, 4, []float32{0, 0, 0, 0})
	if wantOutput, err = segmented.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if wantGradient, err = segmented.Backward(input, outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	if forwardAlias, err = input.Clone(); err != nil {
		t.Fatalf("Clone returned error: %v", err)
	}
	if err = segmented.ForwardInto(forwardAlias, forwardAlias); err != nil {
		t.Fatalf("aliased ForwardInto returned error: %v", err)
	}
	testutil.RequireMatrixAlmostEqual(t, forwardAlias, wantOutput, epsilon)

	if backwardAlias, err = input.Clone(); err != nil {
		t.Fatalf("Clone returned error: %v", err)
	}
	if err = segmented.BackwardInto(backwardAlias, outputGradient, backwardAlias); err != nil {
		t.Fatalf("aliased BackwardInto returned error: %v", err)
	}
	testutil.RequireMatrixAlmostEqual(t, backwardAlias, wantGradient, epsilon)

	if _, err = segmented.Forward(wrongColumns); err == nil {
		t.Fatal("Forward error = nil for wrong column count")
	}
	if _, err = segmented.Backward(input, wrongColumns); err == nil {
		t.Fatal("Backward error = nil for mismatched gradient")
	}
	if err = segmented.BackwardInto(input, outputGradient, outputGradient); err == nil {
		t.Fatal("BackwardInto error = nil for output-gradient alias")
	}
	if _, err = segmented.Forward(nil); err == nil {
		t.Fatal("Forward error = nil for nil input")
	}
}

func Test_SegmentedSoftmax_DestinationAllocations(t *testing.T) {
	var (
		segmented      *activation.SegmentedSoftmax
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	if segmented, err = activation.NewSegmentedSoftmax(2, 3); err != nil {
		t.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	input = mustMatrix(t, 2, 5, []float32{1, 2, 3, 1, -1, -1, 0, 2, 4, 3})
	output = mustMatrix(t, 2, 5, make([]float32, 10))
	outputGradient = mustMatrix(t, 2, 5, []float32{1, -2, 0.5, 2, -1, -1, 3, 2, 1, 0})
	inputGradient = mustMatrix(t, 2, 5, make([]float32, 10))

	requireActivationMaxAllocs(t, "ForwardInto", 0, func() {
		if err = segmented.ForwardInto(input, output); err != nil {
			panic(err)
		}
	})
	requireActivationMaxAllocs(t, "BackwardInto", 0, func() {
		if err = segmented.BackwardInto(input, outputGradient, inputGradient); err != nil {
			panic(err)
		}
	})
}
