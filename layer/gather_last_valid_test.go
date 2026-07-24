package layer_test

import (
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_GatherLastValid_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.GatherLastValid)(nil)
}

func Test_NewGatherLastValid_ValidatesShapeAndExposesAccessors(t *testing.T) {
	var (
		shape         layer.SequenceShape
		gather        *layer.GatherLastValid
		nilGather     *layer.GatherLastValid
		boundaryShape layer.SequenceShape
		boundary      *layer.GatherLastValid
		maxInt        int
		err           error
	)

	shape = mustSequenceShape(t, 3, 4)
	gather, err = layer.NewGatherLastValid(shape)
	if err != nil {
		t.Fatalf("NewGatherLastValid returned error: %v", err)
	}

	if gather.InputShape() != shape {
		t.Fatalf("InputShape = %#v, want %#v", gather.InputShape(), shape)
	}

	if gather.OutputSize() != 4 {
		t.Fatalf("OutputSize = %d, want 4", gather.OutputSize())
	}

	if nilGather.InputShape() != (layer.SequenceShape{}) {
		t.Fatal("nil InputShape returned a nonzero shape")
	}

	if nilGather.OutputSize() != 0 {
		t.Fatalf("nil OutputSize = %d, want 0", nilGather.OutputSize())
	}

	gather, err = layer.NewGatherLastValid(layer.SequenceShape{})
	if err == nil {
		t.Fatal("NewGatherLastValid error = nil, want invalid shape error")
	}

	if gather != nil {
		t.Fatal("NewGatherLastValid returned layer on error")
	}

	if !strings.HasPrefix(err.Error(), "layer: gather last valid input shape invalid:") {
		t.Fatalf("NewGatherLastValid error = %q, want gather context", err)
	}

	maxInt = int(^uint(0) >> 1)
	boundaryShape, err = layer.NewSequenceShape(maxInt, 1)
	if err != nil {
		t.Fatalf("NewSequenceShape returned error at overflow boundary: %v", err)
	}

	boundary, err = layer.NewGatherLastValid(boundaryShape)
	if err != nil {
		t.Fatalf("NewGatherLastValid returned error at overflow boundary: %v", err)
	}

	if boundary.InputShape().Size() != maxInt || boundary.OutputSize() != 1 {
		t.Fatalf(
			"boundary gather shape = size %d output %d, want size %d output 1",
			boundary.InputShape().Size(),
			boundary.OutputSize(),
			maxInt,
		)
	}
}

func Test_GatherLastValid_ForwardWithLengthsSelectsFinalValidStep(t *testing.T) {
	type testcase struct {
		name        string
		steps       int
		featureSize int
		rows        int
		input       []float32
		lengths     []int
		want        []float32
	}

	tests := []testcase{
		{
			name:        "one step and one feature",
			steps:       1,
			featureSize: 1,
			rows:        1,
			input:       []float32{7},
			lengths:     []int{1},
			want:        []float32{7},
		},
		{
			name:        "one step preserves every feature",
			steps:       1,
			featureSize: 3,
			rows:        2,
			input:       []float32{1, 2, 3, 4, 5, 6},
			lengths:     []int{1, 1},
			want:        []float32{1, 2, 3, 4, 5, 6},
		},
		{
			name:        "mixed boundary and interior lengths",
			steps:       3,
			featureSize: 2,
			rows:        3,
			input: []float32{
				1, 2, 91, 92, 93, 94,
				3, 4, 5, 6, 95, 96,
				7, 8, 9, 10, 11, 12,
			},
			lengths: []int{1, 2, 3},
			want:    []float32{1, 2, 5, 6, 11, 12},
		},
		{
			name:        "all maximum lengths",
			steps:       3,
			featureSize: 1,
			rows:        2,
			input:       []float32{1, 2, 3, 4, 5, 6},
			lengths:     []int{3, 3},
			want:        []float32{3, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				gather *layer.GatherLastValid
				input  *matrix.Matrix
				output *matrix.Matrix
				err    error
			)

			gather = mustGatherLastValid(t, tt.steps, tt.featureSize)
			input = mustMatrix(t, tt.rows, tt.steps*tt.featureSize, tt.input)
			output, err = gather.ForwardWithLengths(input, tt.lengths)
			if err != nil {
				t.Fatalf("ForwardWithLengths returned error: %v", err)
			}

			if output == input {
				t.Fatal("ForwardWithLengths output aliases input")
			}

			if output.Rows() != tt.rows || output.Cols() != tt.featureSize {
				t.Fatalf(
					"ForwardWithLengths output shape = %dx%d, want %dx%d",
					output.Rows(),
					output.Cols(),
					tt.rows,
					tt.featureSize,
				)
			}

			requireMatrixValues(t, output, tt.want)
		})
	}
}

func Test_GatherLastValid_BackwardWithLengthsRoutesOnlyToSelectedStep(t *testing.T) {
	type testcase struct {
		name           string
		steps          int
		featureSize    int
		rows           int
		lengths        []int
		outputGradient []float32
		want           []float32
	}

	tests := []testcase{
		{
			name:           "one step and one feature",
			steps:          1,
			featureSize:    1,
			rows:           1,
			lengths:        []int{1},
			outputGradient: []float32{-2},
			want:           []float32{-2},
		},
		{
			name:           "one step preserves every gradient",
			steps:          1,
			featureSize:    3,
			rows:           2,
			lengths:        []int{1, 1},
			outputGradient: []float32{1, -2, 3, -4, 5, -6},
			want:           []float32{1, -2, 3, -4, 5, -6},
		},
		{
			name:           "mixed boundary and interior lengths",
			steps:          3,
			featureSize:    2,
			rows:           3,
			lengths:        []int{1, 2, 3},
			outputGradient: []float32{1, 2, 3, 4, 5, 6},
			want: []float32{
				1, 2, 0, 0, 0, 0,
				0, 0, 3, 4, 0, 0,
				0, 0, 0, 0, 5, 6,
			},
		},
		{
			name:           "all maximum lengths",
			steps:          3,
			featureSize:    1,
			rows:           2,
			lengths:        []int{3, 3},
			outputGradient: []float32{2, -3},
			want:           []float32{0, 0, 2, 0, 0, -3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				gather         *layer.GatherLastValid
				outputGradient *matrix.Matrix
				inputGradient  *matrix.Matrix
				err            error
			)

			gather = mustGatherLastValid(t, tt.steps, tt.featureSize)
			if _, err = gather.ForwardWithLengths(
				mustMatrix(t, tt.rows, tt.steps*tt.featureSize, make([]float32, tt.rows*tt.steps*tt.featureSize)),
				tt.lengths,
			); err != nil {
				t.Fatalf("ForwardWithLengths returned error: %v", err)
			}

			outputGradient = mustMatrix(t, tt.rows, tt.featureSize, tt.outputGradient)
			inputGradient, err = gather.BackwardWithLengths(outputGradient)
			if err != nil {
				t.Fatalf("BackwardWithLengths returned error: %v", err)
			}

			if inputGradient == outputGradient {
				t.Fatal("BackwardWithLengths input gradient aliases output gradient")
			}

			if inputGradient.Rows() != tt.rows || inputGradient.Cols() != tt.steps*tt.featureSize {
				t.Fatalf(
					"BackwardWithLengths input gradient shape = %dx%d, want %dx%d",
					inputGradient.Rows(),
					inputGradient.Cols(),
					tt.rows,
					tt.steps*tt.featureSize,
				)
			}

			requireMatrixValues(t, inputGradient, tt.want)
		})
	}
}

func Test_GatherLastValid_OrdinaryMethodsRequireLengthAwarePath(t *testing.T) {
	var (
		gather *layer.GatherLastValid
		input  *matrix.Matrix
		err    error
	)

	gather = mustGatherLastValid(t, 2, 2)
	input = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
	if _, err = gather.Forward(input); err == nil {
		t.Fatal("Forward error = nil, want explicit length-aware error")
	} else if !strings.Contains(err.Error(), "ForwardWithLengths") {
		t.Fatalf("Forward error = %q, want ForwardWithLengths direction", err)
	}

	if _, err = gather.Backward(mustMatrix(t, 1, 2, []float32{1, 2})); err == nil {
		t.Fatal("Backward error = nil, want explicit length-aware error")
	} else if !strings.Contains(err.Error(), "BackwardWithLengths") {
		t.Fatalf("Backward error = %q, want BackwardWithLengths direction", err)
	}
}

func Test_GatherLastValid_ForwardWithLengthsValidatesReceiverInputAndLengths(t *testing.T) {
	type testcase struct {
		name      string
		gather    *layer.GatherLastValid
		input     *matrix.Matrix
		lengths   []int
		wantError string
	}

	validGather := mustGatherLastValid(t, 3, 2)
	tests := []testcase{
		{
			name:      "nil receiver",
			gather:    nil,
			input:     mustMatrix(t, 1, 6, make([]float32, 6)),
			lengths:   []int{1},
			wantError: "gather last valid layer is nil",
		},
		{
			name:      "zero value",
			gather:    &layer.GatherLastValid{},
			input:     mustMatrix(t, 1, 6, make([]float32, 6)),
			lengths:   []int{1},
			wantError: "gather last valid input shape invalid",
		},
		{
			name:      "nil input",
			gather:    validGather,
			input:     nil,
			lengths:   []int{1},
			wantError: "gather last valid input is nil",
		},
		{
			name:      "invalid input",
			gather:    validGather,
			input:     &matrix.Matrix{},
			lengths:   []int{1},
			wantError: "gather last valid input invalid",
		},
		{
			name:      "input column mismatch",
			gather:    validGather,
			input:     mustMatrix(t, 2, 5, make([]float32, 10)),
			lengths:   []int{1, 2},
			wantError: "got 2x5, want batch rows x 6",
		},
		{
			name:      "nil lengths",
			gather:    validGather,
			input:     mustMatrix(t, 2, 6, make([]float32, 12)),
			lengths:   nil,
			wantError: "length count mismatch: got=0 want=2",
		},
		{
			name:      "length count mismatch",
			gather:    validGather,
			input:     mustMatrix(t, 2, 6, make([]float32, 12)),
			lengths:   []int{1},
			wantError: "length count mismatch: got=1 want=2",
		},
		{
			name:      "negative length",
			gather:    validGather,
			input:     mustMatrix(t, 2, 6, make([]float32, 12)),
			lengths:   []int{-1, 2},
			wantError: "row=0 value=-1 want=1..3",
		},
		{
			name:      "zero length",
			gather:    validGather,
			input:     mustMatrix(t, 2, 6, make([]float32, 12)),
			lengths:   []int{1, 0},
			wantError: "row=1 value=0 want=1..3",
		},
		{
			name:      "length above steps",
			gather:    validGather,
			input:     mustMatrix(t, 2, 6, make([]float32, 12)),
			lengths:   []int{1, 4},
			wantError: "row=1 value=4 want=1..3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				output *matrix.Matrix
				err    error
			)

			output, err = tt.gather.ForwardWithLengths(tt.input, tt.lengths)
			if err == nil {
				t.Fatal("ForwardWithLengths error = nil, want error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("ForwardWithLengths error = %q, want substring %q", err, tt.wantError)
			}

			if output != nil {
				t.Fatal("ForwardWithLengths returned output on error")
			}
		})
	}
}

func Test_GatherLastValid_BackwardWithLengthsValidatesReceiverStateAndGradient(t *testing.T) {
	type testcase struct {
		name           string
		outputGradient *matrix.Matrix
		wantError      string
	}

	var (
		nilGather *layer.GatherLastValid
		gather    *layer.GatherLastValid
		err       error
	)

	if _, err = nilGather.BackwardWithLengths(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("nil BackwardWithLengths error = nil, want error")
	} else if !strings.Contains(err.Error(), "gather last valid layer is nil") {
		t.Fatalf("nil BackwardWithLengths error = %q, want receiver context", err)
	}

	gather = mustGatherLastValid(t, 2, 2)
	if _, err = gather.BackwardWithLengths(mustMatrix(t, 1, 2, []float32{1, 2})); err == nil {
		t.Fatal("BackwardWithLengths before forward error = nil, want error")
	} else if !strings.Contains(err.Error(), "backward called before length-aware forward") {
		t.Fatalf("BackwardWithLengths before forward error = %q, want state context", err)
	}

	if _, err = gather.ForwardWithLengths(mustMatrix(t, 2, 4, make([]float32, 8)), []int{1, 2}); err != nil {
		t.Fatalf("ForwardWithLengths returned error: %v", err)
	}

	tests := []testcase{
		{
			name:           "nil gradient",
			outputGradient: nil,
			wantError:      "gather last valid output gradient is nil",
		},
		{
			name:           "invalid gradient",
			outputGradient: &matrix.Matrix{},
			wantError:      "gather last valid output gradient invalid",
		},
		{
			name:           "batch mismatch",
			outputGradient: mustMatrix(t, 1, 2, make([]float32, 2)),
			wantError:      "got 1x2, want 2x2",
		},
		{
			name:           "column mismatch",
			outputGradient: mustMatrix(t, 2, 3, make([]float32, 6)),
			wantError:      "got 2x3, want 2x2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				inputGradient *matrix.Matrix
				err           error
			)

			inputGradient, err = gather.BackwardWithLengths(tt.outputGradient)
			if err == nil {
				t.Fatal("BackwardWithLengths error = nil, want error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("BackwardWithLengths error = %q, want substring %q", err, tt.wantError)
			}

			if inputGradient != nil {
				t.Fatal("BackwardWithLengths returned input gradient on error")
			}
		})
	}

	if _, err = gather.BackwardWithLengths(mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})); err != nil {
		t.Fatalf("corrected BackwardWithLengths returned error after validation failures: %v", err)
	}
}

func Test_GatherLastValid_ForwardFailureInvalidatesStateAndBackwardFailurePreservesIt(t *testing.T) {
	var (
		gather        *layer.GatherLastValid
		input         *matrix.Matrix
		inputGradient *matrix.Matrix
		err           error
	)

	gather = mustGatherLastValid(t, 2, 1)
	input = mustMatrix(t, 1, 2, []float32{3, 7})
	if _, err = gather.ForwardWithLengths(input, []int{1}); err != nil {
		t.Fatalf("first ForwardWithLengths returned error: %v", err)
	}

	if _, err = gather.BackwardWithLengths(mustMatrix(t, 1, 2, []float32{1, 2})); err == nil {
		t.Fatal("invalid BackwardWithLengths error = nil, want error")
	}

	inputGradient, err = gather.BackwardWithLengths(mustMatrix(t, 1, 1, []float32{5}))
	if err != nil {
		t.Fatalf("corrected BackwardWithLengths returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{5, 0})

	if _, err = gather.Backward(mustMatrix(t, 1, 1, []float32{5})); err == nil {
		t.Fatal("ordinary Backward error = nil, want error")
	}

	if _, err = gather.BackwardWithLengths(mustMatrix(t, 1, 1, []float32{6})); err != nil {
		t.Fatalf("BackwardWithLengths returned error after ordinary Backward: %v", err)
	}

	if _, err = gather.ForwardWithLengths(input, nil); err == nil {
		t.Fatal("invalid ForwardWithLengths error = nil, want error")
	}

	if _, err = gather.BackwardWithLengths(mustMatrix(t, 1, 1, []float32{7})); err == nil {
		t.Fatal("BackwardWithLengths after invalid forward error = nil, want state error")
	} else if !strings.Contains(err.Error(), "backward called before length-aware forward") {
		t.Fatalf("BackwardWithLengths error = %q, want invalidated state context", err)
	}

	if _, err = gather.ForwardWithLengths(input, []int{2}); err != nil {
		t.Fatalf("replacement ForwardWithLengths returned error: %v", err)
	}

	if _, err = gather.Forward(input); err == nil {
		t.Fatal("ordinary Forward error = nil, want error")
	}

	if _, err = gather.BackwardWithLengths(mustMatrix(t, 1, 1, []float32{8})); err == nil {
		t.Fatal("BackwardWithLengths after ordinary Forward error = nil, want state error")
	}
}

func Test_GatherLastValid_ReplacesSnapshotAcrossBatchShapes(t *testing.T) {
	var (
		gather        *layer.GatherLastValid
		output        *matrix.Matrix
		inputGradient *matrix.Matrix
		err           error
	)

	gather = mustGatherLastValid(t, 3, 1)
	if _, err = gather.ForwardWithLengths(
		mustMatrix(t, 1, 3, []float32{1, 2, 3}),
		[]int{1},
	); err != nil {
		t.Fatalf("single-row ForwardWithLengths returned error: %v", err)
	}

	output, err = gather.ForwardWithLengths(
		mustMatrix(t, 3, 3, []float32{
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		}),
		[]int{3, 1, 2},
	)
	if err != nil {
		t.Fatalf("three-row ForwardWithLengths returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{6, 7, 11})

	inputGradient, err = gather.BackwardWithLengths(mustMatrix(t, 3, 1, []float32{1, 2, 3}))
	if err != nil {
		t.Fatalf("BackwardWithLengths returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{
		0, 0, 1,
		2, 0, 0,
		0, 3, 0,
	})

	if _, err = gather.ForwardWithLengths(
		mustMatrix(t, 2, 3, []float32{13, 14, 15, 16, 17, 18}),
		[]int{2, 3},
	); err != nil {
		t.Fatalf("two-row ForwardWithLengths returned error: %v", err)
	}

	inputGradient, err = gather.BackwardWithLengths(mustMatrix(t, 2, 1, []float32{4, 5}))
	if err != nil {
		t.Fatalf("replacement BackwardWithLengths returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{0, 4, 0, 0, 0, 5})
}

func Test_GatherLastValid_DoesNotRetainOrAliasCallerStorage(t *testing.T) {
	var (
		gather         *layer.GatherLastValid
		input          *matrix.Matrix
		output         *matrix.Matrix
		lengths        []int
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	gather = mustGatherLastValid(t, 2, 2)
	input = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
	lengths = []int{1}
	output, err = gather.ForwardWithLengths(input, lengths)
	if err != nil {
		t.Fatalf("ForwardWithLengths returned error: %v", err)
	}

	lengths[0] = 2
	if err = input.CopyValuesFrom([]float32{9, 9, 9, 9}); err != nil {
		t.Fatalf("input CopyValuesFrom returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{1, 2})

	outputGradient = mustMatrix(t, 1, 2, []float32{5, 6})
	inputGradient, err = gather.BackwardWithLengths(outputGradient)
	if err != nil {
		t.Fatalf("BackwardWithLengths returned error: %v", err)
	}
	requireMatrixValues(t, outputGradient, []float32{5, 6})

	if err = outputGradient.CopyValuesFrom([]float32{8, 8}); err != nil {
		t.Fatalf("output gradient CopyValuesFrom returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{5, 6, 0, 0})
}

func Test_GatherLastValid_OneStepResultsDoNotAliasLiveArguments(t *testing.T) {
	var (
		gather              *layer.GatherLastValid
		input               *matrix.Matrix
		firstOutput         *matrix.Matrix
		secondOutput        *matrix.Matrix
		firstInputGradient  *matrix.Matrix
		secondInputGradient *matrix.Matrix
		err                 error
	)

	gather = mustGatherLastValid(t, 1, 2)
	input = mustMatrix(t, 1, 2, []float32{1, 2})
	firstOutput, err = gather.ForwardWithLengths(input, []int{1})
	if err != nil {
		t.Fatalf("first ForwardWithLengths returned error: %v", err)
	}

	secondOutput, err = gather.ForwardWithLengths(firstOutput, []int{1})
	if err != nil {
		t.Fatalf("second ForwardWithLengths returned error: %v", err)
	}

	if secondOutput == firstOutput {
		t.Fatal("second ForwardWithLengths output aliases its input")
	}
	requireMatrixValues(t, secondOutput, []float32{1, 2})

	firstInputGradient, err = gather.BackwardWithLengths(mustMatrix(t, 1, 2, []float32{3, 4}))
	if err != nil {
		t.Fatalf("first BackwardWithLengths returned error: %v", err)
	}

	if _, err = gather.ForwardWithLengths(input, []int{1}); err != nil {
		t.Fatalf("ForwardWithLengths before second backward returned error: %v", err)
	}

	secondInputGradient, err = gather.BackwardWithLengths(firstInputGradient)
	if err != nil {
		t.Fatalf("second BackwardWithLengths returned error: %v", err)
	}

	if secondInputGradient == firstInputGradient {
		t.Fatal("second BackwardWithLengths input gradient aliases its output gradient")
	}
	requireMatrixValues(t, secondInputGradient, []float32{3, 4})
}

func Test_GatherLastValid_MatchesLastStepOnValidPrefixes(t *testing.T) {
	var (
		gather               *layer.GatherLastValid
		paddedInput          *matrix.Matrix
		gatherOutput         *matrix.Matrix
		gatherInputGradient  *matrix.Matrix
		gatherOutputValues   []float32
		gatherGradientValues []float32
		inputValues          []float32
		lengths              []int
		outputGradientValues []float32
		row                  int
		length               int
		prefixStart          int
		prefixEnd            int
		prefixInput          *matrix.Matrix
		lastStep             *layer.LastStep
		prefixOutput         *matrix.Matrix
		prefixGradient       *matrix.Matrix
		wantGradient         []float32
		err                  error
	)

	inputValues = []float32{
		1, 2, 91, 92, 93, 94,
		3, 4, 5, 6, 95, 96,
		7, 8, 9, 10, 11, 12,
	}
	lengths = []int{1, 2, 3}
	outputGradientValues = []float32{1, -1, 2, -2, 3, -3}
	gather = mustGatherLastValid(t, 3, 2)
	paddedInput = mustMatrix(t, 3, 6, inputValues)
	gatherOutput, err = gather.ForwardWithLengths(paddedInput, lengths)
	if err != nil {
		t.Fatalf("ForwardWithLengths returned error: %v", err)
	}

	gatherInputGradient, err = gather.BackwardWithLengths(
		mustMatrix(t, 3, 2, outputGradientValues),
	)
	if err != nil {
		t.Fatalf("BackwardWithLengths returned error: %v", err)
	}

	gatherOutputValues, err = gatherOutput.Values()
	if err != nil {
		t.Fatalf("gather output Values returned error: %v", err)
	}

	gatherGradientValues, err = gatherInputGradient.Values()
	if err != nil {
		t.Fatalf("gather gradient Values returned error: %v", err)
	}

	for row, length = range lengths {
		prefixStart = row * 6
		prefixEnd = prefixStart + length*2
		prefixInput = mustMatrix(t, 1, length*2, inputValues[prefixStart:prefixEnd])
		lastStep = mustLastStep(t, length, 2)
		prefixOutput, err = lastStep.Forward(prefixInput)
		if err != nil {
			t.Fatalf("row %d prefix Forward returned error: %v", row, err)
		}
		requireMatrixValues(t, prefixOutput, gatherOutputValues[row*2:row*2+2])

		prefixGradient, err = lastStep.Backward(
			mustMatrix(t, 1, 2, outputGradientValues[row*2:row*2+2]),
		)
		if err != nil {
			t.Fatalf("row %d prefix Backward returned error: %v", row, err)
		}

		wantGradient = make([]float32, 6)
		copy(wantGradient[:length*2], mustValuesForLayer(t, prefixGradient))
		testutil.RequireSliceAlmostEqual(
			t,
			gatherGradientValues[row*6:row*6+6],
			wantGradient,
			0,
		)
	}
}

func Test_GatherLastValid_GradientCheckInput(t *testing.T) {
	var (
		gather            *layer.GatherLastValid
		input             *matrix.Matrix
		outputGradient    *matrix.Matrix
		analyticGradient  *matrix.Matrix
		numericalGradient *matrix.Matrix
		lengths           []int
		err               error
	)

	gather = mustGatherLastValid(t, 3, 2)
	input = mustMatrix(t, 2, 6, []float32{
		0.2, -0.4, 0.7, 0.3, -0.5, 0.8,
		-0.1, 0.6, 0.9, -0.2, 0.4, -0.7,
	})
	outputGradient = mustMatrix(t, 2, 2, []float32{0.5, -1.25, 0.75, 1.5})
	lengths = []int{1, 3}
	if _, err = gather.ForwardWithLengths(input, lengths); err != nil {
		t.Fatalf("ForwardWithLengths returned error: %v", err)
	}

	analyticGradient, err = gather.BackwardWithLengths(outputGradient)
	if err != nil {
		t.Fatalf("BackwardWithLengths returned error: %v", err)
	}

	numericalGradient, err = testutil.FiniteDifferenceGradient(
		input,
		1e-3,
		func() (value float32, err error) {
			var output *matrix.Matrix

			if output, err = gather.ForwardWithLengths(input, lengths); err != nil {
				return 0, err
			}

			value, err = testutil.WeightedMatrixSum(output, outputGradient)
			return value, err
		},
	)
	if err != nil {
		t.Fatalf("FiniteDifferenceGradient returned error: %v", err)
	}

	testutil.RequireMatrixAlmostEqual(t, analyticGradient, numericalGradient, 1e-3)
}

func Test_GatherLastValid_ComposesAfterStackedSimpleRNNLayers(t *testing.T) {
	var (
		firstConfig         layer.SimpleRNNConfig
		secondConfig        layer.SimpleRNNConfig
		first               *layer.SimpleRNN
		second              *layer.SimpleRNN
		gather              *layer.GatherLastValid
		network             *model.Sequential
		input               *matrix.Matrix
		firstOutput         *matrix.Matrix
		secondOutput        *matrix.Matrix
		output              *matrix.Matrix
		secondInputGradient *matrix.Matrix
		firstInputGradient  *matrix.Matrix
		err                 error
	)

	firstConfig, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, 3, 2), 2)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig for first layer returned error: %v", err)
	}
	first = mustSimpleRNN(
		t,
		firstConfig,
		[]float32{0.2, -0.1, 0.3, 0.4},
		[]float32{0.1, 0.2, -0.3, 0.25},
		[]float32{0, 0.1},
	)

	secondConfig, err = layer.NewSimpleRNNConfig(first.OutputShape(), 3)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig for second layer returned error: %v", err)
	}
	second = mustSimpleRNN(
		t,
		secondConfig,
		[]float32{0.1, -0.2, 0.3, 0.4, 0.2, -0.1},
		[]float32{
			0.1, 0, 0.2,
			-0.1, 0.3, 0,
			0.2, -0.2, 0.1,
		},
		[]float32{0, 0.05, -0.05},
	)

	gather, err = layer.NewGatherLastValid(second.OutputShape())
	if err != nil {
		t.Fatalf("NewGatherLastValid returned error: %v", err)
	}

	network, err = model.NewSequential(first, second, gather)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	parameters := network.Parameters()
	if len(parameters) != 6 {
		t.Fatalf("Sequential parameter count = %d, want only the recurrent layers' 6 parameters", len(parameters))
	}

	if parameters[0] != first.InputWeights() ||
		parameters[1] != first.RecurrentWeights() ||
		parameters[2] != first.Biases() ||
		parameters[3] != second.InputWeights() ||
		parameters[4] != second.RecurrentWeights() ||
		parameters[5] != second.Biases() {
		t.Fatal("GatherLastValid changed stacked SimpleRNN parameter order")
	}

	input = mustMatrix(t, 2, 6, []float32{
		1, 0, 2, 0, 99, 99,
		0, 1, 1, 1, 2, 1,
	})
	firstOutput, err = first.Forward(input)
	if err != nil {
		t.Fatalf("first Forward returned error: %v", err)
	}

	secondOutput, err = second.Forward(firstOutput)
	if err != nil {
		t.Fatalf("second Forward returned error: %v", err)
	}

	output, err = gather.ForwardWithLengths(secondOutput, []int{2, 3})
	if err != nil {
		t.Fatalf("ForwardWithLengths returned error: %v", err)
	}

	if output.Rows() != 2 || output.Cols() != 3 {
		t.Fatalf("stacked gather output shape = %dx%d, want 2x3", output.Rows(), output.Cols())
	}

	secondInputGradient, err = gather.BackwardWithLengths(
		mustMatrix(t, 2, 3, []float32{1, -1, 2, -2, 1, 0.5}),
	)
	if err != nil {
		t.Fatalf("BackwardWithLengths returned error: %v", err)
	}

	firstInputGradient, err = second.Backward(secondInputGradient)
	if err != nil {
		t.Fatalf("second Backward returned error: %v", err)
	}

	if firstInputGradient.Rows() != 2 || firstInputGradient.Cols() != 6 {
		t.Fatalf(
			"second input gradient shape = %dx%d, want 2x6",
			firstInputGradient.Rows(),
			firstInputGradient.Cols(),
		)
	}

	firstInputGradient, err = first.Backward(firstInputGradient)
	if err != nil {
		t.Fatalf("first Backward returned error: %v", err)
	}

	if firstInputGradient.Rows() != 2 || firstInputGradient.Cols() != 6 {
		t.Fatalf(
			"stacked input gradient shape = %dx%d, want 2x6",
			firstInputGradient.Rows(),
			firstInputGradient.Cols(),
		)
	}
}

func mustGatherLastValid(
	tb testing.TB,
	steps,
	featureSize int,
) (gather *layer.GatherLastValid) {
	var err error

	tb.Helper()
	gather, err = layer.NewGatherLastValid(mustSequenceShape(tb, steps, featureSize))
	if err != nil {
		tb.Fatalf("NewGatherLastValid returned error: %v", err)
	}

	return gather
}

func mustValuesForLayer(tb testing.TB, value *matrix.Matrix) (values []float32) {
	var err error

	tb.Helper()
	values, err = value.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	return values
}
