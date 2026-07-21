package layer_test

import (
	"math"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_LastStep_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.LastStep)(nil)
}

func Test_NewLastStep_ValidatesShapeAndExposesAccessors(t *testing.T) {
	var (
		shape       layer.SequenceShape
		lastStep    *layer.LastStep
		nilLastStep *layer.LastStep
		err         error
	)

	shape = mustSequenceShape(t, 3, 4)
	lastStep, err = layer.NewLastStep(shape)
	if err != nil {
		t.Fatalf("NewLastStep returned error: %v", err)
	}

	if lastStep.InputShape() != shape {
		t.Fatalf("InputShape = %#v, want %#v", lastStep.InputShape(), shape)
	}

	if lastStep.OutputSize() != 4 {
		t.Fatalf("OutputSize = %d, want 4", lastStep.OutputSize())
	}

	if nilLastStep.InputShape() != (layer.SequenceShape{}) {
		t.Fatal("nil InputShape returned a nonzero shape")
	}

	if nilLastStep.OutputSize() != 0 {
		t.Fatalf("nil OutputSize = %d, want 0", nilLastStep.OutputSize())
	}

	if _, err = nilLastStep.Backward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("nil Backward error = nil, want error")
	}

	lastStep, err = layer.NewLastStep(layer.SequenceShape{})
	if err == nil {
		t.Fatal("NewLastStep error = nil, want invalid shape error")
	}

	if lastStep != nil {
		t.Fatal("NewLastStep returned layer on error")
	}

	if !strings.HasPrefix(err.Error(), "layer: last step input shape invalid:") {
		t.Fatalf("NewLastStep error = %q, want last step context", err)
	}
}

func Test_LastStep_ForwardSelectsFinalStep(t *testing.T) {
	type testcase struct {
		name        string
		steps       int
		featureSize int
		rows        int
		input       []float32
		want        []float32
	}

	tests := []testcase{
		{
			name:        "one step preserves every value",
			steps:       1,
			featureSize: 3,
			rows:        2,
			input:       []float32{1, 2, 3, 4, 5, 6},
			want:        []float32{1, 2, 3, 4, 5, 6},
		},
		{
			name:        "multiple steps and one feature",
			steps:       3,
			featureSize: 1,
			rows:        2,
			input:       []float32{1, 2, 3, 4, 5, 6},
			want:        []float32{3, 6},
		},
		{
			name:        "multiple steps batches and features",
			steps:       3,
			featureSize: 2,
			rows:        2,
			input: []float32{
				1, 2, 3, 4, 5, 6,
				7, 8, 9, 10, 11, 12,
			},
			want: []float32{5, 6, 11, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				lastStep *layer.LastStep
				input    *matrix.Matrix
				output   *matrix.Matrix
				err      error
			)

			lastStep = mustLastStep(t, tt.steps, tt.featureSize)
			input = mustMatrix(t, tt.rows, tt.steps*tt.featureSize, tt.input)
			output, err = lastStep.Forward(input)
			if err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			if output == input {
				t.Fatal("Forward output aliases input")
			}

			if output.Rows() != tt.rows || output.Cols() != tt.featureSize {
				t.Fatalf(
					"Forward output shape = %dx%d, want %dx%d",
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

func Test_LastStep_BackwardRoutesOnlyToFinalStep(t *testing.T) {
	type testcase struct {
		name           string
		steps          int
		featureSize    int
		rows           int
		outputGradient []float32
		want           []float32
	}

	tests := []testcase{
		{
			name:           "one step preserves every gradient",
			steps:          1,
			featureSize:    3,
			rows:           2,
			outputGradient: []float32{1, -2, 3, -4, 5, -6},
			want:           []float32{1, -2, 3, -4, 5, -6},
		},
		{
			name:           "multiple steps and one feature",
			steps:          3,
			featureSize:    1,
			rows:           2,
			outputGradient: []float32{2, -3},
			want:           []float32{0, 0, 2, 0, 0, -3},
		},
		{
			name:           "multiple steps batches and features",
			steps:          3,
			featureSize:    2,
			rows:           2,
			outputGradient: []float32{1, 2, 3, 4},
			want: []float32{
				0, 0, 0, 0, 1, 2,
				0, 0, 0, 0, 3, 4,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				lastStep       *layer.LastStep
				outputGradient *matrix.Matrix
				inputGradient  *matrix.Matrix
				err            error
			)

			lastStep = mustLastStep(t, tt.steps, tt.featureSize)
			if _, err = lastStep.Forward(
				mustMatrix(t, tt.rows, tt.steps*tt.featureSize, make([]float32, tt.rows*tt.steps*tt.featureSize)),
			); err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			outputGradient = mustMatrix(t, tt.rows, tt.featureSize, tt.outputGradient)
			inputGradient, err = lastStep.Backward(outputGradient)
			if err != nil {
				t.Fatalf("Backward returned error: %v", err)
			}

			if inputGradient == outputGradient {
				t.Fatal("Backward input gradient aliases output gradient")
			}

			if inputGradient.Rows() != tt.rows || inputGradient.Cols() != tt.steps*tt.featureSize {
				t.Fatalf(
					"Backward input gradient shape = %dx%d, want %dx%d",
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

func Test_LastStep_DoesNotRetainOrAliasCallerStorage(t *testing.T) {
	var (
		lastStep       *layer.LastStep
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	lastStep = mustLastStep(t, 2, 2)
	input = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
	output, err = lastStep.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	requireMatrixValues(t, input, []float32{1, 2, 3, 4})

	if err = input.CopyValuesFrom([]float32{9, 9, 9, 9}); err != nil {
		t.Fatalf("input CopyValuesFrom returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{3, 4})

	outputGradient = mustMatrix(t, 1, 2, []float32{5, 6})
	inputGradient, err = lastStep.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	requireMatrixValues(t, outputGradient, []float32{5, 6})

	if err = outputGradient.CopyValuesFrom([]float32{8, 8}); err != nil {
		t.Fatalf("output gradient CopyValuesFrom returned error: %v", err)
	}
	requireMatrixValues(t, inputGradient, []float32{0, 0, 5, 6})
}

func Test_LastStep_OneStepResultsDoNotAliasLiveArguments(t *testing.T) {
	var (
		lastStep            *layer.LastStep
		input               *matrix.Matrix
		firstOutput         *matrix.Matrix
		secondOutput        *matrix.Matrix
		firstInputGradient  *matrix.Matrix
		secondInputGradient *matrix.Matrix
		err                 error
	)

	lastStep = mustLastStep(t, 1, 2)
	input = mustMatrix(t, 1, 2, []float32{1, 2})
	firstOutput, err = lastStep.Forward(input)
	if err != nil {
		t.Fatalf("first Forward returned error: %v", err)
	}

	secondOutput, err = lastStep.Forward(firstOutput)
	if err != nil {
		t.Fatalf("second Forward returned error: %v", err)
	}

	if secondOutput == firstOutput {
		t.Fatal("second Forward output aliases its input")
	}
	requireMatrixValues(t, secondOutput, []float32{1, 2})

	firstInputGradient, err = lastStep.Backward(mustMatrix(t, 1, 2, []float32{3, 4}))
	if err != nil {
		t.Fatalf("first Backward returned error: %v", err)
	}

	if _, err = lastStep.Forward(input); err != nil {
		t.Fatalf("Forward before second Backward returned error: %v", err)
	}

	secondInputGradient, err = lastStep.Backward(firstInputGradient)
	if err != nil {
		t.Fatalf("second Backward returned error: %v", err)
	}

	if secondInputGradient == firstInputGradient {
		t.Fatal("second Backward input gradient aliases its output gradient")
	}
	requireMatrixValues(t, secondInputGradient, []float32{3, 4})
}

func Test_LastStep_ForwardValidatesReceiverAndInput(t *testing.T) {
	type testcase struct {
		name      string
		lastStep  *layer.LastStep
		input     *matrix.Matrix
		wantError string
	}

	validLastStep := mustLastStep(t, 2, 2)
	tests := []testcase{
		{
			name:      "nil receiver",
			lastStep:  nil,
			input:     mustMatrix(t, 1, 4, []float32{1, 2, 3, 4}),
			wantError: "last step layer is nil",
		},
		{
			name:      "zero value",
			lastStep:  &layer.LastStep{},
			input:     mustMatrix(t, 1, 4, []float32{1, 2, 3, 4}),
			wantError: "last step input shape invalid",
		},
		{
			name:      "nil input",
			lastStep:  validLastStep,
			input:     nil,
			wantError: "last step input is nil",
		},
		{
			name:      "invalid input",
			lastStep:  validLastStep,
			input:     &matrix.Matrix{},
			wantError: "last step input invalid",
		},
		{
			name:      "column mismatch",
			lastStep:  validLastStep,
			input:     mustMatrix(t, 2, 3, []float32{1, 2, 3, 4, 5, 6}),
			wantError: "got 2x3, want batch rows x 4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				output *matrix.Matrix
				err    error
			)

			output, err = tt.lastStep.Forward(tt.input)
			if err == nil {
				t.Fatal("Forward error = nil, want error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Forward error = %q, want substring %q", err, tt.wantError)
			}

			if output != nil {
				t.Fatal("Forward returned output on error")
			}
		})
	}
}

func Test_LastStep_BackwardValidatesStateAndGradient(t *testing.T) {
	var (
		lastStep *layer.LastStep
		err      error
	)

	lastStep = mustLastStep(t, 2, 2)
	if _, err = lastStep.Forward(mustMatrix(t, 1, 3, []float32{1, 2, 3})); err == nil {
		t.Fatal("invalid Forward error = nil, want error")
	}

	if _, err = lastStep.Backward(mustMatrix(t, 1, 2, []float32{1, 2})); err == nil {
		t.Fatal("Backward before valid Forward error = nil, want error")
	}
	if !strings.Contains(err.Error(), "backward called before forward") {
		t.Fatalf("Backward before valid Forward error = %q, want state context", err)
	}

	if _, err = lastStep.Forward(mustMatrix(t, 2, 4, make([]float32, 8))); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	type testcase struct {
		name           string
		outputGradient *matrix.Matrix
		wantError      string
	}

	tests := []testcase{
		{
			name:           "nil gradient",
			outputGradient: nil,
			wantError:      "last step output gradient is nil",
		},
		{
			name:           "invalid gradient",
			outputGradient: &matrix.Matrix{},
			wantError:      "last step output gradient invalid",
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

			inputGradient, err = lastStep.Backward(tt.outputGradient)
			if err == nil {
				t.Fatal("Backward error = nil, want error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Backward error = %q, want substring %q", err, tt.wantError)
			}

			if inputGradient != nil {
				t.Fatal("Backward returned input gradient on error")
			}
		})
	}
}

func Test_LastStep_ComposesBetweenSimpleRNNAndDense(t *testing.T) {
	var (
		inputShape     layer.SequenceShape
		config         layer.SimpleRNNConfig
		recurrent      *layer.SimpleRNN
		lastStep       *layer.LastStep
		dense          *layer.Dense
		network        *model.Sequential
		input          *matrix.Matrix
		output         *matrix.Matrix
		inputGradient  *matrix.Matrix
		wantOutput     *matrix.Matrix
		wantGradient   *matrix.Matrix
		parameterCount int
		firstHidden    float32
		secondHidden   float32
		err            error
	)

	inputShape = mustSequenceShape(t, 2, 1)
	config, err = layer.NewSimpleRNNConfig(inputShape, 2)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	recurrent = mustSimpleRNN(t, config, []float32{1, -1}, make([]float32, 4), []float32{0, 0})
	lastStep, err = layer.NewLastStep(recurrent.OutputShape())
	if err != nil {
		t.Fatalf("NewLastStep returned error: %v", err)
	}
	dense = mustDense(t, 2, 1, []float32{1, 0}, []float32{0})
	network, err = model.NewSequential(recurrent, lastStep, dense)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	parameterCount = len(network.Parameters())
	if parameterCount != 5 {
		t.Fatalf("Sequential parameter count = %d, want SimpleRNN and Dense's 5 parameters", parameterCount)
	}

	input = mustMatrix(t, 2, 2, []float32{0, 1, 1, 2})
	output, err = network.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	firstHidden = float32(math.Tanh(1))
	secondHidden = float32(math.Tanh(2))
	wantOutput = mustMatrix(t, 2, 1, []float32{firstHidden, secondHidden})
	testutil.RequireMatrixAlmostEqual(t, output, wantOutput, 1e-6)

	inputGradient, err = network.Backward(mustMatrix(t, 2, 1, []float32{1, 1}))
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	wantGradient = mustMatrix(t, 2, 2, []float32{
		0, 1 - firstHidden*firstHidden,
		0, 1 - secondHidden*secondHidden,
	})
	testutil.RequireMatrixAlmostEqual(t, inputGradient, wantGradient, 1e-6)
}

func mustLastStep(tb testing.TB, steps, featureSize int) (lastStep *layer.LastStep) {
	var err error

	tb.Helper()
	lastStep, err = layer.NewLastStep(mustSequenceShape(tb, steps, featureSize))
	if err != nil {
		tb.Fatalf("NewLastStep returned error: %v", err)
	}

	return lastStep
}
