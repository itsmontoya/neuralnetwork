package layer_test

import (
	"errors"
	"math/rand"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/testutil"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_SimpleRNN_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.SimpleRNN)(nil)
}

func Test_NewSimpleRNN_UsesDocumentedInitializerOrderAndShapes(t *testing.T) {
	var (
		config                     layer.SimpleRNNConfig
		recurrent                  *layer.SimpleRNN
		inputInitializerSource     *matrix.Matrix
		recurrentInitializerSource *matrix.Matrix
		calls                      []string
		inputShape                 layer.SequenceShape
		value                      float32
		err                        error
	)

	inputShape = mustSequenceShape(t, 3, 2)
	config, err = layer.NewSimpleRNNConfig(inputShape, 4)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}

	recurrent, err = layer.NewSimpleRNN(
		config,
		func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
			calls = append(calls, "input")
			if inputSize != 2 || outputSize != 4 {
				t.Fatalf("input initializer shape = %dx%d, want 2x4", inputSize, outputSize)
			}

			inputInitializerSource, err = matrix.New(inputSize, outputSize)
			return inputInitializerSource, err
		},
		func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
			calls = append(calls, "recurrent")
			if inputSize != 4 || outputSize != 4 {
				t.Fatalf("recurrent initializer shape = %dx%d, want 4x4", inputSize, outputSize)
			}

			recurrentInitializerSource, err = matrix.New(inputSize, outputSize)
			return recurrentInitializerSource, err
		},
	)
	if err != nil {
		t.Fatalf("NewSimpleRNN returned error: %v", err)
	}

	if len(calls) != 2 || calls[0] != "input" || calls[1] != "recurrent" {
		t.Fatalf("initializer calls = %v, want [input recurrent]", calls)
	}

	if recurrent.InputWeights().Values() == inputInitializerSource {
		t.Fatal("input weights retain initializer-owned matrix")
	}

	if recurrent.RecurrentWeights().Values() == recurrentInitializerSource {
		t.Fatal("recurrent weights retain initializer-owned matrix")
	}

	if err = inputInitializerSource.Set(0, 0, 9); err != nil {
		t.Fatalf("input initializer source Set returned error: %v", err)
	}

	if err = recurrentInitializerSource.Set(0, 0, 8); err != nil {
		t.Fatalf("recurrent initializer source Set returned error: %v", err)
	}

	if value, err = recurrent.InputWeights().Values().At(0, 0); err != nil {
		t.Fatalf("input weight At returned error: %v", err)
	}
	if value != 0 {
		t.Fatalf("input weight = %g after initializer source mutation, want 0", value)
	}

	if value, err = recurrent.RecurrentWeights().Values().At(0, 0); err != nil {
		t.Fatalf("recurrent weight At returned error: %v", err)
	}
	if value != 0 {
		t.Fatalf("recurrent weight = %g after initializer source mutation, want 0", value)
	}

	requireMatrixValues(t, recurrent.Biases().Values(), []float32{0, 0, 0, 0})
}

func Test_NewSimpleRNN_ValidatesDependenciesAndInitializerOutputs(t *testing.T) {
	type testcase struct {
		name                 string
		config               layer.SimpleRNNConfig
		inputInitializer     layer.WeightInitializer
		recurrentInitializer layer.WeightInitializer
		wantError            string
	}

	var (
		validConfig layer.SimpleRNNConfig
		err         error
	)

	validConfig, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, 2, 3), 2)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}

	tests := []testcase{
		{
			name:                 "zero configuration",
			config:               layer.SimpleRNNConfig{},
			inputInitializer:     layer.ZeroWeights,
			recurrentInitializer: layer.ZeroWeights,
			wantError:            "configuration invalid",
		},
		{
			name:                 "nil input initializer",
			config:               validConfig,
			inputInitializer:     nil,
			recurrentInitializer: layer.ZeroWeights,
			wantError:            "input weight initializer is nil",
		},
		{
			name:                 "nil recurrent initializer",
			config:               validConfig,
			inputInitializer:     layer.ZeroWeights,
			recurrentInitializer: nil,
			wantError:            "recurrent weight initializer is nil",
		},
		{
			name:   "input initializer error",
			config: validConfig,
			inputInitializer: func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
				err = errors.New("input failed")
				return nil, err
			},
			recurrentInitializer: layer.ZeroWeights,
			wantError:            "initialize input weights: input failed",
		},
		{
			name:   "nil input initializer output",
			config: validConfig,
			inputInitializer: func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
				return nil, nil
			},
			recurrentInitializer: layer.ZeroWeights,
			wantError:            "initializer input weights is nil",
		},
		{
			name:   "input initializer shape mismatch",
			config: validConfig,
			inputInitializer: func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
				weights, err = matrix.New(inputSize-1, outputSize)
				return weights, err
			},
			recurrentInitializer: layer.ZeroWeights,
			wantError:            "got 2x2, want 3x2",
		},
		{
			name:             "recurrent initializer error",
			config:           validConfig,
			inputInitializer: layer.ZeroWeights,
			recurrentInitializer: func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
				err = errors.New("recurrent failed")
				return nil, err
			},
			wantError: "initialize recurrent weights: recurrent failed",
		},
		{
			name:             "nil recurrent initializer output",
			config:           validConfig,
			inputInitializer: layer.ZeroWeights,
			recurrentInitializer: func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
				return nil, nil
			},
			wantError: "initializer recurrent weights is nil",
		},
		{
			name:             "recurrent initializer shape mismatch",
			config:           validConfig,
			inputInitializer: layer.ZeroWeights,
			recurrentInitializer: func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
				weights, err = matrix.New(inputSize, outputSize-1)
				return weights, err
			},
			wantError: "got 2x1, want 2x2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var recurrent *layer.SimpleRNN

			recurrent, err = layer.NewSimpleRNN(tt.config, tt.inputInitializer, tt.recurrentInitializer)
			if err == nil {
				t.Fatal("NewSimpleRNN error = nil, want error")
			}

			if !strings.HasPrefix(err.Error(), "layer: ") {
				t.Fatalf("NewSimpleRNN error = %q, want layer context", err)
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("NewSimpleRNN error = %q, want substring %q", err, tt.wantError)
			}

			if recurrent != nil {
				t.Fatal("NewSimpleRNN returned layer on error")
			}
		})
	}
}

func Test_SimpleRNN_AccessorsAndParameterDiscovery(t *testing.T) {
	var (
		config             layer.SimpleRNNConfig
		recurrent          *layer.SimpleRNN
		network            *model.Sequential
		parameters         []*optimizer.Parameter
		appendedParameters []*optimizer.Parameter
		err                error
	)

	config, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, 4, 3), 2)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	recurrent = mustSimpleRNN(t, config, make([]float32, 6), make([]float32, 4), []float32{0, 0})

	if recurrent.Config() != config {
		t.Fatalf("Config = %#v, want %#v", recurrent.Config(), config)
	}
	if recurrent.InputShape() != config.InputShape() {
		t.Fatalf("InputShape = %#v, want %#v", recurrent.InputShape(), config.InputShape())
	}
	if recurrent.OutputShape() != config.OutputShape() {
		t.Fatalf("OutputShape = %#v, want %#v", recurrent.OutputShape(), config.OutputShape())
	}

	parameters = recurrent.Parameters()
	if len(parameters) != 3 {
		t.Fatalf("Parameters length = %d, want 3", len(parameters))
	}
	if parameters[0] != recurrent.InputWeights() ||
		parameters[1] != recurrent.RecurrentWeights() ||
		parameters[2] != recurrent.Biases() {
		t.Fatal("Parameters did not return input weights, recurrent weights, and biases in order")
	}
	parameters[0] = nil
	if recurrent.InputWeights() == nil {
		t.Fatal("mutating Parameters result changed SimpleRNN input weights")
	}

	appendedParameters = make([]*optimizer.Parameter, 1, 4)
	appendedParameters[0] = recurrent.Biases()
	appendedParameters = recurrent.AppendParameters(appendedParameters)
	if len(appendedParameters) != 4 {
		t.Fatalf("AppendParameters length = %d, want 4", len(appendedParameters))
	}
	if appendedParameters[0] != recurrent.Biases() {
		t.Fatal("AppendParameters changed the existing prefix")
	}
	if appendedParameters[1] != recurrent.InputWeights() ||
		appendedParameters[2] != recurrent.RecurrentWeights() ||
		appendedParameters[3] != recurrent.Biases() {
		t.Fatal("AppendParameters did not append parameters in documented order")
	}
	appendedParameters[1] = nil
	if recurrent.InputWeights() == nil {
		t.Fatal("mutating AppendParameters result changed SimpleRNN input weights")
	}

	network, err = model.NewSequential(recurrent)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	parameters = network.Parameters()
	if len(parameters) != 3 ||
		parameters[0] != recurrent.InputWeights() ||
		parameters[1] != recurrent.RecurrentWeights() ||
		parameters[2] != recurrent.Biases() {
		t.Fatal("Sequential did not discover SimpleRNN parameters in documented order")
	}
}

func Test_SimpleRNN_NilReceiverAndZeroValue(t *testing.T) {
	var (
		nilRecurrent  *layer.SimpleRNN
		zeroRecurrent layer.SimpleRNN
		prefix        []*optimizer.Parameter
		err           error
	)

	if nilRecurrent.Config() != (layer.SimpleRNNConfig{}) {
		t.Fatal("Config returned value for nil receiver")
	}
	if nilRecurrent.InputShape() != (layer.SequenceShape{}) {
		t.Fatal("InputShape returned value for nil receiver")
	}
	if nilRecurrent.OutputShape() != (layer.SequenceShape{}) {
		t.Fatal("OutputShape returned value for nil receiver")
	}
	if nilRecurrent.InputWeights() != nil || nilRecurrent.RecurrentWeights() != nil ||
		nilRecurrent.Biases() != nil || nilRecurrent.Parameters() != nil {
		t.Fatal("parameter accessor returned value for nil receiver")
	}

	prefix = []*optimizer.Parameter{nil}
	prefix = nilRecurrent.AppendParameters(prefix)
	if len(prefix) != 1 {
		t.Fatalf("AppendParameters length = %d, want 1", len(prefix))
	}

	if _, err = nilRecurrent.Forward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("Forward error = nil, want nil receiver error")
	}
	if _, err = nilRecurrent.Backward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("Backward error = nil, want nil receiver error")
	}
	if err = nilRecurrent.ResetGradients(); err == nil {
		t.Fatal("ResetGradients error = nil, want nil receiver error")
	}
	if _, err = zeroRecurrent.Forward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("zero-value Forward error = nil, want invalid state error")
	}
}

func Test_SimpleRNN_Forward(t *testing.T) {
	type testcase struct {
		name             string
		config           layer.SimpleRNNConfig
		inputWeights     []float32
		recurrentWeights []float32
		biases           []float32
		rows             int
		input            []float32
		wantOutput       []float32
	}

	var (
		oneStepConfig   layer.SimpleRNNConfig
		multiStepConfig layer.SimpleRNNConfig
		err             error
	)

	oneStepConfig, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, 1, 1), 1)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	multiStepConfig, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, 3, 2), 2)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}

	tests := []testcase{
		{
			name:             "one step feature and hidden value",
			config:           oneStepConfig,
			inputWeights:     []float32{0.5},
			recurrentWeights: []float32{0.25},
			biases:           []float32{0.1},
			rows:             1,
			input:            []float32{2},
			wantOutput:       []float32{0.800499},
		},
		{
			name:   "multiple steps features hidden values and batches",
			config: multiStepConfig,
			inputWeights: []float32{
				0.5, -0.25,
				1, 0.75,
			},
			recurrentWeights: []float32{
				0.2, -0.1,
				0.3, 0.4,
			},
			biases: []float32{0.1, -0.2},
			rows:   2,
			input: []float32{
				1, 2,
				0.5, -1,
				-0.25, 0.75,
				-1, 0.5,
				0, 1,
				2, -0.5,
			},
			wantOutput: []float32{
				0.9890274, 0.78180635,
				-0.21427959, -0.69686526,
				0.44068816, 0.16612776,
				0.099667996, 0.40113428,
				0.84553367, 0.6046767,
				0.7400137, -0.7247993,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				recurrent *layer.SimpleRNN
				input     *matrix.Matrix
				output    *matrix.Matrix
			)

			recurrent = mustSimpleRNN(t, tt.config, tt.inputWeights, tt.recurrentWeights, tt.biases)
			input = mustMatrix(t, tt.rows, tt.config.InputShape().Size(), tt.input)
			output, err = recurrent.Forward(input)
			if err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			if output.Rows() != tt.rows || output.Cols() != tt.config.OutputShape().Size() {
				t.Fatalf(
					"Forward output shape = %dx%d, want %dx%d",
					output.Rows(),
					output.Cols(),
					tt.rows,
					tt.config.OutputShape().Size(),
				)
			}
			requireMatrixValues(t, output, tt.wantOutput)
		})
	}
}

func Test_SimpleRNN_ResetsStateAcrossRowsAndForwardCalls(t *testing.T) {
	var (
		config    layer.SimpleRNNConfig
		recurrent *layer.SimpleRNN
		input     *matrix.Matrix
		zeros     *matrix.Matrix
		output    *matrix.Matrix
		err       error
	)

	config, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, 2, 1), 1)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	recurrent = mustSimpleRNN(t, config, []float32{1}, []float32{0.9}, []float32{0})
	input = mustMatrix(t, 2, 2, []float32{1, 2, 0, 0})
	output, err = recurrent.Forward(input)
	if err != nil {
		t.Fatalf("batched Forward returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{0.7615942, 0.99074286, 0, 0})

	zeros = mustMatrix(t, 1, 2, []float32{0, 0})
	output, err = recurrent.Forward(zeros)
	if err != nil {
		t.Fatalf("zero Forward returned error: %v", err)
	}
	requireMatrixValues(t, output, []float32{0, 0})
}

func Test_SimpleRNN_ForwardValidatesInput(t *testing.T) {
	type testcase struct {
		name      string
		input     *matrix.Matrix
		wantError string
	}

	var (
		config        layer.SimpleRNNConfig
		recurrent     *layer.SimpleRNN
		invalidMatrix matrix.Matrix
		err           error
	)

	config, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, 2, 2), 1)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	recurrent = mustSimpleRNN(t, config, []float32{1, 1}, []float32{1}, []float32{0})
	tests := []testcase{
		{name: "nil", input: nil, wantError: "input is nil"},
		{name: "invalid matrix", input: &invalidMatrix, wantError: "input invalid"},
		{name: "column mismatch", input: mustMatrix(t, 2, 3, make([]float32, 6)), wantError: "got 2x3, want batch rows x 4"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var output *matrix.Matrix

			output, err = recurrent.Forward(tt.input)
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

func Test_SimpleRNN_BackwardAndGradientAccumulation(t *testing.T) {
	var (
		config         layer.SimpleRNNConfig
		recurrent      *layer.SimpleRNN
		input          *matrix.Matrix
		output         *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	config, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, 2, 1), 1)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	recurrent = mustSimpleRNN(t, config, []float32{0.5}, []float32{0.25}, []float32{0.1})
	input = mustMatrix(t, 1, 2, []float32{0.4, -0.2})
	output, err = recurrent.Forward(input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	if output == input {
		t.Fatal("Forward output aliases input")
	}

	if err = input.CopyValuesFrom([]float32{99, 99}); err != nil {
		t.Fatalf("input CopyValuesFrom returned error: %v", err)
	}
	if err = output.CopyValuesFrom([]float32{99, 99}); err != nil {
		t.Fatalf("output CopyValuesFrom returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 1, 2, []float32{0.7, -0.3})
	inputGradient, err = recurrent.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	if inputGradient == outputGradient {
		t.Fatal("Backward input gradient aliases output gradient")
	}

	requireMatrixValues(t, inputGradient, []float32{0.2861617, -0.14920722})
	requireMatrixValues(t, recurrent.InputWeights().Gradient(), []float32{0.28861222})
	requireMatrixValues(t, recurrent.RecurrentWeights().Gradient(), []float32{-0.086931884})
	requireMatrixValues(t, recurrent.Biases().Gradient(), []float32{0.27390894})

	if _, err = recurrent.Backward(outputGradient); err != nil {
		t.Fatalf("second Backward returned error: %v", err)
	}
	requireMatrixValues(t, recurrent.InputWeights().Gradient(), []float32{0.57722443})
	requireMatrixValues(t, recurrent.RecurrentWeights().Gradient(), []float32{-0.17386377})
	requireMatrixValues(t, recurrent.Biases().Gradient(), []float32{0.5478179})

	if err = recurrent.ResetGradients(); err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}
	requireMatrixValues(t, recurrent.InputWeights().Gradient(), []float32{0})
	requireMatrixValues(t, recurrent.RecurrentWeights().Gradient(), []float32{0})
	requireMatrixValues(t, recurrent.Biases().Gradient(), []float32{0})
}

func Test_SimpleRNN_BackwardValidatesStateAndGradient(t *testing.T) {
	type testcase struct {
		name      string
		gradient  *matrix.Matrix
		wantError string
	}

	var (
		config          layer.SimpleRNNConfig
		recurrent       *layer.SimpleRNN
		invalidGradient matrix.Matrix
		err             error
	)

	config, err = layer.NewSimpleRNNConfig(mustSequenceShape(t, 2, 1), 2)
	if err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	recurrent = mustSimpleRNN(t, config, []float32{1, 1}, make([]float32, 4), []float32{0, 0})
	if _, err = recurrent.Backward(mustMatrix(t, 1, 4, make([]float32, 4))); err == nil {
		t.Fatal("Backward before Forward error = nil, want error")
	}
	if !strings.Contains(err.Error(), "backward called before forward") {
		t.Fatalf("Backward before Forward error = %q, want state context", err)
	}

	if _, err = recurrent.Forward(mustMatrix(t, 2, 2, make([]float32, 4))); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	tests := []testcase{
		{name: "nil", gradient: nil, wantError: "output gradient is nil"},
		{name: "invalid matrix", gradient: &invalidGradient, wantError: "output gradient invalid"},
		{name: "row mismatch", gradient: mustMatrix(t, 1, 4, make([]float32, 4)), wantError: "got 1x4, want 2x4"},
		{name: "column mismatch", gradient: mustMatrix(t, 2, 3, make([]float32, 6)), wantError: "got 2x3, want 2x4"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var inputGradient *matrix.Matrix

			inputGradient, err = recurrent.Backward(tt.gradient)
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

func Test_SimpleRNN_DeterministicInitializationAndPrediction(t *testing.T) {
	var (
		first        *layer.SimpleRNN
		second       *layer.SimpleRNN
		input        *matrix.Matrix
		firstOutput  *matrix.Matrix
		secondOutput *matrix.Matrix
		err          error
	)

	first = mustDeterministicSimpleRNN(t)
	second = mustDeterministicSimpleRNN(t)
	testutil.RequireMatrixAlmostEqual(t, first.InputWeights().Values(), second.InputWeights().Values(), 0)
	testutil.RequireMatrixAlmostEqual(t, first.RecurrentWeights().Values(), second.RecurrentWeights().Values(), 0)
	input = mustMatrix(t, 2, 6, []float32{
		0.2, -0.1, 0.4, 0.3, -0.5, 0.6,
		1, 0, -0.25, 0.75, 0.5, -1,
	})
	firstOutput, err = first.Forward(input)
	if err != nil {
		t.Fatalf("first Forward returned error: %v", err)
	}
	secondOutput, err = second.Forward(input)
	if err != nil {
		t.Fatalf("second Forward returned error: %v", err)
	}
	testutil.RequireMatrixAlmostEqual(t, firstOutput, secondOutput, 0)
}

func mustSimpleRNN(
	tb testing.TB,
	config layer.SimpleRNNConfig,
	inputWeightValues,
	recurrentWeightValues,
	biasValues []float32,
) (recurrent *layer.SimpleRNN) {
	var (
		biases *matrix.Matrix
		err    error
	)

	tb.Helper()
	recurrent, err = layer.NewSimpleRNN(
		config,
		func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
			weights, err = matrix.FromSlice(inputSize, outputSize, inputWeightValues)
			return weights, err
		},
		func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
			weights, err = matrix.FromSlice(inputSize, outputSize, recurrentWeightValues)
			return weights, err
		},
	)
	if err != nil {
		tb.Fatalf("NewSimpleRNN returned error: %v", err)
	}

	biases = mustMatrix(tb, 1, config.HiddenSize(), biasValues)
	if err = recurrent.Biases().Values().CopyFrom(biases); err != nil {
		tb.Fatalf("bias CopyFrom returned error: %v", err)
	}

	return recurrent
}

func mustDeterministicSimpleRNN(tb testing.TB) (recurrent *layer.SimpleRNN) {
	var (
		config layer.SimpleRNNConfig
		err    error
	)

	tb.Helper()
	config, err = layer.NewSimpleRNNConfig(mustSequenceShape(tb, 3, 2), 2)
	if err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	recurrent, err = layer.NewSimpleRNN(
		config,
		layer.UniformWeights(-0.5, 0.5, rand.New(rand.NewSource(17))),
		layer.UniformWeights(-0.25, 0.25, rand.New(rand.NewSource(19))),
	)
	if err != nil {
		tb.Fatalf("NewSimpleRNN returned error: %v", err)
	}

	return recurrent
}
