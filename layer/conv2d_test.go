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

func Test_Conv2D_ImplementsLayer(t *testing.T) {
	var _ layer.Layer = (*layer.Conv2D)(nil)
}

func Test_NewConv2D_UsesDocumentedInitializerShape(t *testing.T) {
	var (
		config            layer.Conv2DConfig
		conv              *layer.Conv2D
		initializerSource *matrix.Matrix
		gotFanIn          int
		gotFanOut         int
		weightValue       float32
		err               error
	)

	config = mustConv2DConfig(t, 2, 3, 4, 3, 2, 3, 1, 1, 0, 0)
	conv, err = layer.NewConv2D(config, func(fanIn, fanOut int) (weights *matrix.Matrix, err error) {
		gotFanIn = fanIn
		gotFanOut = fanOut
		initializerSource, err = matrix.New(fanIn, fanOut)
		return initializerSource, err
	})
	if err != nil {
		t.Fatalf("NewConv2D returned error: %v", err)
	}

	if gotFanIn != 12 {
		t.Fatalf("initializer fanIn = %d, want 12", gotFanIn)
	}

	if gotFanOut != 3 {
		t.Fatalf("initializer fanOut = %d, want 3", gotFanOut)
	}

	if conv.Weights().Values() == initializerSource {
		t.Fatal("weights retain initializer-owned matrix")
	}

	if err = initializerSource.Set(0, 0, 99); err != nil {
		t.Fatalf("initializer source Set returned error: %v", err)
	}

	weightValue, err = conv.Weights().Values().At(0, 0)
	if err != nil {
		t.Fatalf("weight At returned error: %v", err)
	}

	if weightValue != 0 {
		t.Fatalf("weight value = %g after source mutation, want 0", weightValue)
	}

	requireMatrixValues(t, conv.Biases().Values(), []float32{0, 0, 0})
}

func Test_NewConv2D_ValidatesDependenciesAndInitializerOutput(t *testing.T) {
	type testcase struct {
		name        string
		config      layer.Conv2DConfig
		initializer layer.WeightInitializer
		wantError   string
	}

	validConfig := mustConv2DConfig(t, 2, 3, 4, 3, 2, 2, 1, 1, 0, 0)
	tests := []testcase{
		{
			name:        "zero configuration",
			config:      layer.Conv2DConfig{},
			initializer: layer.ZeroWeights,
			wantError:   "configuration invalid",
		},
		{
			name:        "nil initializer",
			config:      validConfig,
			initializer: nil,
			wantError:   "weight initializer is nil",
		},
		{
			name:   "initializer error",
			config: validConfig,
			initializer: func(fanIn, fanOut int) (weights *matrix.Matrix, err error) {
				err = errors.New("initializer failed")
				return nil, err
			},
			wantError: "initialize weights: initializer failed",
		},
		{
			name:   "nil initializer output",
			config: validConfig,
			initializer: func(fanIn, fanOut int) (weights *matrix.Matrix, err error) {
				return nil, nil
			},
			wantError: "initializer weights is nil",
		},
		{
			name:   "initializer row mismatch",
			config: validConfig,
			initializer: func(fanIn, fanOut int) (weights *matrix.Matrix, err error) {
				weights, err = matrix.New(fanIn-1, fanOut)
				return weights, err
			},
			wantError: "got 7x3, want 8x3",
		},
		{
			name:   "initializer column mismatch",
			config: validConfig,
			initializer: func(fanIn, fanOut int) (weights *matrix.Matrix, err error) {
				weights, err = matrix.New(fanIn, fanOut-1)
				return weights, err
			},
			wantError: "got 8x2, want 8x3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				conv *layer.Conv2D
				err  error
			)

			conv, err = layer.NewConv2D(tt.config, tt.initializer)
			if err == nil {
				t.Fatal("NewConv2D error = nil, want error")
			}

			if conv != nil {
				t.Fatal("NewConv2D returned layer on error")
			}

			if !strings.HasPrefix(err.Error(), "layer: ") {
				t.Fatalf("NewConv2D error = %q, want layer context", err)
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("NewConv2D error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func Test_NewConv2D_SeededInitializationIsDeterministic(t *testing.T) {
	var (
		config       layer.Conv2DConfig
		first        *layer.Conv2D
		second       *layer.Conv2D
		firstValues  []float32
		secondValues []float32
		err          error
	)

	config = mustConv2DConfig(t, 2, 3, 3, 2, 2, 2, 1, 1, 0, 0)
	first, err = layer.NewConv2D(config, layer.UniformWeights(-1, 1, rand.New(rand.NewSource(41))))
	if err != nil {
		t.Fatalf("first NewConv2D returned error: %v", err)
	}

	second, err = layer.NewConv2D(config, layer.UniformWeights(-1, 1, rand.New(rand.NewSource(41))))
	if err != nil {
		t.Fatalf("second NewConv2D returned error: %v", err)
	}

	firstValues, err = first.Weights().Values().Values()
	if err != nil {
		t.Fatalf("first Values returned error: %v", err)
	}

	secondValues, err = second.Weights().Values().Values()
	if err != nil {
		t.Fatalf("second Values returned error: %v", err)
	}

	testutil.RequireSliceAlmostEqual(t, firstValues, secondValues, 0)
}

func Test_Conv2D_AccessorsAndParameterOrder(t *testing.T) {
	var (
		config             layer.Conv2DConfig
		conv               *layer.Conv2D
		parameters         []*optimizer.Parameter
		appendedParameters []*optimizer.Parameter
		network            *model.Sequential
		err                error
	)

	config = mustConv2DConfig(t, 2, 4, 5, 3, 2, 3, 2, 1, 1, 0)
	conv = mustConv2D(t, config, make([]float32, 12*3), []float32{0, 0, 0})

	if conv.Config() != config {
		t.Fatalf("Config = %#v, want %#v", conv.Config(), config)
	}

	if conv.InputShape() != config.InputShape() {
		t.Fatalf("InputShape = %#v, want %#v", conv.InputShape(), config.InputShape())
	}

	if conv.OutputShape() != config.OutputShape() {
		t.Fatalf("OutputShape = %#v, want %#v", conv.OutputShape(), config.OutputShape())
	}

	parameters = conv.Parameters()
	if len(parameters) != 2 {
		t.Fatalf("Parameters length = %d, want 2", len(parameters))
	}

	if parameters[0] != conv.Weights() || parameters[1] != conv.Biases() {
		t.Fatal("Parameters did not return weights and biases in order")
	}
	parameters[0] = nil
	if conv.Weights() == nil {
		t.Fatal("mutating Parameters result changed Conv2D weights")
	}

	appendedParameters = make([]*optimizer.Parameter, 1, 3)
	appendedParameters[0] = conv.Biases()
	appendedParameters = conv.AppendParameters(appendedParameters)
	if len(appendedParameters) != 3 {
		t.Fatalf("AppendParameters length = %d, want 3", len(appendedParameters))
	}

	if appendedParameters[0] != conv.Biases() {
		t.Fatal("AppendParameters changed the existing prefix")
	}

	if appendedParameters[1] != conv.Weights() || appendedParameters[2] != conv.Biases() {
		t.Fatal("AppendParameters did not append weights and biases in order")
	}

	appendedParameters[1] = nil
	if conv.Weights() == nil {
		t.Fatal("mutating AppendParameters result changed Conv2D weights")
	}

	network, err = model.NewSequential(conv)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	parameters = network.Parameters()
	if len(parameters) != 2 || parameters[0] != conv.Weights() || parameters[1] != conv.Biases() {
		t.Fatal("Sequential did not discover Conv2D parameters in weight, bias order")
	}
}

func Test_Conv2D_NilReceiverAndZeroValue(t *testing.T) {
	var (
		nilConv  *layer.Conv2D
		zeroConv layer.Conv2D
		prefix   []*optimizer.Parameter
		err      error
	)

	if nilConv.Config() != (layer.Conv2DConfig{}) {
		t.Fatal("Config returned value for nil receiver")
	}

	if nilConv.InputShape() != (layer.SpatialShape{}) {
		t.Fatal("InputShape returned value for nil receiver")
	}

	if nilConv.OutputShape() != (layer.SpatialShape{}) {
		t.Fatal("OutputShape returned value for nil receiver")
	}

	if nilConv.Weights() != nil || nilConv.Biases() != nil || nilConv.Parameters() != nil {
		t.Fatal("parameter accessor returned value for nil receiver")
	}

	prefix = []*optimizer.Parameter{nil}
	prefix = nilConv.AppendParameters(prefix)
	if len(prefix) != 1 {
		t.Fatalf("AppendParameters length = %d, want 1", len(prefix))
	}

	if _, err = nilConv.Forward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("Forward error = nil, want nil receiver error")
	}

	if _, err = nilConv.Backward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("Backward error = nil, want nil receiver error")
	}

	if err = nilConv.ResetGradients(); err == nil {
		t.Fatal("ResetGradients error = nil, want nil receiver error")
	}

	if _, err = zeroConv.Forward(mustMatrix(t, 1, 1, []float32{1})); err == nil {
		t.Fatal("zero-value Forward error = nil, want invalid state error")
	}
}

func Test_Conv2D_Forward(t *testing.T) {
	type testcase struct {
		name       string
		config     layer.Conv2DConfig
		weights    []float32
		biases     []float32
		rows       int
		input      []float32
		wantOutput []float32
	}

	tests := []testcase{
		{
			name:    "single channel rectangular kernel",
			config:  mustConv2DConfig(t, 1, 3, 4, 1, 2, 3, 1, 1, 0, 0),
			weights: []float32{1, 2, 3, 4, 5, 6},
			biases:  []float32{0.5},
			rows:    1,
			input: []float32{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
			},
			wantOutput: []float32{106.5, 127.5, 190.5, 211.5},
		},
		{
			name:   "multiple channels filters and batches",
			config: mustConv2DConfig(t, 2, 1, 2, 2, 1, 1, 1, 1, 0, 0),
			weights: []float32{
				1, 2,
				-1, 0.5,
			},
			biases: []float32{0.25, -0.5},
			rows:   2,
			input: []float32{
				1, 2,
				3, 4,
				-1, 0.5,
				2, -2,
			},
			wantOutput: []float32{
				-1.75, -1.75,
				3, 5.5,
				-2.75, 2.75,
				-1.5, -0.5,
			},
		},
		{
			name:   "rectangular stride and padding",
			config: mustConv2DConfig(t, 1, 2, 3, 1, 2, 2, 2, 1, 1, 1),
			weights: []float32{
				1, 2,
				3, 4,
			},
			biases: []float32{0},
			rows:   1,
			input: []float32{
				1, 2, 3,
				4, 5, 6,
			},
			wantOutput: []float32{4, 11, 18, 9, 8, 14, 17, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				conv   *layer.Conv2D
				input  *matrix.Matrix
				output *matrix.Matrix
				err    error
			)

			conv = mustConv2D(t, tt.config, tt.weights, tt.biases)
			input = mustMatrix(t, tt.rows, tt.config.InputShape().Size(), tt.input)
			output, err = conv.Forward(input)
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

func Test_Conv2D_BackwardAndGradientAccumulation(t *testing.T) {
	var (
		config         layer.Conv2DConfig
		conv           *layer.Conv2D
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		inputGradient  *matrix.Matrix
		err            error
	)

	config = mustConv2DConfig(t, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1)
	conv = mustConv2D(
		t,
		config,
		[]float32{
			1, -1,
			2, 0.5,
			-0.5, 2,
			1.5, -2,
			0.25, 1,
			-1, 0.75,
			2, -0.25,
			0.5, 1.25,
		},
		[]float32{0, 0},
	)
	input = mustMatrix(t, 2, config.InputShape().Size(), []float32{
		1, 2, 3, 4,
		-1, 0, 2, -2,
		0.5, -1, 1.5, 2,
		3, -0.5, 1, 2.5,
	})
	outputGradient = mustMatrix(t, 2, config.OutputShape().Size(), []float32{
		1, -2, 0.5, 1.5,
		-1, 0.25, 2, -0.5,
		0.75, 1.25, -1.5, 0.5,
		0.5, -2, 1, 1.5,
	})

	if _, err = conv.Forward(input); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	inputGradient, err = conv.Backward(outputGradient)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{
		3.5, 1.5, 2, 2,
		-0.75, -4.0625, 1, -0.125,
		0.125, -4.625, -2.5, -1,
		1, 3, 2.25, 1.625,
	})
	requireMatrixValues(t, conv.Weights().Gradient(), []float32{
		7, 1,
		-0.75, 7.5,
		-5.25, 2.5,
		1.375, -0.75,
		-1.75, 4.75,
		-0.5, 5,
		-0.625, 1,
		1.25, 2.5,
	})
	requireMatrixValues(t, conv.Biases().Gradient(), []float32{2, 1.75})

	if inputGradient, err = conv.Backward(outputGradient); err != nil {
		t.Fatalf("second Backward returned error: %v", err)
	}

	requireMatrixValues(t, inputGradient, []float32{
		3.5, 1.5, 2, 2,
		-0.75, -4.0625, 1, -0.125,
		0.125, -4.625, -2.5, -1,
		1, 3, 2.25, 1.625,
	})
	requireMatrixValues(t, conv.Weights().Gradient(), []float32{
		14, 2,
		-1.5, 15,
		-10.5, 5,
		2.75, -1.5,
		-3.5, 9.5,
		-1, 10,
		-1.25, 2,
		2.5, 5,
	})
	requireMatrixValues(t, conv.Biases().Gradient(), []float32{4, 3.5})

	if err = conv.ResetGradients(); err != nil {
		t.Fatalf("ResetGradients returned error: %v", err)
	}

	requireMatrixValues(t, conv.Weights().Gradient(), make([]float32, 16))
	requireMatrixValues(t, conv.Biases().Gradient(), []float32{0, 0})
}

func Test_Conv2D_ValidatesForwardAndBackwardShapes(t *testing.T) {
	var (
		config layer.Conv2DConfig
		conv   *layer.Conv2D
		err    error
	)

	config = mustConv2DConfig(t, 1, 2, 3, 2, 1, 2, 1, 1, 0, 0)
	conv = mustConv2D(t, config, make([]float32, 2*2), []float32{0, 0})

	if _, err = conv.Forward(nil); err == nil || !strings.Contains(err.Error(), "input is nil") {
		t.Fatalf("nil Forward error = %v, want input error", err)
	}

	if _, err = conv.Forward(mustMatrix(t, 2, config.InputShape().Size()-1, make([]float32, 10))); err == nil {
		t.Fatal("Forward error = nil, want input shape error")
	} else if !strings.Contains(err.Error(), "got 2x5, want batch rows x 6") {
		t.Fatalf("Forward error = %q, want diagnostic shape", err)
	}

	if _, err = conv.Backward(mustMatrix(t, 1, config.OutputShape().Size(), make([]float32, config.OutputShape().Size()))); err == nil {
		t.Fatal("Backward error = nil, want backward-before-forward error")
	} else if !strings.Contains(err.Error(), "backward called before forward") {
		t.Fatalf("Backward error = %q, want state context", err)
	}

	if _, err = conv.Forward(mustMatrix(t, 2, config.InputShape().Size(), make([]float32, 12))); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	if _, err = conv.Backward(nil); err == nil || !strings.Contains(err.Error(), "output gradient is nil") {
		t.Fatalf("nil Backward error = %v, want output gradient error", err)
	}

	if _, err = conv.Backward(mustMatrix(t, 1, config.OutputShape().Size(), make([]float32, config.OutputShape().Size()))); err == nil {
		t.Fatal("Backward error = nil, want batch shape error")
	} else if !strings.Contains(err.Error(), "got 1x8, want 2x8") {
		t.Fatalf("Backward error = %q, want diagnostic batch shape", err)
	}

	if _, err = conv.Backward(mustMatrix(t, 2, config.OutputShape().Size()-1, make([]float32, 14))); err == nil {
		t.Fatal("Backward error = nil, want column shape error")
	} else if !strings.Contains(err.Error(), "got 2x7, want 2x8") {
		t.Fatalf("Backward error = %q, want diagnostic column shape", err)
	}
}

func Test_Conv2D_DoesNotAliasOrRetainCallerStorage(t *testing.T) {
	var (
		config              layer.Conv2DConfig
		conv                *layer.Conv2D
		input               *matrix.Matrix
		outputGradient      *matrix.Matrix
		firstOutput         *matrix.Matrix
		secondOutput        *matrix.Matrix
		firstInputGradient  *matrix.Matrix
		secondInputGradient *matrix.Matrix
		err                 error
	)

	config = mustConv2DConfig(t, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0)
	conv = mustConv2D(t, config, []float32{2}, []float32{0})
	input = mustMatrix(t, 1, 2, []float32{3, 4})
	outputGradient = mustMatrix(t, 1, 2, []float32{1, 2})

	if firstOutput, err = conv.Forward(input); err != nil {
		t.Fatalf("first Forward returned error: %v", err)
	}

	if firstOutput == input {
		t.Fatal("Forward output aliases input")
	}
	requireMatrixValues(t, input, []float32{3, 4})

	if err = input.Fill(100); err != nil {
		t.Fatalf("input Fill returned error: %v", err)
	}

	if firstInputGradient, err = conv.Backward(outputGradient); err != nil {
		t.Fatalf("first Backward returned error: %v", err)
	}

	requireMatrixValues(t, conv.Weights().Gradient(), []float32{11})
	requireMatrixValues(t, firstInputGradient, []float32{2, 4})
	requireMatrixValues(t, outputGradient, []float32{1, 2})
	if firstInputGradient == outputGradient {
		t.Fatal("Backward input gradient aliases output gradient")
	}

	if secondOutput, err = conv.Forward(firstOutput); err != nil {
		t.Fatalf("aliased Forward returned error: %v", err)
	}

	if secondOutput == firstOutput {
		t.Fatal("Forward scratch output aliases its live input")
	}

	if _, err = conv.Forward(mustMatrix(t, 1, 2, []float32{3, 4})); err != nil {
		t.Fatalf("Forward before aliased Backward returned error: %v", err)
	}

	if secondInputGradient, err = conv.Backward(firstInputGradient); err != nil {
		t.Fatalf("aliased Backward returned error: %v", err)
	}

	if secondInputGradient == firstInputGradient {
		t.Fatal("Backward scratch output aliases its live output gradient")
	}
}

func mustConv2DConfig(
	tb testing.TB,
	inputChannels, inputHeight, inputWidth int,
	outputChannels, kernelHeight, kernelWidth int,
	strideHeight, strideWidth, paddingHeight, paddingWidth int,
) (config layer.Conv2DConfig) {
	var (
		inputShape layer.SpatialShape
		err        error
	)

	tb.Helper()
	inputShape, err = layer.NewSpatialShape(inputChannels, inputHeight, inputWidth)
	if err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}

	config, err = layer.NewConv2DConfig(
		inputShape,
		outputChannels,
		kernelHeight,
		kernelWidth,
		strideHeight,
		strideWidth,
		paddingHeight,
		paddingWidth,
	)
	if err != nil {
		tb.Fatalf("NewConv2DConfig returned error: %v", err)
	}

	return config
}

func mustConv2D(
	tb testing.TB,
	config layer.Conv2DConfig,
	weightValues, biasValues []float32,
) (conv *layer.Conv2D) {
	var (
		biases *matrix.Matrix
		err    error
	)

	tb.Helper()
	conv, err = layer.NewConv2D(config, func(fanIn, fanOut int) (weights *matrix.Matrix, err error) {
		weights, err = matrix.FromSlice(fanIn, fanOut, weightValues)
		return weights, err
	})
	if err != nil {
		tb.Fatalf("NewConv2D returned error: %v", err)
	}

	biases = mustMatrix(tb, 1, config.OutputChannels(), biasValues)
	if err = conv.Biases().Values().CopyFrom(biases); err != nil {
		tb.Fatalf("bias CopyFrom returned error: %v", err)
	}

	return conv
}
