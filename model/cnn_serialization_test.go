package model_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Sequential_SaveLoadRoundTripWithCNNLayers(t *testing.T) {
	type testcase struct {
		name           string
		currentLayer   layer.Layer
		input          *matrix.Matrix
		wantType       string
		wantFields     []string
		wantParameters [][]float32
	}

	var (
		convShape  layer.SpatialShape
		convConfig layer.Conv2DConfig
		poolShape  layer.SpatialShape
		poolConfig layer.MaxPool2DConfig
		tests      []testcase
		tt         testcase
	)

	convShape = mustSerializationSpatialShape(t, 1, 2, 3)
	convConfig = mustSerializationConv2DConfig(t, convShape, 2, 2, 2, 1, 2, 1, 1)
	poolShape = mustSerializationSpatialShape(t, 2, 3, 4)
	poolConfig = mustSerializationMaxPool2DConfig(t, poolShape, 2, 3, 1, 1)
	tests = []testcase{
		{
			name: "conv2d",
			currentLayer: mustSerializationConv2D(
				t,
				convConfig,
				[]float32{
					0.1, 0.2,
					0.3, 0.4,
					0.5, 0.6,
					0.7, 0.8,
				},
				[]float32{0.25, -0.5},
			),
			input: mustMatrix(t, 2, 6, []float32{
				1, 2, 3, 4, 5, 6,
				-1, -2, -3, -4, -5, -6,
			}),
			wantType: "conv2d",
			wantFields: []string{
				`"input_channels": 1`,
				`"input_height": 2`,
				`"input_width": 3`,
				`"output_channels": 2`,
				`"kernel_height": 2`,
				`"kernel_width": 2`,
				`"stride_height": 1`,
				`"stride_width": 2`,
				`"padding_height": 1`,
				`"padding_width": 1`,
			},
			wantParameters: [][]float32{
				{
					0.1, 0.2,
					0.3, 0.4,
					0.5, 0.6,
					0.7, 0.8,
				},
				{0.25, -0.5},
			},
		},
		{
			name:         "max pool2d",
			currentLayer: mustSerializationMaxPool2D(t, poolConfig),
			input: mustMatrix(t, 2, 24, []float32{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
				-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12,
				12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
				-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
			}),
			wantType: "max_pool2d",
			wantFields: []string{
				`"input_channels": 2`,
				`"input_height": 3`,
				`"input_width": 4`,
				`"window_height": 2`,
				`"window_width": 3`,
				`"stride_height": 1`,
				`"stride_width": 1`,
			},
		},
		{
			name:         "flatten",
			currentLayer: mustSerializationFlatten(t, 2, 2, 3),
			input: mustMatrix(t, 2, 12, []float32{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
				13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
			}),
			wantType: "flatten",
			wantFields: []string{
				`"input_channels": 2`,
				`"input_height": 2`,
				`"input_width": 3`,
			},
		},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				network          *model.Sequential
				loaded           *model.Sequential
				before           *matrix.Matrix
				after            *matrix.Matrix
				parameters       []*optimizer.Parameter
				firstDocument    bytes.Buffer
				secondDocument   bytes.Buffer
				field            string
				parameterValues  []float32
				parameterIndex   int
				documentContents string
				err              error
			)

			network, err = model.NewSequential(tt.currentLayer)
			if err != nil {
				t.Fatalf("NewSequential returned error: %v", err)
			}

			before, err = network.Predict(tt.input)
			if err != nil {
				t.Fatalf("Predict returned error: %v", err)
			}

			if err = network.Save(&firstDocument); err != nil {
				t.Fatalf("Save returned error: %v", err)
			}

			documentContents = firstDocument.String()
			if !strings.Contains(documentContents, `"type": "`+tt.wantType+`"`) {
				t.Fatalf("serialized model missing %s layer: %s", tt.wantType, documentContents)
			}

			for _, field = range tt.wantFields {
				if !strings.Contains(documentContents, field) {
					t.Fatalf("serialized model missing field %q: %s", field, documentContents)
				}
			}

			loaded, err = model.LoadSequential(bytes.NewReader(firstDocument.Bytes()))
			if err != nil {
				t.Fatalf("LoadSequential returned error: %v", err)
			}

			after, err = loaded.Predict(tt.input)
			if err != nil {
				t.Fatalf("loaded Predict returned error: %v", err)
			}

			requireMatrixValues(t, after, mustValues(t, before))
			parameters = loaded.Parameters()
			if len(parameters) != len(tt.wantParameters) {
				t.Fatalf("loaded Parameters length = %d, want %d", len(parameters), len(tt.wantParameters))
			}

			for parameterIndex, parameterValues = range tt.wantParameters {
				requireMatrixValues(t, parameters[parameterIndex].Values(), parameterValues)
			}

			if err = loaded.Save(&secondDocument); err != nil {
				t.Fatalf("loaded Save returned error: %v", err)
			}

			if !bytes.Equal(secondDocument.Bytes(), firstDocument.Bytes()) {
				t.Fatalf("loaded Save bytes differ:\nfirst:\n%s\nsecond:\n%s", firstDocument.String(), secondDocument.String())
			}
		})
	}
}

func Test_Sequential_SaveLoadRoundTripWithMixedCNNModel(t *testing.T) {
	var (
		network          *model.Sequential
		loaded           *model.Sequential
		input            *matrix.Matrix
		before           *matrix.Matrix
		after            *matrix.Matrix
		parameters       []*optimizer.Parameter
		document         bytes.Buffer
		documentContents string
		typeName         string
		err              error
	)

	network = mustCNNSerializationModel(t)
	input = mustMatrix(t, 2, 9, []float32{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
		9, 8, 7, 6, 5, 4, 3, 2, 1,
	})
	before, err = network.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	if err = network.Save(&document); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	documentContents = document.String()
	for _, typeName = range []string{"conv2d", "activation", "max_pool2d", "flatten", "dense"} {
		if !strings.Contains(documentContents, `"type": "`+typeName+`"`) {
			t.Fatalf("serialized mixed model missing %s layer: %s", typeName, documentContents)
		}
	}

	for _, typeName = range []string{"gradient", "optimizer", "scratch", "argmax", "forward_rows"} {
		if strings.Contains(documentContents, typeName) {
			t.Fatalf("serialized mixed model contains runtime state %q: %s", typeName, documentContents)
		}
	}

	loaded, err = model.LoadSequential(&document)
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	after, err = loaded.Predict(input)
	if err != nil {
		t.Fatalf("loaded Predict returned error: %v", err)
	}

	requireMatrixValues(t, after, mustValues(t, before))
	parameters = loaded.Parameters()
	if len(parameters) != 4 {
		t.Fatalf("loaded Parameters length = %d, want 4", len(parameters))
	}

	requireMatrixValues(t, parameters[0].Values(), []float32{0.1, 0.2, 0.3, 0.4})
	requireMatrixValues(t, parameters[1].Values(), []float32{0.5})
	requireMatrixValues(t, parameters[2].Values(), []float32{0.75, -0.5})
	requireMatrixValues(t, parameters[3].Values(), []float32{0.1, -0.2})
}

func Test_LoadSequential_CNNRuntimeStateStartsFreshAndCanTrain(t *testing.T) {
	var (
		network             *model.Sequential
		loaded              *model.Sequential
		input               *matrix.Matrix
		outputGradient      *matrix.Matrix
		targets             *matrix.Matrix
		adam                *optimizer.Adam
		parameters          []*optimizer.Parameter
		parameter           *optimizer.Parameter
		gradientValues      []float32
		gradientValue       float32
		convWeightsBefore   []float32
		convWeightsAfter    []float32
		hasOriginalGradient bool
		convWeightsChanged  bool
		document            bytes.Buffer
		index               int
		err                 error
	)

	network = mustCNNSerializationModel(t)
	input = mustMatrix(t, 1, 9, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 1, 2, []float32{1, -2})
	if _, err = network.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	for _, parameter = range network.Parameters() {
		gradientValues = mustValues(t, parameter.Gradient())
		for _, gradientValue = range gradientValues {
			if gradientValue != 0 {
				hasOriginalGradient = true
			}
		}
	}

	if !hasOriginalGradient {
		t.Fatal("original model gradients are all zero, want accumulated state before Save")
	}

	if err = network.Save(&document); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	loaded, err = model.LoadSequential(&document)
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	parameters = loaded.Parameters()
	for _, parameter = range parameters {
		requireMatrixValues(t, parameter.Gradient(), make([]float32, parameter.Gradient().Rows()*parameter.Gradient().Cols()))
	}

	if _, err = loaded.Backward(outputGradient); err == nil {
		t.Fatal("loaded Backward error = nil, want fresh forward-state error")
	} else if !strings.Contains(err.Error(), "backward called before forward") {
		t.Fatalf("loaded Backward error = %q, want fresh forward-state context", err)
	}

	convWeightsBefore = mustValues(t, parameters[0].Values())
	targets = mustMatrix(t, 1, 2, []float32{0, 0})
	adam, err = optimizer.NewAdam(0.01)
	if err != nil {
		t.Fatalf("NewAdam returned error: %v", err)
	}

	if _, err = loaded.TrainBatch(input, targets, loss.MeanSquaredError{}, adam); err != nil {
		t.Fatalf("loaded TrainBatch returned error: %v", err)
	}

	convWeightsAfter = mustValues(t, parameters[0].Values())
	for index = range convWeightsBefore {
		if convWeightsBefore[index] != convWeightsAfter[index] {
			convWeightsChanged = true
		}
	}

	if !convWeightsChanged {
		t.Fatal("loaded TrainBatch did not update convolution weights")
	}

	for _, parameter = range parameters {
		requireMatrixValues(t, parameter.Gradient(), make([]float32, parameter.Gradient().Rows()*parameter.Gradient().Cols()))
	}
}

func Test_LoadSequential_RejectsMalformedCNNLayers(t *testing.T) {
	type testcase struct {
		name      string
		layerJSON string
		wantError string
	}

	var (
		tests    []testcase
		tt       testcase
		document string
		loaded   *model.Sequential
		err      error
	)

	tests = []testcase{
		{
			name: "conv2d missing input shape",
			layerJSON: `{
				"type": "conv2d",
				"output_channels": 1,
				"kernel_height": 2,
				"kernel_width": 2,
				"stride_height": 1,
				"stride_width": 1,
				"weights": {"rows": 4, "cols": 1, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 1, "values": [0]}
			}`,
			wantError: "conv2d input shape is missing",
		},
		{
			name: "conv2d invalid input dimension",
			layerJSON: `{
				"type": "conv2d",
				"input_channels": -1,
				"input_height": 3,
				"input_width": 3,
				"output_channels": 1,
				"kernel_height": 2,
				"kernel_width": 2,
				"stride_height": 1,
				"stride_width": 1,
				"weights": {"rows": 4, "cols": 1, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 1, "values": [0]}
			}`,
			wantError: "conv2d input shape load failed",
		},
		{
			name: "conv2d missing configuration field",
			layerJSON: `{
				"type": "conv2d",
				"input_channels": 1,
				"input_height": 3,
				"input_width": 3,
				"kernel_height": 2,
				"kernel_width": 2,
				"stride_height": 1,
				"stride_width": 1,
				"weights": {"rows": 4, "cols": 1, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 1, "values": [0]}
			}`,
			wantError: "output channels must be positive",
		},
		{
			name: "conv2d missing weights",
			layerJSON: `{
				"type": "conv2d",
				"input_channels": 1,
				"input_height": 3,
				"input_width": 3,
				"output_channels": 1,
				"kernel_height": 2,
				"kernel_width": 2,
				"stride_height": 1,
				"stride_width": 1,
				"biases": {"rows": 1, "cols": 1, "values": [0]}
			}`,
			wantError: "conv2d weights are missing",
		},
		{
			name: "conv2d missing biases",
			layerJSON: `{
				"type": "conv2d",
				"input_channels": 1,
				"input_height": 3,
				"input_width": 3,
				"output_channels": 1,
				"kernel_height": 2,
				"kernel_width": 2,
				"stride_height": 1,
				"stride_width": 1,
				"weights": {"rows": 4, "cols": 1, "values": [1, 1, 1, 1]}
			}`,
			wantError: "conv2d biases are missing",
		},
		{
			name: "conv2d malformed weights",
			layerJSON: `{
				"type": "conv2d",
				"input_channels": 1,
				"input_height": 3,
				"input_width": 3,
				"output_channels": 1,
				"kernel_height": 2,
				"kernel_width": 2,
				"stride_height": 1,
				"stride_width": 1,
				"weights": {"rows": 4, "cols": 1, "values": [1]},
				"biases": {"rows": 1, "cols": 1, "values": [0]}
			}`,
			wantError: "conv2d weights load failed",
		},
		{
			name: "conv2d wrong weight shape",
			layerJSON: `{
				"type": "conv2d",
				"input_channels": 1,
				"input_height": 3,
				"input_width": 3,
				"output_channels": 1,
				"kernel_height": 2,
				"kernel_width": 2,
				"stride_height": 1,
				"stride_width": 1,
				"weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 1, "values": [0]}
			}`,
			wantError: "initializer weights shape mismatch",
		},
		{
			name: "conv2d wrong bias shape",
			layerJSON: `{
				"type": "conv2d",
				"input_channels": 1,
				"input_height": 3,
				"input_width": 3,
				"output_channels": 1,
				"kernel_height": 2,
				"kernel_width": 2,
				"stride_height": 1,
				"stride_width": 1,
				"weights": {"rows": 4, "cols": 1, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "conv2d biases copy failed",
		},
		{
			name:      "max pool2d missing input shape",
			layerJSON: `{"type": "max_pool2d", "window_height": 2, "window_width": 2, "stride_height": 1, "stride_width": 1}`,
			wantError: "max_pool2d input shape is missing",
		},
		{
			name:      "max pool2d invalid configuration",
			layerJSON: `{"type": "max_pool2d", "input_channels": 1, "input_height": 3, "input_width": 3, "window_height": 4, "window_width": 2, "stride_height": 1, "stride_width": 1}`,
			wantError: "max pool2d configuration load failed",
		},
		{
			name:      "flatten missing input shape",
			layerJSON: `{"type": "flatten"}`,
			wantError: "flatten input shape is missing",
		},
		{
			name:      "flatten invalid input dimension",
			layerJSON: `{"type": "flatten", "input_channels": 1, "input_height": -2, "input_width": 3}`,
			wantError: "flatten input shape load failed",
		},
		{
			name:      "unknown layer type has index context",
			layerJSON: `{"type": "average_pool2d"}`,
			wantError: `unknown layer type "average_pool2d"`,
		},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			document = `{"format":"neuralnetwork.sequential","version":1,"layers":[` + tt.layerJSON + `]}`
			loaded, err = model.LoadSequential(strings.NewReader(document))
			if err == nil {
				t.Fatal("LoadSequential error = nil, want error")
			}

			if loaded != nil {
				t.Fatal("LoadSequential returned model on error")
			}

			if !strings.Contains(err.Error(), "layer 0") {
				t.Fatalf("LoadSequential error = %q, want layer index context", err)
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("LoadSequential error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func Test_Sequential_SaveRejectsInvalidCNNLayers(t *testing.T) {
	type testcase struct {
		name         string
		currentLayer layer.Layer
		wantError    string
	}

	var (
		tests []testcase
		tt    testcase
	)

	tests = []testcase{
		{name: "conv2d zero value", currentLayer: &layer.Conv2D{}, wantError: "conv2d configuration serialize failed"},
		{name: "max pool2d zero value", currentLayer: &layer.MaxPool2D{}, wantError: "max pool2d configuration serialize failed"},
		{name: "flatten zero value", currentLayer: &layer.Flatten{}, wantError: "flatten input shape serialize failed"},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				network *model.Sequential
				output  bytes.Buffer
				err     error
			)

			network, err = model.NewSequential(tt.currentLayer)
			if err != nil {
				t.Fatalf("NewSequential returned error: %v", err)
			}

			if err = network.Save(&output); err == nil {
				t.Fatal("Save error = nil, want error")
			}

			if !strings.Contains(err.Error(), "layer 0") {
				t.Fatalf("Save error = %q, want layer index context", err)
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Save error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func Test_LoadSequential_PreservesANNFixtureBytes(t *testing.T) {
	const fixture = `{
  "format": "neuralnetwork.sequential",
  "version": 1,
  "layers": [
    {
      "type": "activation",
      "activation": "relu"
    }
  ]
}
`

	var (
		loaded *model.Sequential
		output bytes.Buffer
		err    error
	)

	loaded, err = model.LoadSequential(strings.NewReader(fixture))
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	if err = loaded.Save(&output); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	if output.String() != fixture {
		t.Fatalf("ANN fixture bytes changed:\nwant:\n%s\ngot:\n%s", fixture, output.String())
	}
}

func mustCNNSerializationModel(tb testing.TB) (network *model.Sequential) {
	var (
		inputShape      layer.SpatialShape
		convConfig      layer.Conv2DConfig
		convLayer       *layer.Conv2D
		activationLayer *layer.Activation
		poolConfig      layer.MaxPool2DConfig
		poolLayer       *layer.MaxPool2D
		flattenLayer    *layer.Flatten
		denseLayer      *layer.Dense
		err             error
	)

	tb.Helper()
	inputShape = mustSerializationSpatialShape(tb, 1, 3, 3)
	convConfig = mustSerializationConv2DConfig(tb, inputShape, 1, 2, 2, 1, 1, 0, 0)
	convLayer = mustSerializationConv2D(tb, convConfig, []float32{0.1, 0.2, 0.3, 0.4}, []float32{0.5})
	activationLayer = mustActivationLayer(tb, activation.ReLU{})
	poolConfig = mustSerializationMaxPool2DConfig(tb, convConfig.OutputShape(), 2, 2, 1, 1)
	poolLayer = mustSerializationMaxPool2D(tb, poolConfig)
	flattenLayer = mustSerializationFlatten(tb, 1, 1, 1)
	denseLayer = mustSerializationDense(tb, 1, 2, []float32{0.75, -0.5}, []float32{0.1, -0.2})

	network, err = model.NewSequential(convLayer, activationLayer, poolLayer, flattenLayer, denseLayer)
	if err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}

func mustSerializationConv2D(
	tb testing.TB,
	config layer.Conv2DConfig,
	weightValues,
	biasValues []float32,
) (convLayer *layer.Conv2D) {
	var (
		biases *matrix.Matrix
		err    error
	)

	tb.Helper()
	convLayer, err = layer.NewConv2D(config, func(inputSize, outputSize int) (weights *matrix.Matrix, err error) {
		weights, err = matrix.FromSlice(inputSize, outputSize, weightValues)
		return weights, err
	})
	if err != nil {
		tb.Fatalf("NewConv2D returned error: %v", err)
	}

	biases = mustMatrix(tb, 1, config.OutputChannels(), biasValues)
	if err = convLayer.Biases().Values().CopyFrom(biases); err != nil {
		tb.Fatalf("conv2d bias CopyFrom returned error: %v", err)
	}

	return convLayer
}

func mustSerializationConv2DConfig(
	tb testing.TB,
	inputShape layer.SpatialShape,
	outputChannels,
	kernelHeight,
	kernelWidth,
	strideHeight,
	strideWidth,
	paddingHeight,
	paddingWidth int,
) (config layer.Conv2DConfig) {
	var err error

	tb.Helper()
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

func mustSerializationMaxPool2D(tb testing.TB, config layer.MaxPool2DConfig) (poolLayer *layer.MaxPool2D) {
	var err error

	tb.Helper()
	poolLayer, err = layer.NewMaxPool2D(config)
	if err != nil {
		tb.Fatalf("NewMaxPool2D returned error: %v", err)
	}

	return poolLayer
}

func mustSerializationMaxPool2DConfig(
	tb testing.TB,
	inputShape layer.SpatialShape,
	windowHeight,
	windowWidth,
	strideHeight,
	strideWidth int,
) (config layer.MaxPool2DConfig) {
	var err error

	tb.Helper()
	config, err = layer.NewMaxPool2DConfig(inputShape, windowHeight, windowWidth, strideHeight, strideWidth)
	if err != nil {
		tb.Fatalf("NewMaxPool2DConfig returned error: %v", err)
	}

	return config
}

func mustSerializationFlatten(tb testing.TB, channels, height, width int) (flattenLayer *layer.Flatten) {
	var (
		shape layer.SpatialShape
		err   error
	)

	tb.Helper()
	shape = mustSerializationSpatialShape(tb, channels, height, width)
	flattenLayer, err = layer.NewFlatten(shape)
	if err != nil {
		tb.Fatalf("NewFlatten returned error: %v", err)
	}

	return flattenLayer
}

func mustSerializationSpatialShape(tb testing.TB, channels, height, width int) (shape layer.SpatialShape) {
	var err error

	tb.Helper()
	shape, err = layer.NewSpatialShape(channels, height, width)
	if err != nil {
		tb.Fatalf("NewSpatialShape returned error: %v", err)
	}

	return shape
}
