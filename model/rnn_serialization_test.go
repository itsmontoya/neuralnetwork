package model_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Sequential_SaveLoadRoundTripWithRNNLayers(t *testing.T) {
	type testcase struct {
		name           string
		currentLayer   layer.Layer
		input          *matrix.Matrix
		wantFields     []string
		wantParameters [][]float32
	}

	var (
		tests []testcase
		tt    testcase
	)

	tests = []testcase{
		{
			name: "simple rnn",
			currentLayer: mustSerializationSimpleRNN(
				t,
				3,
				2,
				2,
				[]float32{0.1, -0.2, 0.3, 0.4},
				[]float32{0.5, 0.1, -0.25, 0.2},
				[]float32{0.05, -0.1},
			),
			input: mustMatrix(t, 2, 6, []float32{
				1, 2, -1, 0.5, 0.25, -0.75,
				-0.5, 1.5, 2, -1, 0.75, 0.25,
			}),
			wantFields: []string{
				"type",
				"steps",
				"feature_size",
				"hidden_size",
				"input_weights",
				"recurrent_weights",
				"biases",
			},
			wantParameters: [][]float32{
				{0.1, -0.2, 0.3, 0.4},
				{0.5, 0.1, -0.25, 0.2},
				{0.05, -0.1},
			},
		},
		{
			name:         "last step",
			currentLayer: mustSerializationLastStep(t, 3, 2),
			input: mustMatrix(t, 2, 6, []float32{
				1, 2, 3, 4, 5, 6,
				-1, -2, -3, -4, -5, -6,
			}),
			wantFields: []string{"type", "steps", "feature_size"},
		},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				network         *model.Sequential
				loaded          *model.Sequential
				before          *matrix.Matrix
				after           *matrix.Matrix
				parameters      []*optimizer.Parameter
				firstDocument   bytes.Buffer
				secondDocument  bytes.Buffer
				fields          map[string]json.RawMessage
				field           string
				parameterValues []float32
				parameterIndex  int
				err             error
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

			fields = serializedLayerFields(t, firstDocument.Bytes())
			if len(fields) != len(tt.wantFields) {
				t.Fatalf("serialized layer field count = %d, want %d: %s", len(fields), len(tt.wantFields), firstDocument.String())
			}

			for _, field = range tt.wantFields {
				if _, ok := fields[field]; !ok {
					t.Fatalf("serialized layer missing field %q: %s", field, firstDocument.String())
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

func Test_Sequential_SaveLoadRoundTripWithMixedRNNModel(t *testing.T) {
	var (
		network          *model.Sequential
		loaded           *model.Sequential
		input            *matrix.Matrix
		before           *matrix.Matrix
		after            *matrix.Matrix
		repeated         *matrix.Matrix
		afterValues      []float32
		parameters       []*optimizer.Parameter
		document         bytes.Buffer
		documentContents string
		typeName         string
		err              error
	)

	network = mustRNNSerializationModel(t)
	input = mustMatrix(t, 2, 4, []float32{
		1, 2, -1, 0.5,
		-0.5, 1.5, 2, -1,
	})
	before, err = network.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	if err = network.Save(&document); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	documentContents = document.String()
	for _, typeName = range []string{"simple_rnn", "last_step", "dense"} {
		if !strings.Contains(documentContents, `"type": "`+typeName+`"`) {
			t.Fatalf("serialized mixed model missing %s layer: %s", typeName, documentContents)
		}
	}

	for _, typeName = range []string{"gradient", "optimizer", "scratch", "hidden_cache", "forward_rows"} {
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

	afterValues = mustValues(t, after)
	requireMatrixValues(t, after, mustValues(t, before))
	repeated, err = loaded.Predict(input)
	if err != nil {
		t.Fatalf("loaded repeated Predict returned error: %v", err)
	}

	requireMatrixValues(t, repeated, afterValues)
	parameters = loaded.Parameters()
	if len(parameters) != 5 {
		t.Fatalf("loaded Parameters length = %d, want 5", len(parameters))
	}

	requireMatrixValues(t, parameters[0].Values(), []float32{0.1, -0.2, 0.3, 0.4})
	requireMatrixValues(t, parameters[1].Values(), []float32{0.5, 0.1, -0.25, 0.2})
	requireMatrixValues(t, parameters[2].Values(), []float32{0.05, -0.1})
	requireMatrixValues(t, parameters[3].Values(), []float32{0.75, -0.5})
	requireMatrixValues(t, parameters[4].Values(), []float32{0.1})
}

func Test_LoadSequential_LastStepRuntimeStateStartsFresh(t *testing.T) {
	var (
		network        *model.Sequential
		loaded         *model.Sequential
		input          *matrix.Matrix
		outputGradient *matrix.Matrix
		document       bytes.Buffer
		err            error
	)

	network, err = model.NewSequential(mustSerializationLastStep(t, 2, 2))
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	input = mustMatrix(t, 1, 4, []float32{1, 2, 3, 4})
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	if err = network.Save(&document); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	loaded, err = model.LoadSequential(&document)
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 1, 2, []float32{1, -1})
	if _, err = loaded.Backward(outputGradient); err == nil {
		t.Fatal("loaded Backward error = nil, want fresh forward-state error")
	} else if !strings.Contains(err.Error(), "last step backward called before forward") {
		t.Fatalf("loaded Backward error = %q, want last step forward-state context", err)
	}
}

func Test_LoadSequential_RNNRuntimeStateStartsFreshAndCanTrain(t *testing.T) {
	var (
		network             *model.Sequential
		loaded              *model.Sequential
		input               *matrix.Matrix
		outputGradient      *matrix.Matrix
		targets             *matrix.Matrix
		sgd                 *optimizer.SGD
		parameters          []*optimizer.Parameter
		parameter           *optimizer.Parameter
		gradientValues      []float32
		gradientValue       float32
		inputWeightsBefore  []float32
		inputWeightsAfter   []float32
		recurrentBefore     []float32
		recurrentAfter      []float32
		hasOriginalGradient bool
		inputWeightsChanged bool
		recurrentChanged    bool
		document            bytes.Buffer
		index               int
		err                 error
	)

	network, err = model.NewSequential(mustSerializationSimpleRNN(
		t,
		2,
		2,
		2,
		[]float32{0.1, -0.2, 0.3, 0.4},
		[]float32{0.5, 0.1, -0.25, 0.2},
		[]float32{0.05, -0.1},
	))
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	input = mustMatrix(t, 1, 4, []float32{1, 2, -1, 0.5})
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	outputGradient = mustMatrix(t, 1, 4, []float32{1, -2, 0.5, 1.5})
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
	if len(parameters) != 3 {
		t.Fatalf("loaded Parameters length = %d, want 3", len(parameters))
	}

	for _, parameter = range parameters {
		requireMatrixValues(t, parameter.Gradient(), make([]float32, parameter.Gradient().Rows()*parameter.Gradient().Cols()))
	}

	if _, err = loaded.Backward(outputGradient); err == nil {
		t.Fatal("loaded Backward error = nil, want fresh forward-state error")
	} else if !strings.Contains(err.Error(), "backward called before forward") {
		t.Fatalf("loaded Backward error = %q, want fresh forward-state context", err)
	}

	if _, err = loaded.Predict(input); err != nil {
		t.Fatalf("loaded Predict returned error: %v", err)
	}

	if _, err = loaded.Backward(outputGradient); err != nil {
		t.Fatalf("loaded Backward returned error: %v", err)
	}

	for _, parameter = range parameters {
		if err = parameter.ResetGradient(); err != nil {
			t.Fatalf("ResetGradient returned error: %v", err)
		}
	}

	inputWeightsBefore = mustValues(t, parameters[0].Values())
	recurrentBefore = mustValues(t, parameters[1].Values())
	targets = mustMatrix(t, 1, 4, []float32{0, 0, 0, 0})
	sgd, err = optimizer.NewSGD(0.05)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	if _, err = loaded.TrainBatch(input, targets, loss.MeanSquaredError{}, sgd); err != nil {
		t.Fatalf("loaded TrainBatch returned error: %v", err)
	}

	inputWeightsAfter = mustValues(t, parameters[0].Values())
	recurrentAfter = mustValues(t, parameters[1].Values())
	for index = range inputWeightsBefore {
		if inputWeightsBefore[index] != inputWeightsAfter[index] {
			inputWeightsChanged = true
		}
	}

	for index = range recurrentBefore {
		if recurrentBefore[index] != recurrentAfter[index] {
			recurrentChanged = true
		}
	}

	if !inputWeightsChanged {
		t.Fatal("loaded TrainBatch did not update simple rnn input weights")
	}

	if !recurrentChanged {
		t.Fatal("loaded TrainBatch did not update simple rnn recurrent weights")
	}
}

func Test_LoadSequential_RejectsMalformedRNNLayers(t *testing.T) {
	type testcase struct {
		name      string
		layerJSON string
		wantError string
	}

	var (
		maxInt   int
		tests    []testcase
		tt       testcase
		document string
		loaded   *model.Sequential
		err      error
	)

	maxInt = int(^uint(0) >> 1)
	tests = []testcase{
		{
			name: "simple rnn missing input shape",
			layerJSON: `{
				"type": "simple_rnn",
				"hidden_size": 2,
				"input_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "simple_rnn input shape is missing",
		},
		{
			name: "simple rnn missing steps",
			layerJSON: `{
				"type": "simple_rnn",
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "simple_rnn input shape load failed",
		},
		{
			name: "simple rnn invalid feature size",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": -1,
				"hidden_size": 2,
				"input_weights": {"rows": 1, "cols": 2, "values": [1, 1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "simple_rnn input shape load failed",
		},
		{
			name: "simple rnn input shape overflow",
			layerJSON: fmt.Sprintf(`{
				"type": "simple_rnn",
				"steps": %d,
				"feature_size": 2,
				"hidden_size": 1,
				"input_weights": {"rows": 2, "cols": 1, "values": [1, 1]},
				"recurrent_weights": {"rows": 1, "cols": 1, "values": [1]},
				"biases": {"rows": 1, "cols": 1, "values": [0]}
			}`, maxInt),
			wantError: "sequence shape size overflows int",
		},
		{
			name: "simple rnn output shape overflow",
			layerJSON: fmt.Sprintf(`{
				"type": "simple_rnn",
				"steps": %d,
				"feature_size": 1,
				"hidden_size": 2,
				"input_weights": {"rows": 1, "cols": 2, "values": [1, 1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`, maxInt),
			wantError: "simple rnn output shape invalid",
		},
		{
			name: "simple rnn missing hidden size",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"input_weights": {"rows": 2, "cols": 1, "values": [1, 1]},
				"recurrent_weights": {"rows": 1, "cols": 1, "values": [1]},
				"biases": {"rows": 1, "cols": 1, "values": [0]}
			}`,
			wantError: "hidden size must be positive",
		},
		{
			name: "simple rnn missing input weights",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "simple rnn input weights are missing",
		},
		{
			name: "simple rnn missing recurrent weights",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "simple rnn recurrent weights are missing",
		},
		{
			name: "simple rnn missing biases",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]}
			}`,
			wantError: "simple rnn biases are missing",
		},
		{
			name: "simple rnn malformed input weights",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": 2, "cols": 2, "values": [1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "simple rnn input weights load failed",
		},
		{
			name: "simple rnn malformed recurrent weights",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "simple rnn recurrent weights load failed",
		},
		{
			name: "simple rnn malformed biases",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0]}
			}`,
			wantError: "simple rnn biases load failed",
		},
		{
			name: "simple rnn input weight shape mismatch",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": 1, "cols": 2, "values": [1, 1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "initializer input weights shape mismatch",
		},
		{
			name: "simple rnn recurrent weight shape mismatch",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"recurrent_weights": {"rows": 1, "cols": 4, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`,
			wantError: "initializer recurrent weights shape mismatch",
		},
		{
			name: "simple rnn bias shape mismatch",
			layerJSON: `{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 2, "cols": 1, "values": [0, 0]}
			}`,
			wantError: "simple rnn biases copy failed",
		},
		{
			name: "simple rnn matrix dimension overflow",
			layerJSON: fmt.Sprintf(`{
				"type": "simple_rnn",
				"steps": 2,
				"feature_size": 2,
				"hidden_size": 2,
				"input_weights": {"rows": %d, "cols": 2, "values": []},
				"recurrent_weights": {"rows": 2, "cols": 2, "values": [1, 1, 1, 1]},
				"biases": {"rows": 1, "cols": 2, "values": [0, 0]}
			}`, maxInt),
			wantError: "simple rnn input weights load failed",
		},
		{
			name:      "last step missing input shape",
			layerJSON: `{"type": "last_step"}`,
			wantError: "last_step input shape is missing",
		},
		{
			name:      "last step invalid steps",
			layerJSON: `{"type": "last_step", "steps": -1, "feature_size": 2}`,
			wantError: "last_step input shape load failed",
		},
		{
			name:      "last step input shape overflow",
			layerJSON: fmt.Sprintf(`{"type": "last_step", "steps": %d, "feature_size": 2}`, maxInt),
			wantError: "sequence shape size overflows int",
		},
		{
			name:      "unknown rnn layer type",
			layerJSON: `{"type": "gru"}`,
			wantError: `unknown layer type "gru"`,
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

func Test_Sequential_SaveRejectsInvalidRNNLayers(t *testing.T) {
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
		{name: "simple rnn zero value", currentLayer: &layer.SimpleRNN{}, wantError: "simple rnn input shape serialize failed"},
		{name: "last step zero value", currentLayer: &layer.LastStep{}, wantError: "last step input shape serialize failed"},
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

func mustRNNSerializationModel(tb testing.TB) (network *model.Sequential) {
	var (
		recurrentLayer *layer.SimpleRNN
		lastStepLayer  *layer.LastStep
		denseLayer     *layer.Dense
		err            error
	)

	tb.Helper()
	recurrentLayer = mustSerializationSimpleRNN(
		tb,
		2,
		2,
		2,
		[]float32{0.1, -0.2, 0.3, 0.4},
		[]float32{0.5, 0.1, -0.25, 0.2},
		[]float32{0.05, -0.1},
	)
	lastStepLayer, err = layer.NewLastStep(recurrentLayer.OutputShape())
	if err != nil {
		tb.Fatalf("NewLastStep returned error: %v", err)
	}

	denseLayer = mustSerializationDense(tb, 2, 1, []float32{0.75, -0.5}, []float32{0.1})
	network, err = model.NewSequential(recurrentLayer, lastStepLayer, denseLayer)
	if err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}

func mustSerializationSimpleRNN(
	tb testing.TB,
	steps,
	featureSize,
	hiddenSize int,
	inputWeightValues,
	recurrentWeightValues,
	biasValues []float32,
) (recurrentLayer *layer.SimpleRNN) {
	var (
		inputShape layer.SequenceShape
		config     layer.SimpleRNNConfig
		biases     *matrix.Matrix
		err        error
	)

	tb.Helper()
	inputShape, err = layer.NewSequenceShape(steps, featureSize)
	if err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}

	config, err = layer.NewSimpleRNNConfig(inputShape, hiddenSize)
	if err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}

	recurrentLayer, err = layer.NewSimpleRNN(
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

	biases = mustMatrix(tb, 1, hiddenSize, biasValues)
	if err = recurrentLayer.Biases().Values().CopyFrom(biases); err != nil {
		tb.Fatalf("simple rnn bias CopyFrom returned error: %v", err)
	}

	return recurrentLayer
}

func mustSerializationLastStep(tb testing.TB, steps, featureSize int) (lastStepLayer *layer.LastStep) {
	var (
		inputShape layer.SequenceShape
		err        error
	)

	tb.Helper()
	inputShape, err = layer.NewSequenceShape(steps, featureSize)
	if err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}

	lastStepLayer, err = layer.NewLastStep(inputShape)
	if err != nil {
		tb.Fatalf("NewLastStep returned error: %v", err)
	}

	return lastStepLayer
}

func serializedLayerFields(tb testing.TB, document []byte) (fields map[string]json.RawMessage) {
	var (
		decoded struct {
			Layers []map[string]json.RawMessage `json:"layers"`
		}
		err error
	)

	tb.Helper()
	if err = json.Unmarshal(document, &decoded); err != nil {
		tb.Fatalf("json.Unmarshal returned error: %v", err)
	}

	if len(decoded.Layers) != 1 {
		tb.Fatalf("serialized layer count = %d, want 1", len(decoded.Layers))
	}

	fields = decoded.Layers[0]
	return fields
}
