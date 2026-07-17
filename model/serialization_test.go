package model_test

import (
	"bytes"
	"math/rand"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Sequential_SaveLoadRoundTrip(t *testing.T) {
	var (
		hidden           *layer.Dense
		hiddenActivation *layer.Activation
		output           *layer.Dense
		outputActivation *layer.Activation
		network          *model.Sequential
		loaded           *model.Sequential
		input            *matrix.Matrix
		before           *matrix.Matrix
		after            *matrix.Matrix
		loadedParameters []*optimizer.Parameter
		buffer           bytes.Buffer
		err              error
	)

	hidden = mustSerializationDense(
		t,
		2,
		3,
		[]float32{
			0.5, -0.25, 0.75,
			-1, 0.4, 0.2,
		},
		[]float32{0.1, -0.2, 0.3},
	)
	hiddenActivation = mustActivationLayer(t, activation.Tanh{})
	output = mustSerializationDense(
		t,
		3,
		1,
		[]float32{
			0.8,
			-0.6,
			0.25,
		},
		[]float32{-0.1},
	)
	outputActivation = mustActivationLayer(t, activation.Sigmoid{})

	network, err = model.NewSequential(hidden, hiddenActivation, output, outputActivation)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	input = mustMatrix(t, 3, 2, []float32{
		0, 0,
		1, 0,
		0, 1,
	})
	before, err = network.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	err = network.Save(&buffer)
	if err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	if !strings.Contains(buffer.String(), `"format": "neuralnetwork.sequential"`) {
		t.Fatalf("serialized model missing sequential format: %s", buffer.String())
	}

	if strings.Contains(buffer.String(), "parameter_buffer") {
		t.Fatalf("serialized model contains parameter scratch: %s", buffer.String())
	}

	loaded, err = model.LoadSequential(&buffer)
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	if !loaded.Training() {
		t.Fatal("loaded Training = false, want true")
	}

	loadedParameters = loaded.Parameters()
	if len(loadedParameters) != 4 {
		t.Fatalf("loaded Parameters length = %d, want 4", len(loadedParameters))
	}

	requireMatrixValues(t, loadedParameters[0].Values(), []float32{
		0.5, -0.25, 0.75,
		-1, 0.4, 0.2,
	})
	requireMatrixValues(t, loadedParameters[1].Values(), []float32{0.1, -0.2, 0.3})
	requireMatrixValues(t, loadedParameters[2].Values(), []float32{0.8, -0.6, 0.25})
	requireMatrixValues(t, loadedParameters[3].Values(), []float32{-0.1})

	after, err = loaded.Predict(input)
	if err != nil {
		t.Fatalf("loaded Predict returned error: %v", err)
	}

	requireMatrixValues(t, after, mustValues(t, before))
}

func Test_Sequential_SaveLoadRoundTripWithBatchNormalization(t *testing.T) {
	var (
		batchNorm *layer.BatchNormalization
		network   *model.Sequential
		loaded    *model.Sequential
		input     *matrix.Matrix
		before    *matrix.Matrix
		after     *matrix.Matrix
		buffer    bytes.Buffer
		err       error
	)

	batchNorm, err = layer.NewBatchNormalizationWithConfig(2, 0.8, 1e-4)
	if err != nil {
		t.Fatalf("NewBatchNormalizationWithConfig returned error: %v", err)
	}

	err = batchNorm.Gamma().Values().CopyFrom(mustMatrix(t, 1, 2, []float32{2, 3}))
	if err != nil {
		t.Fatalf("gamma CopyFrom returned error: %v", err)
	}

	err = batchNorm.Beta().Values().CopyFrom(mustMatrix(t, 1, 2, []float32{0.5, -1}))
	if err != nil {
		t.Fatalf("beta CopyFrom returned error: %v", err)
	}

	err = batchNorm.RunningMean().CopyFrom(mustMatrix(t, 1, 2, []float32{1, 2}))
	if err != nil {
		t.Fatalf("running mean CopyFrom returned error: %v", err)
	}

	err = batchNorm.RunningVariance().CopyFrom(mustMatrix(t, 1, 2, []float32{4, 9}))
	if err != nil {
		t.Fatalf("running variance CopyFrom returned error: %v", err)
	}

	network, err = model.NewSequential(batchNorm)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	err = network.SetTraining(false)
	if err != nil {
		t.Fatalf("SetTraining returned error: %v", err)
	}

	input = mustMatrix(t, 2, 2, []float32{
		3, 8,
		5, 11,
	})
	before, err = network.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	err = network.Save(&buffer)
	if err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	if !strings.Contains(buffer.String(), `"type": "batch_normalization"`) {
		t.Fatalf("serialized model missing batch normalization layer: %s", buffer.String())
	}

	if !strings.Contains(buffer.String(), `"running_mean"`) {
		t.Fatalf("serialized model missing batch normalization running mean: %s", buffer.String())
	}

	loaded, err = model.LoadSequential(&buffer)
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	err = loaded.SetTraining(false)
	if err != nil {
		t.Fatalf("loaded SetTraining returned error: %v", err)
	}

	after, err = loaded.Predict(input)
	if err != nil {
		t.Fatalf("loaded Predict returned error: %v", err)
	}

	requireMatrixValues(t, after, mustValues(t, before))
}

func Test_Sequential_SaveLoadRoundTripWithDropout(t *testing.T) {
	var (
		dropout *layer.Dropout
		network *model.Sequential
		loaded  *model.Sequential
		input   *matrix.Matrix
		before  *matrix.Matrix
		after   *matrix.Matrix
		buffer  bytes.Buffer
		err     error
	)

	dropout, err = layer.NewDropout(0.25, rand.New(rand.NewSource(9)))
	if err != nil {
		t.Fatalf("NewDropout returned error: %v", err)
	}

	network, err = model.NewSequential(dropout)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	err = network.SetTraining(false)
	if err != nil {
		t.Fatalf("SetTraining returned error: %v", err)
	}

	input = mustMatrix(t, 2, 2, []float32{
		1, 2,
		3, 4,
	})
	before, err = network.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	err = network.Save(&buffer)
	if err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	if !strings.Contains(buffer.String(), `"type": "dropout"`) {
		t.Fatalf("serialized model missing dropout layer: %s", buffer.String())
	}

	if !strings.Contains(buffer.String(), `"rate": 0.25`) {
		t.Fatalf("serialized model missing dropout rate: %s", buffer.String())
	}

	loaded, err = model.LoadSequential(&buffer)
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	err = loaded.SetTraining(false)
	if err != nil {
		t.Fatalf("loaded SetTraining returned error: %v", err)
	}

	after, err = loaded.Predict(input)
	if err != nil {
		t.Fatalf("loaded Predict returned error: %v", err)
	}

	requireMatrixValues(t, after, mustValues(t, before))
}

func Test_Sequential_SaveLoadRoundTripAfterTraining(t *testing.T) {
	var (
		dense   *layer.Dense
		network *model.Sequential
		loaded  *model.Sequential
		dataset *data.Dataset
		inputs  *matrix.Matrix
		before  *matrix.Matrix
		after   *matrix.Matrix
		sgd     *optimizer.SGD
		buffer  bytes.Buffer
		err     error
	)

	dense = mustDense(t)
	network, err = model.NewSequential(dense)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	dataset = mustFitDataset(t)
	sgd, err = optimizer.NewSGD(0.1)
	if err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	_, err = network.Fit(dataset, model.FitConfig{
		Epochs:    30,
		BatchSize: 4,
		Optimizer: sgd,
		Loss:      loss.MeanSquaredError{},
	})
	if err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}

	inputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	before, err = network.Predict(inputs)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	err = network.Save(&buffer)
	if err != nil {
		t.Fatalf("Save returned error: %v", err)
	}

	loaded, err = model.LoadSequential(&buffer)
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	after, err = loaded.Predict(inputs)
	if err != nil {
		t.Fatalf("loaded Predict returned error: %v", err)
	}

	requireMatrixValues(t, after, mustValues(t, before))
}

func Test_Sequential_SaveRejectsUnsupportedLayer(t *testing.T) {
	var (
		network *model.Sequential
		buffer  bytes.Buffer
		err     error
	)

	network, err = model.NewSequential(&recordingLayer{})
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	err = network.Save(&buffer)
	if err == nil {
		t.Fatal("Save error = nil, want error")
	}

	if !strings.Contains(err.Error(), "unsupported layer type") {
		t.Fatalf("Save error = %q, want unsupported layer type", err.Error())
	}
}

func Test_Sequential_SaveRejectsNilWriter(t *testing.T) {
	var (
		network *model.Sequential
		err     error
	)

	network, err = model.NewSequential(mustDense(t))
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	err = network.Save(nil)
	if err == nil {
		t.Fatal("Save error = nil, want error")
	}

	if !strings.Contains(err.Error(), "save writer is nil") {
		t.Fatalf("Save error = %q, want save writer is nil", err.Error())
	}
}

func Test_LoadSequential_RejectsMalformedDocuments(t *testing.T) {
	type testcase struct {
		name      string
		document  string
		wantError string
	}

	var (
		tests  []testcase
		tt     testcase
		loaded *model.Sequential
		err    error
	)

	tests = []testcase{
		{
			name: "unsupported format",
			document: `{
				"format": "neuralnetwork.feedforward",
				"version": 1,
				"layers": []
			}`,
			wantError: "unsupported serialization format",
		},
		{
			name: "unsupported version",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 2,
				"layers": []
			}`,
			wantError: "unsupported serialization version",
		},
		{
			name: "trailing json",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": []
			}
			{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": []
			}`,
			wantError: "JSON contains multiple values",
		},
		{
			name: "missing dense weights",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "dense",
						"input_size": 2,
						"output_size": 1,
						"biases": {"rows": 1, "cols": 1, "values": [0.1]}
					}
				]
			}`,
			wantError: "dense weights are missing",
		},
		{
			name: "missing dense biases",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "dense",
						"input_size": 2,
						"output_size": 1,
						"weights": {"rows": 2, "cols": 1, "values": [0.2, 0.3]}
					}
				]
			}`,
			wantError: "dense biases are missing",
		},
		{
			name: "invalid dense weight shape",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "dense",
						"input_size": 2,
						"output_size": 1,
						"weights": {"rows": 1, "cols": 2, "values": [0.2, 0.3]},
						"biases": {"rows": 1, "cols": 1, "values": [0.1]}
					}
				]
			}`,
			wantError: "dense weights shape mismatch",
		},
		{
			name: "invalid dense bias shape",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "dense",
						"input_size": 2,
						"output_size": 1,
						"weights": {"rows": 2, "cols": 1, "values": [0.2, 0.3]},
						"biases": {"rows": 1, "cols": 2, "values": [0.1, 0.2]}
					}
				]
			}`,
			wantError: "dense biases copy failed",
		},
		{
			name: "invalid dense bias value count",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "dense",
						"input_size": 2,
						"output_size": 1,
						"weights": {"rows": 2, "cols": 1, "values": [0.2, 0.3]},
						"biases": {"rows": 1, "cols": 1, "values": []}
					}
				]
			}`,
			wantError: "dense biases load failed",
		},
		{
			name: "missing batch normalization gamma",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_normalization",
						"feature_size": 2,
						"momentum": 0.8,
						"epsilon": 0.0001,
						"beta": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_mean": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_variance": {"rows": 1, "cols": 2, "values": [1, 1]}
					}
				]
			}`,
			wantError: "batch normalization gamma is missing",
		},
		{
			name: "missing batch normalization beta",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_normalization",
						"feature_size": 2,
						"momentum": 0.8,
						"epsilon": 0.0001,
						"gamma": {"rows": 1, "cols": 2, "values": [1, 1]},
						"running_mean": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_variance": {"rows": 1, "cols": 2, "values": [1, 1]}
					}
				]
			}`,
			wantError: "batch normalization beta is missing",
		},
		{
			name: "missing batch normalization running mean",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_normalization",
						"feature_size": 2,
						"momentum": 0.8,
						"epsilon": 0.0001,
						"gamma": {"rows": 1, "cols": 2, "values": [1, 1]},
						"beta": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_variance": {"rows": 1, "cols": 2, "values": [1, 1]}
					}
				]
			}`,
			wantError: "batch normalization running mean is missing",
		},
		{
			name: "missing batch normalization running variance",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_normalization",
						"feature_size": 2,
						"momentum": 0.8,
						"epsilon": 0.0001,
						"gamma": {"rows": 1, "cols": 2, "values": [1, 1]},
						"beta": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_mean": {"rows": 1, "cols": 2, "values": [0, 0]}
					}
				]
			}`,
			wantError: "batch normalization running variance is missing",
		},
		{
			name: "invalid batch normalization gamma shape",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_normalization",
						"feature_size": 2,
						"momentum": 0.8,
						"epsilon": 0.0001,
						"gamma": {"rows": 1, "cols": 3, "values": [1, 1, 1]},
						"beta": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_mean": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_variance": {"rows": 1, "cols": 2, "values": [1, 1]}
					}
				]
			}`,
			wantError: "batch normalization gamma copy failed",
		},
		{
			name: "invalid batch normalization gamma value count",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_normalization",
						"feature_size": 2,
						"momentum": 0.8,
						"epsilon": 0.0001,
						"gamma": {"rows": 1, "cols": 2, "values": [1]},
						"beta": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_mean": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_variance": {"rows": 1, "cols": 2, "values": [1, 1]}
					}
				]
			}`,
			wantError: "batch normalization gamma load failed",
		},
		{
			name: "invalid batch normalization beta shape",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_normalization",
						"feature_size": 2,
						"momentum": 0.8,
						"epsilon": 0.0001,
						"gamma": {"rows": 1, "cols": 2, "values": [1, 1]},
						"beta": {"rows": 1, "cols": 3, "values": [0, 0, 0]},
						"running_mean": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_variance": {"rows": 1, "cols": 2, "values": [1, 1]}
					}
				]
			}`,
			wantError: "batch normalization beta copy failed",
		},
		{
			name: "invalid batch normalization running mean shape",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_normalization",
						"feature_size": 2,
						"momentum": 0.8,
						"epsilon": 0.0001,
						"gamma": {"rows": 1, "cols": 2, "values": [1, 1]},
						"beta": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_mean": {"rows": 1, "cols": 3, "values": [0, 0, 0]},
						"running_variance": {"rows": 1, "cols": 2, "values": [1, 1]}
					}
				]
			}`,
			wantError: "batch normalization running mean copy failed",
		},
		{
			name: "invalid batch normalization running variance shape",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_normalization",
						"feature_size": 2,
						"momentum": 0.8,
						"epsilon": 0.0001,
						"gamma": {"rows": 1, "cols": 2, "values": [1, 1]},
						"beta": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_mean": {"rows": 1, "cols": 2, "values": [0, 0]},
						"running_variance": {"rows": 1, "cols": 3, "values": [1, 1, 1]}
					}
				]
			}`,
			wantError: "batch normalization running variance copy failed",
		},
		{
			name: "invalid dropout rate",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "dropout",
						"rate": 1
					}
				]
			}`,
			wantError: "dropout construct failed",
		},
		{
			name: "empty activation name",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "activation"
					}
				]
			}`,
			wantError: "unknown activation name",
		},
		{
			name: "unsupported activation name",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "activation",
						"activation": "swish"
					}
				]
			}`,
			wantError: "unknown activation name",
		},
		{
			name: "unknown layer type",
			document: `{
				"format": "neuralnetwork.sequential",
				"version": 1,
				"layers": [
					{
						"type": "batch_norm"
					}
				]
			}`,
			wantError: "unknown layer type",
		},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			loaded, err = model.LoadSequential(strings.NewReader(tt.document))
			if err == nil {
				t.Fatal("LoadSequential error = nil, want error")
			}

			if loaded != nil {
				t.Fatal("LoadSequential returned model on error")
			}

			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("LoadSequential error = %q, want %q", err.Error(), tt.wantError)
			}
		})
	}
}

func Test_LoadSequential_RejectsNilReader(t *testing.T) {
	var (
		loaded *model.Sequential
		err    error
	)

	loaded, err = model.LoadSequential(nil)
	if err == nil {
		t.Fatal("LoadSequential error = nil, want error")
	}

	if loaded != nil {
		t.Fatal("LoadSequential returned model on error")
	}

	if !strings.Contains(err.Error(), "load reader is nil") {
		t.Fatalf("LoadSequential error = %q, want load reader is nil", err.Error())
	}
}

func Test_LoadSequential_RejectsInvalidJSON(t *testing.T) {
	var (
		loaded *model.Sequential
		err    error
	)

	loaded, err = model.LoadSequential(strings.NewReader(`{
		"format": "neuralnetwork.sequential",
			"version": 1,
			"layers": [
	}`))
	if err == nil {
		t.Fatal("LoadSequential error = nil, want error")
	}

	if loaded != nil {
		t.Fatal("LoadSequential returned model on error")
	}

	if !strings.Contains(err.Error(), "decode JSON") {
		t.Fatalf("LoadSequential error = %q, want decode JSON", err.Error())
	}
}

func mustSerializationDense(
	tb testing.TB,
	inputSize,
	outputSize int,
	weightValues,
	biasValues []float32,
) (dense *layer.Dense) {
	var (
		biases *matrix.Matrix
		err    error
	)

	tb.Helper()

	dense, err = layer.NewDense(inputSize, outputSize, func(layerInputSize, layerOutputSize int) (weights *matrix.Matrix, err error) {
		weights, err = matrix.FromSlice(layerInputSize, layerOutputSize, weightValues)
		return weights, err
	})
	if err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	biases = mustMatrix(tb, 1, outputSize, biasValues)
	err = dense.Biases().Values().CopyFrom(biases)
	if err != nil {
		tb.Fatalf("CopyFrom returned error: %v", err)
	}

	return dense
}

func mustActivationLayer(tb testing.TB, function activation.Activation) (activationLayer *layer.Activation) {
	var err error

	tb.Helper()

	activationLayer, err = layer.NewActivation(function)
	if err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	return activationLayer
}
