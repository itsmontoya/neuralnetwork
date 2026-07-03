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
		buffer           bytes.Buffer
		err              error
	)

	hidden = mustSerializationDense(
		t,
		2,
		3,
		[]float64{
			0.5, -0.25, 0.75,
			-1, 0.4, 0.2,
		},
		[]float64{0.1, -0.2, 0.3},
	)
	hiddenActivation = mustActivationLayer(t, activation.Tanh{})
	output = mustSerializationDense(
		t,
		3,
		1,
		[]float64{
			0.8,
			-0.6,
			0.25,
		},
		[]float64{-0.1},
	)
	outputActivation = mustActivationLayer(t, activation.Sigmoid{})

	network, err = model.NewSequential(hidden, hiddenActivation, output, outputActivation)
	if err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}

	input = mustMatrix(t, 3, 2, []float64{
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

	loaded, err = model.LoadSequential(&buffer)
	if err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	if !loaded.Training() {
		t.Fatal("loaded Training = false, want true")
	}

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

	err = batchNorm.Gamma().Values().CopyFrom(mustMatrix(t, 1, 2, []float64{2, 3}))
	if err != nil {
		t.Fatalf("gamma CopyFrom returned error: %v", err)
	}

	err = batchNorm.Beta().Values().CopyFrom(mustMatrix(t, 1, 2, []float64{0.5, -1}))
	if err != nil {
		t.Fatalf("beta CopyFrom returned error: %v", err)
	}

	err = batchNorm.RunningMean().CopyFrom(mustMatrix(t, 1, 2, []float64{1, 2}))
	if err != nil {
		t.Fatalf("running mean CopyFrom returned error: %v", err)
	}

	err = batchNorm.RunningVariance().CopyFrom(mustMatrix(t, 1, 2, []float64{4, 9}))
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

	input = mustMatrix(t, 2, 2, []float64{
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

	input = mustMatrix(t, 2, 2, []float64{
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

func Test_LoadSequential_RejectsUnknownLayerType(t *testing.T) {
	var (
		loaded *model.Sequential
		err    error
	)

	loaded, err = model.LoadSequential(strings.NewReader(`{
		"format": "neuralnetwork.sequential",
			"version": 1,
			"layers": [
				{
					"type": "batch_norm"
				}
			]
	}`))
	if err == nil {
		t.Fatal("LoadSequential error = nil, want error")
	}

	if loaded != nil {
		t.Fatal("LoadSequential returned model on error")
	}

	if !strings.Contains(err.Error(), "unknown layer type") {
		t.Fatalf("LoadSequential error = %q, want unknown layer type", err.Error())
	}
}

func Test_LoadSequential_RejectsUnknownActivationName(t *testing.T) {
	var (
		loaded *model.Sequential
		err    error
	)

	loaded, err = model.LoadSequential(strings.NewReader(`{
		"format": "neuralnetwork.sequential",
		"version": 1,
		"layers": [
			{
				"type": "activation",
				"activation": "swish"
			}
		]
	}`))
	if err == nil {
		t.Fatal("LoadSequential error = nil, want error")
	}

	if loaded != nil {
		t.Fatal("LoadSequential returned model on error")
	}

	if !strings.Contains(err.Error(), "unknown activation name") {
		t.Fatalf("LoadSequential error = %q, want unknown activation name", err.Error())
	}
}

func mustSerializationDense(
	tb testing.TB,
	inputSize,
	outputSize int,
	weightValues,
	biasValues []float64,
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
