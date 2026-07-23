//go:build darwin && cgo && metal && !purego

package model_test

import (
	"bytes"
	"errors"
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
	"github.com/itsmontoya/neuralnetwork/model"
)

const metalInferenceTolerance = 2e-4

type metalInferenceShape struct {
	name       string
	batchSize  int
	inputSize  int
	hiddenSize int
	classCount int
}

type metalFallbackActivation struct {
	delta float32
	err   error
}

func Test_SequentialResidentPredictParity(t *testing.T) {
	var tests []metalInferenceShape
	tests = []metalInferenceShape{
		{name: "below threshold", batchSize: 8, inputSize: 32, hiddenSize: 64, classCount: 10},
		{name: "at threshold", batchSize: 256, inputSize: 128, hiddenSize: 128, classCount: 16},
		{name: "uneven", batchSize: 127, inputSize: 257, hiddenSize: 263, classCount: 19},
	}

	var test metalInferenceShape
	requireModelMetal(t)
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			var (
				network       *model.Sequential
				input         *matrix.Matrix
				output        *matrix.Matrix
				inputValues   []float32
				hiddenWeights []float32
				hiddenBiases  []float32
				outputWeights []float32
				outputBiases  []float32
				want          []float32
				counters      metaltest.Counters
				err           error
			)

			network, input, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases =
				metalInferenceModel(t, test, activation.ReLU{})
			want = metalInferenceReference(
				inputValues,
				test,
				hiddenWeights,
				hiddenBiases,
				outputWeights,
				outputBiases,
				true,
			)

			metaltest.Enable()
			defer metaltest.Disable()
			if output, err = network.Predict(input); err != nil {
				t.Fatalf("Predict returned error: %v", err)
			}
			counters = metaltest.Snapshot()
			if test.name == "below threshold" {
				requireModelMetalCounters(t, counters, 0, 0, 0, 0, 0)
			} else if counters.CommandSubmissions != 1 || counters.Waits != 1 ||
				counters.ResultDownloads != 0 {
				t.Fatalf("resident prediction counters = %+v, want one command/wait and no downloads", counters)
			}
			requireMetalInferenceValues(t, output, want)
		})
	}
}

func Test_SequentialResidentPredictTransfersAndMutations(t *testing.T) {
	var (
		shape         metalInferenceShape
		network       *model.Sequential
		hidden        *layer.Dense
		input         *matrix.Matrix
		output        *matrix.Matrix
		inputValues   []float32
		hiddenWeights []float32
		hiddenBiases  []float32
		outputWeights []float32
		outputBiases  []float32
		changedInput  []float32
		want          []float32
		counters      metaltest.Counters
		err           error
	)

	requireModelMetal(t)
	shape = metalInferenceShape{
		name:       "large",
		batchSize:  256,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 16,
	}
	network, hidden, input, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases =
		metalInferenceModelWithLayers(t, shape, activation.ReLU{})

	metaltest.Enable()
	defer metaltest.Disable()
	if output, err = network.Predict(input); err != nil {
		t.Fatalf("first Predict returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 5 || counters.ResultDownloads != 0 ||
		counters.CommandSubmissions != 1 || counters.Waits != 1 {
		t.Fatalf("cold resident prediction counters = %+v, want five uploads and one command/wait", counters)
	}
	if counters.InputUploadBytes != metalInferenceUploadBytes(shape) {
		t.Fatalf(
			"cold upload bytes = %d, want %d",
			counters.InputUploadBytes,
			metalInferenceUploadBytes(shape),
		)
	}

	metaltest.Reset()
	if output, err = network.Predict(input); err != nil {
		t.Fatalf("warmed Predict returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 0 || counters.InputUploadBytes != 0 ||
		counters.ResultDownloads != 0 || counters.CommandSubmissions != 1 || counters.Waits != 1 {
		t.Fatalf("warmed resident prediction counters = %+v, want no transfers and one command/wait", counters)
	}

	changedInput = append([]float32(nil), inputValues...)
	changedInput[0] += 0.75
	if err = input.CopyValuesFrom(changedInput); err != nil {
		t.Fatalf("CopyValuesFrom input returned error: %v", err)
	}
	metaltest.Reset()
	if output, err = network.Predict(input); err != nil {
		t.Fatalf("Predict after input mutation returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 1 || counters.InputUploadBytes != uint64(len(changedInput))*4 {
		t.Fatalf("input mutation counters = %+v, want one input re-upload", counters)
	}

	hiddenWeights[0] -= 0.5
	if err = hidden.Weights().Values().Set(0, 0, hiddenWeights[0]); err != nil {
		t.Fatalf("Set hidden weight returned error: %v", err)
	}
	metaltest.Reset()
	if output, err = network.Predict(input); err != nil {
		t.Fatalf("Predict after parameter mutation returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 1 || counters.InputUploadBytes != uint64(len(hiddenWeights))*4 {
		t.Fatalf("parameter mutation counters = %+v, want one parameter re-upload", counters)
	}

	want = metalInferenceReference(
		changedInput,
		shape,
		hiddenWeights,
		hiddenBiases,
		outputWeights,
		outputBiases,
		true,
	)
	requireMetalInferenceValues(t, output, want)
}

func Test_SequentialResidentPredictObservationBoundaries(t *testing.T) {
	type testcase struct {
		name string
		run  func(*matrix.Matrix) error
	}

	var (
		shape        metalInferenceShape
		network      *model.Sequential
		input        *matrix.Matrix
		output       *matrix.Matrix
		target       *matrix.Matrix
		outputValues []float32
		targetValues []float32
		tests        []testcase
		test         testcase
		counters     metaltest.Counters
		document     bytes.Buffer
		err          error
	)

	requireModelMetal(t)
	shape = metalInferenceShape{
		name:       "observation",
		batchSize:  256,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 16,
	}
	network, input, _, _, _, _, _ = metalInferenceModel(t, shape, activation.ReLU{})
	outputValues = make([]float32, shape.batchSize*shape.classCount)
	tests = []testcase{
		{
			name: "Values",
			run: func(value *matrix.Matrix) (runErr error) {
				_, runErr = value.Values()
				return runErr
			},
		},
		{
			name: "ValuesInto",
			run: func(value *matrix.Matrix) (runErr error) {
				runErr = value.ValuesInto(outputValues)
				return runErr
			},
		},
		{
			name: "At",
			run: func(value *matrix.Matrix) (runErr error) {
				_, runErr = value.At(0, 0)
				return runErr
			},
		},
	}

	metaltest.Enable()
	defer metaltest.Disable()
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			if output, err = network.Predict(input); err != nil {
				t.Fatalf("Predict returned error: %v", err)
			}
			metaltest.Reset()
			if err = test.run(output); err != nil {
				t.Fatalf("observation returned error: %v", err)
			}
			counters = metaltest.Snapshot()
			if counters.ResultDownloads != 1 ||
				counters.ResultDownloadBytes != uint64(shape.batchSize*shape.classCount)*4 {
				t.Fatalf("observation counters = %+v, want one final-output download", counters)
			}
		})
	}

	if output, err = network.Predict(input); err != nil {
		t.Fatalf("Predict before metric returned error: %v", err)
	}
	targetValues = make([]float32, shape.batchSize*shape.classCount)
	var row int
	for row = 0; row < shape.batchSize; row++ {
		targetValues[row*shape.classCount] = 1
	}
	if target, err = matrix.FromSlice(shape.batchSize, shape.classCount, targetValues); err != nil {
		t.Fatalf("FromSlice target returned error: %v", err)
	}
	metaltest.Reset()
	if _, err = (metric.CategoricalAccuracy{}).Value(output, target); err != nil {
		t.Fatalf("CategoricalAccuracy returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads != 1 {
		t.Fatalf("metric counters = %+v, want one prediction download", counters)
	}

	if output, err = network.Predict(input); err != nil {
		t.Fatalf("Predict before Save returned error: %v", err)
	}
	metaltest.Reset()
	if err = network.Save(&document); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads != 0 {
		t.Fatalf("Save counters = %+v, want no download for unchanged synchronized parameters", counters)
	}
	if document.Len() == 0 {
		t.Fatal("Save wrote an empty document")
	}
}

func Test_SequentialResidentPredictCustomActivationFallback(t *testing.T) {
	var (
		shape         metalInferenceShape
		custom        *metalFallbackActivation
		network       *model.Sequential
		input         *matrix.Matrix
		output        *matrix.Matrix
		inputValues   []float32
		hiddenWeights []float32
		hiddenBiases  []float32
		outputWeights []float32
		outputBiases  []float32
		want          []float32
		counters      metaltest.Counters
		err           error
	)

	requireModelMetal(t)
	shape = metalInferenceShape{
		name:       "custom activation",
		batchSize:  256,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 16,
	}
	custom = &metalFallbackActivation{delta: 0.25, err: errors.New("injected custom activation failure")}
	network, input, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases =
		metalInferenceModel(t, shape, custom)

	metaltest.Enable()
	defer metaltest.Disable()
	if _, err = network.Predict(input); err == nil {
		t.Fatal("Predict custom activation error = nil")
	}
	custom.err = nil
	metaltest.Reset()
	if output, err = network.Predict(input); err != nil {
		t.Fatalf("Predict after custom activation recovery returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads != 1 || counters.CommandSubmissions != 2 || counters.Waits != 2 {
		t.Fatalf("custom activation counters = %+v, want one download and two command boundaries", counters)
	}
	want = metalInferenceReference(
		inputValues,
		shape,
		hiddenWeights,
		hiddenBiases,
		outputWeights,
		outputBiases,
		false,
	)
	requireMetalInferenceValues(t, output, want)
}

func Test_SequentialResidentPredictNonFinitePropagation(t *testing.T) {
	var (
		shape         metalInferenceShape
		network       *model.Sequential
		input         *matrix.Matrix
		output        *matrix.Matrix
		inputValues   []float32
		hiddenWeights []float32
		hiddenBiases  []float32
		outputWeights []float32
		outputBiases  []float32
		want          []float32
		err           error
	)

	requireModelMetal(t)
	shape = metalInferenceShape{
		name:       "non-finite",
		batchSize:  256,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 16,
	}
	network, input, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases =
		metalInferenceModel(t, shape, activation.ReLU{})
	inputValues[0] = float32(math.NaN())
	inputValues[shape.inputSize] = float32(math.Inf(1))
	inputValues[2*shape.inputSize] = float32(math.Inf(-1))
	if err = input.CopyValuesFrom(inputValues); err != nil {
		t.Fatalf("CopyValuesFrom non-finite input returned error: %v", err)
	}
	want = metalInferenceReference(
		inputValues,
		shape,
		hiddenWeights,
		hiddenBiases,
		outputWeights,
		outputBiases,
		true,
	)
	if output, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	requireMetalInferenceClassifications(t, output, want)
}

func (a *metalFallbackActivation) Forward(
	input *matrix.Matrix,
) (output *matrix.Matrix, err error) {
	if a.err != nil {
		return nil, a.err
	}

	output, err = input.AddScalar(a.delta)
	return output, err
}

func (a *metalFallbackActivation) Backward(
	_ *matrix.Matrix,
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = outputGradient.Clone()
	return inputGradient, err
}

func metalInferenceModel(
	tb testing.TB,
	shape metalInferenceShape,
	function activation.Activation,
) (
	network *model.Sequential,
	input *matrix.Matrix,
	inputValues,
	hiddenWeights,
	hiddenBiases,
	outputWeights,
	outputBiases []float32,
) {
	network, _, input, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases =
		metalInferenceModelWithLayers(tb, shape, function)
	return network, input, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases
}

func metalInferenceModelWithLayers(
	tb testing.TB,
	shape metalInferenceShape,
	function activation.Activation,
) (
	network *model.Sequential,
	hidden *layer.Dense,
	input *matrix.Matrix,
	inputValues,
	hiddenWeights,
	hiddenBiases,
	outputWeights,
	outputBiases []float32,
) {
	var (
		hiddenActivation *layer.Activation
		output           *layer.Dense
		outputActivation *layer.Activation
		err              error
	)

	tb.Helper()
	if hidden, err = layer.NewDense(shape.inputSize, shape.hiddenSize, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewDense hidden returned error: %v", err)
	}
	if hiddenActivation, err = layer.NewActivation(function); err != nil {
		tb.Fatalf("NewActivation hidden returned error: %v", err)
	}
	if output, err = layer.NewDense(shape.hiddenSize, shape.classCount, layer.ZeroWeights); err != nil {
		tb.Fatalf("NewDense output returned error: %v", err)
	}
	if outputActivation, err = layer.NewActivation(activation.Softmax{}); err != nil {
		tb.Fatalf("NewActivation output returned error: %v", err)
	}

	inputValues = metalInferenceValues(shape.batchSize*shape.inputSize, 11, 0.4)
	hiddenWeights = metalInferenceValues(shape.inputSize*shape.hiddenSize, 17, 0.08)
	hiddenBiases = metalInferenceValues(shape.hiddenSize, 7, 0.03)
	outputWeights = metalInferenceValues(shape.hiddenSize*shape.classCount, 13, 0.08)
	outputBiases = metalInferenceValues(shape.classCount, 5, 0.03)
	if input, err = matrix.FromSlice(shape.batchSize, shape.inputSize, inputValues); err != nil {
		tb.Fatalf("FromSlice input returned error: %v", err)
	}
	if err = hidden.Weights().Values().CopyValuesFrom(hiddenWeights); err != nil {
		tb.Fatalf("CopyValuesFrom hidden weights returned error: %v", err)
	}
	if err = hidden.Biases().Values().CopyValuesFrom(hiddenBiases); err != nil {
		tb.Fatalf("CopyValuesFrom hidden biases returned error: %v", err)
	}
	if err = output.Weights().Values().CopyValuesFrom(outputWeights); err != nil {
		tb.Fatalf("CopyValuesFrom output weights returned error: %v", err)
	}
	if err = output.Biases().Values().CopyValuesFrom(outputBiases); err != nil {
		tb.Fatalf("CopyValuesFrom output biases returned error: %v", err)
	}
	if network, err = model.NewSequential(hidden, hiddenActivation, output, outputActivation); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network, hidden, input, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases
}

func metalInferenceReference(
	input []float32,
	shape metalInferenceShape,
	hiddenWeights,
	hiddenBiases,
	outputWeights,
	outputBiases []float32,
	relu bool,
) (output []float32) {
	var hidden []float32

	hidden = metalInferenceDense(
		input,
		shape.batchSize,
		shape.inputSize,
		hiddenWeights,
		shape.hiddenSize,
		hiddenBiases,
	)
	var index int
	if relu {
		for index = range hidden {
			if !(hidden[index] > 0) {
				hidden[index] = 0
			}
		}
	} else {
		for index = range hidden {
			hidden[index] += 0.25
		}
	}

	output = metalInferenceDense(
		hidden,
		shape.batchSize,
		shape.hiddenSize,
		outputWeights,
		shape.classCount,
		outputBiases,
	)
	metalInferenceSoftmax(output, shape.batchSize, shape.classCount)
	return output
}

func metalInferenceDense(
	input []float32,
	rows,
	inputSize int,
	weights []float32,
	outputSize int,
	biases []float32,
) (output []float32) {
	var (
		row   int
		col   int
		inner int
		sum   float32
	)

	output = make([]float32, rows*outputSize)
	for row = 0; row < rows; row++ {
		for col = 0; col < outputSize; col++ {
			sum = 0
			for inner = 0; inner < inputSize; inner++ {
				sum += input[row*inputSize+inner] * weights[inner*outputSize+col]
			}
			output[row*outputSize+col] = sum + biases[col]
		}
	}
	return output
}

func metalInferenceSoftmax(values []float32, rows, cols int) {
	var (
		row     int
		col     int
		offset  int
		maximum float32
		value   float32
		sum     float32
	)

	for row = 0; row < rows; row++ {
		offset = row * cols
		maximum = values[offset]
		for col = 1; col < cols; col++ {
			value = values[offset+col]
			if value > maximum {
				maximum = value
			}
		}
		sum = 0
		for col = 0; col < cols; col++ {
			value = float32(math.Exp(float64(values[offset+col] - maximum)))
			values[offset+col] = value
			sum += value
		}
		for col = 0; col < cols; col++ {
			values[offset+col] /= sum
		}
	}
}

func metalInferenceValues(length, period int, scale float32) (values []float32) {
	var index int

	values = make([]float32, length)
	for index = range values {
		values[index] = scale * float32(index%period-period/2) / float32(period)
	}
	return values
}

func metalInferenceUploadBytes(shape metalInferenceShape) (bytes uint64) {
	var elements int

	elements = shape.batchSize*shape.inputSize +
		shape.inputSize*shape.hiddenSize +
		shape.hiddenSize +
		shape.hiddenSize*shape.classCount +
		shape.classCount
	bytes = uint64(elements) * 4
	return bytes
}

func requireMetalInferenceValues(tb testing.TB, got *matrix.Matrix, want []float32) {
	tb.Helper()

	var (
		values []float32
		index  int
		err    error
	)

	if values, err = got.Values(); err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}
	if len(values) != len(want) {
		tb.Fatalf("value length = %d, want %d", len(values), len(want))
	}
	for index = range want {
		if math.IsNaN(float64(values[index])) || math.IsInf(float64(values[index]), 0) {
			tb.Fatalf("value %d = %g, want finite %g", index, values[index], want[index])
		}
		if math.Abs(float64(values[index]-want[index])) >
			metalInferenceTolerance+metalInferenceTolerance*math.Abs(float64(want[index])) {
			tb.Fatalf("value %d = %g, want %g", index, values[index], want[index])
		}
	}
}

func requireMetalInferenceClassifications(tb testing.TB, got *matrix.Matrix, want []float32) {
	tb.Helper()

	var (
		values []float32
		index  int
		err    error
	)

	if values, err = got.Values(); err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}
	if len(values) != len(want) {
		tb.Fatalf("value length = %d, want %d", len(values), len(want))
	}
	for index = range want {
		if math.IsNaN(float64(want[index])) {
			if !math.IsNaN(float64(values[index])) {
				tb.Fatalf("value %d = %g, want NaN", index, values[index])
			}
			continue
		}
		if math.IsInf(float64(want[index]), 1) {
			if !math.IsInf(float64(values[index]), 1) {
				tb.Fatalf("value %d = %g, want +Inf", index, values[index])
			}
			continue
		}
		if math.IsInf(float64(want[index]), -1) {
			if !math.IsInf(float64(values[index]), -1) {
				tb.Fatalf("value %d = %g, want -Inf", index, values[index])
			}
			continue
		}
		if math.IsNaN(float64(values[index])) || math.IsInf(float64(values[index]), 0) ||
			math.Abs(float64(values[index]-want[index])) > metalInferenceTolerance {
			tb.Fatalf("value %d = %g, want %g", index, values[index], want[index])
		}
	}
}
