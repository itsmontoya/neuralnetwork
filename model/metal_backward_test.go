//go:build darwin && cgo && metal && !purego

package model_test

import (
	"math"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const (
	metalBackwardTolerance      = 2e-3
	metalFiniteDifferenceStep   = 1e-2
	metalFiniteDifferenceAbsTol = 5e-3
	metalFiniteDifferenceRelTol = 5e-2
)

type metalBackwardReference struct {
	inputGradient        []float32
	hiddenWeightGradient []float32
	hiddenBiasGradient   []float32
	outputWeightGradient []float32
	outputBiasGradient   []float32
}

type metalIdentityFallback struct{}

func Test_SequentialResidentBackwardParityAndTransfers(t *testing.T) {
	var (
		shape          metalInferenceShape
		network        *model.Sequential
		input          *matrix.Matrix
		inputGradient  *matrix.Matrix
		outputGradient *matrix.Matrix
		inputValues    []float32
		hiddenWeights  []float32
		hiddenBiases   []float32
		outputWeights  []float32
		outputBiases   []float32
		gradientValues []float32
		parameters     []*optimizer.Parameter
		reference      metalBackwardReference
		counters       metaltest.Counters
		err            error
	)

	requireModelMetal(t)
	shape = metalInferenceShape{
		name:       "threshold",
		batchSize:  256,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 16,
	}
	network, input, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases =
		metalInferenceModel(t, shape, activation.ReLU{})
	gradientValues = metalInferenceValues(shape.batchSize*shape.classCount, 19, 0.3)
	if outputGradient, err = matrix.FromSlice(shape.batchSize, shape.classCount, gradientValues); err != nil {
		t.Fatalf("FromSlice output gradient returned error: %v", err)
	}
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if inputGradient, err = network.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	requireModelMetalCounters(t, counters, 15, 5, 0, 1, 1)

	parameters = network.Parameters()
	metaltest.Reset()
	if _, err = parameters[0].Gradient().Values(); err != nil {
		t.Fatalf("weight gradient Values returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads != 1 ||
		counters.ResultDownloadBytes != uint64(shape.inputSize*shape.hiddenSize)*4 {
		t.Fatalf("weight gradient observation counters = %+v, want only the weight gradient", counters)
	}

	metaltest.Reset()
	if _, err = inputGradient.Values(); err != nil {
		t.Fatalf("input gradient Values returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads != 1 ||
		counters.ResultDownloadBytes != uint64(shape.batchSize*shape.inputSize)*4 {
		t.Fatalf("input gradient observation counters = %+v, want only the input gradient", counters)
	}

	reference = metalBackwardReferenceValues(
		inputValues,
		gradientValues,
		shape,
		hiddenWeights,
		hiddenBiases,
		outputWeights,
		outputBiases,
		true,
	)
	requireBackwardMatrixValues(t, inputGradient, reference.inputGradient, metalBackwardTolerance)
	requireBackwardMatrixValues(t, parameters[0].Gradient(), reference.hiddenWeightGradient, metalBackwardTolerance)
	requireBackwardMatrixValues(t, parameters[1].Gradient(), reference.hiddenBiasGradient, metalBackwardTolerance)
	requireBackwardMatrixValues(t, parameters[2].Gradient(), reference.outputWeightGradient, metalBackwardTolerance)
	requireBackwardMatrixValues(t, parameters[3].Gradient(), reference.outputBiasGradient, metalBackwardTolerance)

	metaltest.Reset()
	if _, err = network.Backward(outputGradient); err != nil {
		t.Fatalf("repeated Backward returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 0 || counters.ResultDownloads != 0 ||
		counters.CommandSubmissions != 1 || counters.Waits != 1 {
		t.Fatalf("repeated backward counters = %+v, want resident reuse without transfers", counters)
	}
}

func Test_SequentialResidentBackwardAccumulationResetAndRecovery(t *testing.T) {
	var (
		shape           metalInferenceShape
		network         *model.Sequential
		hidden          *layer.Dense
		input           *matrix.Matrix
		outputGradient  *matrix.Matrix
		invalidGradient *matrix.Matrix
		changedInput    *matrix.Matrix
		firstGradient   []float32
		values          []float32
		parameters      []*optimizer.Parameter
		counters        metaltest.Counters
		parameter       *optimizer.Parameter
		index           int
		err             error
	)

	requireModelMetal(t)
	shape = metalInferenceShape{
		name:       "accumulation",
		batchSize:  256,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 16,
	}
	network, hidden, input, _, _, _, _, _ =
		metalInferenceModelWithLayers(t, shape, activation.ReLU{})
	outputGradient = metalBackwardMatrix(
		t,
		shape.batchSize,
		shape.classCount,
		metalInferenceValues(shape.batchSize*shape.classCount, 11, 0.2),
	)
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	if _, err = network.Backward(outputGradient); err != nil {
		t.Fatalf("first Backward returned error: %v", err)
	}
	parameters = network.Parameters()
	if firstGradient, err = parameters[2].Gradient().Values(); err != nil {
		t.Fatalf("first output weight gradient Values returned error: %v", err)
	}
	if _, err = network.Backward(outputGradient); err != nil {
		t.Fatalf("second Backward returned error: %v", err)
	}
	if values, err = parameters[2].Gradient().Values(); err != nil {
		t.Fatalf("accumulated output weight gradient Values returned error: %v", err)
	}
	for index = range firstGradient {
		requireBackwardFloat(t, values[index], 2*firstGradient[index], metalBackwardTolerance)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	for _, parameter = range parameters {
		if err = parameter.ResetGradient(); err != nil {
			t.Fatalf("ResetGradient returned error: %v", err)
		}
	}
	counters = metaltest.Snapshot()
	requireModelMetalCounters(t, counters, 4, 0, 0, 4, 4)
	for _, parameter = range parameters {
		requireBackwardMatrixZero(t, parameter.Gradient())
	}

	if invalidGradient, err = matrix.New(shape.batchSize-1, shape.classCount); err != nil {
		t.Fatalf("New invalid gradient returned error: %v", err)
	}
	metaltest.Reset()
	if _, err = network.Backward(invalidGradient); err == nil ||
		!strings.Contains(err.Error(), "shape mismatch") {
		t.Fatalf("invalid Backward error = %v, want shape mismatch", err)
	}
	counters = metaltest.Snapshot()
	if counters.CommandSubmissions != 0 || counters.Waits != 0 {
		t.Fatalf("invalid backward counters = %+v, want validation before encoding", counters)
	}
	if _, err = network.Backward(outputGradient); err != nil {
		t.Fatalf("Backward after rejected gradient returned error: %v", err)
	}

	if err = hidden.Weights().Values().Set(0, 0, 0.125); err != nil {
		t.Fatalf("Set changed hidden weight returned error: %v", err)
	}
	changedInput = metalBackwardMatrix(
		t,
		shape.batchSize+1,
		shape.inputSize,
		metalInferenceValues((shape.batchSize+1)*shape.inputSize, 23, 0.4),
	)
	outputGradient = metalBackwardMatrix(
		t,
		shape.batchSize+1,
		shape.classCount,
		metalInferenceValues((shape.batchSize+1)*shape.classCount, 13, 0.2),
	)
	if _, err = network.Predict(changedInput); err != nil {
		t.Fatalf("Predict changed batch returned error: %v", err)
	}
	if _, err = network.Backward(outputGradient); err != nil {
		t.Fatalf("Backward changed batch returned error: %v", err)
	}
}

func Test_SequentialResidentBackwardFiniteDifferences(t *testing.T) {
	var (
		shape          metalInferenceShape
		network        *model.Sequential
		input          *matrix.Matrix
		inputGradient  *matrix.Matrix
		outputGradient *matrix.Matrix
		inputValues    []float32
		gradientValues []float32
		parameters     []*optimizer.Parameter
		analyticInput  float32
		analyticWeight float32
		analyticBias   float32
		numericInput   float32
		numericWeight  float32
		numericBias    float32
		err            error
	)

	requireModelMetal(t)
	shape = metalInferenceShape{
		name:       "uneven",
		batchSize:  127,
		inputSize:  257,
		hiddenSize: 263,
		classCount: 19,
	}
	network, input, inputValues, _, _, _, _ =
		metalInferenceModel(t, shape, activation.ReLU{})
	gradientValues = metalInferenceValues(shape.batchSize*shape.classCount, 17, 0.15)
	outputGradient = metalBackwardMatrix(t, shape.batchSize, shape.classCount, gradientValues)
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	if inputGradient, err = network.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	parameters = network.Parameters()
	if analyticInput, err = inputGradient.At(0, 0); err != nil {
		t.Fatalf("input gradient At returned error: %v", err)
	}
	if analyticWeight, err = parameters[2].Gradient().At(0, 0); err != nil {
		t.Fatalf("weight gradient At returned error: %v", err)
	}
	if analyticBias, err = parameters[3].Gradient().At(0, 0); err != nil {
		t.Fatalf("bias gradient At returned error: %v", err)
	}

	numericInput = metalCentralDifference(
		t,
		network,
		input,
		outputGradient,
		input,
		0,
		0,
		inputValues[0],
	)
	var outputWeightBase float32
	if outputWeightBase, err = parameters[2].Values().At(0, 0); err != nil {
		t.Fatalf("output weight At returned error: %v", err)
	}
	numericWeight = metalCentralDifference(
		t,
		network,
		input,
		outputGradient,
		parameters[2].Values(),
		0,
		0,
		outputWeightBase,
	)
	var outputBiasBase float32
	if outputBiasBase, err = parameters[3].Values().At(0, 0); err != nil {
		t.Fatalf("output bias At returned error: %v", err)
	}
	numericBias = metalCentralDifference(
		t,
		network,
		input,
		outputGradient,
		parameters[3].Values(),
		0,
		0,
		outputBiasBase,
	)

	requireFiniteDifference(t, "input", analyticInput, numericInput)
	requireFiniteDifference(t, "weight", analyticWeight, numericWeight)
	requireFiniteDifference(t, "bias", analyticBias, numericBias)
}

func Test_SequentialResidentBackwardCustomFallback(t *testing.T) {
	var (
		shape          metalInferenceShape
		network        *model.Sequential
		input          *matrix.Matrix
		inputGradient  *matrix.Matrix
		outputGradient *matrix.Matrix
		inputValues    []float32
		hiddenWeights  []float32
		hiddenBiases   []float32
		outputWeights  []float32
		outputBiases   []float32
		gradientValues []float32
		parameters     []*optimizer.Parameter
		reference      metalBackwardReference
		counters       metaltest.Counters
		err            error
	)

	requireModelMetal(t)
	shape = metalInferenceShape{
		name:       "custom fallback",
		batchSize:  256,
		inputSize:  128,
		hiddenSize: 128,
		classCount: 16,
	}
	network, input, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases =
		metalInferenceModel(t, shape, metalIdentityFallback{})
	gradientValues = metalInferenceValues(shape.batchSize*shape.classCount, 17, 0.2)
	outputGradient = metalBackwardMatrix(t, shape.batchSize, shape.classCount, gradientValues)
	if _, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if inputGradient, err = network.Backward(outputGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads != 1 ||
		counters.CommandSubmissions != 2 || counters.Waits != 2 {
		t.Fatalf("custom fallback counters = %+v, want one exact download boundary and two scopes", counters)
	}

	reference = metalBackwardReferenceValues(
		inputValues,
		gradientValues,
		shape,
		hiddenWeights,
		hiddenBiases,
		outputWeights,
		outputBiases,
		false,
	)
	parameters = network.Parameters()
	requireBackwardMatrixValues(t, inputGradient, reference.inputGradient, metalBackwardTolerance)
	requireBackwardMatrixValues(t, parameters[0].Gradient(), reference.hiddenWeightGradient, metalBackwardTolerance)
	requireBackwardMatrixValues(t, parameters[1].Gradient(), reference.hiddenBiasGradient, metalBackwardTolerance)
	requireBackwardMatrixValues(t, parameters[2].Gradient(), reference.outputWeightGradient, metalBackwardTolerance)
	requireBackwardMatrixValues(t, parameters[3].Gradient(), reference.outputBiasGradient, metalBackwardTolerance)
}

func (metalIdentityFallback) Forward(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	output, err = input.AddScalar(0)
	return output, err
}

func (metalIdentityFallback) Backward(
	_ *matrix.Matrix,
	outputGradient *matrix.Matrix,
) (inputGradient *matrix.Matrix, err error) {
	inputGradient, err = outputGradient.AddScalar(0)
	return inputGradient, err
}

func metalBackwardReferenceValues(
	input,
	outputGradient []float32,
	shape metalInferenceShape,
	hiddenWeights,
	hiddenBiases,
	outputWeights,
	outputBiases []float32,
	relu bool,
) (reference metalBackwardReference) {
	var (
		hiddenInputGradient []float32
		hiddenPreactivation []float32
		hiddenValues        []float32
		outputValues        []float32
		outputInputGradient []float32
		row                 int
		col                 int
		inner               int
		offset              int
		dot                 float32
	)

	hiddenPreactivation = metalInferenceDense(
		input,
		shape.batchSize,
		shape.inputSize,
		hiddenWeights,
		shape.hiddenSize,
		hiddenBiases,
	)
	hiddenValues = append([]float32(nil), hiddenPreactivation...)
	outputInputGradient = make([]float32, len(hiddenValues))
	if relu {
		for offset = range hiddenValues {
			if !(hiddenValues[offset] > 0) {
				hiddenValues[offset] = 0
			}
		}
	}
	outputValues = metalInferenceDense(
		hiddenValues,
		shape.batchSize,
		shape.hiddenSize,
		outputWeights,
		shape.classCount,
		outputBiases,
	)
	metalInferenceSoftmax(outputValues, shape.batchSize, shape.classCount)
	for row = 0; row < shape.batchSize; row++ {
		offset = row * shape.classCount
		dot = 0
		for col = 0; col < shape.classCount; col++ {
			dot += outputGradient[offset+col] * outputValues[offset+col]
		}
		for col = 0; col < shape.classCount; col++ {
			outputValues[offset+col] *= outputGradient[offset+col] - dot
		}
	}

	reference.outputWeightGradient = make([]float32, shape.hiddenSize*shape.classCount)
	reference.outputBiasGradient = make([]float32, shape.classCount)
	for row = 0; row < shape.batchSize; row++ {
		for col = 0; col < shape.classCount; col++ {
			reference.outputBiasGradient[col] += outputValues[row*shape.classCount+col]
			for inner = 0; inner < shape.hiddenSize; inner++ {
				reference.outputWeightGradient[inner*shape.classCount+col] +=
					hiddenValues[row*shape.hiddenSize+inner] * outputValues[row*shape.classCount+col]
				outputInputGradient[row*shape.hiddenSize+inner] +=
					outputValues[row*shape.classCount+col] * outputWeights[inner*shape.classCount+col]
			}
		}
	}

	hiddenInputGradient = outputInputGradient
	if relu {
		for offset = range hiddenInputGradient {
			if !(hiddenPreactivation[offset] > 0) {
				hiddenInputGradient[offset] = 0
			}
		}
	}

	reference.hiddenWeightGradient = make([]float32, shape.inputSize*shape.hiddenSize)
	reference.hiddenBiasGradient = make([]float32, shape.hiddenSize)
	reference.inputGradient = make([]float32, len(input))
	for row = 0; row < shape.batchSize; row++ {
		for col = 0; col < shape.hiddenSize; col++ {
			reference.hiddenBiasGradient[col] += hiddenInputGradient[row*shape.hiddenSize+col]
			for inner = 0; inner < shape.inputSize; inner++ {
				reference.hiddenWeightGradient[inner*shape.hiddenSize+col] +=
					input[row*shape.inputSize+inner] * hiddenInputGradient[row*shape.hiddenSize+col]
				reference.inputGradient[row*shape.inputSize+inner] +=
					hiddenInputGradient[row*shape.hiddenSize+col] * hiddenWeights[inner*shape.hiddenSize+col]
			}
		}
	}
	return reference
}

func metalCentralDifference(
	tb testing.TB,
	network *model.Sequential,
	input,
	outputGradient,
	value *matrix.Matrix,
	row,
	col int,
	base float32,
) (gradient float32) {
	var (
		positive float32
		negative float32
		err      error
	)

	tb.Helper()
	if err = value.Set(row, col, base+metalFiniteDifferenceStep); err != nil {
		tb.Fatalf("Set positive perturbation returned error: %v", err)
	}
	positive = metalBackwardObjective(tb, network, input, outputGradient)
	if err = value.Set(row, col, base-metalFiniteDifferenceStep); err != nil {
		tb.Fatalf("Set negative perturbation returned error: %v", err)
	}
	negative = metalBackwardObjective(tb, network, input, outputGradient)
	if err = value.Set(row, col, base); err != nil {
		tb.Fatalf("Restore perturbation returned error: %v", err)
	}

	gradient = (positive - negative) / (2 * metalFiniteDifferenceStep)
	return gradient
}

func metalBackwardObjective(
	tb testing.TB,
	network *model.Sequential,
	input,
	outputGradient *matrix.Matrix,
) (value float32) {
	var (
		output         *matrix.Matrix
		outputValues   []float32
		gradientValues []float32
		index          int
		err            error
	)

	tb.Helper()
	if output, err = network.Predict(input); err != nil {
		tb.Fatalf("Predict objective returned error: %v", err)
	}
	if outputValues, err = output.Values(); err != nil {
		tb.Fatalf("output Values returned error: %v", err)
	}
	if gradientValues, err = outputGradient.Values(); err != nil {
		tb.Fatalf("output gradient Values returned error: %v", err)
	}
	for index = range outputValues {
		value += outputValues[index] * gradientValues[index]
	}
	return value
}

func metalBackwardMatrix(
	tb testing.TB,
	rows,
	cols int,
	values []float32,
) (value *matrix.Matrix) {
	var err error

	tb.Helper()
	if value, err = matrix.FromSlice(rows, cols, values); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}
	return value
}

func requireBackwardMatrixValues(
	tb testing.TB,
	got *matrix.Matrix,
	want []float32,
	tolerance float32,
) {
	var (
		values []float32
		index  int
		err    error
	)

	tb.Helper()
	if values, err = got.Values(); err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}
	if len(values) != len(want) {
		tb.Fatalf("value length = %d, want %d", len(values), len(want))
	}
	for index = range want {
		requireBackwardFloat(tb, values[index], want[index], tolerance)
	}
}

func requireBackwardMatrixZero(tb testing.TB, value *matrix.Matrix) {
	var (
		values []float32
		index  int
		err    error
	)

	tb.Helper()
	if values, err = value.Values(); err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}
	for index = range values {
		if values[index] != 0 {
			tb.Fatalf("value %d = %g, want zero", index, values[index])
		}
	}
}

func requireBackwardFloat(tb testing.TB, got, want, tolerance float32) {
	var difference float64

	tb.Helper()
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
		tb.Fatalf("value = %g, want finite %g", got, want)
	}
	difference = math.Abs(float64(got - want))
	if difference > float64(tolerance)+float64(tolerance)*math.Abs(float64(want)) {
		tb.Fatalf("value = %g, want %g, difference %g", got, want, difference)
	}
}

func requireFiniteDifference(tb testing.TB, name string, analytic, numeric float32) {
	var tolerance float64

	tb.Helper()
	tolerance = metalFiniteDifferenceAbsTol +
		metalFiniteDifferenceRelTol*math.Abs(float64(numeric))
	if math.Abs(float64(analytic-numeric)) > tolerance {
		tb.Fatalf(
			"%s gradient = %g, finite difference %g, tolerance %g",
			name,
			analytic,
			numeric,
			tolerance,
		)
	}
}
