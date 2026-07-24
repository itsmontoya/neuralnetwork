//go:build darwin && cgo && metal && !purego

package model_test

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const metalTrainingTolerance = 2e-4

func Test_SequentialResidentTrainingValidatesCategoricalTargets(t *testing.T) {
	type testcase struct {
		name       string
		targetRow  []float32
		wantDetail string
	}

	var tests []testcase
	tests = []testcase{
		{
			name:       "no class",
			targetRow:  []float32{0, 0, 0, 0},
			wantDetail: "categorical target row 0 must contain exactly one class: ones=0",
		},
		{
			name:       "multiple classes",
			targetRow:  []float32{1, 0, 1, 0},
			wantDetail: "categorical target row 0 must contain exactly one class: ones=2",
		},
		{
			name:       "fractional class",
			targetRow:  []float32{0, 0.5, 0.5, 0},
			wantDetail: "categorical target at row 0 column 1 must be 0 or 1: value=0.5",
		},
	}

	requireModelMetal(t)
	var test testcase
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			var (
				shape         metalInferenceShape
				network       *model.Sequential
				input         *matrix.Matrix
				targets       *matrix.Matrix
				optimizerRule *optimizer.SGD
				before        [][]float32
				targetValues  []float32
				parameters    []*optimizer.Parameter
				counters      metaltest.Counters
				row           int
				err           error
			)

			shape = metalTrainingShape()
			network, input, _, _, _, _, _ =
				metalInferenceModel(t, shape, activation.ReLU{})
			parameters = network.Parameters()
			before = parameterValues(t, parameters)
			targetValues = metalTrainingTargetValues(shape)
			for row = 0; row < len(test.targetRow); row++ {
				targetValues[row] = test.targetRow[row]
			}
			targets = metalBackwardMatrix(t, shape.batchSize, shape.classCount, targetValues)
			if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
				t.Fatalf("NewSGD returned error: %v", err)
			}

			metaltest.Enable()
			defer metaltest.Disable()
			if _, err = network.TrainBatch(
				input,
				targets,
				loss.CategoricalCrossEntropy{},
				optimizerRule,
			); err == nil || !strings.Contains(err.Error(), test.wantDetail) {
				t.Fatalf("TrainBatch error = %v, want %q", err, test.wantDetail)
			}
			counters = metaltest.Snapshot()
			if counters.ResultDownloads != 1 || counters.ResultDownloadBytes != 20 {
				t.Fatalf("validation counters = %+v, want one 20-byte diagnostic result", counters)
			}
			requireParameterValues(t, parameters, before, 0)
		})
	}
}

func Test_SequentialResidentTrainingTransfersAndParameterReuse(t *testing.T) {
	var (
		shape           metalInferenceShape
		network         *model.Sequential
		warmInput       *matrix.Matrix
		warmTargets     *matrix.Matrix
		input           *matrix.Matrix
		targets         *matrix.Matrix
		optimizerRule   *optimizer.SGD
		counters        metaltest.Counters
		wantUploadBytes uint64
		err             error
	)

	requireModelMetal(t)
	shape = metalTrainingShape()
	network = metalBaselineModel(t, metalBaselineShape(shape))
	warmInput, warmTargets = metalTrainingMatrices(t, shape)
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	if _, err = network.TrainBatch(
		warmInput,
		warmTargets,
		loss.CategoricalCrossEntropy{},
		optimizerRule,
	); err != nil {
		t.Fatalf("warm-up TrainBatch returned error: %v", err)
	}
	input, targets = metalTrainingMatrices(t, shape)
	wantUploadBytes = uint64(shape.batchSize*(shape.inputSize+shape.classCount)) * 4

	metaltest.Enable()
	defer metaltest.Disable()
	if _, err = network.TrainBatch(
		input,
		targets,
		loss.CategoricalCrossEntropy{},
		optimizerRule,
	); err != nil {
		t.Fatalf("warmed TrainBatch returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 2 || counters.InputUploadBytes != wantUploadBytes {
		t.Fatalf(
			"warmed uploads = %+v, want only one input and one target totaling %d bytes",
			counters,
			wantUploadBytes,
		)
	}
	if counters.ResultDownloads != 1 || counters.ResultDownloadBytes != 20 {
		t.Fatalf("warmed downloads = %+v, want only the scalar diagnostic result", counters)
	}
	if counters.CommandSubmissions != 2 || counters.Waits != 2 {
		t.Fatalf("warmed commands = %+v, want loss and backward/update scopes", counters)
	}

	metaltest.Reset()
	if _, err = network.TrainBatch(
		input,
		targets,
		loss.CategoricalCrossEntropy{},
		optimizerRule,
	); err != nil {
		t.Fatalf("repeated TrainBatch returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 0 || counters.ResultDownloads != 1 {
		t.Fatalf("repeated transfers = %+v, want resident input, targets, and parameters", counters)
	}
}

func Test_SequentialResidentTrainingOneStepParity(t *testing.T) {
	var (
		shape               metalInferenceShape
		gradientNetwork     *model.Sequential
		trainingNetwork     *model.Sequential
		gradientInput       *matrix.Matrix
		trainingInput       *matrix.Matrix
		targets             *matrix.Matrix
		predictions         *matrix.Matrix
		predictionGradient  *matrix.Matrix
		optimizerRule       *optimizer.SGD
		metrics             model.TrainMetrics
		reference           metalBackwardReference
		inputValues         []float32
		hiddenWeights       []float32
		hiddenBiases        []float32
		outputWeights       []float32
		outputBiases        []float32
		predictionValues    []float32
		targetValues        []float32
		categoricalGradient []float32
		expectedParameters  [][]float32
		gradientParameters  []*optimizer.Parameter
		trainingParameters  []*optimizer.Parameter
		expectedLoss        float32
		learningRate        float32
		err                 error
	)

	requireModelMetal(t)
	shape = metalTrainingShape()
	gradientNetwork, gradientInput, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases =
		metalInferenceModel(t, shape, activation.ReLU{})
	trainingNetwork, trainingInput, _, _, _, _, _ =
		metalInferenceModel(t, shape, activation.ReLU{})
	targetValues = metalTrainingTargetValues(shape)
	targets = metalBackwardMatrix(t, shape.batchSize, shape.classCount, targetValues)
	predictionValues = metalInferenceReference(
		inputValues,
		shape,
		hiddenWeights,
		hiddenBiases,
		outputWeights,
		outputBiases,
		true,
	)
	expectedLoss = categoricalLossReference(predictionValues, targetValues, shape)
	categoricalGradient = categoricalGradientReference(predictionValues, targetValues, shape)
	reference = metalBackwardReferenceValues(
		inputValues,
		categoricalGradient,
		shape,
		hiddenWeights,
		hiddenBiases,
		outputWeights,
		outputBiases,
		true,
	)

	if predictions, err = gradientNetwork.Predict(gradientInput); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	requireMetalInferenceValues(t, predictions, predictionValues)
	if predictionGradient, err = (loss.CategoricalCrossEntropy{}).Gradient(
		predictions,
		targets,
	); err != nil {
		t.Fatalf("Gradient returned error: %v", err)
	}
	if _, err = gradientNetwork.Backward(predictionGradient); err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}
	gradientParameters = gradientNetwork.Parameters()
	requireBackwardMatrixValues(
		t,
		gradientParameters[0].Gradient(),
		reference.hiddenWeightGradient,
		metalTrainingTolerance,
	)
	requireBackwardMatrixValues(
		t,
		gradientParameters[1].Gradient(),
		reference.hiddenBiasGradient,
		metalTrainingTolerance,
	)
	requireBackwardMatrixValues(
		t,
		gradientParameters[2].Gradient(),
		reference.outputWeightGradient,
		metalTrainingTolerance,
	)
	requireBackwardMatrixValues(
		t,
		gradientParameters[3].Gradient(),
		reference.outputBiasGradient,
		metalTrainingTolerance,
	)

	learningRate = 0.01
	expectedParameters = [][]float32{
		updatedParameterReference(hiddenWeights, reference.hiddenWeightGradient, learningRate),
		updatedParameterReference(hiddenBiases, reference.hiddenBiasGradient, learningRate),
		updatedParameterReference(outputWeights, reference.outputWeightGradient, learningRate),
		updatedParameterReference(outputBiases, reference.outputBiasGradient, learningRate),
	}
	if optimizerRule, err = optimizer.NewSGD(learningRate); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	if metrics, err = trainingNetwork.TrainBatch(
		trainingInput,
		targets,
		loss.CategoricalCrossEntropy{},
		optimizerRule,
	); err != nil {
		t.Fatalf("TrainBatch returned error: %v", err)
	}
	requireTrainingFloat(t, metrics.Loss, expectedLoss, metalTrainingTolerance)
	trainingParameters = trainingNetwork.Parameters()
	requireParameterValues(
		t,
		trainingParameters,
		expectedParameters,
		metalTrainingTolerance,
	)
	var parameter *optimizer.Parameter
	for _, parameter = range trainingParameters {
		requireBackwardMatrixZero(t, parameter.Gradient())
	}
}

func Test_SequentialResidentTrainingIsDeterministicAndTracksCPU(t *testing.T) {
	var (
		shape            metalInferenceShape
		first            *model.Sequential
		second           *model.Sequential
		firstInput       *matrix.Matrix
		secondInput      *matrix.Matrix
		targets          *matrix.Matrix
		firstOptimizer   *optimizer.SGD
		secondOptimizer  *optimizer.SGD
		firstMetrics     model.TrainMetrics
		secondMetrics    model.TrainMetrics
		firstPrediction  *matrix.Matrix
		secondPrediction *matrix.Matrix
		reference        metalBackwardReference
		inputValues      []float32
		hiddenWeights    []float32
		hiddenBiases     []float32
		outputWeights    []float32
		outputBiases     []float32
		predictionValues []float32
		targetValues     []float32
		gradientValues   []float32
		expected         [][]float32
		firstLosses      []float32
		secondLosses     []float32
		secondValues     []float32
		learningRate     float32
		step             int
		err              error
	)

	requireModelMetal(t)
	shape = metalTrainingShape()
	first, firstInput, inputValues, hiddenWeights, hiddenBiases, outputWeights, outputBiases =
		metalInferenceModel(t, shape, activation.ReLU{})
	second, secondInput, _, _, _, _, _ =
		metalInferenceModel(t, shape, activation.ReLU{})
	targetValues = metalTrainingTargetValues(shape)
	targets = metalBackwardMatrix(t, shape.batchSize, shape.classCount, targetValues)
	learningRate = 0.01
	if firstOptimizer, err = optimizer.NewSGD(learningRate); err != nil {
		t.Fatalf("NewSGD first returned error: %v", err)
	}
	if secondOptimizer, err = optimizer.NewSGD(learningRate); err != nil {
		t.Fatalf("NewSGD second returned error: %v", err)
	}

	for step = 0; step < 3; step++ {
		predictionValues = metalInferenceReference(
			inputValues,
			shape,
			hiddenWeights,
			hiddenBiases,
			outputWeights,
			outputBiases,
			true,
		)
		gradientValues = categoricalGradientReference(predictionValues, targetValues, shape)
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
		hiddenWeights = updatedParameterReference(
			hiddenWeights,
			reference.hiddenWeightGradient,
			learningRate,
		)
		hiddenBiases = updatedParameterReference(
			hiddenBiases,
			reference.hiddenBiasGradient,
			learningRate,
		)
		outputWeights = updatedParameterReference(
			outputWeights,
			reference.outputWeightGradient,
			learningRate,
		)
		outputBiases = updatedParameterReference(
			outputBiases,
			reference.outputBiasGradient,
			learningRate,
		)
		if firstMetrics, err = first.TrainBatch(
			firstInput,
			targets,
			loss.CategoricalCrossEntropy{},
			firstOptimizer,
		); err != nil {
			t.Fatalf("first TrainBatch step %d returned error: %v", step, err)
		}
		if secondMetrics, err = second.TrainBatch(
			secondInput,
			targets,
			loss.CategoricalCrossEntropy{},
			secondOptimizer,
		); err != nil {
			t.Fatalf("second TrainBatch step %d returned error: %v", step, err)
		}
		firstLosses = append(firstLosses, firstMetrics.Loss)
		secondLosses = append(secondLosses, secondMetrics.Loss)
		if math.Float32bits(firstMetrics.Loss) != math.Float32bits(secondMetrics.Loss) {
			t.Fatalf(
				"step %d losses differ: first=%g second=%g",
				step,
				firstMetrics.Loss,
				secondMetrics.Loss,
			)
		}
	}

	for step = range firstLosses {
		if math.Float32bits(firstLosses[step]) != math.Float32bits(secondLosses[step]) {
			t.Fatalf(
				"loss history step %d differs: first=%g second=%g",
				step,
				firstLosses[step],
				secondLosses[step],
			)
		}
	}
	if firstPrediction, err = first.Predict(firstInput); err != nil {
		t.Fatalf("first final Predict returned error: %v", err)
	}
	if secondPrediction, err = second.Predict(secondInput); err != nil {
		t.Fatalf("second final Predict returned error: %v", err)
	}
	if secondValues, err = secondPrediction.Values(); err != nil {
		t.Fatalf("second final prediction Values returned error: %v", err)
	}
	requireBackwardMatrixValues(t, firstPrediction, secondValues, 0)

	expected = [][]float32{hiddenWeights, hiddenBiases, outputWeights, outputBiases}
	requireParameterValues(t, first.Parameters(), expected, metalTrainingTolerance)
	requireParameterValues(t, second.Parameters(), expected, metalTrainingTolerance)
	requireParameterValues(t, first.Parameters(), parameterValues(t, second.Parameters()), 0)
}

func Test_SequentialResidentTrainingReducesLossThroughTrainBatchAndFit(t *testing.T) {
	var (
		shape         metalInferenceShape
		network       *model.Sequential
		fitNetwork    *model.Sequential
		input         *matrix.Matrix
		targets       *matrix.Matrix
		fitInput      *matrix.Matrix
		fitTargets    *matrix.Matrix
		dataset       *data.Dataset
		optimizerRule *optimizer.SGD
		fitOptimizer  *optimizer.SGD
		metrics       model.TrainMetrics
		history       model.TrainingHistory
		firstLoss     float32
		lastLoss      float32
		step          int
		err           error
	)

	requireModelMetal(t)
	shape = metalTrainingShape()
	network = metalBaselineModel(t, metalBaselineShape(shape))
	fitNetwork = metalBaselineModel(t, metalBaselineShape(shape))
	input, targets = metalClassificationMatrices(t, shape)
	fitInput, fitTargets = metalClassificationMatrices(t, shape)
	if optimizerRule, err = optimizer.NewSGD(0.2); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	for step = 0; step < 8; step++ {
		if metrics, err = network.TrainBatch(
			input,
			targets,
			loss.CategoricalCrossEntropy{},
			optimizerRule,
		); err != nil {
			t.Fatalf("TrainBatch step %d returned error: %v", step, err)
		}
		if step == 0 {
			firstLoss = metrics.Loss
		}
		lastLoss = metrics.Loss
	}
	if !(lastLoss < firstLoss) {
		t.Fatalf("TrainBatch loss did not decrease: first=%g last=%g", firstLoss, lastLoss)
	}

	if dataset, err = data.NewDataset(fitInput, fitTargets); err != nil {
		t.Fatalf("NewDataset returned error: %v", err)
	}
	if fitOptimizer, err = optimizer.NewSGD(0.2); err != nil {
		t.Fatalf("NewSGD fit returned error: %v", err)
	}
	if history, err = fitNetwork.Fit(dataset, model.FitConfig{
		Epochs:    4,
		BatchSize: shape.batchSize,
		Optimizer: fitOptimizer,
		Loss:      loss.CategoricalCrossEntropy{},
	}); err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}
	if len(history.Epochs) != 4 {
		t.Fatalf("Fit epoch count = %d, want 4", len(history.Epochs))
	}
	if !(history.Epochs[3].Loss < history.Epochs[0].Loss) {
		t.Fatalf(
			"Fit loss did not decrease: first=%g last=%g",
			history.Epochs[0].Loss,
			history.Epochs[3].Loss,
		)
	}
}

func Test_SequentialResidentFitObservationBoundaries(t *testing.T) {
	var (
		shape             metalInferenceShape
		network           *model.Sequential
		input             *matrix.Matrix
		targets           *matrix.Matrix
		validationInput   *matrix.Matrix
		validationTargets *matrix.Matrix
		trainingData      *data.Dataset
		validationData    *data.Dataset
		optimizerRule     *optimizer.SGD
		schedule          *optimizer.StepDecay
		earlyStopping     *model.EarlyStopping
		history           model.TrainingHistory
		callbackRates     []float32
		counters          metaltest.Counters
		err               error
	)

	requireModelMetal(t)
	shape = metalTrainingShape()
	network = metalBaselineModel(t, metalBaselineShape(shape))
	input, targets = metalClassificationMatrices(t, shape)
	validationInput, validationTargets = metalClassificationMatrices(t, shape)
	if trainingData, err = data.NewDataset(input, targets); err != nil {
		t.Fatalf("NewDataset training returned error: %v", err)
	}
	if validationData, err = data.NewDataset(validationInput, validationTargets); err != nil {
		t.Fatalf("NewDataset validation returned error: %v", err)
	}
	if optimizerRule, err = optimizer.NewSGD(0.1); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	if schedule, err = optimizer.NewStepDecay(0.1, 0.5, 1); err != nil {
		t.Fatalf("NewStepDecay returned error: %v", err)
	}
	if earlyStopping, err = model.NewEarlyStopping(1, 100); err != nil {
		t.Fatalf("NewEarlyStopping returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:               5,
		BatchSize:            shape.batchSize,
		Optimizer:            optimizerRule,
		LearningRateSchedule: schedule,
		EarlyStopping:        earlyStopping,
		Loss:                 loss.CategoricalCrossEntropy{},
		ValidationData:       validationData,
		Accuracy:             (metric.CategoricalAccuracy{}).Value,
		Callback: func(model.EpochMetrics) (callbackErr error) {
			callbackRates = append(callbackRates, optimizerRule.LearningRate())
			return nil
		},
	}); err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}
	if len(history.Epochs) != 2 || len(callbackRates) != 2 {
		t.Fatalf(
			"early-stopped epochs=%d callbacks=%d, want 2 each",
			len(history.Epochs),
			len(callbackRates),
		)
	}
	if callbackRates[0] != 0.1 || callbackRates[1] != 0.05 {
		t.Fatalf("callback learning rates = %v, want [0.1 0.05]", callbackRates)
	}
	var metrics model.EpochMetrics
	for _, metrics = range history.Epochs {
		if !metrics.HasAccuracy || !metrics.HasValidationLoss ||
			!metrics.HasValidationAccuracy {
			t.Fatalf("Fit metrics = %+v, want training and validation observations", metrics)
		}
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads == 0 || counters.CommandSubmissions == 0 ||
		counters.Waits != counters.CommandSubmissions {
		t.Fatalf("Fit observation counters = %+v, want completed observation barriers", counters)
	}
}

func Test_SequentialResidentTrainingSerialization(t *testing.T) {
	var (
		shape         metalInferenceShape
		network       *model.Sequential
		loaded        *model.Sequential
		input         *matrix.Matrix
		targets       *matrix.Matrix
		loadedInput   *matrix.Matrix
		optimizerRule *optimizer.SGD
		document      bytes.Buffer
		reencoded     bytes.Buffer
		counters      metaltest.Counters
		err           error
	)

	requireModelMetal(t)
	shape = metalTrainingShape()
	network = metalBaselineModel(t, metalBaselineShape(shape))
	input, targets = metalTrainingMatrices(t, shape)
	if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	if _, err = network.TrainBatch(
		input,
		targets,
		loss.CategoricalCrossEntropy{},
		optimizerRule,
	); err != nil {
		t.Fatalf("TrainBatch returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if err = network.Save(&document); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads != 4 {
		t.Fatalf("immediate Save counters = %+v, want four current parameter downloads", counters)
	}
	if strings.Contains(document.String(), "residency") ||
		strings.Contains(document.String(), "device") {
		t.Fatalf("serialized document contains runtime state: %s", document.String())
	}
	if loaded, err = model.LoadSequential(bytes.NewReader(document.Bytes())); err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}
	if err = loaded.Save(&reencoded); err != nil {
		t.Fatalf("loaded Save returned error: %v", err)
	}
	if !bytes.Equal(document.Bytes(), reencoded.Bytes()) {
		t.Fatal("version 1 bytes changed after loading an equivalent logical model")
	}

	loadedInput, _ = metalTrainingMatrices(t, shape)
	metaltest.Reset()
	if _, err = loaded.Predict(loadedInput); err != nil {
		t.Fatalf("loaded Predict returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 5 {
		t.Fatalf("loaded Predict counters = %+v, want fresh input and four parameter uploads", counters)
	}
}

func Test_SequentialResidentTrainingUnsupportedFallbacks(t *testing.T) {
	type testcase struct {
		name          string
		lossFunc      loss.Loss
		optimizerFunc func(testing.TB) optimizer.Optimizer
	}

	var tests []testcase
	tests = []testcase{
		{
			name:     "unsupported loss",
			lossFunc: loss.MeanSquaredError{},
			optimizerFunc: func(tb testing.TB) (optimizerRule optimizer.Optimizer) {
				var err error
				if optimizerRule, err = optimizer.NewSGD(0.01); err != nil {
					tb.Fatalf("NewSGD returned error: %v", err)
				}
				return optimizerRule
			},
		},
		{
			name:     "regularized optimizer",
			lossFunc: loss.CategoricalCrossEntropy{},
			optimizerFunc: func(tb testing.TB) (optimizerRule optimizer.Optimizer) {
				var (
					sgd         *optimizer.SGD
					regularizer *optimizer.L2WeightDecay
					err         error
				)
				if sgd, err = optimizer.NewSGD(0.01); err != nil {
					tb.Fatalf("NewSGD returned error: %v", err)
				}
				if regularizer, err = optimizer.NewL2WeightDecay(0.001); err != nil {
					tb.Fatalf("NewL2WeightDecay returned error: %v", err)
				}
				if optimizerRule, err = optimizer.NewRegularized(sgd, regularizer); err != nil {
					tb.Fatalf("NewRegularized returned error: %v", err)
				}
				return optimizerRule
			},
		},
	}

	requireModelMetal(t)
	var test testcase
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			var (
				shape         metalInferenceShape
				network       *model.Sequential
				input         *matrix.Matrix
				targets       *matrix.Matrix
				optimizerRule optimizer.Optimizer
				before        [][]float32
				after         [][]float32
				parameters    []*optimizer.Parameter
				counters      metaltest.Counters
				err           error
			)

			shape = metalTrainingShape()
			network = metalBaselineModel(t, metalBaselineShape(shape))
			input, targets = metalTrainingMatrices(t, shape)
			optimizerRule = test.optimizerFunc(t)
			parameters = network.Parameters()
			before = parameterValues(t, parameters)

			metaltest.Enable()
			defer metaltest.Disable()
			if _, err = network.TrainBatch(
				input,
				targets,
				test.lossFunc,
				optimizerRule,
			); err != nil {
				t.Fatalf("TrainBatch returned error: %v", err)
			}
			counters = metaltest.Snapshot()
			if counters.ResultDownloads == 0 {
				t.Fatalf("fallback counters = %+v, want an explicit CPU observation", counters)
			}
			after = parameterValues(t, parameters)
			if parameterSlicesEqual(before, after) {
				t.Fatal("fallback training did not update parameters")
			}
			var parameter *optimizer.Parameter
			for _, parameter = range parameters {
				requireBackwardMatrixZero(t, parameter.Gradient())
			}
		})
	}
}

func metalTrainingShape() (shape metalInferenceShape) {
	shape.name = "training"
	shape.batchSize = 256
	shape.inputSize = 128
	shape.hiddenSize = 128
	shape.classCount = 16
	return shape
}

func metalTrainingMatrices(
	tb testing.TB,
	shape metalInferenceShape,
) (input, targets *matrix.Matrix) {
	var inputValues []float32

	tb.Helper()
	inputValues = metalInferenceValues(shape.batchSize*shape.inputSize, 11, 0.4)
	input = metalBackwardMatrix(tb, shape.batchSize, shape.inputSize, inputValues)
	targets = metalBackwardMatrix(
		tb,
		shape.batchSize,
		shape.classCount,
		metalTrainingTargetValues(shape),
	)
	return input, targets
}

func metalClassificationMatrices(
	tb testing.TB,
	shape metalInferenceShape,
) (input, targets *matrix.Matrix) {
	var (
		inputValues  []float32
		targetValues []float32
		row          int
		class        int
	)

	tb.Helper()
	inputValues = make([]float32, shape.batchSize*shape.inputSize)
	targetValues = make([]float32, shape.batchSize*shape.classCount)
	for row = 0; row < shape.batchSize; row++ {
		class = row % shape.classCount
		inputValues[row*shape.inputSize+class] = 1
		inputValues[row*shape.inputSize+shape.classCount+class] = 0.5
		targetValues[row*shape.classCount+class] = 1
	}
	input = metalBackwardMatrix(tb, shape.batchSize, shape.inputSize, inputValues)
	targets = metalBackwardMatrix(tb, shape.batchSize, shape.classCount, targetValues)
	return input, targets
}

func metalTrainingTargetValues(shape metalInferenceShape) (values []float32) {
	var row int

	values = make([]float32, shape.batchSize*shape.classCount)
	for row = 0; row < shape.batchSize; row++ {
		values[row*shape.classCount+row%shape.classCount] = 1
	}
	return values
}

func categoricalLossReference(
	predictions,
	targets []float32,
	shape metalInferenceShape,
) (value float32) {
	var (
		row        int
		col        int
		prediction float32
	)

	for row = 0; row < shape.batchSize; row++ {
		for col = 0; col < shape.classCount; col++ {
			if targets[row*shape.classCount+col] != 1 {
				continue
			}
			prediction = predictions[row*shape.classCount+col]
			if prediction < 1e-7 {
				prediction = 1e-7
			} else if prediction > 1-1e-7 {
				prediction = 1 - 1e-7
			}
			value -= float32(math.Log(float64(prediction)))
		}
	}
	value /= float32(shape.batchSize)
	return value
}

func categoricalGradientReference(
	predictions,
	targets []float32,
	shape metalInferenceShape,
) (gradient []float32) {
	var (
		index      int
		prediction float32
	)

	gradient = make([]float32, len(predictions))
	for index = range predictions {
		if targets[index] == 0 {
			continue
		}
		prediction = predictions[index]
		if prediction < 1e-7 {
			prediction = 1e-7
		} else if prediction > 1-1e-7 {
			prediction = 1 - 1e-7
		}
		gradient[index] = -targets[index] / prediction / float32(shape.batchSize)
	}
	return gradient
}

func updatedParameterReference(
	values,
	gradient []float32,
	learningRate float32,
) (updated []float32) {
	var index int

	updated = append([]float32(nil), values...)
	for index = range updated {
		updated[index] -= learningRate * gradient[index]
	}
	return updated
}

func parameterValues(
	tb testing.TB,
	parameters []*optimizer.Parameter,
) (values [][]float32) {
	var (
		parameter *optimizer.Parameter
		current   []float32
		err       error
	)

	tb.Helper()
	values = make([][]float32, 0, len(parameters))
	for _, parameter = range parameters {
		if current, err = parameter.Values().Values(); err != nil {
			tb.Fatalf("parameter Values returned error: %v", err)
		}
		values = append(values, current)
	}
	return values
}

func requireParameterValues(
	tb testing.TB,
	parameters []*optimizer.Parameter,
	want [][]float32,
	tolerance float32,
) {
	var (
		parameter *optimizer.Parameter
		index     int
	)

	tb.Helper()
	if len(parameters) != len(want) {
		tb.Fatalf("parameter count = %d, want %d", len(parameters), len(want))
	}
	for index, parameter = range parameters {
		requireBackwardMatrixValues(tb, parameter.Values(), want[index], tolerance)
	}
}

func parameterSlicesEqual(left, right [][]float32) (equal bool) {
	var (
		row int
		col int
	)

	if len(left) != len(right) {
		return false
	}
	for row = range left {
		if len(left[row]) != len(right[row]) {
			return false
		}
		for col = range left[row] {
			if left[row][col] != right[row][col] {
				return false
			}
		}
	}
	return true
}

func requireTrainingFloat(tb testing.TB, got, want, tolerance float32) {
	var difference float64

	tb.Helper()
	difference = math.Abs(float64(got - want))
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) ||
		difference > float64(tolerance)+float64(tolerance)*math.Abs(float64(want)) {
		tb.Fatalf("value = %g, want %g, difference %g", got, want, difference)
	}
}
