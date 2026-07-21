package main

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_NewTemporalOrderDatasetStoresFlattenedTimeMajorRows(t *testing.T) {
	var (
		random       *rand.Rand
		dataset      *data.Dataset
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		inputValues  []float32
		targetValues []float32
		row          int
		step         int
		feature      int
		classIndex   int
		offset       int
		value        float32
		wantPositive bool
		err          error
	)

	random = rand.New(rand.NewSource(103))
	if dataset, err = newTemporalOrderDataset(random); err != nil {
		t.Fatalf("newTemporalOrderDataset returned error: %v", err)
	}

	if dataset.SampleCount() != classCount*samplesPerClass {
		t.Fatalf("SampleCount = %d, want %d", dataset.SampleCount(), classCount*samplesPerClass)
	}

	if dataset.InputSize() != sequenceSize {
		t.Fatalf("InputSize = %d, want %d", dataset.InputSize(), sequenceSize)
	}

	if dataset.TargetSize() != classCount {
		t.Fatalf("TargetSize = %d, want %d", dataset.TargetSize(), classCount)
	}

	if sequenceOffset(1, 1) != 3 {
		t.Fatalf("sequenceOffset(1, 1) = %d, want time-major column 3", sequenceOffset(1, 1))
	}

	if inputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	if inputValues, err = inputs.Values(); err != nil {
		t.Fatalf("input Values returned error: %v", err)
	}

	if targets, err = dataset.Targets(); err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	if targetValues, err = targets.Values(); err != nil {
		t.Fatalf("target Values returned error: %v", err)
	}

	for row = 0; row < dataset.SampleCount(); row++ {
		classIndex = row / samplesPerClass
		for step = 0; step < sequenceSteps; step++ {
			for feature = 0; feature < sequenceFeatureSize; feature++ {
				offset = row*sequenceSize + sequenceOffset(step, feature)
				value = inputValues[offset]
				wantPositive = step == 0 && feature == classIndex
				wantPositive = wantPositive || step == 1 && feature == 1-classIndex
				if wantPositive {
					if value < 0.8 || value >= 1.2 {
						t.Fatalf("input row %d step %d feature %d = %g, want amplitude in [0.8, 1.2)", row, step, feature, value)
					}
					continue
				}

				if value != 0 {
					t.Fatalf("input row %d step %d feature %d = %g, want 0", row, step, feature, value)
				}
			}
		}

		for feature = 0; feature < classCount; feature++ {
			value = targetValues[row*classCount+feature]
			if feature == classIndex && value != 1 {
				t.Fatalf("target row %d class %d = %g, want 1", row, feature, value)
			}

			if feature != classIndex && value != 0 {
				t.Fatalf("target row %d class %d = %g, want 0", row, feature, value)
			}
		}
	}
}

func Test_NewRNNModelPredictsDeterministicallyAndStatelessly(t *testing.T) {
	var (
		firstRandom        *rand.Rand
		secondRandom       *rand.Rand
		firstNetwork       *model.Sequential
		secondNetwork      *model.Sequential
		recurrent          *layer.SimpleRNN
		parameters         []*optimizer.Parameter
		inputs             *matrix.Matrix
		inputValues        []float32
		firstPrediction    *matrix.Matrix
		repeatedPrediction *matrix.Matrix
		secondPrediction   *matrix.Matrix
		singleInput        *matrix.Matrix
		singlePrediction   *matrix.Matrix
		firstValues        []float32
		repeatedValues     []float32
		secondValues       []float32
		singleValues       []float32
		row                int
		col                int
		err                error
	)

	firstRandom = rand.New(rand.NewSource(107))
	secondRandom = rand.New(rand.NewSource(107))
	if inputs, err = newCanonicalInputs(); err != nil {
		t.Fatalf("newCanonicalInputs returned error: %v", err)
	}

	if inputValues, err = inputs.Values(); err != nil {
		t.Fatalf("input Values returned error: %v", err)
	}

	if firstNetwork, recurrent, err = newRNNModel(firstRandom); err != nil {
		t.Fatalf("first newRNNModel returned error: %v", err)
	}

	if secondNetwork, _, err = newRNNModel(secondRandom); err != nil {
		t.Fatalf("second newRNNModel returned error: %v", err)
	}

	if recurrent.InputShape().Steps() != sequenceSteps || recurrent.InputShape().FeatureSize() != sequenceFeatureSize {
		t.Fatalf(
			"recurrent input shape = %dx%d, want %dx%d",
			recurrent.InputShape().Steps(),
			recurrent.InputShape().FeatureSize(),
			sequenceSteps,
			sequenceFeatureSize,
		)
	}

	parameters = firstNetwork.Parameters()
	if len(parameters) != 5 {
		t.Fatalf("Parameters length = %d, want 5", len(parameters))
	}

	if parameters[0] != recurrent.InputWeights() || parameters[1] != recurrent.RecurrentWeights() || parameters[2] != recurrent.Biases() {
		t.Fatal("first model parameters are not recurrent input weights, recurrent weights, and biases")
	}

	if firstPrediction, err = firstNetwork.Predict(inputs); err != nil {
		t.Fatalf("first Predict returned error: %v", err)
	}

	if firstPrediction.Rows() != classCount || firstPrediction.Cols() != classCount {
		t.Fatalf("prediction shape = %dx%d, want %dx%d", firstPrediction.Rows(), firstPrediction.Cols(), classCount, classCount)
	}

	if firstValues, err = firstPrediction.Values(); err != nil {
		t.Fatalf("first prediction Values returned error: %v", err)
	}

	if repeatedPrediction, err = firstNetwork.Predict(inputs); err != nil {
		t.Fatalf("repeated Predict returned error: %v", err)
	}

	if repeatedValues, err = repeatedPrediction.Values(); err != nil {
		t.Fatalf("repeated prediction Values returned error: %v", err)
	}

	if secondPrediction, err = secondNetwork.Predict(inputs); err != nil {
		t.Fatalf("second model Predict returned error: %v", err)
	}

	if secondValues, err = secondPrediction.Values(); err != nil {
		t.Fatalf("second prediction Values returned error: %v", err)
	}

	for col = range firstValues {
		if repeatedValues[col] != firstValues[col] {
			t.Fatalf("repeated prediction value %d = %g, want stateless %g", col, repeatedValues[col], firstValues[col])
		}

		if secondValues[col] != firstValues[col] {
			t.Fatalf("second model prediction value %d = %g, want deterministic %g", col, secondValues[col], firstValues[col])
		}
	}

	for row = 0; row < classCount; row++ {
		if singleInput, err = matrix.FromSlice(
			1,
			sequenceSize,
			inputValues[row*sequenceSize:(row+1)*sequenceSize],
		); err != nil {
			t.Fatalf("row %d FromSlice returned error: %v", row, err)
		}

		if singlePrediction, err = firstNetwork.Predict(singleInput); err != nil {
			t.Fatalf("row %d Predict returned error: %v", row, err)
		}

		if singleValues, err = singlePrediction.Values(); err != nil {
			t.Fatalf("row %d prediction Values returned error: %v", row, err)
		}

		for col = 0; col < classCount; col++ {
			if singleValues[col] == firstValues[row*classCount+col] {
				continue
			}

			t.Fatalf(
				"isolated row %d class %d prediction = %g, want batched %g",
				row,
				col,
				singleValues[col],
				firstValues[row*classCount+col],
			)
		}
	}
}

func Test_RNNFitLearnsTemporalOrderWithValidationAndPartialBatch(t *testing.T) {
	const integrationEpochs = 50

	var (
		trainingRandom    *rand.Rand
		validationRandom  *rand.Rand
		modelRandom       *rand.Rand
		shuffleRandom     *rand.Rand
		trainingData      *data.Dataset
		validationData    *data.Dataset
		batches           []*data.Batch
		finalBatchInputs  *matrix.Matrix
		network           *model.Sequential
		optimizerRule     optimizer.Optimizer
		validationInputs  *matrix.Matrix
		validationTargets *matrix.Matrix
		initialPrediction *matrix.Matrix
		finalPrediction   *matrix.Matrix
		history           model.TrainingHistory
		finalMetrics      model.EpochMetrics
		lossFunction      loss.CategoricalCrossEntropy
		accuracyMetric    metric.CategoricalAccuracy
		initialLoss       float32
		finalAccuracy     float32
		err               error
	)

	trainingRandom = rand.New(rand.NewSource(109))
	validationRandom = rand.New(rand.NewSource(113))
	modelRandom = rand.New(rand.NewSource(127))
	shuffleRandom = rand.New(rand.NewSource(131))

	if trainingData, err = newTemporalOrderDataset(trainingRandom); err != nil {
		t.Fatalf("training newTemporalOrderDataset returned error: %v", err)
	}

	if validationData, err = newTemporalOrderDataset(validationRandom); err != nil {
		t.Fatalf("validation newTemporalOrderDataset returned error: %v", err)
	}

	if batches, err = trainingData.Batches(batchSize, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	if batches[0].SampleCount() != batchSize || batches[0].SampleCount() <= 1 {
		t.Fatalf("first batch sample count = %d, want %d and greater than one", batches[0].SampleCount(), batchSize)
	}

	if batches[len(batches)-1].SampleCount() != trainingData.SampleCount()%batchSize {
		t.Fatalf(
			"final batch sample count = %d, want %d",
			batches[len(batches)-1].SampleCount(),
			trainingData.SampleCount()%batchSize,
		)
	}

	if finalBatchInputs, err = batches[len(batches)-1].Inputs(); err != nil {
		t.Fatalf("final batch Inputs returned error: %v", err)
	}

	if finalBatchInputs.Cols() != sequenceSize {
		t.Fatalf("final batch input columns = %d, want complete sequence size %d", finalBatchInputs.Cols(), sequenceSize)
	}

	if network, _, err = newRNNModel(modelRandom); err != nil {
		t.Fatalf("newRNNModel returned error: %v", err)
	}

	if validationInputs, err = validationData.Inputs(); err != nil {
		t.Fatalf("validation Inputs returned error: %v", err)
	}

	if validationTargets, err = validationData.Targets(); err != nil {
		t.Fatalf("validation Targets returned error: %v", err)
	}

	if initialPrediction, err = network.Predict(validationInputs); err != nil {
		t.Fatalf("initial Predict returned error: %v", err)
	}

	if initialLoss, err = lossFunction.Value(initialPrediction, validationTargets); err != nil {
		t.Fatalf("initial loss Value returned error: %v", err)
	}

	if optimizerRule, err = optimizer.NewAdam(learningRate); err != nil {
		t.Fatalf("NewAdam returned error: %v", err)
	}

	history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:         integrationEpochs,
		BatchSize:      batchSize,
		Shuffle:        true,
		Random:         shuffleRandom,
		Optimizer:      optimizerRule,
		Loss:           lossFunction,
		ValidationData: validationData,
		Accuracy:       accuracyMetric.Value,
	})
	if err != nil {
		t.Fatalf("Fit returned error: %v", err)
	}

	if len(history.Epochs) != integrationEpochs {
		t.Fatalf("history epoch count = %d, want %d", len(history.Epochs), integrationEpochs)
	}

	finalMetrics = history.Epochs[len(history.Epochs)-1]
	if !finalMetrics.HasAccuracy || finalMetrics.Accuracy < 0.95 {
		t.Fatalf("final training accuracy = %g with flag %t, want at least 0.95", finalMetrics.Accuracy, finalMetrics.HasAccuracy)
	}

	if !finalMetrics.HasValidationLoss {
		t.Fatal("HasValidationLoss = false, want true")
	}

	if finalMetrics.ValidationLoss >= initialLoss*0.5 {
		t.Fatalf(
			"final validation loss = %g, want less than half initial loss %g",
			finalMetrics.ValidationLoss,
			initialLoss,
		)
	}

	if !finalMetrics.HasValidationAccuracy || finalMetrics.ValidationAccuracy < 0.95 {
		t.Fatalf(
			"final validation accuracy = %g with flag %t, want at least 0.95",
			finalMetrics.ValidationAccuracy,
			finalMetrics.HasValidationAccuracy,
		)
	}

	if finalPrediction, err = network.Predict(validationInputs); err != nil {
		t.Fatalf("final Predict returned error: %v", err)
	}

	if finalAccuracy, err = accuracyMetric.Value(finalPrediction, validationTargets); err != nil {
		t.Fatalf("final accuracy Value returned error: %v", err)
	}

	if finalAccuracy < 0.95 {
		t.Fatalf("evaluated validation accuracy = %g, want at least 0.95", finalAccuracy)
	}
}

func Test_RNNTrainBatchUpdatesRecurrentParameters(t *testing.T) {
	type parameterCase struct {
		name      string
		parameter *optimizer.Parameter
		before    []float32
	}

	var (
		dataRandom    *rand.Rand
		modelRandom   *rand.Rand
		dataset       *data.Dataset
		batches       []*data.Batch
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		network       *model.Sequential
		recurrent     *layer.SimpleRNN
		parameters    []*optimizer.Parameter
		tests         []parameterCase
		test          parameterCase
		currentValues []float32
		optimizerRule optimizer.Optimizer
		index         int
		changed       bool
		err           error
	)

	dataRandom = rand.New(rand.NewSource(137))
	modelRandom = rand.New(rand.NewSource(139))
	if dataset, err = newTemporalOrderDataset(dataRandom); err != nil {
		t.Fatalf("newTemporalOrderDataset returned error: %v", err)
	}

	if batches, err = dataset.Batches(batchSize, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	if inputs, err = batches[0].Inputs(); err != nil {
		t.Fatalf("batch Inputs returned error: %v", err)
	}

	if targets, err = batches[0].Targets(); err != nil {
		t.Fatalf("batch Targets returned error: %v", err)
	}

	if network, recurrent, err = newRNNModel(modelRandom); err != nil {
		t.Fatalf("newRNNModel returned error: %v", err)
	}

	parameters = network.Parameters()
	if len(parameters) != 5 {
		t.Fatalf("Parameters length = %d, want 5", len(parameters))
	}

	if parameters[0] != recurrent.InputWeights() || parameters[1] != recurrent.RecurrentWeights() || parameters[2] != recurrent.Biases() {
		t.Fatal("first model parameters are not recurrent input weights, recurrent weights, and biases")
	}

	tests = []parameterCase{
		{name: "input weights", parameter: recurrent.InputWeights()},
		{name: "recurrent weights", parameter: recurrent.RecurrentWeights()},
		{name: "biases", parameter: recurrent.Biases()},
	}
	for index = range tests {
		if tests[index].before, err = tests[index].parameter.Values().Values(); err != nil {
			t.Fatalf("%s Values returned error: %v", tests[index].name, err)
		}
	}

	if optimizerRule, err = optimizer.NewSGD(0.05); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	if _, err = network.TrainBatch(inputs, targets, loss.CategoricalCrossEntropy{}, optimizerRule); err != nil {
		t.Fatalf("TrainBatch returned error: %v", err)
	}

	for _, test = range tests {
		if currentValues, err = test.parameter.Values().Values(); err != nil {
			t.Fatalf("updated %s Values returned error: %v", test.name, err)
		}

		changed = false
		for index = range test.before {
			if currentValues[index] == test.before[index] {
				continue
			}

			changed = true
			break
		}

		if !changed {
			t.Fatalf("%s did not change after TrainBatch", test.name)
		}
	}
}

func Test_RNNExampleRequiresCallerRandomSources(t *testing.T) {
	var err error

	if _, err = newTemporalOrderDataset(nil); err == nil {
		t.Fatal("newTemporalOrderDataset error = nil, want error for nil random source")
	}

	if _, _, err = newRNNModel(nil); err == nil {
		t.Fatal("newRNNModel error = nil, want error for nil random source")
	}
}
