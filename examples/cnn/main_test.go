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

func Test_NewSyntheticDatasetStoresFlattenedCHWRows(t *testing.T) {
	type lineCase struct {
		name       string
		sampleRow  int
		horizontal bool
	}

	var (
		random           *rand.Rand
		dataset          *data.Dataset
		inputs           *matrix.Matrix
		targets          *matrix.Matrix
		inputValues      []float32
		targetValues     []float32
		tests            []lineCase
		test             lineCase
		row              int
		col              int
		positiveCount    int
		firstPositive    int
		firstPositiveRow int
		firstPositiveCol int
		value            float32
		err              error
	)

	random = rand.New(rand.NewSource(31))
	dataset, err = newSyntheticDataset(random)
	if err != nil {
		t.Fatalf("newSyntheticDataset returned error: %v", err)
	}

	if dataset.SampleCount() != classCount*samplesPerClass {
		t.Fatalf("SampleCount = %d, want %d", dataset.SampleCount(), classCount*samplesPerClass)
	}

	if dataset.InputSize() != imageSize {
		t.Fatalf("InputSize = %d, want %d", dataset.InputSize(), imageSize)
	}

	if dataset.TargetSize() != classCount {
		t.Fatalf("TargetSize = %d, want %d", dataset.TargetSize(), classCount)
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
		for col = 0; col < classCount; col++ {
			value = targetValues[row*classCount+col]
			if col == row/samplesPerClass && value != 1 {
				t.Fatalf("target row %d col %d = %g, want 1", row, col, value)
			}

			if col != row/samplesPerClass && value != 0 {
				t.Fatalf("target row %d col %d = %g, want 0", row, col, value)
			}
		}
	}

	tests = []lineCase{
		{name: "horizontal", sampleRow: 0, horizontal: true},
		{name: "vertical", sampleRow: samplesPerClass, horizontal: false},
	}

	for _, test = range tests {
		positiveCount = 0
		firstPositive = -1
		for row = 0; row < imageHeight; row++ {
			for col = 0; col < imageWidth; col++ {
				value = inputValues[test.sampleRow*imageSize+imageOffset(0, row, col)]
				if value <= 0 {
					continue
				}

				positiveCount++
				if firstPositive >= 0 {
					if test.horizontal && row != firstPositiveRow {
						t.Fatalf("%s sample has positive values in rows %d and %d", test.name, firstPositiveRow, row)
					}

					if !test.horizontal && col != firstPositiveCol {
						t.Fatalf("%s sample has positive values in columns %d and %d", test.name, firstPositiveCol, col)
					}
					continue
				}

				firstPositive = imageOffset(0, row, col)
				firstPositiveRow = row
				firstPositiveCol = col
			}
		}

		if positiveCount != imageWidth {
			t.Fatalf("%s positive value count = %d, want %d", test.name, positiveCount, imageWidth)
		}
	}
}

func Test_NewCNNModelPredictsDeterministically(t *testing.T) {
	var (
		dataRandom        *rand.Rand
		firstModelRandom  *rand.Rand
		secondModelRandom *rand.Rand
		dataset           *data.Dataset
		inputs            *matrix.Matrix
		firstNetwork      *model.Sequential
		secondNetwork     *model.Sequential
		firstPrediction   *matrix.Matrix
		secondPrediction  *matrix.Matrix
		firstValues       []float32
		secondValues      []float32
		index             int
		err               error
	)

	dataRandom = rand.New(rand.NewSource(47))
	firstModelRandom = rand.New(rand.NewSource(53))
	secondModelRandom = rand.New(rand.NewSource(53))

	if dataset, err = newSyntheticDataset(dataRandom); err != nil {
		t.Fatalf("newSyntheticDataset returned error: %v", err)
	}

	if inputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	if firstNetwork, _, err = newCNNModel(firstModelRandom); err != nil {
		t.Fatalf("first newCNNModel returned error: %v", err)
	}

	if secondNetwork, _, err = newCNNModel(secondModelRandom); err != nil {
		t.Fatalf("second newCNNModel returned error: %v", err)
	}

	if firstPrediction, err = firstNetwork.Predict(inputs); err != nil {
		t.Fatalf("first Predict returned error: %v", err)
	}

	if secondPrediction, err = secondNetwork.Predict(inputs); err != nil {
		t.Fatalf("second Predict returned error: %v", err)
	}

	if firstPrediction.Rows() != dataset.SampleCount() {
		t.Fatalf("prediction Rows = %d, want %d", firstPrediction.Rows(), dataset.SampleCount())
	}

	if firstPrediction.Cols() != classCount {
		t.Fatalf("prediction Cols = %d, want %d", firstPrediction.Cols(), classCount)
	}

	if firstValues, err = firstPrediction.Values(); err != nil {
		t.Fatalf("first prediction Values returned error: %v", err)
	}

	if secondValues, err = secondPrediction.Values(); err != nil {
		t.Fatalf("second prediction Values returned error: %v", err)
	}

	for index = range firstValues {
		if firstValues[index] == secondValues[index] {
			continue
		}

		t.Fatalf("prediction value %d = %g, want deterministic %g", index, firstValues[index], secondValues[index])
	}
}

func Test_CNNFitLearnsWithValidationAndPartialBatch(t *testing.T) {
	const integrationEpochs = 60

	var (
		trainingRandom    *rand.Rand
		validationRandom  *rand.Rand
		modelRandom       *rand.Rand
		shuffleRandom     *rand.Rand
		trainingData      *data.Dataset
		validationData    *data.Dataset
		batches           []*data.Batch
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

	trainingRandom = rand.New(rand.NewSource(59))
	validationRandom = rand.New(rand.NewSource(61))
	modelRandom = rand.New(rand.NewSource(67))
	shuffleRandom = rand.New(rand.NewSource(71))

	if trainingData, err = newSyntheticDataset(trainingRandom); err != nil {
		t.Fatalf("training newSyntheticDataset returned error: %v", err)
	}

	if validationData, err = newSyntheticDataset(validationRandom); err != nil {
		t.Fatalf("validation newSyntheticDataset returned error: %v", err)
	}

	if batches, err = trainingData.Batches(batchSize, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}

	if batches[0].SampleCount() <= 1 {
		t.Fatalf("first batch sample count = %d, want greater than one", batches[0].SampleCount())
	}

	if batches[len(batches)-1].SampleCount() != trainingData.SampleCount()%batchSize {
		t.Fatalf(
			"final batch sample count = %d, want %d",
			batches[len(batches)-1].SampleCount(),
			trainingData.SampleCount()%batchSize,
		)
	}

	if network, _, err = newCNNModel(modelRandom); err != nil {
		t.Fatalf("newCNNModel returned error: %v", err)
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
	if !finalMetrics.HasAccuracy || finalMetrics.Accuracy < 0.9 {
		t.Fatalf("final training accuracy = %g with flag %t, want at least 0.9", finalMetrics.Accuracy, finalMetrics.HasAccuracy)
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

	if !finalMetrics.HasValidationAccuracy || finalMetrics.ValidationAccuracy < 0.9 {
		t.Fatalf(
			"final validation accuracy = %g with flag %t, want at least 0.9",
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

	if finalAccuracy < 0.9 {
		t.Fatalf("evaluated validation accuracy = %g, want at least 0.9", finalAccuracy)
	}
}

func Test_CNNTrainBatchUpdatesConvolutionParameters(t *testing.T) {
	var (
		dataRandom     *rand.Rand
		modelRandom    *rand.Rand
		dataset        *data.Dataset
		inputs         *matrix.Matrix
		targets        *matrix.Matrix
		network        *model.Sequential
		convolution    *layer.Conv2D
		parameters     []*optimizer.Parameter
		weightValues   []float32
		biasValues     []float32
		currentValues  []float32
		optimizerRule  optimizer.Optimizer
		index          int
		weightsChanged bool
		biasesChanged  bool
		err            error
	)

	dataRandom = rand.New(rand.NewSource(73))
	modelRandom = rand.New(rand.NewSource(79))

	if dataset, err = newSyntheticDataset(dataRandom); err != nil {
		t.Fatalf("newSyntheticDataset returned error: %v", err)
	}

	if inputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	if targets, err = dataset.Targets(); err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	if network, convolution, err = newCNNModel(modelRandom); err != nil {
		t.Fatalf("newCNNModel returned error: %v", err)
	}

	parameters = network.Parameters()
	if len(parameters) != 4 {
		t.Fatalf("Parameters length = %d, want 4", len(parameters))
	}

	if parameters[0] != convolution.Weights() || parameters[1] != convolution.Biases() {
		t.Fatal("first model parameters are not convolution weights and biases")
	}

	if weightValues, err = convolution.Weights().Values().Values(); err != nil {
		t.Fatalf("weight Values returned error: %v", err)
	}

	if biasValues, err = convolution.Biases().Values().Values(); err != nil {
		t.Fatalf("bias Values returned error: %v", err)
	}

	if optimizerRule, err = optimizer.NewSGD(0.05); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}

	if _, err = network.TrainBatch(inputs, targets, loss.CategoricalCrossEntropy{}, optimizerRule); err != nil {
		t.Fatalf("TrainBatch returned error: %v", err)
	}

	if currentValues, err = convolution.Weights().Values().Values(); err != nil {
		t.Fatalf("updated weight Values returned error: %v", err)
	}

	for index = range weightValues {
		if currentValues[index] == weightValues[index] {
			continue
		}

		weightsChanged = true
		break
	}

	if currentValues, err = convolution.Biases().Values().Values(); err != nil {
		t.Fatalf("updated bias Values returned error: %v", err)
	}

	for index = range biasValues {
		if currentValues[index] == biasValues[index] {
			continue
		}

		biasesChanged = true
		break
	}

	if !weightsChanged {
		t.Fatal("convolution weights did not change after TrainBatch")
	}

	if !biasesChanged {
		t.Fatal("convolution biases did not change after TrainBatch")
	}
}
