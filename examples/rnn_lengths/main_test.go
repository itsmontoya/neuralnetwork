package main

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_NewVariableLengthDatasetKeepsMixedLengthsAligned(t *testing.T) {
	var (
		dataset      *data.SequenceDataset
		inputs       *matrix.Matrix
		lengths      *data.SequenceLengths
		inputValues  []float32
		lengthValues []int
		row          int
		sample       int
		step         int
		wantLength   int
		nonZeroPad   bool
		err          error
	)

	if dataset, err = newVariableLengthDataset(rand.New(rand.NewSource(233))); err != nil {
		t.Fatalf("newVariableLengthDataset returned error: %v", err)
	}

	if dataset.SampleCount() != exampleClassCount*exampleSamplesPerClass {
		t.Fatalf(
			"SampleCount = %d, want %d",
			dataset.SampleCount(),
			exampleClassCount*exampleSamplesPerClass,
		)
	}

	if dataset.InputSize() != exampleSequenceSize {
		t.Fatalf("InputSize = %d, want %d", dataset.InputSize(), exampleSequenceSize)
	}

	if dataset.TargetSize() != exampleClassCount {
		t.Fatalf("TargetSize = %d, want %d", dataset.TargetSize(), exampleClassCount)
	}

	if dataset.Steps() != exampleSequenceSteps {
		t.Fatalf("Steps = %d, want %d", dataset.Steps(), exampleSequenceSteps)
	}

	if inputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	if inputValues, err = inputs.Values(); err != nil {
		t.Fatalf("input Values returned error: %v", err)
	}

	if lengths, err = dataset.Lengths(); err != nil {
		t.Fatalf("Lengths returned error: %v", err)
	}

	if lengthValues, err = lengths.Values(); err != nil {
		t.Fatalf("length Values returned error: %v", err)
	}

	for row = range lengthValues {
		sample = row % exampleSamplesPerClass
		wantLength = 2 + sample%3
		if lengthValues[row] != wantLength {
			t.Fatalf(
				"length row %d = %d, want %d",
				row,
				lengthValues[row],
				wantLength,
			)
		}

		for step = lengthValues[row]; step < exampleSequenceSteps; step++ {
			if inputValues[row*exampleSequenceSize+sequenceOffset(step, 0)] != 0 ||
				inputValues[row*exampleSequenceSize+sequenceOffset(step, 1)] != 0 {
				nonZeroPad = true
			}
		}
	}

	if !nonZeroPad {
		t.Fatal("dataset contains no non-zero padded suffix, want deliberate distractors")
	}
}

func Test_LengthAwarePredictionsAreDeterministicAndIgnorePaddedSuffixes(t *testing.T) {
	var (
		firstNetwork     *model.Sequential
		secondNetwork    *model.Sequential
		cleanInputs      *matrix.Matrix
		paddedInputs     *matrix.Matrix
		lengths          *data.SequenceLengths
		firstPrediction  *matrix.Matrix
		paddedPrediction *matrix.Matrix
		secondPrediction *matrix.Matrix
		firstValues      []float32
		paddedValues     []float32
		secondValues     []float32
		err              error
	)

	if firstNetwork, err = newLengthAwareRNNModel(rand.New(rand.NewSource(239))); err != nil {
		t.Fatalf("first newLengthAwareRNNModel returned error: %v", err)
	}

	if secondNetwork, err = newLengthAwareRNNModel(rand.New(rand.NewSource(239))); err != nil {
		t.Fatalf("second newLengthAwareRNNModel returned error: %v", err)
	}

	if cleanInputs, lengths, err = newCanonicalInputs(false); err != nil {
		t.Fatalf("clean newCanonicalInputs returned error: %v", err)
	}

	if paddedInputs, _, err = newCanonicalInputs(true); err != nil {
		t.Fatalf("padded newCanonicalInputs returned error: %v", err)
	}

	if firstPrediction, err = firstNetwork.PredictWithLengths(cleanInputs, lengths); err != nil {
		t.Fatalf("clean PredictWithLengths returned error: %v", err)
	}
	if firstValues, err = firstPrediction.Values(); err != nil {
		t.Fatalf("clean prediction Values returned error: %v", err)
	}

	if paddedPrediction, err = firstNetwork.PredictWithLengths(paddedInputs, lengths); err != nil {
		t.Fatalf("padded PredictWithLengths returned error: %v", err)
	}
	if paddedValues, err = paddedPrediction.Values(); err != nil {
		t.Fatalf("padded prediction Values returned error: %v", err)
	}

	if secondPrediction, err = secondNetwork.PredictWithLengths(cleanInputs, lengths); err != nil {
		t.Fatalf("second PredictWithLengths returned error: %v", err)
	}
	if secondValues, err = secondPrediction.Values(); err != nil {
		t.Fatalf("second prediction Values returned error: %v", err)
	}

	requireExactValues(t, paddedValues, firstValues)
	requireExactValues(t, secondValues, firstValues)
}

func Test_LengthAwareRNNFitIsDeterministicWithMixedLengthsAndPartialBatch(t *testing.T) {
	const integrationEpochs = 50

	var (
		trainingData      *data.SequenceDataset
		validationData    *data.SequenceDataset
		batches           []*data.SequenceBatch
		firstNetwork      *model.Sequential
		secondNetwork     *model.Sequential
		firstOptimizer    optimizer.Optimizer
		secondOptimizer   optimizer.Optimizer
		firstHistory      model.TrainingHistory
		secondHistory     model.TrainingHistory
		validationInputs  *matrix.Matrix
		validationTargets *matrix.Matrix
		validationLengths *data.SequenceLengths
		firstPrediction   *matrix.Matrix
		secondPrediction  *matrix.Matrix
		firstValues       []float32
		secondValues      []float32
		accuracyMetric    metric.CategoricalAccuracy
		accuracy          float32
		epochIndex        int
		err               error
	)

	if trainingData, err = newVariableLengthDataset(rand.New(rand.NewSource(241))); err != nil {
		t.Fatalf("training newVariableLengthDataset returned error: %v", err)
	}
	if validationData, err = newVariableLengthDataset(rand.New(rand.NewSource(251))); err != nil {
		t.Fatalf("validation newVariableLengthDataset returned error: %v", err)
	}

	if batches, err = trainingData.Batches(exampleBatchSize, nil); err != nil {
		t.Fatalf("Batches returned error: %v", err)
	}
	if batches[len(batches)-1].SampleCount() != trainingData.SampleCount()%exampleBatchSize {
		t.Fatalf(
			"final batch sample count = %d, want %d",
			batches[len(batches)-1].SampleCount(),
			trainingData.SampleCount()%exampleBatchSize,
		)
	}

	if firstNetwork, err = newLengthAwareRNNModel(rand.New(rand.NewSource(257))); err != nil {
		t.Fatalf("first newLengthAwareRNNModel returned error: %v", err)
	}
	if secondNetwork, err = newLengthAwareRNNModel(rand.New(rand.NewSource(257))); err != nil {
		t.Fatalf("second newLengthAwareRNNModel returned error: %v", err)
	}
	if firstOptimizer, err = optimizer.NewAdam(exampleLearningRate); err != nil {
		t.Fatalf("first NewAdam returned error: %v", err)
	}
	if secondOptimizer, err = optimizer.NewAdam(exampleLearningRate); err != nil {
		t.Fatalf("second NewAdam returned error: %v", err)
	}

	firstHistory, err = firstNetwork.FitWithLengths(trainingData, model.SequenceFitConfig{
		Epochs:         integrationEpochs,
		BatchSize:      exampleBatchSize,
		Shuffle:        true,
		Random:         rand.New(rand.NewSource(263)),
		Optimizer:      firstOptimizer,
		Loss:           loss.CategoricalCrossEntropy{},
		ValidationData: validationData,
		Accuracy:       accuracyMetric.Value,
	})
	if err != nil {
		t.Fatalf("first FitWithLengths returned error: %v", err)
	}

	secondHistory, err = secondNetwork.FitWithLengths(trainingData, model.SequenceFitConfig{
		Epochs:         integrationEpochs,
		BatchSize:      exampleBatchSize,
		Shuffle:        true,
		Random:         rand.New(rand.NewSource(263)),
		Optimizer:      secondOptimizer,
		Loss:           loss.CategoricalCrossEntropy{},
		ValidationData: validationData,
		Accuracy:       accuracyMetric.Value,
	})
	if err != nil {
		t.Fatalf("second FitWithLengths returned error: %v", err)
	}

	if len(firstHistory.Epochs) != integrationEpochs {
		t.Fatalf("first history epoch count = %d, want %d", len(firstHistory.Epochs), integrationEpochs)
	}
	if len(secondHistory.Epochs) != len(firstHistory.Epochs) {
		t.Fatalf(
			"second history epoch count = %d, want %d",
			len(secondHistory.Epochs),
			len(firstHistory.Epochs),
		)
	}
	for epochIndex = range firstHistory.Epochs {
		if secondHistory.Epochs[epochIndex] != firstHistory.Epochs[epochIndex] {
			t.Fatalf(
				"second epoch %d metrics = %+v, want %+v",
				epochIndex,
				secondHistory.Epochs[epochIndex],
				firstHistory.Epochs[epochIndex],
			)
		}
	}

	if validationInputs, err = validationData.Inputs(); err != nil {
		t.Fatalf("validation Inputs returned error: %v", err)
	}
	if validationTargets, err = validationData.Targets(); err != nil {
		t.Fatalf("validation Targets returned error: %v", err)
	}
	if validationLengths, err = validationData.Lengths(); err != nil {
		t.Fatalf("validation Lengths returned error: %v", err)
	}

	if firstPrediction, err = firstNetwork.PredictWithLengths(
		validationInputs,
		validationLengths,
	); err != nil {
		t.Fatalf("first PredictWithLengths returned error: %v", err)
	}
	if firstValues, err = firstPrediction.Values(); err != nil {
		t.Fatalf("first prediction Values returned error: %v", err)
	}

	if secondPrediction, err = secondNetwork.PredictWithLengths(
		validationInputs,
		validationLengths,
	); err != nil {
		t.Fatalf("second PredictWithLengths returned error: %v", err)
	}
	if secondValues, err = secondPrediction.Values(); err != nil {
		t.Fatalf("second prediction Values returned error: %v", err)
	}
	requireExactValues(t, secondValues, firstValues)

	if accuracy, err = accuracyMetric.Value(firstPrediction, validationTargets); err != nil {
		t.Fatalf("accuracy Value returned error: %v", err)
	}
	if accuracy < 0.95 {
		t.Fatalf("validation accuracy = %g, want at least 0.95", accuracy)
	}
}

func Test_LengthAwareRNNFirstPredictionAfterLoadUsesSuppliedLengths(t *testing.T) {
	var (
		network      *model.Sequential
		restored     *model.Sequential
		inputs       *matrix.Matrix
		lengths      *data.SequenceLengths
		before       *matrix.Matrix
		after        *matrix.Matrix
		beforeValues []float32
		afterValues  []float32
		document     bytes.Buffer
		err          error
	)

	if network, err = newLengthAwareRNNModel(rand.New(rand.NewSource(269))); err != nil {
		t.Fatalf("newLengthAwareRNNModel returned error: %v", err)
	}
	if inputs, lengths, err = newCanonicalInputs(true); err != nil {
		t.Fatalf("newCanonicalInputs returned error: %v", err)
	}

	if before, err = network.PredictWithLengths(inputs, lengths); err != nil {
		t.Fatalf("PredictWithLengths before Save returned error: %v", err)
	}
	if beforeValues, err = before.Values(); err != nil {
		t.Fatalf("prediction before Save Values returned error: %v", err)
	}

	if err = network.Save(&document); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	if restored, err = model.LoadSequential(bytes.NewReader(document.Bytes())); err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}

	if after, err = restored.PredictWithLengths(inputs, lengths); err != nil {
		t.Fatalf("first restored PredictWithLengths returned error: %v", err)
	}
	if afterValues, err = after.Values(); err != nil {
		t.Fatalf("restored prediction Values returned error: %v", err)
	}
	requireExactValues(t, afterValues, beforeValues)
}

func Test_LengthAwareRNNExampleRequiresCallerRandomSources(t *testing.T) {
	var err error

	if _, err = newVariableLengthDataset(nil); err == nil {
		t.Fatal("newVariableLengthDataset error = nil, want error for nil random source")
	}
	if _, err = newLengthAwareRNNModel(nil); err == nil {
		t.Fatal("newLengthAwareRNNModel error = nil, want error for nil random source")
	}
}

func requireExactValues(tb testing.TB, got, want []float32) {
	var index int

	tb.Helper()
	if len(got) != len(want) {
		tb.Fatalf("value count = %d, want %d", len(got), len(want))
	}

	for index = range want {
		if got[index] != want[index] {
			tb.Fatalf("value %d = %g, want %g", index, got[index], want[index])
		}
	}
}
