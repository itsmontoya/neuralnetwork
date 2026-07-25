// Command rnn_lengths trains a recurrent network on padded sequences with
// mixed logical lengths. Synthetic data keeps the example deterministic and
// independent of downloads while exercising the complete length-aware path.
package main

import (
	"errors"
	"fmt"
	"log"
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const (
	exampleEpochs              = 80
	exampleLogInterval         = 20
	exampleBatchSize           = 8
	exampleClassCount          = 2
	exampleSamplesPerClass     = 18
	exampleSequenceSteps       = 4
	exampleSequenceFeatureSize = 2
	exampleSequenceSize        = exampleSequenceSteps * exampleSequenceFeatureSize
	exampleHiddenSize          = 6
	exampleLearningRate        = 0.03
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	var (
		trainingRandom   *rand.Rand
		validationRandom *rand.Rand
		modelRandom      *rand.Rand
		shuffleRandom    *rand.Rand
		trainingData     *data.SequenceDataset
		validationData   *data.SequenceDataset
		network          *model.Sequential
		optimizerRule    optimizer.Optimizer
		history          model.TrainingHistory
		accuracyMetric   metric.CategoricalAccuracy
		finalMetrics     model.EpochMetrics
	)

	trainingRandom = rand.New(rand.NewSource(211))
	validationRandom = rand.New(rand.NewSource(223))
	modelRandom = rand.New(rand.NewSource(227))
	shuffleRandom = rand.New(rand.NewSource(229))

	if trainingData, err = newVariableLengthDataset(trainingRandom); err != nil {
		return err
	}

	if validationData, err = newVariableLengthDataset(validationRandom); err != nil {
		return err
	}

	if network, err = newLengthAwareRNNModel(modelRandom); err != nil {
		return err
	}

	if optimizerRule, err = optimizer.NewAdam(exampleLearningRate); err != nil {
		return err
	}

	history, err = network.FitWithLengths(trainingData, model.SequenceFitConfig{
		Epochs:         exampleEpochs,
		BatchSize:      exampleBatchSize,
		Shuffle:        true,
		Random:         shuffleRandom,
		Optimizer:      optimizerRule,
		Loss:           loss.CategoricalCrossEntropy{},
		ValidationData: validationData,
		Accuracy:       accuracyMetric.Value,
		Callback:       printEpochMetrics,
	})
	if err != nil {
		return err
	}

	finalMetrics = history.Epochs[len(history.Epochs)-1]
	fmt.Printf(
		"final loss %.6f accuracy %.3f validation loss %.6f validation accuracy %.3f\n",
		finalMetrics.Loss,
		finalMetrics.Accuracy,
		finalMetrics.ValidationLoss,
		finalMetrics.ValidationAccuracy,
	)

	err = printLengthAwarePredictions(network)
	return err
}

// newVariableLengthDataset builds event-order sequences whose valid prefixes
// have two, three, or four steps. Non-zero suffix values are deliberate
// distractors and must not affect the last-valid prediction.
func newVariableLengthDataset(random *rand.Rand) (dataset *data.SequenceDataset, err error) {
	var (
		sampleCount     int
		inputValues     []float32
		targetValues    []float32
		lengthValues    []int
		inputs          *matrix.Matrix
		targets         *matrix.Matrix
		lengths         *data.SequenceLengths
		classIndex      int
		sample          int
		row             int
		rowOffset       int
		logicalLength   int
		firstStep       int
		firstFeature    int
		secondFeature   int
		firstAmplitude  float32
		secondAmplitude float32
		paddingStep     int
		paddingFeature  int
		paddingValue    float32
	)

	if random == nil {
		err = errors.New("rnn lengths example: data random source is nil")
		return nil, err
	}

	sampleCount = exampleClassCount * exampleSamplesPerClass
	inputValues = make([]float32, sampleCount*exampleSequenceSize)
	targetValues = make([]float32, sampleCount*exampleClassCount)
	lengthValues = make([]int, sampleCount)

	for classIndex = 0; classIndex < exampleClassCount; classIndex++ {
		for sample = 0; sample < exampleSamplesPerClass; sample++ {
			row = classIndex*exampleSamplesPerClass + sample
			rowOffset = row * exampleSequenceSize
			logicalLength = 2 + sample%3
			firstStep = logicalLength - 2
			firstFeature = classIndex
			secondFeature = 1 - classIndex
			firstAmplitude = 0.8 + 0.4*random.Float32()
			secondAmplitude = 0.8 + 0.4*random.Float32()

			inputValues[rowOffset+sequenceOffset(firstStep, firstFeature)] = firstAmplitude
			inputValues[rowOffset+sequenceOffset(firstStep+1, secondFeature)] = secondAmplitude
			for paddingStep = logicalLength; paddingStep < exampleSequenceSteps; paddingStep++ {
				paddingFeature = (paddingStep + classIndex) % exampleSequenceFeatureSize
				paddingValue = 3 + random.Float32()
				inputValues[rowOffset+sequenceOffset(paddingStep, paddingFeature)] = paddingValue
			}

			targetValues[row*exampleClassCount+classIndex] = 1
			lengthValues[row] = logicalLength
		}
	}

	if inputs, err = matrix.FromSlice(sampleCount, exampleSequenceSize, inputValues); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(sampleCount, exampleClassCount, targetValues); err != nil {
		return nil, err
	}

	if lengths, err = data.NewSequenceLengths(exampleSequenceSteps, lengthValues); err != nil {
		return nil, err
	}

	dataset, err = data.NewSequenceDataset(inputs, targets, lengths)
	return dataset, err
}

func newLengthAwareRNNModel(random *rand.Rand) (network *model.Sequential, err error) {
	var (
		inputShape       layer.SequenceShape
		recurrentConfig  layer.SimpleRNNConfig
		recurrent        *layer.SimpleRNN
		gather           *layer.GatherLastValid
		output           *layer.Dense
		outputActivation *layer.Activation
	)

	if random == nil {
		err = errors.New("rnn lengths example: model random source is nil")
		return nil, err
	}

	if inputShape, err = layer.NewSequenceShape(
		exampleSequenceSteps,
		exampleSequenceFeatureSize,
	); err != nil {
		return nil, err
	}

	if recurrentConfig, err = layer.NewSimpleRNNConfig(inputShape, exampleHiddenSize); err != nil {
		return nil, err
	}

	if recurrent, err = layer.NewSimpleRNN(
		recurrentConfig,
		layer.XavierUniformWeights(random),
		layer.XavierUniformWeights(random),
	); err != nil {
		return nil, err
	}

	if gather, err = layer.NewGatherLastValid(recurrent.OutputShape()); err != nil {
		return nil, err
	}

	if output, err = layer.NewDense(
		gather.OutputSize(),
		exampleClassCount,
		layer.XavierUniformWeights(random),
	); err != nil {
		return nil, err
	}

	if outputActivation, err = layer.NewActivation(activation.Softmax{}); err != nil {
		return nil, err
	}

	network, err = model.NewSequential(recurrent, gather, output, outputActivation)
	return network, err
}

func printEpochMetrics(metrics model.EpochMetrics) (err error) {
	if metrics.Epoch == 1 ||
		metrics.Epoch%exampleLogInterval == 0 ||
		metrics.Epoch == exampleEpochs {
		fmt.Printf(
			"epoch %3d loss %.6f accuracy %.3f validation loss %.6f validation accuracy %.3f\n",
			metrics.Epoch,
			metrics.Loss,
			metrics.Accuracy,
			metrics.ValidationLoss,
			metrics.ValidationAccuracy,
		)
	}

	return nil
}

func printLengthAwarePredictions(network *model.Sequential) (err error) {
	var (
		inputs           *matrix.Matrix
		lengths          *data.SequenceLengths
		predictions      *matrix.Matrix
		predictionValues []float32
		lengthValues     []int
		classIndex       int
		name             string
		row              int
	)

	if inputs, lengths, err = newCanonicalInputs(false); err != nil {
		return err
	}

	if predictions, err = network.PredictWithLengths(inputs, lengths); err != nil {
		return err
	}

	if predictionValues, err = predictions.Values(); err != nil {
		return err
	}

	if lengthValues, err = lengths.Values(); err != nil {
		return err
	}

	fmt.Println("predictions:")
	for row, name = range []string{"A then B", "B then A"} {
		classIndex = argmax(
			predictionValues[row*exampleClassCount : row*exampleClassCount+exampleClassCount],
		)
		fmt.Printf(
			"%s with %d valid steps => %s\n",
			name,
			lengthValues[row],
			className(classIndex),
		)
	}

	return nil
}

// newCanonicalInputs returns the same two valid prefixes with either zero or
// deliberately large padded suffixes.
func newCanonicalInputs(withDistractorPadding bool) (
	inputs *matrix.Matrix,
	lengths *data.SequenceLengths,
	err error,
) {
	var (
		values       []float32
		lengthValues []int
	)

	values = []float32{
		1, 0,
		0, 1,
		0, 0,
		0, 0,
		0, 0,
		0, 1,
		1, 0,
		0, 0,
	}
	if withDistractorPadding {
		values[sequenceOffset(2, 0)] = -13
		values[sequenceOffset(2, 1)] = 17
		values[sequenceOffset(3, 0)] = 19
		values[sequenceOffset(3, 1)] = -23
		values[exampleSequenceSize+sequenceOffset(3, 0)] = -29
		values[exampleSequenceSize+sequenceOffset(3, 1)] = 31
	}
	lengthValues = []int{2, 3}

	if inputs, err = matrix.FromSlice(exampleClassCount, exampleSequenceSize, values); err != nil {
		return nil, nil, err
	}

	if lengths, err = data.NewSequenceLengths(exampleSequenceSteps, lengthValues); err != nil {
		return nil, nil, err
	}

	return inputs, lengths, nil
}

func sequenceOffset(step, feature int) (offset int) {
	offset = step*exampleSequenceFeatureSize + feature
	return offset
}

func className(index int) (name string) {
	switch index {
	case 0:
		name = "A then B"
	case 1:
		name = "B then A"
	default:
		name = "unknown"
	}

	return name
}

func argmax(values []float32) (index int) {
	var (
		column int
		value  float32
		best   float32
	)

	best = values[0]
	for column = 1; column < len(values); column++ {
		value = values[column]
		if value <= best {
			continue
		}

		best = value
		index = column
	}

	return index
}
