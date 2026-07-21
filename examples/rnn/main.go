// Command rnn trains a recurrent network to classify the order of two events.
// Synthetic data keeps the example fast, deterministic, and independent of
// downloads while exercising flattened time-major sequence rows through the
// complete RNN training path.
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
	epochs              = 80
	logInterval         = 20
	batchSize           = 8
	classCount          = 2
	samplesPerClass     = 17
	sequenceSteps       = 3
	sequenceFeatureSize = 2
	sequenceSize        = sequenceSteps * sequenceFeatureSize
	hiddenSize          = 6
	learningRate        = 0.03
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
		trainingData     *data.Dataset
		validationData   *data.Dataset
		network          *model.Sequential
		optimizerRule    optimizer.Optimizer
		history          model.TrainingHistory
		accuracyMetric   metric.CategoricalAccuracy
		finalMetrics     model.EpochMetrics
	)

	trainingRandom = rand.New(rand.NewSource(83))
	validationRandom = rand.New(rand.NewSource(89))
	modelRandom = rand.New(rand.NewSource(97))
	shuffleRandom = rand.New(rand.NewSource(101))

	if trainingData, err = newTemporalOrderDataset(trainingRandom); err != nil {
		return err
	}

	if validationData, err = newTemporalOrderDataset(validationRandom); err != nil {
		return err
	}

	if network, _, err = newRNNModel(modelRandom); err != nil {
		return err
	}

	if optimizerRule, err = optimizer.NewAdam(learningRate); err != nil {
		return err
	}

	history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:         epochs,
		BatchSize:      batchSize,
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

	err = printPredictions(network)
	return err
}

// newTemporalOrderDataset builds sequences containing the same event pair in
// opposite orders. The final blank step requires the recurrent state to retain
// that order for classification. Every sequence remains in one matrix row.
func newTemporalOrderDataset(random *rand.Rand) (dataset *data.Dataset, err error) {
	var (
		sampleCount     int
		inputValues     []float32
		targetValues    []float32
		inputs          *matrix.Matrix
		targets         *matrix.Matrix
		classIndex      int
		sample          int
		row             int
		rowOffset       int
		firstFeature    int
		secondFeature   int
		firstAmplitude  float32
		secondAmplitude float32
	)

	if random == nil {
		err = errors.New("rnn example: data random source is nil")
		return nil, err
	}

	sampleCount = classCount * samplesPerClass
	inputValues = make([]float32, sampleCount*sequenceSize)
	targetValues = make([]float32, sampleCount*classCount)

	for classIndex = 0; classIndex < classCount; classIndex++ {
		for sample = 0; sample < samplesPerClass; sample++ {
			row = classIndex*samplesPerClass + sample
			rowOffset = row * sequenceSize
			firstFeature = classIndex
			secondFeature = 1 - classIndex
			firstAmplitude = 0.8 + 0.4*random.Float32()
			secondAmplitude = 0.8 + 0.4*random.Float32()

			inputValues[rowOffset+sequenceOffset(0, firstFeature)] = firstAmplitude
			inputValues[rowOffset+sequenceOffset(1, secondFeature)] = secondAmplitude
			targetValues[row*classCount+classIndex] = 1
		}
	}

	if inputs, err = matrix.FromSlice(sampleCount, sequenceSize, inputValues); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(sampleCount, classCount, targetValues); err != nil {
		return nil, err
	}

	dataset, err = data.NewDataset(inputs, targets)
	return dataset, err
}

func newRNNModel(random *rand.Rand) (
	network *model.Sequential,
	recurrent *layer.SimpleRNN,
	err error,
) {
	var (
		inputShape       layer.SequenceShape
		recurrentConfig  layer.SimpleRNNConfig
		lastStep         *layer.LastStep
		output           *layer.Dense
		outputActivation *layer.Activation
	)

	if random == nil {
		err = errors.New("rnn example: model random source is nil")
		return nil, nil, err
	}

	if inputShape, err = layer.NewSequenceShape(sequenceSteps, sequenceFeatureSize); err != nil {
		return nil, nil, err
	}

	if recurrentConfig, err = layer.NewSimpleRNNConfig(inputShape, hiddenSize); err != nil {
		return nil, nil, err
	}

	if recurrent, err = layer.NewSimpleRNN(
		recurrentConfig,
		layer.XavierUniformWeights(random),
		layer.XavierUniformWeights(random),
	); err != nil {
		return nil, nil, err
	}

	if lastStep, err = layer.NewLastStep(recurrent.OutputShape()); err != nil {
		return nil, nil, err
	}

	if output, err = layer.NewDense(lastStep.OutputSize(), classCount, layer.XavierUniformWeights(random)); err != nil {
		return nil, nil, err
	}

	if outputActivation, err = layer.NewActivation(activation.Softmax{}); err != nil {
		return nil, nil, err
	}

	network, err = model.NewSequential(recurrent, lastStep, output, outputActivation)
	return network, recurrent, err
}

func printEpochMetrics(metrics model.EpochMetrics) (err error) {
	if metrics.Epoch == 1 || metrics.Epoch%logInterval == 0 || metrics.Epoch == epochs {
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

func printPredictions(network *model.Sequential) (err error) {
	var (
		inputs           *matrix.Matrix
		predictions      *matrix.Matrix
		predictionValues []float32
		classIndex       int
		name             string
		row              int
	)

	if inputs, err = newCanonicalInputs(); err != nil {
		return err
	}

	if predictions, err = network.Predict(inputs); err != nil {
		return err
	}

	if predictionValues, err = predictions.Values(); err != nil {
		return err
	}

	fmt.Println("predictions:")
	for row, name = range []string{"A then B", "B then A"} {
		classIndex = argmax(predictionValues[row*classCount : row*classCount+classCount])
		fmt.Printf("%s => %s\n", name, className(classIndex))
	}

	return nil
}

func newCanonicalInputs() (inputs *matrix.Matrix, err error) {
	var values []float32

	values = make([]float32, classCount*sequenceSize)
	values[sequenceOffset(0, 0)] = 1
	values[sequenceOffset(1, 1)] = 1
	values[sequenceSize+sequenceOffset(0, 1)] = 1
	values[sequenceSize+sequenceOffset(1, 0)] = 1

	inputs, err = matrix.FromSlice(classCount, sequenceSize, values)
	return inputs, err
}

func sequenceOffset(step, feature int) (offset int) {
	offset = step*sequenceFeatureSize + feature
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
		col   int
		value float32
		best  float32
	)

	best = values[0]
	for col = 1; col < len(values); col++ {
		value = values[col]
		if value <= best {
			continue
		}

		best = value
		index = col
	}

	return index
}
