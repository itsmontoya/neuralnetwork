// Command cnn trains a convolutional network to classify synthetic horizontal
// and vertical line images. Synthetic data keeps the example fast,
// deterministic, and independent of downloads while exercising flattened CHW
// input rows through the complete CNN training path.
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
	epochs          = 80
	logInterval     = 20
	batchSize       = 8
	classCount      = 2
	samplesPerClass = 17
	imageChannels   = 1
	imageHeight     = 5
	imageWidth      = 5
	imageSize       = imageChannels * imageHeight * imageWidth
	learningRate    = 0.02
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	var (
		dataRandom       *rand.Rand
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

	dataRandom = rand.New(rand.NewSource(31))
	validationRandom = rand.New(rand.NewSource(37))
	modelRandom = rand.New(rand.NewSource(41))
	shuffleRandom = rand.New(rand.NewSource(43))

	if trainingData, err = newSyntheticDataset(dataRandom); err != nil {
		return err
	}

	if validationData, err = newSyntheticDataset(validationRandom); err != nil {
		return err
	}

	if network, _, err = newCNNModel(modelRandom); err != nil {
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

func newSyntheticDataset(random *rand.Rand) (dataset *data.Dataset, err error) {
	var (
		sampleCount  int
		inputValues  []float32
		targetValues []float32
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		classIndex   int
		sample       int
		row          int
		linePosition int
		intensity    float32
		pixel        int
	)

	if random == nil {
		err = errors.New("cnn example: data random source is nil")
		return nil, err
	}

	sampleCount = classCount * samplesPerClass
	inputValues = make([]float32, sampleCount*imageSize)
	targetValues = make([]float32, sampleCount*classCount)

	for classIndex = 0; classIndex < classCount; classIndex++ {
		for sample = 0; sample < samplesPerClass; sample++ {
			row = classIndex*samplesPerClass + sample
			linePosition = 1 + random.Intn(imageHeight-2)
			intensity = 0.8 + 0.4*random.Float32()

			for pixel = 0; pixel < imageWidth; pixel++ {
				if classIndex == 0 {
					inputValues[row*imageSize+imageOffset(0, linePosition, pixel)] = intensity
					continue
				}

				inputValues[row*imageSize+imageOffset(0, pixel, linePosition)] = intensity
			}

			targetValues[row*classCount+classIndex] = 1
		}
	}

	if inputs, err = matrix.FromSlice(sampleCount, imageSize, inputValues); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(sampleCount, classCount, targetValues); err != nil {
		return nil, err
	}

	dataset, err = data.NewDataset(inputs, targets)
	return dataset, err
}

func newCNNModel(random *rand.Rand) (
	network *model.Sequential,
	convolution *layer.Conv2D,
	err error,
) {
	var (
		inputShape        layer.SpatialShape
		convolutionConfig layer.Conv2DConfig
		convolutionReLU   *layer.Activation
		poolConfig        layer.MaxPool2DConfig
		pool              *layer.MaxPool2D
		flatten           *layer.Flatten
		output            *layer.Dense
		outputActivation  *layer.Activation
	)

	if random == nil {
		err = errors.New("cnn example: model random source is nil")
		return nil, nil, err
	}

	if inputShape, err = layer.NewSpatialShape(imageChannels, imageHeight, imageWidth); err != nil {
		return nil, nil, err
	}

	if convolutionConfig, err = layer.NewConv2DConfig(inputShape, 4, 3, 3, 1, 1, 1, 1); err != nil {
		return nil, nil, err
	}

	if convolution, err = layer.NewConv2D(convolutionConfig, layer.HeNormalWeights(random)); err != nil {
		return nil, nil, err
	}

	if convolutionReLU, err = layer.NewActivation(activation.ReLU{}); err != nil {
		return nil, nil, err
	}

	if poolConfig, err = layer.NewMaxPool2DConfig(convolution.OutputShape(), 2, 2, 2, 2); err != nil {
		return nil, nil, err
	}

	if pool, err = layer.NewMaxPool2D(poolConfig); err != nil {
		return nil, nil, err
	}

	if flatten, err = layer.NewFlatten(pool.OutputShape()); err != nil {
		return nil, nil, err
	}

	if output, err = layer.NewDense(flatten.OutputSize(), classCount, layer.XavierUniformWeights(random)); err != nil {
		return nil, nil, err
	}

	if outputActivation, err = layer.NewActivation(activation.Softmax{}); err != nil {
		return nil, nil, err
	}

	network, err = model.NewSequential(
		convolution,
		convolutionReLU,
		pool,
		flatten,
		output,
		outputActivation,
	)
	return network, convolution, err
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
	for row, name = range []string{"horizontal", "vertical"} {
		classIndex = argmax(predictionValues[row*classCount : row*classCount+classCount])
		fmt.Printf("%s => %s\n", name, className(classIndex))
	}

	return nil
}

func newCanonicalInputs() (inputs *matrix.Matrix, err error) {
	var (
		values []float32
		center int
		pixel  int
	)

	values = make([]float32, classCount*imageSize)
	center = imageHeight / 2
	for pixel = 0; pixel < imageWidth; pixel++ {
		values[imageOffset(0, center, pixel)] = 1
		values[imageSize+imageOffset(0, pixel, center)] = 1
	}

	inputs, err = matrix.FromSlice(classCount, imageSize, values)
	return inputs, err
}

func imageOffset(channel, row, col int) (offset int) {
	offset = channel*imageHeight*imageWidth + row*imageWidth + col
	return offset
}

func className(index int) (name string) {
	switch index {
	case 0:
		name = "horizontal"
	case 1:
		name = "vertical"
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
