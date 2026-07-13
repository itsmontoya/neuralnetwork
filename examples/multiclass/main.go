package main

import (
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
	epochs          = 500
	logInterval     = 100
	classCount      = 3
	samplesPerClass = 40
	clusterStddev   = 0.18
	learningRate    = 0.03
)

type point struct {
	x float32
	y float32
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	var (
		dataRandom     *rand.Rand
		modelRandom    *rand.Rand
		shuffleRandom  *rand.Rand
		trainingData   *data.Dataset
		network        *model.Sequential
		optimizerRule  optimizer.Optimizer
		history        model.TrainingHistory
		accuracyMetric metric.CategoricalAccuracy
		finalMetrics   model.EpochMetrics
	)

	dataRandom = rand.New(rand.NewSource(21))
	modelRandom = rand.New(rand.NewSource(23))
	shuffleRandom = rand.New(rand.NewSource(29))

	if trainingData, err = newClusterDataset(dataRandom); err != nil {
		return err
	}

	if network, err = newClusterModel(modelRandom); err != nil {
		return err
	}

	if optimizerRule, err = optimizer.NewAdam(learningRate); err != nil {
		return err
	}

	history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:    epochs,
		BatchSize: 12,
		Shuffle:   true,
		Random:    shuffleRandom,
		Optimizer: optimizerRule,
		Loss:      loss.CategoricalCrossEntropy{},
		Accuracy:  accuracyMetric.Value,
		Callback:  printEpochMetrics,
	})
	if err != nil {
		return err
	}

	finalMetrics = history.Epochs[len(history.Epochs)-1]
	fmt.Printf("final loss %.6f accuracy %.3f\n", finalMetrics.Loss, finalMetrics.Accuracy)

	err = printClassPredictions(network)
	return err
}

func newClusterDataset(random *rand.Rand) (dataset *data.Dataset, err error) {
	var (
		centers      []point
		inputValues  []float32
		targetValues []float32
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		classIndex   int
		sample       int
		col          int
		x            float32
		y            float32
		center       point
	)

	centers = clusterCenters()
	inputValues = make([]float32, 0, len(centers)*samplesPerClass*2)
	targetValues = make([]float32, 0, len(centers)*samplesPerClass*classCount)

	for classIndex, center = range centers {
		for sample = 0; sample < samplesPerClass; sample++ {
			x = center.x + float32(random.NormFloat64())*clusterStddev
			y = center.y + float32(random.NormFloat64())*clusterStddev

			inputValues = append(inputValues, x, y)
			for col = 0; col < classCount; col++ {
				if col == classIndex {
					targetValues = append(targetValues, 1)
					continue
				}

				targetValues = append(targetValues, 0)
			}
		}
	}

	if inputs, err = matrix.FromSlice(len(centers)*samplesPerClass, 2, inputValues); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(len(centers)*samplesPerClass, classCount, targetValues); err != nil {
		return nil, err
	}

	dataset, err = data.NewDataset(inputs, targets)
	return dataset, err
}

func newClusterModel(random *rand.Rand) (network *model.Sequential, err error) {
	var (
		hidden           *layer.Dense
		hiddenActivation *layer.Activation
		output           *layer.Dense
		outputActivation *layer.Activation
	)

	if hidden, err = layer.NewDense(2, 8, layer.HeNormalWeights(random)); err != nil {
		return nil, err
	}

	if hiddenActivation, err = layer.NewActivation(activation.ReLU{}); err != nil {
		return nil, err
	}

	if output, err = layer.NewDense(8, classCount, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if outputActivation, err = layer.NewActivation(activation.Softmax{}); err != nil {
		return nil, err
	}

	network, err = model.NewSequential(hidden, hiddenActivation, output, outputActivation)
	return network, err
}

func printEpochMetrics(metrics model.EpochMetrics) (err error) {
	if metrics.Epoch == 1 || metrics.Epoch%logInterval == 0 || metrics.Epoch == epochs {
		fmt.Printf("epoch %4d loss %.6f accuracy %.3f\n", metrics.Epoch, metrics.Loss, metrics.Accuracy)
	}

	return nil
}

func printClassPredictions(network *model.Sequential) (err error) {
	var (
		centers          []point
		inputValues      []float32
		inputs           *matrix.Matrix
		predictions      *matrix.Matrix
		predictionValues []float32
		index            int
		center           point
		classIndex       int
	)

	centers = clusterCenters()
	inputValues = make([]float32, 0, len(centers)*2)
	for _, center = range centers {
		inputValues = append(inputValues, center.x, center.y)
	}

	if inputs, err = matrix.FromSlice(len(centers), 2, inputValues); err != nil {
		return err
	}

	if predictions, err = network.Predict(inputs); err != nil {
		return err
	}

	if predictionValues, err = predictions.Values(); err != nil {
		return err
	}

	fmt.Println("center predictions:")
	for index, center = range centers {
		classIndex = argmax(predictionValues[index*classCount : index*classCount+classCount])
		fmt.Printf("(%.1f, %.1f) => %s\n", center.x, center.y, className(classIndex))
	}

	return nil
}

func clusterCenters() (centers []point) {
	centers = []point{
		{x: -1.2, y: -0.8},
		{x: 1.2, y: -0.8},
		{x: 0, y: 1.2},
	}
	return centers
}

func className(index int) (name string) {
	switch index {
	case 0:
		name = "left"
	case 1:
		name = "right"
	case 2:
		name = "top"
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
