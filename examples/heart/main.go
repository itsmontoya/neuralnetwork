package main

import (
	"fmt"
	"log"
	"math"
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
	epochs       = 800
	logInterval  = 200
	sampleCount  = 1800
	batchSize    = 60
	learningRate = 0.01
	renderRows   = 29
	renderCols   = 61
	minX         = -1.5
	maxX         = 1.5
	minY         = -1.3
	maxY         = 1.4
)

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
		accuracyMetric metric.BinaryAccuracy
		finalMetrics   model.EpochMetrics
	)

	dataRandom = rand.New(rand.NewSource(31))
	modelRandom = rand.New(rand.NewSource(37))
	shuffleRandom = rand.New(rand.NewSource(41))

	if trainingData, err = newHeartDataset(dataRandom); err != nil {
		return err
	}

	if network, err = newHeartModel(modelRandom); err != nil {
		return err
	}

	if optimizerRule, err = optimizer.NewAdam(learningRate); err != nil {
		return err
	}

	history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:    epochs,
		BatchSize: batchSize,
		Shuffle:   true,
		Random:    shuffleRandom,
		Optimizer: optimizerRule,
		Loss:      loss.BinaryCrossEntropy{},
		Accuracy:  accuracyMetric.Value,
		Callback:  printEpochMetrics,
	})
	if err != nil {
		return err
	}

	finalMetrics = history.Epochs[len(history.Epochs)-1]
	fmt.Printf("final loss %.6f accuracy %.3f\n", finalMetrics.Loss, finalMetrics.Accuracy)
	fmt.Println()

	err = renderHeart(network)
	return err
}

func newHeartDataset(random *rand.Rand) (dataset *data.Dataset, err error) {
	var (
		inputValues  []float64
		targetValues []float64
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		index        int
		x            float64
		y            float64
		target       float64
	)

	inputValues = make([]float64, 0, sampleCount*2)
	targetValues = make([]float64, 0, sampleCount)

	for index = 0; index < sampleCount; index++ {
		x = minX + (maxX-minX)*random.Float64()
		y = minY + (maxY-minY)*random.Float64()
		target = 0
		if insideHeart(x, y) {
			target = 1
		}

		inputValues = append(inputValues, x, y)
		targetValues = append(targetValues, target)
	}

	if inputs, err = matrix.FromSlice(sampleCount, 2, inputValues); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(sampleCount, 1, targetValues); err != nil {
		return nil, err
	}

	dataset, err = data.NewDataset(inputs, targets)
	return dataset, err
}

func newHeartModel(random *rand.Rand) (network *model.Sequential, err error) {
	var (
		first            *layer.Dense
		firstActivation  *layer.Activation
		second           *layer.Dense
		secondActivation *layer.Activation
		output           *layer.Dense
		outputActivation *layer.Activation
	)

	if first, err = layer.NewDense(2, 24, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if firstActivation, err = layer.NewActivation(activation.Tanh{}); err != nil {
		return nil, err
	}

	if second, err = layer.NewDense(24, 24, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if secondActivation, err = layer.NewActivation(activation.Tanh{}); err != nil {
		return nil, err
	}

	if output, err = layer.NewDense(24, 1, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if outputActivation, err = layer.NewActivation(activation.Sigmoid{}); err != nil {
		return nil, err
	}

	network, err = model.NewSequential(first, firstActivation, second, secondActivation, output, outputActivation)
	return network, err
}

func printEpochMetrics(metrics model.EpochMetrics) (err error) {
	if metrics.Epoch == 1 || metrics.Epoch%logInterval == 0 || metrics.Epoch == epochs {
		fmt.Printf("epoch %4d loss %.6f accuracy %.3f\n", metrics.Epoch, metrics.Loss, metrics.Accuracy)
	}

	return nil
}

func renderHeart(network *model.Sequential) (err error) {
	var (
		inputValues      []float64
		inputs           *matrix.Matrix
		predictions      *matrix.Matrix
		predictionValues []float64
		row              int
		col              int
		x                float64
		y                float64
		index            int
	)

	inputValues = make([]float64, 0, renderRows*renderCols*2)
	for row = 0; row < renderRows; row++ {
		y = maxY - (maxY-minY)*float64(row)/float64(renderRows-1)
		for col = 0; col < renderCols; col++ {
			x = minX + (maxX-minX)*float64(col)/float64(renderCols-1)
			inputValues = append(inputValues, x, y)
		}
	}

	if inputs, err = matrix.FromSlice(renderRows*renderCols, 2, inputValues); err != nil {
		return err
	}

	if predictions, err = network.Predict(inputs); err != nil {
		return err
	}

	if predictionValues, err = predictions.Values(); err != nil {
		return err
	}

	for row = 0; row < renderRows; row++ {
		for col = 0; col < renderCols; col++ {
			index = row*renderCols + col
			fmt.Print(shade(predictionValues[index]))
		}
		fmt.Println()
	}

	return nil
}

func insideHeart(x, y float64) (inside bool) {
	var value float64

	y += 0.12
	value = math.Pow(x*x+y*y-1, 3) - x*x*y*y*y
	inside = value <= 0
	return inside
}

func shade(value float64) (char string) {
	switch {
	case value >= 0.85:
		char = "@"
	case value >= 0.65:
		char = "#"
	case value >= 0.45:
		char = "+"
	case value >= 0.25:
		char = "."
	default:
		char = " "
	}

	return char
}
