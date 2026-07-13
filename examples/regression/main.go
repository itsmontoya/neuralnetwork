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
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const (
	epochs       = 800
	logInterval  = 100
	sampleCount  = 41
	noiseStddev  = 0.03
	learningRate = 0.05
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	var (
		dataRandom    *rand.Rand
		modelRandom   *rand.Rand
		trainingData  *data.Dataset
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		history       model.TrainingHistory
	)

	dataRandom = rand.New(rand.NewSource(11))
	modelRandom = rand.New(rand.NewSource(13))

	if trainingData, err = newRegressionDataset(dataRandom); err != nil {
		return err
	}

	if network, err = newRegressionModel(modelRandom); err != nil {
		return err
	}

	if optimizerRule, err = optimizer.NewAdam(learningRate); err != nil {
		return err
	}

	history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:    epochs,
		BatchSize: sampleCount,
		Shuffle:   false,
		Optimizer: optimizerRule,
		Loss:      loss.MeanSquaredError{},
		Callback:  printEpochLoss,
	})
	if err != nil {
		return err
	}

	fmt.Printf("final loss %.6f\n", history.Epochs[len(history.Epochs)-1].Loss)
	err = printRegressionPredictions(network)
	return err
}

func newRegressionDataset(random *rand.Rand) (dataset *data.Dataset, err error) {
	var (
		inputValues  []float32
		targetValues []float32
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		index        int
		x            float32
		y            float32
		noise        float32
	)

	inputValues = make([]float32, 0, sampleCount)
	targetValues = make([]float32, 0, sampleCount)

	for index = 0; index < sampleCount; index++ {
		x = -1 + 2*float32(index)/float32(sampleCount-1)
		noise = float32(random.NormFloat64()) * noiseStddev
		y = 2*x + 1 + noise

		inputValues = append(inputValues, x)
		targetValues = append(targetValues, y)
	}

	if inputs, err = matrix.FromSlice(sampleCount, 1, inputValues); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(sampleCount, 1, targetValues); err != nil {
		return nil, err
	}

	dataset, err = data.NewDataset(inputs, targets)
	return dataset, err
}

func newRegressionModel(random *rand.Rand) (network *model.Sequential, err error) {
	var (
		output           *layer.Dense
		outputActivation *layer.Activation
	)

	if output, err = layer.NewDense(1, 1, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if outputActivation, err = layer.NewActivation(activation.Linear{}); err != nil {
		return nil, err
	}

	network, err = model.NewSequential(output, outputActivation)
	return network, err
}

func printEpochLoss(metrics model.EpochMetrics) (err error) {
	if metrics.Epoch == 1 || metrics.Epoch%logInterval == 0 || metrics.Epoch == epochs {
		fmt.Printf("epoch %4d loss %.6f\n", metrics.Epoch, metrics.Loss)
	}

	return nil
}

func printRegressionPredictions(network *model.Sequential) (err error) {
	var (
		inputValues      []float32
		inputs           *matrix.Matrix
		predictions      *matrix.Matrix
		predictionValues []float32
		index            int
		x                float32
	)

	inputValues = []float32{-1, 0, 1}
	if inputs, err = matrix.FromSlice(len(inputValues), 1, inputValues); err != nil {
		return err
	}

	if predictions, err = network.Predict(inputs); err != nil {
		return err
	}

	if predictionValues, err = predictions.Values(); err != nil {
		return err
	}

	fmt.Println("predictions:")
	for index, x = range inputValues {
		fmt.Printf("x %.1f target %.3f prediction %.3f\n", x, 2*x+1, predictionValues[index])
	}

	return nil
}
