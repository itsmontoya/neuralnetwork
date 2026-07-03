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
	epochs      = 5000
	logInterval = 500
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	var (
		random           *rand.Rand
		trainingData     *data.Dataset
		network          *model.Sequential
		optimizerRule    optimizer.Optimizer
		history          model.TrainingHistory
		inputs           *matrix.Matrix
		inputValues      []float64
		predictions      *matrix.Matrix
		predictionValues []float64
		index            int
	)

	random = rand.New(rand.NewSource(1))

	if trainingData, err = newXORDataset(); err != nil {
		return err
	}

	if network, err = newXORModel(random); err != nil {
		return err
	}

	if optimizerRule, err = optimizer.NewAdam(0.05); err != nil {
		return err
	}

	history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:    epochs,
		BatchSize: 4,
		Shuffle:   false,
		Optimizer: optimizerRule,
		Loss:      loss.BinaryCrossEntropy{},
		Callback:  printEpochLoss,
	})
	if err != nil {
		return err
	}

	fmt.Printf("final loss %.6f\n", history.Epochs[len(history.Epochs)-1].Loss)
	fmt.Println("predictions:")

	if inputs, err = trainingData.Inputs(); err != nil {
		return err
	}

	if inputValues, err = inputs.Values(); err != nil {
		return err
	}

	if predictions, err = network.Predict(inputs); err != nil {
		return err
	}

	if predictionValues, err = predictions.Values(); err != nil {
		return err
	}

	for index = range predictionValues {
		fmt.Printf("%.0f xor %.0f = %.4f\n", inputValues[index*2], inputValues[index*2+1], predictionValues[index])
	}

	return nil
}

func newXORDataset() (dataset *data.Dataset, err error) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
	)

	if inputs, err = matrix.FromSlice(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(4, 1, []float64{
		0,
		1,
		1,
		0,
	}); err != nil {
		return nil, err
	}

	dataset, err = data.NewDataset(inputs, targets)
	return dataset, err
}

func newXORModel(random *rand.Rand) (network *model.Sequential, err error) {
	var (
		hidden           *layer.Dense
		hiddenActivation *layer.Activation
		output           *layer.Dense
		outputActivation *layer.Activation
	)

	if hidden, err = layer.NewDense(2, 4, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if hiddenActivation, err = layer.NewActivation(activation.Tanh{}); err != nil {
		return nil, err
	}

	if output, err = layer.NewDense(4, 1, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if outputActivation, err = layer.NewActivation(activation.Sigmoid{}); err != nil {
		return nil, err
	}

	network, err = model.NewSequential(hidden, hiddenActivation, output, outputActivation)
	return network, err
}

func printEpochLoss(metrics model.EpochMetrics) (err error) {
	if metrics.Epoch == 1 || metrics.Epoch%logInterval == 0 || metrics.Epoch == epochs {
		fmt.Printf("epoch %4d loss %.6f\n", metrics.Epoch, metrics.Loss)
	}

	return nil
}
