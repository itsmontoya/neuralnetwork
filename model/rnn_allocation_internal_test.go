package model

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_Sequential_RNNTrainBatchDoesNotAllocateAfterWarmUp(t *testing.T) {
	var (
		inputShape    layer.SequenceShape
		config        layer.SimpleRNNConfig
		recurrent     *layer.SimpleRNN
		lastStep      *layer.LastStep
		output        *layer.Dense
		network       *Sequential
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		optimizerRule *optimizer.SGD
		allocations   float64
		err           error
	)

	if inputShape, err = layer.NewSequenceShape(8, 16); err != nil {
		t.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if config, err = layer.NewSimpleRNNConfig(inputShape, 32); err != nil {
		t.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	if recurrent, err = layer.NewSimpleRNN(config, layer.ZeroWeights, layer.ZeroWeights); err != nil {
		t.Fatalf("NewSimpleRNN returned error: %v", err)
	}
	if lastStep, err = layer.NewLastStep(recurrent.OutputShape()); err != nil {
		t.Fatalf("NewLastStep returned error: %v", err)
	}
	if output, err = layer.NewDense(lastStep.OutputSize(), 8, layer.ZeroWeights); err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = NewSequential(recurrent, lastStep, output); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if inputs, err = matrix.New(16, inputShape.Size()); err != nil {
		t.Fatalf("New inputs returned error: %v", err)
	}
	if targets, err = matrix.New(16, output.OutputSize()); err != nil {
		t.Fatalf("New targets returned error: %v", err)
	}
	if optimizerRule, err = optimizer.NewSGD(0.001); err != nil {
		t.Fatalf("NewSGD returned error: %v", err)
	}
	if _, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule); err != nil {
		t.Fatalf("warm-up TrainBatch returned error: %v", err)
	}

	allocations = testing.AllocsPerRun(100, func() {
		allocationTrainMetrics, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule)
		if err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf("warmed RNN TrainBatch allocations = %g, want 0", allocations)
	}
}
