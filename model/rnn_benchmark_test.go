package model_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Benchmark_SequentialTrainBatch_RNN(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule *optimizer.SGD
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		metrics       model.TrainMetrics
		err           error
		index         int
	)

	network = benchmarkRNNModel(b)
	inputs, targets = benchmarkSyntheticMatrices(b, 16, 8*16, 8)
	if optimizerRule, err = optimizer.NewSGD(0.001); err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}
	if _, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule); err != nil {
		b.Fatalf("warm-up TrainBatch returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		if metrics, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule); err != nil {
			b.Fatalf("TrainBatch returned error: %v", err)
		}
	}

	benchmarkTrainMetrics = metrics
}

func benchmarkRNNModel(tb testing.TB) (network *model.Sequential) {
	var (
		random      *rand.Rand
		inputShape  layer.SequenceShape
		config      layer.SimpleRNNConfig
		initializer layer.WeightInitializer
		recurrent   *layer.SimpleRNN
		lastStep    *layer.LastStep
		output      *layer.Dense
		err         error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(43))
	if inputShape, err = layer.NewSequenceShape(8, 16); err != nil {
		tb.Fatalf("NewSequenceShape returned error: %v", err)
	}
	if config, err = layer.NewSimpleRNNConfig(inputShape, 32); err != nil {
		tb.Fatalf("NewSimpleRNNConfig returned error: %v", err)
	}
	initializer = layer.UniformWeights(-0.1, 0.1, random)
	if recurrent, err = layer.NewSimpleRNN(config, initializer, initializer); err != nil {
		tb.Fatalf("NewSimpleRNN returned error: %v", err)
	}
	if lastStep, err = layer.NewLastStep(recurrent.OutputShape()); err != nil {
		tb.Fatalf("NewLastStep returned error: %v", err)
	}
	if output, err = layer.NewDense(lastStep.OutputSize(), 8, initializer); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(recurrent, lastStep, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}
