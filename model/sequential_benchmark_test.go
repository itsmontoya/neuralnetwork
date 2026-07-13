package model_test

import (
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

var benchmarkTrainMetrics model.TrainMetrics
var benchmarkTrainingHistory model.TrainingHistory

func Benchmark_SequentialTrainBatch_XOR(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		metrics       model.TrainMetrics
		err           error
		index         int
	)

	inputs, targets = benchmarkXORMatrices(b)
	network = benchmarkXORModel(b)
	optimizerRule, err = optimizer.NewAdam(0.05)
	if err != nil {
		b.Fatalf("NewAdam returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		metrics, err = network.TrainBatch(inputs, targets, loss.BinaryCrossEntropy{}, optimizerRule)
		if err != nil {
			b.Fatalf("TrainBatch returned error: %v", err)
		}
	}

	benchmarkTrainMetrics = metrics
}

func Benchmark_SequentialFit_XOR(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		dataset       *data.Dataset
		config        model.FitConfig
		history       model.TrainingHistory
		err           error
		index         int
	)

	inputs, targets = benchmarkXORMatrices(b)
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		b.Fatalf("NewDataset returned error: %v", err)
	}

	network = benchmarkXORModel(b)
	optimizerRule, err = optimizer.NewAdam(0.05)
	if err != nil {
		b.Fatalf("NewAdam returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = 4
	config.Optimizer = optimizerRule
	config.Loss = loss.BinaryCrossEntropy{}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		history, err = network.Fit(dataset, config)
		if err != nil {
			b.Fatalf("Fit returned error: %v", err)
		}
	}

	benchmarkTrainingHistory = history
}

func Benchmark_SequentialTrainBatch_SyntheticDense(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		metrics       model.TrainMetrics
		err           error
		index         int
	)

	inputs, targets = benchmarkSyntheticMatrices(b, 128, 32, 16)
	network = benchmarkSyntheticModel(b)
	optimizerRule, err = optimizer.NewSGD(0.01)
	if err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		metrics, err = network.TrainBatch(inputs, targets, loss.MeanSquaredError{}, optimizerRule)
		if err != nil {
			b.Fatalf("TrainBatch returned error: %v", err)
		}
	}

	benchmarkTrainMetrics = metrics
}

func Benchmark_SequentialFit_SyntheticDense(b *testing.B) {
	var (
		network       *model.Sequential
		optimizerRule optimizer.Optimizer
		inputs        *matrix.Matrix
		targets       *matrix.Matrix
		dataset       *data.Dataset
		config        model.FitConfig
		history       model.TrainingHistory
		err           error
		index         int
	)

	inputs, targets = benchmarkSyntheticMatrices(b, 128, 32, 16)
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		b.Fatalf("NewDataset returned error: %v", err)
	}

	network = benchmarkSyntheticModel(b)
	optimizerRule, err = optimizer.NewSGD(0.01)
	if err != nil {
		b.Fatalf("NewSGD returned error: %v", err)
	}

	config.Epochs = 1
	config.BatchSize = 32
	config.Optimizer = optimizerRule
	config.Loss = loss.MeanSquaredError{}

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		history, err = network.Fit(dataset, config)
		if err != nil {
			b.Fatalf("Fit returned error: %v", err)
		}
	}

	benchmarkTrainingHistory = history
}

func benchmarkXORMatrices(tb testing.TB) (inputs, targets *matrix.Matrix) {
	var err error

	tb.Helper()

	if inputs, err = matrix.FromSlice(4, 2, []float32{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	if targets, err = matrix.FromSlice(4, 1, []float32{
		0,
		1,
		1,
		0,
	}); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return inputs, targets
}

func benchmarkXORModel(tb testing.TB) (network *model.Sequential) {
	var (
		random           *rand.Rand
		hidden           *layer.Dense
		hiddenActivation *layer.Activation
		output           *layer.Dense
		outputActivation *layer.Activation
		err              error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(1))
	if hidden, err = layer.NewDense(2, 4, layer.XavierUniformWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if hiddenActivation, err = layer.NewActivation(activation.Tanh{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	if output, err = layer.NewDense(4, 1, layer.XavierUniformWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if outputActivation, err = layer.NewActivation(activation.Sigmoid{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	if network, err = model.NewSequential(hidden, hiddenActivation, output, outputActivation); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}

func benchmarkSyntheticMatrices(tb testing.TB, samples, inputSize, targetSize int) (inputs, targets *matrix.Matrix) {
	var (
		inputValues  []float32
		targetValues []float32
		row          int
		col          int
		err          error
	)

	tb.Helper()

	inputValues = make([]float32, samples*inputSize)
	for row = 0; row < samples; row++ {
		for col = 0; col < inputSize; col++ {
			inputValues[row*inputSize+col] = float32((row+1)*(col+3)%17) / 17
		}
	}

	targetValues = make([]float32, samples*targetSize)
	for row = 0; row < samples; row++ {
		for col = 0; col < targetSize; col++ {
			targetValues[row*targetSize+col] = float32((row+col*2)%11) / 11
		}
	}

	if inputs, err = matrix.FromSlice(samples, inputSize, inputValues); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	if targets, err = matrix.FromSlice(samples, targetSize, targetValues); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return inputs, targets
}

func benchmarkSyntheticModel(tb testing.TB) (network *model.Sequential) {
	var (
		random           *rand.Rand
		hidden           *layer.Dense
		hiddenActivation *layer.Activation
		output           *layer.Dense
		err              error
	)

	tb.Helper()

	random = rand.New(rand.NewSource(3))
	if hidden, err = layer.NewDense(32, 64, layer.HeNormalWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if hiddenActivation, err = layer.NewActivation(activation.ReLU{}); err != nil {
		tb.Fatalf("NewActivation returned error: %v", err)
	}

	if output, err = layer.NewDense(64, 16, layer.XavierUniformWeights(random)); err != nil {
		tb.Fatalf("NewDense returned error: %v", err)
	}

	if network, err = model.NewSequential(hidden, hiddenActivation, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}

	return network
}
